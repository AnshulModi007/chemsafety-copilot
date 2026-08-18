"""Per-query telemetry scope.

Why a ContextVar and not a global: the FastAPI handlers are sync `def`, so
Starlette runs each one in an anyio worker thread. A module-level global would
interleave concurrent queries' LLM calls into one another's records; a
ContextVar gives each request its own accumulator, and anyio copies the calling
context into the worker thread so the handler sees it correctly.

The streaming path needs more care. `StreamingResponse` over a sync generator
drives it via `anyio.to_thread.run_sync(next, gen)` once per chunk, and each of
those steps starts from the *async task's* context -- so a ContextVar set inside
the generator is not visible on the following step, and may land on a different
worker thread. `reactivating()` solves that by re-binding the scope around every
single `next()`, which is correct regardless of which thread serves it.
"""
import contextvars
import time
import uuid
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

from src.telemetry.record import LlmCall, QueryTelemetry


class QueryScope:
    """Mutable accumulator for one in-flight query."""

    def __init__(self, *, streamed: bool = False) -> None:
        self.query_id = uuid.uuid4().hex
        self.started_at = time.perf_counter()
        self.streamed = streamed
        self.llm_calls: list[LlmCall] = []
        self.stage_latency_ms: dict[str, float] = {}
        # Captured as soon as the router decides, so a query that fails later
        # is still attributable to a branch. Reading intent only off the final
        # response would file every failure under "unrouted" and make
        # "which intent fails most" unanswerable -- the question an error
        # breakdown exists to answer.
        self.intent: str | None = None
        # Stack, not a single value: retrieval opens around rerank, and a
        # comparative query re-enters the same stage once per sub-query.
        self._stage_stack: list[str] = []

    @property
    def current_stage(self) -> str | None:
        return self._stage_stack[-1] if self._stage_stack else None

    def add_llm_call(self, call: LlmCall) -> None:
        self.llm_calls.append(call)

    def add_stage_time(self, stage: str, elapsed_ms: float) -> None:
        # Accumulate rather than overwrite: a comparative query runs retrieval
        # once per sub-query, and the total time in retrieval is what matters.
        self.stage_latency_ms[stage] = self.stage_latency_ms.get(stage, 0.0) + elapsed_ms

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0


_current: contextvars.ContextVar[QueryScope | None] = contextvars.ContextVar(
    "chemsafety_telemetry_scope", default=None
)


def current_scope() -> QueryScope | None:
    """The active scope, or None when the pipeline runs outside a request
    (CLI entrypoints, tests, the eval harness) -- in which case every
    instrumentation hook becomes a no-op."""
    return _current.get()


def _bind(scope: QueryScope | None) -> None:
    """Set the active scope by value.

    Deliberately not `ContextVar.reset(token)`: a token may only be reset in the
    exact Context that created it, and both the streaming generator and the
    handler that wraps it can be resumed on a different anyio worker thread --
    which raises "Token was created in a different Context" and kills the
    response mid-stream. Assigning the previous value has the same effect here
    and is safe across contexts.
    """
    _current.set(scope)


@contextmanager
def query_scope(*, streamed: bool = False) -> Iterator[QueryScope]:
    """Open a telemetry scope for the duration of one query."""
    scope = QueryScope(streamed=streamed)
    previous = _current.get()
    _bind(scope)
    try:
        yield scope
    finally:
        _bind(previous)


_T = TypeVar("_T")


def reactivating(scope: QueryScope, iterable: Iterable[_T]) -> Iterator[_T]:
    """Re-bind `scope` around each step of `iterable`.

    Required for the SSE path: Starlette pulls one item at a time via
    `anyio.to_thread.run_sync`, so without this the scope set on step N is gone
    by step N+1 and every LLM call made mid-stream would be dropped.
    """
    iterator = iter(iterable)
    while True:
        previous = _current.get()
        _bind(scope)
        try:
            item = next(iterator)
        except StopIteration:
            _bind(previous)
            return
        except BaseException:
            _bind(previous)
            raise
        # Restored before yielding: the consumer's own work between steps is
        # not part of this query, and any LLM call happens inside next().
        _bind(previous)
        yield item


def build_record(
    scope: QueryScope,
    *,
    timestamp: str,
    query_text: str | None,
    response: dict | None,
    error: BaseException | None = None,
) -> QueryTelemetry:
    """Fold a finished scope plus the response envelope into one row.

    Every quality signal here is read off the envelope the pipeline already
    returns (see src/agent/copilot.py) -- none of it is re-derived or inferred.
    """
    from src.telemetry.record import classify_exception

    record = QueryTelemetry(
        query_id=scope.query_id,
        timestamp=timestamp,
        query_text=query_text,
        total_latency_ms=scope.elapsed_ms(),
        stage_latency_ms=dict(scope.stage_latency_ms),
        llm_calls=list(scope.llm_calls),
        streamed=scope.streamed,
        success=error is None,
    )

    # Whatever the router decided, even if the query failed downstream.
    record.intent = scope.intent

    if error is not None:
        record.error_category = classify_exception(error)
        record.error_type = type(error).__name__
        return record

    if not response:
        return record

    data = response.get("data") or {}
    record.intent = response.get("intent") or record.intent
    record.resolved_query = response.get("resolved_query")
    record.from_cache = bool(response.get("from_cache"))

    record.crag_insufficient = bool(data.get("crag_insufficient"))
    # A non-null rewritten query is the only signal that the retry actually ran.
    record.crag_retry_fired = data.get("crag_rewritten_query") is not None
    record.web_fallback_fired = data.get("source") == "web"

    faithfulness = data.get("faithfulness")
    if isinstance(faithfulness, dict):
        record.faithful = bool(faithfulness.get("faithful"))
        record.unsupported_claim_count = len(faithfulness.get("unsupported_claims") or [])

    confidence = data.get("confidence")
    if isinstance(confidence, (int, float)):
        record.retrieval_confidence = float(confidence)

    sources = data.get("sources") or []
    record.source_count = len(sources)
    scores = [s.get("rerank_score") for s in sources if isinstance(s.get("rerank_score"), (int, float))]
    if scores:
        record.top_rerank_score = float(max(scores))

    return record
