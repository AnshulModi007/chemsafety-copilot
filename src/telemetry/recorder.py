"""The request-path entrypoints for telemetry.

Two shapes, because the two endpoints differ structurally:

  * `record_query()` -- a context manager for the plain `/ask` handler, which
    runs start to finish inside one worker thread.
  * `instrumented_stream()` -- a generator wrapper for `/ask/stream`, whose
    body runs *after* the handler returns, one step at a time, potentially on
    different worker threads. It finalises at the terminal `done` event.

Neither can raise into the request. Every failure path here logs and returns.
"""
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from config import LOG_QUERY_TEXT
from src.telemetry.context import QueryScope, build_record, query_scope, reactivating
from src.telemetry.store import writer

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist(
    scope: QueryScope,
    *,
    timestamp: str,
    query_text: str | None,
    response: dict | None,
    error: BaseException | None = None,
) -> None:
    """Build and hand off a record. Swallows everything: a telemetry failure
    must never surface to the user, and must never mask a real error that is
    already propagating."""
    try:
        record = build_record(
            scope,
            timestamp=timestamp,
            query_text=query_text if LOG_QUERY_TEXT else None,
            response=response,
            error=error,
        )
        writer.submit(record)
    except Exception:
        logger.warning("failed to build telemetry record", exc_info=True)


class _Outcome:
    """Lets the caller hand back the response envelope once it exists, or the
    underlying exception when it intends to re-raise a translated one."""

    def __init__(self) -> None:
        self.response: dict | None = None
        self.error: BaseException | None = None

    def set_response(self, response: dict | None) -> None:
        self.response = response

    def set_error(self, error: BaseException) -> None:
        """Record the *original* failure.

        The handler translates pipeline exceptions into HTTPException for the
        client. If telemetry only saw the translated one, every failure would
        classify as "validation" and the error breakdown would be useless --
        a Groq outage has to read as `llm_provider`.
        """
        self.error = error


@contextmanager
def record_query(query_text: str) -> Iterator[_Outcome]:
    """Wrap one non-streaming query. Re-raises whatever the body raised, after
    recording it as a failed row."""
    timestamp = _now()
    outcome = _Outcome()
    with query_scope(streamed=False) as scope:
        try:
            yield outcome
        except BaseException as exc:
            _persist(scope, timestamp=timestamp, query_text=query_text,
                     response=None, error=outcome.error or exc)
            raise
        _persist(scope, timestamp=timestamp, query_text=query_text,
                 response=outcome.response, error=outcome.error)


def instrumented_stream(query_text: str, events: Any) -> Iterator[tuple[str, Any]]:
    """Wrap the (kind, payload) generator behind `/ask/stream`.

    `reactivating` re-binds the telemetry scope around every step, because
    Starlette pulls one item at a time via `anyio.to_thread.run_sync` and a
    ContextVar set on one step is not visible on the next -- without it, every
    LLM call made mid-stream would be attributed to no query at all.
    """
    timestamp = _now()
    with query_scope(streamed=True) as scope:
        response: dict | None = None
        try:
            for kind, payload in reactivating(scope, events):
                if kind == "done" and isinstance(payload, dict):
                    response = payload
                yield (kind, payload)
        except BaseException as exc:
            _persist(scope, timestamp=timestamp, query_text=query_text,
                     response=None, error=exc)
            raise
        _persist(scope, timestamp=timestamp, query_text=query_text,
                 response=response, error=None)
