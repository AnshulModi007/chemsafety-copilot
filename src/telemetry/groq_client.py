"""An instrumented stand-in for `groq.Groq()`.

This is the whole reason the instrumentation is non-invasive. Every module in
the pipeline does `_client = Groq()` at import time and then calls
`_client.chat.completions.create(...)`. Swapping that one line for
`instrumented_groq()` captures the model, token counts, and latency of every
Groq request in the pipeline without touching a single line of business logic.

Token usage was already being returned by Groq and thrown away at all 12 call
sites. Verified against the installed SDK (0.37.1):
  * non-streaming -> `response.usage` carries prompt/completion/total_tokens
    plus Groq's own `total_time` server-side inference figure;
  * streaming -> the final chunk carries the same `usage` object with no
    `stream_options` needed (that kwarg is not accepted by this SDK version).

Failure policy: instrumentation never changes call semantics. If recording
raises for any reason, the underlying response is still returned unchanged.
"""
import logging
import time
from typing import Any, Iterator

from groq import Groq

from src.telemetry.context import current_scope
from src.telemetry.record import LlmCall

logger = logging.getLogger(__name__)


def _server_time_ms(usage: Any) -> float | None:
    total_time = getattr(usage, "total_time", None)
    return total_time * 1000.0 if isinstance(total_time, (int, float)) else None


def _record(model: str, usage: Any, latency_ms: float, *, streamed: bool) -> None:
    """Attach one LlmCall to the active scope. Silent no-op outside a request.

    The entire body is guarded, including the scope lookup: this runs inside
    every LLM call in the pipeline, and an exception escaping here would turn a
    telemetry problem into a failed user query.
    """
    try:
        scope = current_scope()
        if scope is None:
            return
        scope.add_llm_call(
            LlmCall(
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                latency_ms=latency_ms,
                server_time_ms=_server_time_ms(usage),
                stage=scope.current_stage,
                streamed=streamed,
            )
        )
    except Exception:
        logger.warning("failed to record LLM call telemetry", exc_info=True)


class _InstrumentedCompletions:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", "unknown"))
        started = time.perf_counter()

        response = self._inner.create(*args, **kwargs)

        if kwargs.get("stream"):
            # Cannot measure here -- the request has only been opened. Wrap the
            # iterator so timing ends when the caller finishes consuming it.
            return _instrumented_stream(response, model, started)

        _record(model, getattr(response, "usage", None), (time.perf_counter() - started) * 1000.0,
                streamed=False)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _instrumented_stream(stream: Any, model: str, started: float) -> Iterator[Any]:
    """Pass chunks through untouched, keeping the last usage-bearing one.

    Recorded in a `finally` so an abandoned stream (client disconnect, or a
    caller that breaks early) still contributes what it used rather than
    vanishing from the cost figures.
    """
    usage = None
    try:
        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            yield chunk
    finally:
        _record(model, usage, (time.perf_counter() - started) * 1000.0, streamed=True)


class _InstrumentedChat:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.completions = _InstrumentedCompletions(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class InstrumentedGroq:
    """Proxy exposing the same surface as `Groq`, with `chat.completions.create`
    instrumented. Any other attribute falls through to the real client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._client = Groq(*args, **kwargs)
        self.chat = _InstrumentedChat(self._client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def instrumented_groq(*args: Any, **kwargs: Any) -> InstrumentedGroq:
    return InstrumentedGroq(*args, **kwargs)
