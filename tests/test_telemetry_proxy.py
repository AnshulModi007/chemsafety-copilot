"""Tests for the instrumented Groq client.

The proxy sits on the request path of every LLM call in the pipeline, so the
thing that matters most is that it is transparent: identical return values,
identical exceptions, identical streaming semantics. Capturing telemetry is the
secondary job, and it must never be the reason a query fails.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.telemetry import groq_client  # noqa: E402
from src.telemetry.context import current_scope, query_scope  # noqa: E402
from src.telemetry.stages import stage  # noqa: E402


def _usage(prompt=100, completion=20, total_time=0.05):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        total_tokens=prompt + completion, total_time=total_time,
    )


class FakeCompletions:
    """Stands in for groq's completions resource."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


def _proxy(completions):
    return groq_client._InstrumentedCompletions(completions)


def test_non_streaming_call_returns_the_response_untouched():
    response = SimpleNamespace(usage=_usage(), choices=["original"])
    proxy = _proxy(FakeCompletions(result=response))

    with query_scope():
        returned = proxy.create(model="openai/gpt-oss-20b", messages=[])

    assert returned is response  # same object, not a copy or wrapper


def test_non_streaming_call_records_model_tokens_and_latency():
    proxy = _proxy(FakeCompletions(result=SimpleNamespace(usage=_usage(150, 30, 0.25))))

    with query_scope() as scope:
        proxy.create(model="openai/gpt-oss-120b", messages=[])

    assert len(scope.llm_calls) == 1
    call = scope.llm_calls[0]
    assert call.model == "openai/gpt-oss-120b"
    assert call.prompt_tokens == 150
    assert call.completion_tokens == 30
    assert call.total_tokens == 180
    assert call.latency_ms >= 0
    assert call.server_time_ms == pytest.approx(250.0)  # total_time seconds -> ms
    assert call.streamed is False


def test_calls_are_attributed_to_the_enclosing_stage():
    proxy = _proxy(FakeCompletions(result=SimpleNamespace(usage=_usage())))

    with query_scope() as scope:
        with stage("routing"):
            proxy.create(model="openai/gpt-oss-20b", messages=[])
        with stage("generation"):
            proxy.create(model="openai/gpt-oss-120b", messages=[])
        proxy.create(model="openai/gpt-oss-20b", messages=[])  # outside any stage

    assert [c.stage for c in scope.llm_calls] == ["routing", "generation", None]


def test_nested_stages_attribute_to_the_innermost():
    proxy = _proxy(FakeCompletions(result=SimpleNamespace(usage=_usage())))

    with query_scope() as scope:
        with stage("retrieval"):
            with stage("rerank"):
                proxy.create(model="openai/gpt-oss-20b", messages=[])

    assert scope.llm_calls[0].stage == "rerank"
    assert set(scope.stage_latency_ms) == {"retrieval", "rerank"}


def test_streaming_passes_chunks_through_and_records_the_final_usage():
    chunks = [
        SimpleNamespace(usage=None, delta="a"),
        SimpleNamespace(usage=None, delta="b"),
        # Groq attaches usage to the final chunk -- verified against SDK 0.37.1.
        SimpleNamespace(usage=_usage(75, 40, 0.049), delta=""),
    ]
    proxy = _proxy(FakeCompletions(result=iter(chunks)))

    with query_scope() as scope:
        stream = proxy.create(model="openai/gpt-oss-20b", messages=[], stream=True)
        # Nothing is recorded until the caller actually consumes the stream.
        assert scope.llm_calls == []
        received = list(stream)

    assert received == chunks
    assert len(scope.llm_calls) == 1
    call = scope.llm_calls[0]
    assert call.prompt_tokens == 75
    assert call.completion_tokens == 40
    assert call.streamed is True


def test_an_abandoned_stream_still_records_what_it_used():
    """A client disconnect or an early break must not make the tokens vanish
    from the cost figures."""
    chunks = [SimpleNamespace(usage=_usage(60, 5), delta="a"),
              SimpleNamespace(usage=None, delta="b")]
    proxy = _proxy(FakeCompletions(result=iter(chunks)))

    with query_scope() as scope:
        stream = proxy.create(model="openai/gpt-oss-20b", messages=[], stream=True)
        next(stream)
        stream.close()  # abandon before exhaustion

    assert len(scope.llm_calls) == 1
    assert scope.llm_calls[0].prompt_tokens == 60


def test_provider_errors_propagate_unchanged():
    error = RuntimeError("Error code: 413 - request too large")
    proxy = _proxy(FakeCompletions(error=error))

    with query_scope() as scope:
        with pytest.raises(RuntimeError, match="413"):
            proxy.create(model="openai/gpt-oss-120b", messages=[])

    # A call that never returned has no usage to record.
    assert scope.llm_calls == []


def test_a_response_without_usage_is_still_returned():
    """Some responses carry no usage object; the call must succeed anyway and
    be recorded with zero tokens rather than dropped or raised on."""
    response = SimpleNamespace(choices=["x"])  # no .usage attribute at all
    proxy = _proxy(FakeCompletions(result=response))

    with query_scope() as scope:
        assert proxy.create(model="openai/gpt-oss-20b", messages=[]) is response

    assert scope.llm_calls[0].total_tokens == 0


def test_recording_failure_never_breaks_the_call(monkeypatch):
    response = SimpleNamespace(usage=_usage())
    proxy = _proxy(FakeCompletions(result=response))

    def explode(*_a, **_k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(groq_client, "current_scope", explode)
    # No scope needed: the point is that the proxy swallows its own failure.
    assert proxy.create(model="openai/gpt-oss-20b", messages=[]) is response


def test_proxy_is_inert_outside_a_request_scope():
    proxy = _proxy(FakeCompletions(result=SimpleNamespace(usage=_usage())))
    assert current_scope() is None
    proxy.create(model="openai/gpt-oss-20b", messages=[])  # must not raise


def test_kwargs_reach_the_underlying_client_unmodified():
    """Instrumentation must not alter the request -- response_format and
    stream in particular drive the pipeline's parsing contract."""
    fake = FakeCompletions(result=SimpleNamespace(usage=_usage()))
    proxy = _proxy(fake)

    with query_scope():
        proxy.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )

    assert fake.calls[0] == {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    }


def test_concurrent_queries_do_not_leak_calls_into_each_other():
    """Handlers are sync `def`, so FastAPI runs them on separate worker
    threads; a ContextVar (not a global) is what keeps records separate."""
    import threading

    proxy = _proxy(FakeCompletions(result=SimpleNamespace(usage=_usage())))
    counts = {}

    def run(name, n):
        with query_scope() as scope:
            for _ in range(n):
                proxy.create(model="openai/gpt-oss-20b", messages=[])
            counts[name] = len(scope.llm_calls)

    threads = [threading.Thread(target=run, args=(f"t{i}", i + 1)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counts == {"t0": 1, "t1": 2, "t2": 3, "t3": 4, "t4": 5}


def test_streaming_survives_each_step_running_on_a_different_thread():
    """Starlette drives an SSE generator with `anyio.to_thread.run_sync(next,
    gen)`, so consecutive steps run on different worker threads in different
    Contexts. This drives it the same way -- one step per thread, each from a
    fresh copy of the caller's context -- and asserts that every mid-stream LLM
    call is still attributed to the right query.
    """
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from src.telemetry.context import QueryScope, reactivating

    proxy = _proxy(FakeCompletions(result=SimpleNamespace(usage=_usage())))
    scope = QueryScope(streamed=True)

    def producer():
        for i in range(4):
            proxy.create(model="openai/gpt-oss-120b", messages=[])
            yield i

    wrapped = reactivating(scope, producer())
    received = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            # Mirrors anyio.to_thread.run_sync: a fresh copy of the current
            # context, executed on some other thread.
            ctx = contextvars.copy_context()
            try:
                received.append(pool.submit(ctx.run, next, wrapped).result())
            except StopIteration:
                break

    assert received == [0, 1, 2, 3]
    assert len(scope.llm_calls) == 4  # every mid-stream call was attributed


def test_query_scope_can_be_exited_from_a_different_context():
    """Regression test for the bug that broke live streaming.

    The scope wrapping the whole SSE response is entered on one worker thread
    and exited on another once the generator finishes. Restoring it with
    `ContextVar.reset(token)` raised "Token was created in a different Context"
    and killed the response *after* 110 deltas had already reached the client.
    Verified to fail against the previous token-based implementation.
    """
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from src.telemetry.context import query_scope

    cm = query_scope(streamed=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        scope = pool.submit(contextvars.copy_context().run, cm.__enter__).result()
        assert scope is not None
        # Must not raise.
        pool.submit(contextvars.copy_context().run, cm.__exit__, None, None, None).result()


def test_reactivating_keeps_the_scope_bound_across_generator_steps():
    """Mirrors what Starlette does to the SSE generator: each step is pulled
    separately, and without re-binding the scope is lost after step one."""
    from src.telemetry.context import QueryScope, reactivating

    scope = QueryScope(streamed=True)
    seen = []

    def producer():
        for i in range(3):
            seen.append(current_scope())
            yield i

    assert list(reactivating(scope, producer())) == [0, 1, 2]
    assert seen == [scope, scope, scope]
    assert current_scope() is None  # and it is unbound again afterwards
