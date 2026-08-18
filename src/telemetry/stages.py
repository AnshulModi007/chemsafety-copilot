"""Per-stage latency timing.

Per-stage numbers are the point of the whole exercise -- a single
end-to-end figure cannot tell you whether a slow query is slow because of
retrieval, a CRAG retry, or generation. This is the one part of the
instrumentation that has to appear inside the pipeline, so it is deliberately
kept to a `with` wrapper around calls that already exist: it reads as a label on
existing work rather than logging interleaved with logic.

A no-op outside a request scope, so CLI entrypoints, the eval harness, and unit
tests are unaffected.
"""
import time
from contextlib import contextmanager
from typing import Iterator

from src.telemetry.context import current_scope


def note_intent(intent: str) -> None:
    """Record the router's decision the moment it is made.

    Called from the orchestrator right after classification so a query that
    fails during retrieval or generation is still filed under the branch it was
    routed to. A no-op outside a request scope, and it never raises.
    """
    try:
        scope = current_scope()
        if scope is not None:
            scope.intent = intent
    except Exception:
        pass


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Attribute the enclosed work, and any LLM calls inside it, to `name`."""
    scope = current_scope()
    if scope is None:
        yield
        return

    scope._stage_stack.append(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        scope._stage_stack.pop()
        # Recorded even when the body raised: a stage that fails still consumed
        # time, and hiding it would misattribute the latency of failed queries.
        scope.add_stage_time(name, elapsed_ms)
