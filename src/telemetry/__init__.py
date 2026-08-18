"""Production observability for the ChemSafety Copilot pipeline.

Instrumentation is deliberately kept out of business logic:

  * `groq_client.instrumented_groq()` is a drop-in for `Groq()`. Swapping the
    one-line client construction in each pipeline module captures model, token
    usage, and latency for every LLM call without touching any logic.
  * `stages.stage(name)` is a `with` wrapper for per-stage latency -- the only
    piece that appears inside the pipeline, and it wraps existing calls rather
    than interleaving logging with them.
  * `context.query_scope()` opens a per-request accumulator; everything is a
    no-op outside one, so CLI entrypoints, tests, and the eval harness are
    unaffected.
  * `store.writer` persists asynchronously on a background thread, so a metrics
    failure can never slow or break a user's query.
"""
from src.telemetry.context import (  # noqa: F401
    QueryScope, build_record, current_scope, query_scope, reactivating,
)
from src.telemetry.groq_client import instrumented_groq  # noqa: F401
from src.telemetry.record import LlmCall, QueryTelemetry, cost_for  # noqa: F401
from src.telemetry.stages import stage  # noqa: F401
