"""Typed telemetry records: one QueryTelemetry per user query, plus one
LlmCall per Groq request made while serving it.

A single historical query fans out to 8-12 Groq calls across 7 modules
(reformulation, routing, three expansion calls, CRAG grading, an optional
rewrite, generation, faithfulness, diagram extraction), so cost and token
usage are always a sum over LlmCall rows -- never a single reading.
"""
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_PRICING_USD_PER_MTOK  # noqa: E402

# The pipeline stages worth timing separately. `stage` is Optional on LlmCall
# because a call can happen outside any instrumented stage (e.g. routing runs
# before the retrieval stage opens).
Stage = Literal[
    "reformulate",
    "routing",
    "cache_lookup",
    "expansion",
    "retrieval",
    "rerank",
    "crag_grading",
    "generation",
    "faithfulness",
    "web_search",
    "diagram",
    "tool_call",
]

# Coarse, stable buckets -- an exception type name alone is too granular to
# chart, and a raw message is unbounded and can carry query content.
ErrorCategory = Literal[
    "llm_provider",       # Groq 4xx/5xx: rate limit, decommissioned model, auth
    "llm_output_invalid",  # model returned unparseable/invalid structured output
    "retrieval",          # Chroma/BM25/index failure
    "external_tool",      # PubChem / Tavily unreachable
    "validation",         # bad user input
    "internal",           # anything else
]


class LlmCall(BaseModel):
    """One Groq chat-completion request, captured by the client proxy."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Wall-clock as measured client-side, including network.
    latency_ms: float = 0.0
    # Groq reports its own inference time in usage.total_time; keeping both lets
    # network overhead be separated from model time.
    server_time_ms: float | None = None
    stage: str | None = None
    streamed: bool = False

    @property
    def cost_usd(self) -> float:
        return cost_for(self.model, self.prompt_tokens, self.completion_tokens)

    @property
    def is_priced(self) -> bool:
        return self.model in GROQ_PRICING_USD_PER_MTOK


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Notional USD cost from the config price table.

    Returns 0.0 for a model with no price entry -- the caller reports those
    separately as "unpriced" rather than treating them as free (see
    config.GROQ_PRICING_USD_PER_MTOK).
    """
    prices = GROQ_PRICING_USD_PER_MTOK.get(model)
    if prices is None:
        return 0.0
    return (
        prompt_tokens / 1_000_000 * prices["input"]
        + completion_tokens / 1_000_000 * prices["output"]
    )


class QueryTelemetry(BaseModel):
    """The single row written per query."""

    query_id: str
    # ISO-8601 UTC, stored as text so SQLite date functions work on it directly.
    timestamp: str
    # None when LOG_QUERY_TEXT is disabled.
    query_text: str | None = None
    resolved_query: str | None = None

    # Routing / outcome
    intent: str | None = None
    success: bool = True
    error_category: ErrorCategory | None = None
    # Exception class name only -- never the message, which can echo query text.
    error_type: str | None = None

    # Timing
    total_latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)

    # Cost & usage, summed over llm_calls
    llm_calls: list[LlmCall] = Field(default_factory=list)

    # Quality signals, all read straight off the response envelope
    from_cache: bool = False
    crag_retry_fired: bool = False
    crag_insufficient: bool = False
    web_fallback_fired: bool = False
    faithful: bool | None = None
    unsupported_claim_count: int = 0
    retrieval_confidence: float | None = None
    top_rerank_score: float | None = None
    source_count: int = 0
    streamed: bool = False

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)

    @property
    def models_used(self) -> list[str]:
        seen: list[str] = []
        for call in self.llm_calls:
            if call.model not in seen:
                seen.append(call.model)
        return seen


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Map an exception onto a chartable bucket.

    Matches on type name and module rather than importing every provider's
    exception classes, so this keeps working if a dependency reorganises them
    (and so it can run without those packages installed, e.g. in tests).
    """
    name = type(exc).__name__
    module = type(exc).__module__.split(".")[0]
    text = f"{name} {exc}".lower()

    if module == "groq" or "groq" in text:
        return "llm_provider"
    if name in ("ValidationError", "JSONDecodeError") or "json" in name.lower():
        return "llm_output_invalid"
    if "model failed to produce valid" in text:
        return "llm_output_invalid"
    if module == "chromadb" or "chroma" in text or name == "IndexError":
        return "retrieval"
    if name in ("CompoundNotFound", "PubChemUnavailable", "WebSearchUnavailable"):
        return "external_tool"
    if name in ("HTTPException", "RequestValidationError", "ValueError"):
        return "validation"
    return "internal"
