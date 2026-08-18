"""Read-only metrics API.

Kept on its own router, separate from the user-facing query API, and
deliberately aggregate-only: every endpoint returns counts, rates, and
percentiles. The one endpoint that touches individual queries
(`/metrics/failures`) returns no exception messages, because a message can echo
the query text that `LOG_QUERY_TEXT` exists to control.

Nothing here reads a secret, and nothing here writes.
"""
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import METRICS_EXCLUDE_CACHED_BY_DEFAULT  # noqa: E402
from src.telemetry import store  # noqa: E402

router = APIRouter(prefix="/metrics", tags=["metrics"])

# Bounded so a crafted request cannot ask for an unbounded scan.
MAX_WINDOW_HOURS = 24 * 90


def _conn() -> Iterator[sqlite3.Connection]:
    conn = store.connect()
    try:
        store.init_db(conn)
        yield conn
    finally:
        conn.close()


def _window(hours: int) -> int:
    if hours < 1 or hours > MAX_WINDOW_HOURS:
        raise HTTPException(status_code=400, detail=f"hours must be between 1 and {MAX_WINDOW_HOURS}")
    return hours


# --- Response models ----------------------------------------------------------

class StageLatency(BaseModel):
    stage: str
    mean_ms: float
    p95_ms: float
    # Fraction of queries that entered this stage at all -- a cached hit or a
    # PubChem lookup never reaches retrieval, and the mean is over those that did.
    share_of_queries: float


class LatencySummary(BaseModel):
    sample_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    stages: list[StageLatency]


class ModelCost(BaseModel):
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    # False when the model has no entry in config.GROQ_PRICING_USD_PER_MTOK,
    # meaning its cost reads 0 and the total understates actual spend.
    priced: bool


class CostSummary(BaseModel):
    sample_size: int
    total_cost_usd: float
    mean_cost_usd: float
    total_tokens: int
    mean_tokens: float
    mean_llm_calls: float
    by_model: list[ModelCost]


class IntentCount(BaseModel):
    intent: str
    count: int


class QualitySummary(BaseModel):
    sample_size: int
    faithfulness_checked: int
    # None when nothing in the window was checked -- distinct from a 0% pass rate.
    faithfulness_pass_rate: float | None
    crag_retry_rate: float
    crag_insufficient_rate: float
    web_fallback_rate: float
    cache_hit_rate: float
    mean_retrieval_confidence: float | None
    intent_distribution: list[IntentCount]


class ErrorCount(BaseModel):
    category: str
    error_type: str | None
    count: int


class ReliabilitySummary(BaseModel):
    sample_size: int
    failures: int
    error_rate: float
    by_category: list[ErrorCount]


class TimeseriesPoint(BaseModel):
    t: str
    queries: int
    failures: int
    mean_latency_ms: float
    cost_usd: float
    faithfulness_pass_rate: float | None


class Timeseries(BaseModel):
    bucket_minutes: int
    points: list[TimeseriesPoint]


class FailureRow(BaseModel):
    query_id: str
    timestamp: str
    intent: str | None
    error_category: str | None
    error_type: str | None
    total_latency_ms: float
    query_text: str | None


class WriterHealth(BaseModel):
    written: int
    dropped: int
    failed: int
    queue_depth: int
    thread_alive: bool


class Overview(BaseModel):
    window_hours: int
    include_cached: bool
    latency: LatencySummary
    cost: CostSummary
    quality: QualitySummary
    reliability: ReliabilitySummary
    timeseries: Timeseries
    writer: WriterHealth


# --- Shared query params ------------------------------------------------------

_HOURS = Query(default=24, description="Look-back window in hours.")
_CACHED = Query(
    default=None,
    description=(
        "Include semantic-cache hits. Excluded by default: they run zero LLM "
        "calls and return in milliseconds, so including them flatters latency "
        "and cost percentiles."
    ),
)


def _include_cached(value: bool | None) -> bool:
    return (not METRICS_EXCLUDE_CACHED_BY_DEFAULT) if value is None else value


# --- Endpoints ----------------------------------------------------------------

@router.get("/latency", response_model=LatencySummary)
def latency(hours: int = _HOURS, include_cached: bool | None = _CACHED) -> dict:
    for conn in _conn():
        return store.latency_summary(conn, _window(hours), _include_cached(include_cached))
    raise HTTPException(status_code=500, detail="metrics database unavailable")


@router.get("/cost", response_model=CostSummary)
def cost(hours: int = _HOURS, include_cached: bool | None = _CACHED) -> dict:
    for conn in _conn():
        return store.cost_summary(conn, _window(hours), _include_cached(include_cached))
    raise HTTPException(status_code=500, detail="metrics database unavailable")


@router.get("/quality", response_model=QualitySummary)
def quality(hours: int = _HOURS, include_cached: bool | None = _CACHED) -> dict:
    for conn in _conn():
        return store.quality_summary(conn, _window(hours), _include_cached(include_cached))
    raise HTTPException(status_code=500, detail="metrics database unavailable")


@router.get("/reliability", response_model=ReliabilitySummary)
def reliability(hours: int = _HOURS, include_cached: bool | None = _CACHED) -> dict:
    for conn in _conn():
        return store.reliability_summary(conn, _window(hours), _include_cached(include_cached))
    raise HTTPException(status_code=500, detail="metrics database unavailable")


@router.get("/timeseries", response_model=Timeseries)
def timeseries(
    hours: int = _HOURS,
    include_cached: bool | None = _CACHED,
    bucket_minutes: int = Query(default=60, ge=1, le=1440),
) -> dict:
    for conn in _conn():
        return store.timeseries(conn, _window(hours), _include_cached(include_cached), bucket_minutes)
    raise HTTPException(status_code=500, detail="metrics database unavailable")


@router.get("/failures", response_model=list[FailureRow])
def failures(hours: int = _HOURS, limit: int = Query(default=20, ge=1, le=200)) -> list[dict]:
    for conn in _conn():
        return store.recent_failures(conn, _window(hours), limit)
    raise HTTPException(status_code=500, detail="metrics database unavailable")


@router.get("/overview", response_model=Overview)
def overview(
    hours: int = _HOURS,
    include_cached: bool | None = _CACHED,
    bucket_minutes: int = Query(default=60, ge=1, le=1440),
) -> dict:
    """Everything the dashboard needs in one round-trip, so a page load is a
    single request rather than six racing ones."""
    window = _window(hours)
    cached = _include_cached(include_cached)
    for conn in _conn():
        return {
            "window_hours": window,
            "include_cached": cached,
            "latency": store.latency_summary(conn, window, cached),
            "cost": store.cost_summary(conn, window, cached),
            "quality": store.quality_summary(conn, window, cached),
            "reliability": store.reliability_summary(conn, window, cached),
            "timeseries": store.timeseries(conn, window, cached, bucket_minutes),
            "writer": store.writer_health(),
        }
    raise HTTPException(status_code=500, detail="metrics database unavailable")
