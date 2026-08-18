"""SQLite persistence and aggregation for query telemetry.

Two design constraints drove this:

1. A metrics write must never slow down or break a user's query. The handlers
   are sync `def`, so there is no event loop to await on -- instead the request
   path only does `queue.put_nowait(...)` (microseconds, and it swallows a full
   queue rather than raising), and a single daemon thread owns the one SQLite
   connection and performs the inserts. That makes the fail-safe structural
   rather than a hopeful try/except, and satisfies SQLite's
   one-connection-per-thread rule by construction.

2. Aggregates must be queryable by time window, intent, model, and outcome. So
   per-call rows live in their own table with an indexed foreign key, letting
   "cost by model" be a plain GROUP BY instead of JSON unpacking in Python.
"""
import json
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from config import METRICS_DB_PATH
from src.telemetry.record import QueryTelemetry

logger = logging.getLogger(__name__)

# Bounded on purpose: if the writer thread dies or stalls, telemetry is dropped
# (and counted) rather than growing until the process runs out of memory.
_QUEUE_MAXSIZE = 2048

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_telemetry (
    query_id                TEXT PRIMARY KEY,
    timestamp               TEXT NOT NULL,
    query_text              TEXT,
    resolved_query          TEXT,
    intent                  TEXT,
    success                 INTEGER NOT NULL,
    error_category          TEXT,
    error_type              TEXT,
    total_latency_ms        REAL NOT NULL,
    stage_latency_json      TEXT NOT NULL,
    prompt_tokens           INTEGER NOT NULL,
    completion_tokens       INTEGER NOT NULL,
    total_tokens            INTEGER NOT NULL,
    cost_usd                REAL NOT NULL,
    llm_call_count          INTEGER NOT NULL,
    from_cache              INTEGER NOT NULL,
    crag_retry_fired        INTEGER NOT NULL,
    crag_insufficient       INTEGER NOT NULL,
    web_fallback_fired      INTEGER NOT NULL,
    faithful                INTEGER,
    unsupported_claim_count INTEGER NOT NULL,
    retrieval_confidence    REAL,
    top_rerank_score        REAL,
    source_count            INTEGER NOT NULL,
    streamed                INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qt_timestamp ON query_telemetry(timestamp);
CREATE INDEX IF NOT EXISTS idx_qt_intent    ON query_telemetry(intent);
CREATE INDEX IF NOT EXISTS idx_qt_success   ON query_telemetry(success);
CREATE INDEX IF NOT EXISTS idx_qt_cached    ON query_telemetry(from_cache);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id          TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    model             TEXT NOT NULL,
    stage             TEXT,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    priced            INTEGER NOT NULL,
    latency_ms        REAL NOT NULL,
    server_time_ms    REAL,
    streamed          INTEGER NOT NULL,
    FOREIGN KEY (query_id) REFERENCES query_telemetry(query_id)
);

CREATE INDEX IF NOT EXISTS idx_lc_query ON llm_calls(query_id);
CREATE INDEX IF NOT EXISTS idx_lc_model ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_lc_time  ON llm_calls(timestamp);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else METRICS_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL lets the metrics API read while the writer thread is inserting.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_record(conn: sqlite3.Connection, record: QueryTelemetry) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO query_telemetry VALUES (
            :query_id, :timestamp, :query_text, :resolved_query, :intent, :success,
            :error_category, :error_type, :total_latency_ms, :stage_latency_json,
            :prompt_tokens, :completion_tokens, :total_tokens, :cost_usd, :llm_call_count,
            :from_cache, :crag_retry_fired, :crag_insufficient, :web_fallback_fired,
            :faithful, :unsupported_claim_count, :retrieval_confidence, :top_rerank_score,
            :source_count, :streamed
        )
        """,
        {
            "query_id": record.query_id,
            "timestamp": record.timestamp,
            "query_text": record.query_text,
            "resolved_query": record.resolved_query,
            "intent": record.intent,
            "success": int(record.success),
            "error_category": record.error_category,
            "error_type": record.error_type,
            "total_latency_ms": record.total_latency_ms,
            "stage_latency_json": json.dumps(record.stage_latency_ms),
            "prompt_tokens": record.total_prompt_tokens,
            "completion_tokens": record.total_completion_tokens,
            "total_tokens": record.total_tokens,
            "cost_usd": record.total_cost_usd,
            "llm_call_count": record.llm_call_count,
            "from_cache": int(record.from_cache),
            "crag_retry_fired": int(record.crag_retry_fired),
            "crag_insufficient": int(record.crag_insufficient),
            "web_fallback_fired": int(record.web_fallback_fired),
            "faithful": None if record.faithful is None else int(record.faithful),
            "unsupported_claim_count": record.unsupported_claim_count,
            "retrieval_confidence": record.retrieval_confidence,
            "top_rerank_score": record.top_rerank_score,
            "source_count": record.source_count,
            "streamed": int(record.streamed),
        },
    )
    conn.executemany(
        """
        INSERT INTO llm_calls (
            query_id, timestamp, model, stage, prompt_tokens, completion_tokens,
            total_tokens, cost_usd, priced, latency_ms, server_time_ms, streamed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.query_id,
                record.timestamp,
                call.model,
                call.stage,
                call.prompt_tokens,
                call.completion_tokens,
                call.total_tokens,
                call.cost_usd,
                int(call.is_priced),
                call.latency_ms,
                call.server_time_ms,
                int(call.streamed),
            )
            for call in record.llm_calls
        ],
    )
    conn.commit()


class TelemetryWriter:
    """Queue-backed background writer. `submit` is the only method the request
    path touches, and it is designed never to raise."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = db_path
        self._queue: queue.Queue[QueryTelemetry | None] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.failed = 0
        self.written = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="telemetry-writer", daemon=True
            )
            self._thread.start()

    def submit(self, record: QueryTelemetry) -> None:
        """Hand a record off for writing. Never blocks; never raises.

        A dropped or failed write is counted and logged -- the query response is
        unaffected either way, which is the entire contract.
        """
        try:
            self.start()
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            logger.warning("telemetry queue full -- dropped record %s", record.query_id)
        except Exception:
            self.dropped += 1
            logger.warning("failed to enqueue telemetry", exc_info=True)

    def _run(self) -> None:
        try:
            conn = connect(self._db_path)
            init_db(conn)
        except Exception:
            logger.error("telemetry writer could not open the database", exc_info=True)
            return

        while True:
            record = self._queue.get()
            if record is None:
                break
            try:
                insert_record(conn, record)
                self.written += 1
            except Exception:
                self.failed += 1
                logger.warning("failed to persist telemetry record", exc_info=True)
            finally:
                self._queue.task_done()

        conn.close()

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue drains. For tests and shutdown only -- the
        request path must never call this."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._thread = None


# Process-wide writer used by the request path.
writer = TelemetryWriter()


# --- Aggregation --------------------------------------------------------------

def _window_start(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Deliberately not interpolated: with the small
    sample sizes this dashboard shows, an interpolated p99 invents a latency
    that no query actually had."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _where(hours: int, include_cached: bool, extra: Iterable[str] = ()) -> tuple[str, list[Any]]:
    clauses = ["timestamp >= ?"]
    params: list[Any] = [_window_start(hours)]
    if not include_cached:
        clauses.append("from_cache = 0")
    clauses.extend(extra)
    return " AND ".join(clauses), params


def latency_summary(conn: sqlite3.Connection, hours: int, include_cached: bool) -> dict:
    where, params = _where(hours, include_cached)
    rows = conn.execute(
        f"SELECT total_latency_ms, stage_latency_json FROM query_telemetry WHERE {where}",
        params,
    ).fetchall()

    totals = [r["total_latency_ms"] for r in rows]

    # Averaged over the queries that actually ran each stage -- a cached hit or
    # a PubChem lookup never enters retrieval, and folding those in as zeros
    # would understate how slow retrieval is when it does run.
    stage_totals: dict[str, list[float]] = {}
    for row in rows:
        for name, value in json.loads(row["stage_latency_json"]).items():
            stage_totals.setdefault(name, []).append(value)

    return {
        "sample_size": len(totals),
        "p50_ms": round(_percentile(totals, 0.50), 1),
        "p95_ms": round(_percentile(totals, 0.95), 1),
        "p99_ms": round(_percentile(totals, 0.99), 1),
        "mean_ms": round(sum(totals) / len(totals), 1) if totals else 0.0,
        "max_ms": round(max(totals), 1) if totals else 0.0,
        "stages": [
            {
                "stage": name,
                "mean_ms": round(sum(values) / len(values), 1),
                "p95_ms": round(_percentile(values, 0.95), 1),
                "share_of_queries": round(len(values) / len(totals), 3) if totals else 0.0,
            }
            for name, values in sorted(
                stage_totals.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])
            )
        ],
    }


def cost_summary(conn: sqlite3.Connection, hours: int, include_cached: bool) -> dict:
    where, params = _where(hours, include_cached)
    row = conn.execute(
        f"""SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd), 0) AS total,
                   COALESCE(SUM(total_tokens), 0) AS tokens,
                   COALESCE(SUM(llm_call_count), 0) AS calls
            FROM query_telemetry WHERE {where}""",
        params,
    ).fetchone()

    by_model = conn.execute(
        f"""SELECT model,
                   COUNT(*) AS calls,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(cost_usd), 0) AS cost_usd,
                   MIN(priced) AS priced
            FROM llm_calls
            WHERE query_id IN (SELECT query_id FROM query_telemetry WHERE {where})
            GROUP BY model ORDER BY cost_usd DESC""",
        params,
    ).fetchall()

    n = row["n"] or 0
    return {
        "sample_size": n,
        "total_cost_usd": round(row["total"], 6),
        "mean_cost_usd": round(row["total"] / n, 6) if n else 0.0,
        "total_tokens": row["tokens"],
        "mean_tokens": round(row["tokens"] / n, 1) if n else 0.0,
        "mean_llm_calls": round(row["calls"] / n, 2) if n else 0.0,
        "by_model": [
            {
                "model": r["model"],
                "calls": r["calls"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "cost_usd": round(r["cost_usd"], 6),
                # False means this model has no entry in the config price table,
                # so its cost reads 0 and the total is an understatement.
                "priced": bool(r["priced"]),
            }
            for r in by_model
        ],
    }


def quality_summary(conn: sqlite3.Connection, hours: int, include_cached: bool) -> dict:
    where, params = _where(hours, include_cached)

    faith = conn.execute(
        f"""SELECT COUNT(*) AS checked, COALESCE(SUM(faithful), 0) AS passed
            FROM query_telemetry WHERE {where} AND faithful IS NOT NULL""",
        params,
    ).fetchone()

    flags = conn.execute(
        f"""SELECT COUNT(*) AS n,
                   COALESCE(SUM(crag_retry_fired), 0) AS retries,
                   COALESCE(SUM(crag_insufficient), 0) AS insufficient,
                   COALESCE(SUM(web_fallback_fired), 0) AS web,
                   AVG(retrieval_confidence) AS mean_conf
            FROM query_telemetry WHERE {where}""",
        params,
    ).fetchone()

    # Cache-hit rate is deliberately measured over the *unfiltered* window.
    # Every other metric here excludes cache hits by default so they don't
    # flatter the percentiles -- but applying that filter to this metric would
    # divide hits by a population with the hits removed, pinning it at 0%.
    cache_where, cache_params = _where(hours, include_cached=True)
    cache = conn.execute(
        f"""SELECT COUNT(*) AS n, COALESCE(SUM(from_cache), 0) AS cached
            FROM query_telemetry WHERE {cache_where}""",
        cache_params,
    ).fetchone()

    intents = conn.execute(
        f"""SELECT COALESCE(intent, 'unrouted') AS intent, COUNT(*) AS n
            FROM query_telemetry WHERE {where} GROUP BY intent ORDER BY n DESC""",
        params,
    ).fetchall()

    n = flags["n"] or 0
    checked = faith["checked"] or 0
    return {
        "sample_size": n,
        # Only over answers that were actually checked: three of the five intents
        # have no free-text context to verify, so they never produce a verdict.
        "faithfulness_checked": checked,
        "faithfulness_pass_rate": round(faith["passed"] / checked, 3) if checked else None,
        "crag_retry_rate": round(flags["retries"] / n, 3) if n else 0.0,
        "crag_insufficient_rate": round(flags["insufficient"] / n, 3) if n else 0.0,
        "web_fallback_rate": round(flags["web"] / n, 3) if n else 0.0,
        "cache_hit_rate": (
            round(cache["cached"] / cache["n"], 3) if cache["n"] else 0.0
        ),
        "mean_retrieval_confidence": (
            round(flags["mean_conf"], 3) if flags["mean_conf"] is not None else None
        ),
        "intent_distribution": [{"intent": r["intent"], "count": r["n"]} for r in intents],
    }


def reliability_summary(conn: sqlite3.Connection, hours: int, include_cached: bool) -> dict:
    where, params = _where(hours, include_cached)
    row = conn.execute(
        f"""SELECT COUNT(*) AS n, COALESCE(SUM(1 - success), 0) AS failures
            FROM query_telemetry WHERE {where}""",
        params,
    ).fetchone()
    categories = conn.execute(
        f"""SELECT COALESCE(error_category, 'unknown') AS category,
                   error_type, COUNT(*) AS n
            FROM query_telemetry WHERE {where} AND success = 0
            GROUP BY category, error_type ORDER BY n DESC""",
        params,
    ).fetchall()

    n = row["n"] or 0
    return {
        "sample_size": n,
        "failures": row["failures"],
        "error_rate": round(row["failures"] / n, 4) if n else 0.0,
        "by_category": [
            {"category": r["category"], "error_type": r["error_type"], "count": r["n"]}
            for r in categories
        ],
    }


def timeseries(conn: sqlite3.Connection, hours: int, include_cached: bool, bucket_minutes: int) -> dict:
    """Volume, latency, cost, and faithfulness bucketed over the window.

    Buckets are computed by integer-dividing the epoch seconds, which keeps the
    grouping correct across hour and day boundaries (strftime-based grouping on
    '%H' silently merges the same hour from different days).
    """
    where, params = _where(hours, include_cached)
    seconds = max(1, bucket_minutes) * 60
    rows = conn.execute(
        f"""SELECT
                CAST(strftime('%s', timestamp) AS INTEGER) / {seconds} * {seconds} AS bucket,
                COUNT(*) AS queries,
                COALESCE(SUM(1 - success), 0) AS failures,
                AVG(total_latency_ms) AS mean_latency_ms,
                COALESCE(SUM(cost_usd), 0) AS cost_usd,
                SUM(CASE WHEN faithful IS NOT NULL THEN 1 ELSE 0 END) AS checked,
                COALESCE(SUM(faithful), 0) AS faithful
            FROM query_telemetry WHERE {where}
            GROUP BY bucket ORDER BY bucket""",
        params,
    ).fetchall()

    return {
        "bucket_minutes": bucket_minutes,
        "points": [
            {
                "t": datetime.fromtimestamp(r["bucket"], tz=timezone.utc).isoformat(),
                "queries": r["queries"],
                "failures": r["failures"],
                "mean_latency_ms": round(r["mean_latency_ms"] or 0.0, 1),
                "cost_usd": round(r["cost_usd"], 6),
                "faithfulness_pass_rate": (
                    round(r["faithful"] / r["checked"], 3) if r["checked"] else None
                ),
            }
            for r in rows
        ],
    }


def recent_failures(conn: sqlite3.Connection, hours: int, limit: int = 20) -> list[dict]:
    """The incident list. Query text is included only if it was persisted at all
    (LOG_QUERY_TEXT), and never the exception message -- which can echo it."""
    rows = conn.execute(
        """SELECT query_id, timestamp, intent, error_category, error_type,
                  total_latency_ms, query_text
           FROM query_telemetry
           WHERE timestamp >= ? AND success = 0
           ORDER BY timestamp DESC LIMIT ?""",
        [_window_start(hours), limit],
    ).fetchall()
    return [dict(r) for r in rows]


def writer_health() -> dict:
    """Whether telemetry itself is healthy -- a dashboard that silently stops
    recording is worse than no dashboard."""
    return {
        "written": writer.written,
        "dropped": writer.dropped,
        "failed": writer.failed,
        "queue_depth": writer._queue.qsize(),
        "thread_alive": writer._thread is not None and writer._thread.is_alive(),
    }
