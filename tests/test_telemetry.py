"""Tests for the production telemetry layer.

Three things matter most and are covered here:
  1. Cost is computed correctly from the config price table, including the
     unpriced-model case that would otherwise silently under-report spend.
  2. The aggregation queries return what the dashboard claims they do.
  3. A failing metrics write cannot break a query -- the whole point of the
     background writer.
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.telemetry import store  # noqa: E402
from src.telemetry.context import QueryScope, build_record  # noqa: E402
from src.telemetry.record import (  # noqa: E402
    LlmCall, QueryTelemetry, classify_exception, cost_for,
)


# --- Cost calculation ---------------------------------------------------------

def test_cost_matches_the_configured_per_million_rates():
    # gpt-oss-120b: $0.15/1M in, $0.60/1M out.
    assert cost_for("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)
    assert cost_for("openai/gpt-oss-120b", 0, 1_000_000) == pytest.approx(0.60)
    assert cost_for("openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_cost_scales_linearly_at_realistic_token_counts():
    # A representative grounded query: ~8.7k prompt, ~600 completion.
    expected = 8_700 / 1e6 * 0.15 + 600 / 1e6 * 0.60
    assert cost_for("openai/gpt-oss-120b", 8_700, 600) == pytest.approx(expected)


def test_fast_model_is_priced_separately_from_the_main_model():
    main = cost_for("openai/gpt-oss-120b", 10_000, 1_000)
    fast = cost_for("openai/gpt-oss-20b", 10_000, 1_000)
    assert fast == pytest.approx(main / 2)


def test_unpriced_model_costs_zero_but_is_flagged_not_hidden():
    # A model missing from the price table must be visibly unpriced -- reporting
    # it as free would silently understate spend after a Groq model swap.
    call = LlmCall(model="some/newly-launched-model", prompt_tokens=1_000_000)
    assert call.cost_usd == 0.0
    assert call.is_priced is False
    assert LlmCall(model="openai/gpt-oss-20b").is_priced is True


def test_query_cost_sums_every_call_in_the_fan_out():
    # One historical query fans out across the fast and main models.
    record = QueryTelemetry(
        query_id="q", timestamp="2026-01-01T00:00:00+00:00",
        llm_calls=[
            LlmCall(model="openai/gpt-oss-20b", prompt_tokens=500, completion_tokens=50),
            LlmCall(model="openai/gpt-oss-20b", prompt_tokens=800, completion_tokens=80),
            LlmCall(model="openai/gpt-oss-120b", prompt_tokens=8_000, completion_tokens=600),
        ],
    )
    assert record.llm_call_count == 3
    assert record.total_prompt_tokens == 9_300
    assert record.total_completion_tokens == 730
    assert record.total_cost_usd == pytest.approx(
        cost_for("openai/gpt-oss-20b", 1_300, 130) + cost_for("openai/gpt-oss-120b", 8_000, 600)
    )
    assert record.models_used == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


# --- Error classification -----------------------------------------------------

def test_exceptions_map_onto_chartable_categories():
    class CompoundNotFound(Exception):
        pass

    assert classify_exception(ValueError("bad input")) == "validation"
    assert classify_exception(CompoundNotFound("no such chemical")) == "external_tool"
    assert classify_exception(
        RuntimeError("Model failed to produce valid grounded JSON after 3 attempts")
    ) == "llm_output_invalid"
    assert classify_exception(Exception("groq rate_limit_exceeded")) == "llm_provider"
    assert classify_exception(KeyError("x")) == "internal"


# --- Record building ----------------------------------------------------------

def test_quality_signals_are_read_off_the_response_envelope():
    scope = QueryScope()
    scope.add_stage_time("retrieval", 120.0)
    scope.add_stage_time("generation", 900.0)
    scope.add_llm_call(LlmCall(model="openai/gpt-oss-120b", prompt_tokens=100, completion_tokens=10))

    record = build_record(
        scope,
        timestamp="2026-01-01T00:00:00+00:00",
        query_text="what caused it",
        response={
            "intent": "historical",
            "resolved_query": None,
            "from_cache": False,
            "data": {
                "crag_insufficient": False,
                "crag_rewritten_query": "rewritten",
                "source": "internal",
                "confidence": 0.964,
                "faithfulness": {"faithful": False, "unsupported_claims": ["a", "b"]},
                "sources": [{"rerank_score": 0.98}, {"rerank_score": 0.71}],
            },
        },
    )

    assert record.intent == "historical"
    assert record.success is True
    assert record.crag_retry_fired is True  # a non-null rewritten query is the signal
    assert record.web_fallback_fired is False
    assert record.faithful is False
    assert record.unsupported_claim_count == 2
    assert record.retrieval_confidence == pytest.approx(0.964)
    assert record.top_rerank_score == pytest.approx(0.98)
    assert record.source_count == 2
    assert record.stage_latency_ms == {"retrieval": 120.0, "generation": 900.0}


def test_stage_time_accumulates_across_repeated_entries():
    # A comparative query runs retrieval once per sub-question; total time in
    # the stage is what matters, not the last visit.
    scope = QueryScope()
    scope.add_stage_time("retrieval", 100.0)
    scope.add_stage_time("retrieval", 150.0)
    assert scope.stage_latency_ms["retrieval"] == 250.0


def test_a_failed_query_records_the_category_and_no_response_fields():
    scope = QueryScope()
    record = build_record(
        scope, timestamp="2026-01-01T00:00:00+00:00", query_text="q",
        response=None, error=ValueError("bad"),
    )
    assert record.success is False
    assert record.error_category == "validation"
    assert record.error_type == "ValueError"
    assert record.intent is None


# --- Aggregation --------------------------------------------------------------

def _record(**kwargs) -> QueryTelemetry:
    base = dict(
        query_id=kwargs.pop("query_id", "q"),
        timestamp=kwargs.pop("timestamp", datetime.now(timezone.utc).isoformat()),
    )
    return QueryTelemetry(**base, **kwargs)


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "metrics.db")
    store.init_db(c)
    yield c
    c.close()


def test_latency_percentiles_and_stage_breakdown(conn):
    for i, ms in enumerate([100, 200, 300, 400, 5000]):
        store.insert_record(conn, _record(
            query_id=f"q{i}", total_latency_ms=float(ms),
            stage_latency_ms={"retrieval": ms * 0.2, "generation": ms * 0.7},
        ))

    summary = store.latency_summary(conn, hours=24, include_cached=False)
    assert summary["sample_size"] == 5
    assert summary["p50_ms"] == 300.0
    assert summary["max_ms"] == 5000.0
    # Ordered by mean, so the dominant stage is first -- the point of the panel.
    assert summary["stages"][0]["stage"] == "generation"
    assert summary["stages"][0]["share_of_queries"] == 1.0


def test_cached_queries_are_excluded_from_percentiles_by_default(conn):
    store.insert_record(conn, _record(query_id="slow", total_latency_ms=4000.0))
    for i in range(8):
        store.insert_record(conn, _record(
            query_id=f"cached{i}", total_latency_ms=5.0, from_cache=True,
        ))

    excluded = store.latency_summary(conn, hours=24, include_cached=False)
    included = store.latency_summary(conn, hours=24, include_cached=True)

    assert excluded["sample_size"] == 1
    assert excluded["p50_ms"] == 4000.0
    # Including cache hits drags the median to ~nothing, which is exactly the
    # flattering effect the default guards against.
    assert included["sample_size"] == 9
    assert included["p50_ms"] == 5.0


def test_cost_is_grouped_by_model(conn):
    store.insert_record(conn, _record(query_id="a", llm_calls=[
        LlmCall(model="openai/gpt-oss-120b", prompt_tokens=10_000, completion_tokens=1_000),
        LlmCall(model="openai/gpt-oss-20b", prompt_tokens=2_000, completion_tokens=200),
    ]))
    store.insert_record(conn, _record(query_id="b", llm_calls=[
        LlmCall(model="openai/gpt-oss-20b", prompt_tokens=1_000, completion_tokens=100),
    ]))

    summary = store.cost_summary(conn, hours=24, include_cached=False)
    assert summary["sample_size"] == 2
    assert summary["mean_llm_calls"] == 1.5
    by_model = {m["model"]: m for m in summary["by_model"]}
    assert by_model["openai/gpt-oss-20b"]["calls"] == 2
    assert by_model["openai/gpt-oss-20b"]["prompt_tokens"] == 3_000
    assert by_model["openai/gpt-oss-120b"]["cost_usd"] == pytest.approx(
        cost_for("openai/gpt-oss-120b", 10_000, 1_000), abs=1e-9
    )
    assert summary["total_cost_usd"] == pytest.approx(
        sum(m["cost_usd"] for m in summary["by_model"]), abs=1e-9
    )


def test_quality_rates_and_intent_distribution(conn):
    store.insert_record(conn, _record(query_id="a", intent="historical", faithful=True))
    store.insert_record(conn, _record(query_id="b", intent="historical", faithful=False,
                                      crag_retry_fired=True))
    store.insert_record(conn, _record(query_id="c", intent="calculation"))  # never checked
    store.insert_record(conn, _record(query_id="d", intent="historical",
                                      web_fallback_fired=True, faithful=True))

    q = store.quality_summary(conn, hours=24, include_cached=False)
    assert q["sample_size"] == 4
    # Only over answers that were actually checked -- calculation has no
    # free-text context to verify, so it must not count as a failure.
    assert q["faithfulness_checked"] == 3
    assert q["faithfulness_pass_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert q["crag_retry_rate"] == 0.25
    assert q["web_fallback_rate"] == 0.25
    assert q["intent_distribution"][0] == {"intent": "historical", "count": 3}


def test_cache_hit_rate_is_measured_over_the_unfiltered_window(conn):
    """Regression: every other metric excludes cache hits so they don't flatter
    the percentiles, but applying that filter here divides hits by a population
    with the hits removed -- pinning the rate at 0% however many there were."""
    store.insert_record(conn, _record(query_id="live1"))
    store.insert_record(conn, _record(query_id="live2"))
    store.insert_record(conn, _record(query_id="hit1", from_cache=True))
    store.insert_record(conn, _record(query_id="hit2", from_cache=True))

    q = store.quality_summary(conn, hours=24, include_cached=False)
    assert q["sample_size"] == 2      # the other rates are over live queries
    assert q["cache_hit_rate"] == 0.5  # ...but this one is over all four


def test_a_failure_after_routing_keeps_its_intent():
    """Regression: reading intent only off the final response filed every
    failure under "unrouted", making the error breakdown unattributable."""
    scope = QueryScope()
    scope.intent = "historical"  # what note_intent() sets at classification time

    record = build_record(
        scope, timestamp="2026-01-01T00:00:00+00:00", query_text="q",
        response=None, error=RuntimeError("groq rate_limit_exceeded"),
    )
    assert record.intent == "historical"
    assert record.error_category == "llm_provider"


def test_faithfulness_rate_is_none_when_nothing_was_checked(conn):
    store.insert_record(conn, _record(query_id="a", intent="calculation"))
    q = store.quality_summary(conn, hours=24, include_cached=False)
    # None, not 0.0 -- "never checked" and "checked and always failed" must not
    # render identically on the dashboard.
    assert q["faithfulness_pass_rate"] is None


def test_reliability_breaks_failures_down_by_category(conn):
    store.insert_record(conn, _record(query_id="ok", success=True))
    store.insert_record(conn, _record(query_id="e1", success=False,
                                      error_category="llm_provider", error_type="APIStatusError"))
    store.insert_record(conn, _record(query_id="e2", success=False,
                                      error_category="llm_provider", error_type="APIStatusError"))
    store.insert_record(conn, _record(query_id="e3", success=False,
                                      error_category="retrieval", error_type="RuntimeError"))

    r = store.reliability_summary(conn, hours=24, include_cached=False)
    assert r["failures"] == 3
    assert r["error_rate"] == 0.75
    assert r["by_category"][0]["category"] == "llm_provider"
    assert r["by_category"][0]["count"] == 2


def test_time_window_excludes_older_records(conn):
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    store.insert_record(conn, _record(query_id="old", timestamp=old, total_latency_ms=9999.0))
    store.insert_record(conn, _record(query_id="new", total_latency_ms=100.0))

    assert store.latency_summary(conn, hours=24, include_cached=False)["sample_size"] == 1
    assert store.latency_summary(conn, hours=72, include_cached=False)["sample_size"] == 2


def test_timeseries_buckets_do_not_merge_across_days(conn):
    # strftime('%H') grouping would collapse these two into one bucket.
    now = datetime.now(timezone.utc)
    store.insert_record(conn, _record(query_id="t1", timestamp=now.isoformat()))
    store.insert_record(conn, _record(
        query_id="t2", timestamp=(now - timedelta(days=1)).isoformat()))

    series = store.timeseries(conn, hours=48, include_cached=False, bucket_minutes=60)
    assert len(series["points"]) == 2


def test_stage_latency_survives_the_json_round_trip(conn):
    store.insert_record(conn, _record(
        query_id="s", stage_latency_ms={"retrieval": 12.5, "generation": 900.25}))
    row = conn.execute("SELECT stage_latency_json FROM query_telemetry").fetchone()
    assert json.loads(row["stage_latency_json"]) == {"retrieval": 12.5, "generation": 900.25}


def test_llm_calls_are_persisted_one_row_per_call(conn):
    store.insert_record(conn, _record(query_id="q", llm_calls=[
        LlmCall(model="openai/gpt-oss-20b", stage="routing", prompt_tokens=100),
        LlmCall(model="openai/gpt-oss-20b", stage="crag_grading", prompt_tokens=200),
        LlmCall(model="openai/gpt-oss-120b", stage="generation", prompt_tokens=300, streamed=True),
    ]))
    rows = conn.execute("SELECT stage, model, streamed FROM llm_calls ORDER BY id").fetchall()
    assert [r["stage"] for r in rows] == ["routing", "crag_grading", "generation"]
    assert rows[2]["streamed"] == 1


# --- Fail-safe behaviour ------------------------------------------------------

def test_submit_never_raises_when_the_database_is_unusable(tmp_path):
    # Point the writer at a path that cannot be opened as a database.
    broken = tmp_path / "not-a-dir" / "nested" / "metrics.db"
    broken.parent.parent.mkdir()
    broken.parent.write_text("this is a file, not a directory")

    w = store.TelemetryWriter(db_path=broken)
    w.submit(_record(query_id="q"))  # must not raise
    w.flush(timeout=1.0)
    w.stop()


def test_submit_never_raises_when_the_queue_is_full(monkeypatch, tmp_path):
    w = store.TelemetryWriter(db_path=tmp_path / "m.db")
    # Never start the consumer, and shrink the queue so it fills immediately.
    monkeypatch.setattr(w, "start", lambda: None)
    w._queue = __import__("queue").Queue(maxsize=1)

    for i in range(5):
        w.submit(_record(query_id=f"q{i}"))  # must not raise

    assert w.dropped == 4
    assert w._queue.qsize() == 1


def test_a_metrics_failure_does_not_break_the_query_response(monkeypatch):
    """The contract: an exception anywhere in telemetry leaves the caller's
    return value untouched."""
    from src.telemetry import recorder

    def explode(_record):
        raise RuntimeError("metrics backend on fire")

    monkeypatch.setattr(recorder.writer, "submit", explode)

    with recorder.record_query("what caused it") as outcome:
        response = {"intent": "historical", "answer": "grounded answer", "data": {}}
        outcome.set_response(response)

    assert response["answer"] == "grounded answer"


def test_a_metrics_failure_does_not_mask_a_real_pipeline_error(monkeypatch):
    from src.telemetry import recorder

    monkeypatch.setattr(
        recorder.writer, "submit",
        lambda _r: (_ for _ in ()).throw(RuntimeError("metrics down")),
    )

    with pytest.raises(ValueError, match="the real failure"):
        with recorder.record_query("q"):
            raise ValueError("the real failure")


def test_pipeline_hooks_are_inert_outside_a_request_scope():
    """The CLI entrypoints and eval harness run the same pipeline with no
    telemetry scope open; every hook must be a silent no-op there."""
    from src.telemetry.context import current_scope
    from src.telemetry.stages import stage

    assert current_scope() is None
    with stage("retrieval"):
        pass  # must not raise
    assert current_scope() is None


def test_writer_persists_through_the_background_thread(tmp_path):
    w = store.TelemetryWriter(db_path=tmp_path / "m.db")
    w.submit(_record(query_id="q1", total_latency_ms=42.0, intent="historical"))
    w.flush(timeout=5.0)
    w.stop()

    c = sqlite3.connect(str(tmp_path / "m.db"))
    c.row_factory = sqlite3.Row
    row = c.execute("SELECT * FROM query_telemetry WHERE query_id='q1'").fetchone()
    c.close()
    assert row is not None
    assert row["total_latency_ms"] == 42.0
    assert row["intent"] == "historical"
