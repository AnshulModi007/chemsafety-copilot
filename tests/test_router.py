"""Routing-logic tests: given a fixed (mocked) Groq JSON response, does
classify_intent/reformulate_query/extract_psv_params parse it into the
right structured decision? Also covers the retry-once-on-malformed-response
behavior and the deterministic actually_missing_fields override added
during the quality pass. The Groq client itself is mocked via conftest's
fake_groq fixture -- these tests exercise the plumbing, not the LLM's
judgment, so they're deterministic, fast, and free."""
import json

import pytest
from pydantic import ValidationError

from src.agent import router


def test_classify_intent_returns_route_decision(fake_groq):
    fake_groq.queue(json.dumps({
        "intent": "calculation", "chemical_name": None, "sub_queries": [],
        "reasoning": "PSV sizing request",
    }))
    decision = router.classify_intent("size a relief valve for a 5000 lb/hr propane release")
    assert decision.intent == "calculation"
    assert decision.reasoning == "PSV sizing request"


def test_classify_intent_general_knowledge_for_concept_questions(fake_groq):
    # Regression test: "what is a tray tower" / "what is mass transfer" have no
    # named chemical, incident, or calculation -- they must route to
    # general_knowledge, not get forced into chemical_property (see the
    # confusing "couldn't tell which chemical" failure this used to produce).
    fake_groq.queue(json.dumps({
        "intent": "general_knowledge", "chemical_name": None, "sub_queries": [],
        "reasoning": "asks about a general chemical-engineering concept, not a named chemical",
    }))
    decision = router.classify_intent("what is mass transfer?")
    assert decision.intent == "general_knowledge"
    assert decision.chemical_name is None


def test_classify_intent_extracts_chemical_name(fake_groq):
    fake_groq.queue(json.dumps({
        "intent": "chemical_property", "chemical_name": "chlorine", "sub_queries": [],
        "reasoning": "asks for a property of a named chemical",
    }))
    decision = router.classify_intent("what is the molecular weight of chlorine?")
    assert decision.intent == "chemical_property"
    assert decision.chemical_name == "chlorine"


def test_classify_intent_reasoning_defaults_to_empty_string_when_omitted(fake_groq):
    # Observed in practice: the fast model sometimes omits the free-text
    # "reasoning" field even when classification itself succeeded -- this
    # must not crash the request (see RouteDecision.reasoning's default).
    fake_groq.queue(json.dumps({"intent": "historical", "chemical_name": None, "sub_queries": []}))
    decision = router.classify_intent("what caused the incident at plant X?")
    assert decision.intent == "historical"
    assert decision.reasoning == ""


def test_classify_intent_retries_once_on_malformed_response(fake_groq):
    fake_groq.queue(
        json.dumps({"intent": "not_a_real_intent", "sub_queries": []}),
        json.dumps({"intent": "historical", "chemical_name": None, "sub_queries": [], "reasoning": "ok"}),
    )
    decision = router.classify_intent("what happened at plant X?")
    assert decision.intent == "historical"
    assert len(fake_groq.calls) == 2


def test_classify_intent_raises_after_two_bad_responses(fake_groq):
    bad = json.dumps({"intent": "not_a_real_intent"})
    fake_groq.queue(bad, bad)
    with pytest.raises(ValidationError):
        router.classify_intent("what happened at plant X?")


def test_reformulate_query_skips_llm_call_with_no_history(fake_groq):
    # No responses queued -- if this made an LLM call it would raise
    # IndexError popping from an empty queue.
    result = router.reformulate_query("what is chlorine's molecular weight?", [])
    assert result == "what is chlorine's molecular weight?"


def test_reformulate_query_resolves_followup_using_history(fake_groq):
    fake_groq.queue(json.dumps({"resolved_query": "What is the molecular weight of chlorine?"}))
    result = router.reformulate_query(
        "what about its molecular weight?",
        [{"role": "user", "content": "Tell me about chlorine"}],
    )
    assert result == "What is the molecular weight of chlorine?"


def test_reformulate_query_fails_soft_to_raw_query_on_malformed_response(fake_groq):
    bad = json.dumps({"unexpected_key": True})
    fake_groq.queue(bad, bad)
    result = router.reformulate_query(
        "what about its pricing?", [{"role": "user", "content": "Tell me about chlorine"}],
    )
    assert result == "what about its pricing?"


def test_extract_psv_params_converts_fahrenheit_to_rankine(fake_groq):
    fake_groq.queue(json.dumps({
        "mass_flow_lb_hr": 5000, "molecular_weight": 44,
        "relieving_temp_value": 200, "relieving_temp_unit": "F",
        "set_pressure_value": 250, "set_pressure_unit": "psig",
        "k": None, "compressibility_z": None, "missing_required_fields": [],
    }))
    params = router.extract_psv_params("size a PSV: 5000 lb/hr, MW 44, 200F, 250 psig")
    assert params.relieving_temp_rankine == pytest.approx(659.67)
    assert params.set_pressure_psig == 250
    assert params.actually_missing_fields == []


def test_extract_psv_params_converts_psia_to_psig(fake_groq):
    fake_groq.queue(json.dumps({
        "mass_flow_lb_hr": 5000, "molecular_weight": 44,
        "relieving_temp_value": 600, "relieving_temp_unit": "R",
        "set_pressure_value": 264.7, "set_pressure_unit": "psia",
        "missing_required_fields": [],
    }))
    params = router.extract_psv_params("size a PSV at 264.7 psia")
    assert params.set_pressure_psig == pytest.approx(250.0)


def test_extract_psv_params_actually_missing_fields_overrides_model_self_report(fake_groq):
    # The model's own missing_required_fields list is empty here even though
    # two required values are actually null -- actually_missing_fields must
    # catch that deterministically rather than trust the self-report.
    fake_groq.queue(json.dumps({
        "mass_flow_lb_hr": None, "molecular_weight": 44,
        "relieving_temp_value": 200, "relieving_temp_unit": "F",
        "set_pressure_value": None, "set_pressure_unit": "psig",
        "missing_required_fields": [],
    }))
    params = router.extract_psv_params("size a PSV with MW 44 at 200F")
    assert set(params.actually_missing_fields) == {"mass_flow_lb_hr", "set_pressure_value"}
