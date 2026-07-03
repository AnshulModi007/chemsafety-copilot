"""Top-level agentic entrypoint: classify intent, dispatch to the right tool,
return a uniform response envelope regardless of which path was taken.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agent.router import classify_intent, extract_psv_params  # noqa: E402
from src.generation.generate import generate_with_crag  # noqa: E402
from src.tools import calculations  # noqa: E402
from src.tools.pubchem import CompoundNotFound, get_compound_properties  # noqa: E402

PE_DISCLAIMER = (
    "This reflects historical incident findings and reference data, not a stamped engineering "
    "judgment. Consult a licensed Professional Engineer for any real design or safety decision."
)


def _handle_historical_or_comparative(query: str) -> dict:
    result = generate_with_crag(query)
    return {
        "answer": result["answer"],
        "data": {
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"],
            "crag_insufficient": result["crag_insufficient"],
            "crag_rewritten_query": result["crag_rewritten_query"],
        },
    }


def _handle_chemical_property(chemical_name: str | None) -> dict:
    if not chemical_name:
        return {
            "answer": "I couldn't tell which chemical you're asking about -- could you name it explicitly?",
            "data": {},
        }
    try:
        props = get_compound_properties(chemical_name)
    except CompoundNotFound as e:
        return {"answer": str(e), "data": {}}

    hazard_line = (
        "; ".join(props["ghs_hazard_statements"]) if props["ghs_hazard_statements"]
        else "no GHS classification available from PubChem for this compound"
    )
    answer = (
        f"{props['iupac_name'] or chemical_name} ({props['molecular_formula']}, "
        f"MW {props['molecular_weight']}). GHS hazards: {hazard_line}. "
        f"Source: PubChem CID {props['cid']} ({props['pubchem_url']})."
    )
    return {"answer": answer, "data": props}


def _handle_calculation(query: str) -> dict:
    params = extract_psv_params(query)
    if params.missing_required_fields:
        return {
            "answer": (
                "I need a few more values to size this relief valve: "
                + ", ".join(params.missing_required_fields)
                + ". Please provide them (mass flow in lb/hr, molecular weight, relieving "
                "temperature, and PSV set pressure)."
            ),
            "data": {"missing_required_fields": params.missing_required_fields},
        }

    kwargs = dict(
        mass_flow_lb_hr=params.mass_flow_lb_hr,
        molecular_weight=params.molecular_weight,
        relieving_temp_rankine=params.relieving_temp_rankine,
        set_pressure_psig=params.set_pressure_psig,
    )
    if params.k is not None:
        kwargs["k"] = params.k
    if params.compressibility_z is not None:
        kwargs["compressibility_z"] = params.compressibility_z

    result = calculations.size_psv_vapor(**kwargs)
    orifice = result["recommended_orifice"]
    orifice_text = (
        f"API 526 orifice {orifice['designation']} ({orifice['area_in2']} in^2)"
        if orifice else "larger than the largest standard API 526 orifice -- consider multiple valves"
    )
    answer = (
        f"Required effective discharge area: {result['required_area_in2']:.4f} in^2. "
        f"Recommended: {orifice_text}. {result['disclaimer']}"
    )
    return {"answer": answer, "data": result}


def ask(query: str) -> dict:
    decision = classify_intent(query)

    if decision.intent in ("historical", "comparative"):
        result = _handle_historical_or_comparative(query)
    elif decision.intent == "chemical_property":
        result = _handle_chemical_property(decision.chemical_name)
    elif decision.intent == "calculation":
        result = _handle_calculation(query)
        result["answer"] += f" {PE_DISCLAIMER}"
    else:
        raise ValueError(f"Unhandled intent: {decision.intent}")

    return {
        "query": query,
        "intent": decision.intent,
        "routing_reasoning": decision.reasoning,
        **result,
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is the molecular weight of anhydrous ammonia?"
    print(json.dumps(ask(query), indent=2))
