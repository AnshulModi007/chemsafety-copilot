"""Top-level agentic entrypoint: classify intent, dispatch to the right tool,
return a uniform response envelope regardless of which path was taken.

Diagram generation (src/visualization) is wired in here as a post-processing
layer, not a 5th intent: the router's job is picking a data source, and a
diagram is a presentation concern on top of whichever tool already ran, not
an alternative data source itself. For historical/comparative/calculation/
chemical_property it always attempts a diagram (no keyword gate); for
general_knowledge specifically, the diagram is conditional -- an LLM call
decides whether one is actually warranted (explicitly asked for, or the
concept has a natural physical layout) since forcing a diagram onto every
possible chemE concept question would misrepresent purely conceptual ones.
Every path fails soft -- a missing or malformed diagram never blocks the
underlying answer.
"""
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agent import semantic_cache  # noqa: E402
from src.agent.router import classify_intent, extract_psv_params, reformulate_query  # noqa: E402
from src.generation.crag import retrieve_with_crag  # noqa: E402
from src.generation.generate import (  # noqa: E402
    GENERAL_KNOWLEDGE_DISCLAIMER, _web_fallback, generate_from_hits, generate_from_hits_stream,
    generate_from_web_stream, generate_general_knowledge, generate_general_knowledge_stream,
    generate_with_crag, generate_with_crag_stream,
)
from src.tools import calculations  # noqa: E402
from src.tools.pubchem import CompoundNotFound, PubChemUnavailable, get_compound_properties  # noqa: E402
from src.tools.websearch import WebSearchUnavailable, web_search  # noqa: E402
from src.visualization.bowtie import get_incident_diagram  # noqa: E402
from src.visualization.causal_chain import extract_causal_chain, generate_comparison_svg  # noqa: E402
from src.visualization.concept_diagram import extract_concept_diagram, generate_concept_diagram_svg  # noqa: E402
from src.visualization.ghs_pictograms import generate_ghs_svg  # noqa: E402
from src.visualization.psv_schematic import generate_psv_svg  # noqa: E402

logger = logging.getLogger(__name__)

PE_DISCLAIMER = (
    "This reflects historical incident findings and reference data, not a stamped engineering "
    "judgment. Consult a licensed Professional Engineer for any real design or safety decision."
)


def _mean_rerank_score(chunks: list[dict]) -> float:
    """Average cross-encoder rerank score across a set of chunks, 0.0 if empty."""
    scores = [c.get("rerank_score", 0.0) for c in chunks]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _report_title_from_chunks(chunks: list[dict]) -> str | None:
    """The most common report_title among a set of chunks -- used as a
    diagram caption when a single incident's chunks are available."""
    if not chunks:
        return None
    return Counter(c["report_title"] for c in chunks).most_common(1)[0][0]


def _safe_incident_diagram(query: str, used_chunks: list[dict]) -> dict | None:
    """get_incident_diagram, but never lets a diagram-generation failure
    propagate into the main answer path -- diagrams are a nice-to-have."""
    if not used_chunks:
        return None
    try:
        diagram = get_incident_diagram(query, used_chunks, title=_report_title_from_chunks(used_chunks))
    except Exception:
        logger.warning("incident diagram generation failed", exc_info=True)
        return None
    logger.info("incident diagram: %s", diagram["kind"] if diagram else "none")
    return diagram


def _safe_concept_diagram(query: str, answer: str) -> str | None:
    """Best-effort general-knowledge concept diagram: never lets extraction
    or rendering failures propagate, and logs whether a diagram was actually
    judged necessary for this answer (most general_knowledge answers won't
    get one -- see extract_concept_diagram's diagram_needed judgment)."""
    try:
        diagram = extract_concept_diagram(query, answer)
        svg = generate_concept_diagram_svg(diagram) if diagram else None
    except Exception:
        logger.warning("concept diagram generation failed", exc_info=True)
        return None
    logger.info("concept diagram: %s", "generated" if svg else "none")
    return svg


def _handle_historical(query: str) -> dict:
    """Answer a historical (single-incident) question via CRAG, with an
    auto-generated incident diagram (bowtie, falling back to a causal-chain
    flowchart) attached when the answer is grounded in real report chunks."""
    result = generate_with_crag(query)
    diagram = _safe_incident_diagram(query, result.get("used_chunks", []))
    return {
        "answer": result["answer"],
        "data": {
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"],
            "crag_insufficient": result["crag_insufficient"],
            "crag_rewritten_query": result["crag_rewritten_query"],
            "source": result["source"],
            "confidence": result["confidence"],
            "trace": result["trace"],
            "faithfulness": result.get("faithfulness"),
            "diagram": diagram,
        },
    }


def _handle_comparative(query: str, sub_queries: list[str]) -> dict:
    """A single retrieval call over a multi-entity question can starve one
    entity's chunks out of the top-k pool entirely -- retrieve separately per
    sub-question (one per incident/entity being compared) and merge, so the
    generation step actually has grounding for every entity being compared
    instead of silently filling gaps from outside knowledge (see failure gallery).

    Also builds a side-by-side causal-chain diagram, one column per incident,
    from the same per-sub-query chunks (not the merged pool), since a merged
    pool loses which chunk belongs to which incident.
    """
    if len(sub_queries) < 2:
        return _handle_historical(query)

    merged_chunks: dict[str, dict] = {}
    any_insufficient = False
    trace = []
    per_incident_chunks: list[tuple[str, list[dict]]] = []
    for sub_query in sub_queries:
        crag_result = retrieve_with_crag(sub_query)
        any_insufficient = any_insufficient or crag_result["insufficient"]
        for chunk in crag_result["used_chunks"]:
            merged_chunks[chunk["chunk_id"]] = chunk
        trace.append({"sub_query": sub_query, "attempts": crag_result["trace"]})
        per_incident_chunks.append((sub_query, crag_result["used_chunks"]))

    if not merged_chunks:
        web_result = _web_fallback(query)
        if web_result is not None:
            return {
                "answer": web_result["answer"],
                "data": {
                    "citations": web_result["citations"],
                    "retrieved_chunks": [],
                    "crag_insufficient": True,
                    "sub_queries": sub_queries,
                    "source": "web",
                    "confidence": 0.0,
                    "trace": trace,
                    "faithfulness": web_result.get("faithfulness"),
                    "diagram": None,
                },
            }
        return {
            "answer": (
                "None of the incidents in this comparison had sufficient grounded evidence in the "
                "retrieved report excerpts, even after per-incident retrieval and query rewriting."
            ),
            "data": {
                "citations": [], "retrieved_chunks": [], "crag_insufficient": True,
                "sub_queries": sub_queries, "source": "insufficient", "confidence": 0.0,
                "trace": trace, "faithfulness": None, "diagram": None,
            },
        }

    used_chunks = list(merged_chunks.values())
    result = generate_from_hits(query, used_chunks)
    diagram = _comparative_diagram(per_incident_chunks)
    return {
        "answer": result["answer"],
        "data": {
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"],
            "crag_insufficient": any_insufficient,
            "sub_queries": sub_queries,
            "source": "internal",
            "confidence": _mean_rerank_score(used_chunks),
            "trace": trace,
            "faithfulness": result.get("faithfulness"),
            "diagram": diagram,
        },
    }


def _comparative_diagram(per_incident_chunks: list[tuple[str, list[dict]]]) -> dict | None:
    """Best-effort side-by-side causal-chain diagram for a comparative
    answer: one causal-chain extraction per incident's own chunks, then
    laid out in columns. Never raises -- diagrams are optional."""
    try:
        named_chains = [
            (_report_title_from_chunks(chunks) or sub_query, extract_causal_chain(sub_query, chunks))
            for sub_query, chunks in per_incident_chunks
        ]
        svg = generate_comparison_svg(named_chains)
    except Exception:
        logger.warning("comparative diagram generation failed", exc_info=True)
        return None
    logger.info("comparative diagram: %s", "side_by_side" if svg else "none")
    return {"kind": "side_by_side", "svg": svg} if svg else None


def _handle_chemical_property(chemical_name: str | None) -> dict:
    """Answer a chemical-property question via a live PubChem lookup, with
    an auto-generated row of simplified GHS hazard-class pictograms."""
    if not chemical_name:
        return {
            "answer": "I couldn't tell which chemical you're asking about -- could you name it explicitly?",
            "data": {},
        }
    try:
        props = get_compound_properties(chemical_name)
    except CompoundNotFound as e:
        return {"answer": str(e), "data": {}}
    except PubChemUnavailable as e:
        logger.warning("PubChem lookup failed: %s", e)
        return {
            "answer": (
                f"I couldn't reach PubChem to look up {chemical_name} right now ({e}). "
                "Please try again in a moment."
            ),
            "data": {"pubchem_unavailable": True},
        }

    hazard_line = (
        "; ".join(props["ghs_hazard_statements"]) if props["ghs_hazard_statements"]
        else "no GHS classification available from PubChem for this compound"
    )
    answer = (
        f"{props['iupac_name'] or chemical_name} ({props['molecular_formula']}, "
        f"MW {props['molecular_weight']}). GHS hazards: {hazard_line}. "
        f"Source: PubChem CID {props['cid']} ({props['pubchem_url']})."
    )
    try:
        ghs_svg = generate_ghs_svg(props["ghs_hazard_statements"])
    except Exception:
        logger.warning("GHS pictogram generation failed", exc_info=True)
        ghs_svg = None
    return {"answer": answer, "data": {**props, "ghs_diagram_svg": ghs_svg}}


def _handle_calculation(query: str) -> dict:
    """Answer a PSV-sizing question via the API 520 vapor-relief calculation,
    with an auto-generated (illustrative, not certified) cross-section
    schematic scaled to the recommended orifice size."""
    params = extract_psv_params(query)
    missing = params.actually_missing_fields
    if missing:
        return {
            "answer": (
                "I need a few more values to size this relief valve: "
                + ", ".join(missing)
                + ". Please provide them (mass flow in lb/hr, molecular weight, relieving "
                "temperature, and PSV set pressure)."
            ),
            "data": {"missing_required_fields": missing},
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

    try:
        result = calculations.size_psv_vapor(**kwargs)
    except ValueError as e:
        logger.info("PSV sizing rejected invalid input: %s", e)
        return {
            "answer": f"I can't size this PSV: {e}",
            "data": {"invalid_input": str(e)},
        }

    orifice = result["recommended_orifice"]
    orifice_text = (
        f"API 526 orifice {orifice['designation']} ({orifice['area_in2']} in^2)"
        if orifice else "larger than the largest standard API 526 orifice -- consider multiple valves"
    )
    answer = (
        f"Required effective discharge area: {result['required_area_in2']:.4f} in^2. "
        f"Recommended: {orifice_text}. {result['disclaimer']}"
    )
    if result["warnings"]:
        answer += " " + " ".join(result["warnings"])
    try:
        diagram_svg = generate_psv_svg(result["inputs"], orifice)
    except Exception:
        logger.warning("PSV schematic generation failed", exc_info=True)
        diagram_svg = None
    return {"answer": answer, "data": {**result, "diagram_svg": diagram_svg}}


def _handle_general_knowledge(query: str) -> dict:
    """Answer a general chemical-engineering concept question (e.g. "what is
    a tray tower") that isn't about a specific chemical, incident, or
    calculation -- no tool call, just an ungrounded LLM answer, plus a
    diagram only when one is actually judged necessary (explicitly asked
    for, or the concept has a natural physical layout -- see
    _safe_concept_diagram). GENERAL_KNOWLEDGE_DISCLAIMER is appended by the
    caller (ask()/stream_ask()), not generated by the model itself -- kept
    out of the model's own output so an explicit user request for a short
    answer (e.g. "answer in one word") isn't structurally impossible to honor."""
    result = generate_general_knowledge(query)
    diagram_svg = _safe_concept_diagram(query, result["answer"])
    return {
        "answer": result["answer"],
        "data": {"citations": result["citations"], "source": "general_knowledge", "diagram_svg": diagram_svg},
    }


def ask(query: str, history: list[dict] | None = None) -> dict:
    """Classify intent, dispatch to the matching tool, and return a single
    response envelope: {query, resolved_query, intent, routing_reasoning,
    from_cache, answer, data}. See stream_ask for the token-streaming twin."""
    resolved_query = reformulate_query(query, history or [])
    decision = classify_intent(resolved_query)
    logger.info("routed intent=%s reasoning=%s", decision.intent, decision.reasoning)

    cached = semantic_cache.get_cached(resolved_query, decision.intent)
    if cached is not None:
        logger.info("served from semantic cache")
        return {
            **cached,
            "query": query,
            "resolved_query": resolved_query if resolved_query != query else None,
            "from_cache": True,
        }

    start = time.perf_counter()
    if decision.intent == "historical":
        result = _handle_historical(resolved_query)
    elif decision.intent == "comparative":
        result = _handle_comparative(resolved_query, decision.sub_queries)
    elif decision.intent == "chemical_property":
        result = _handle_chemical_property(decision.chemical_name)
    elif decision.intent == "calculation":
        result = _handle_calculation(resolved_query)
        result["answer"] += f" {PE_DISCLAIMER}"
    elif decision.intent == "general_knowledge":
        result = _handle_general_knowledge(resolved_query)
        result["answer"] += f" {GENERAL_KNOWLEDGE_DISCLAIMER}"
    else:
        raise ValueError(f"Unhandled intent: {decision.intent}")
    logger.info("intent=%s handled in %.2fs", decision.intent, time.perf_counter() - start)

    response = {
        "query": query,
        "resolved_query": resolved_query if resolved_query != query else None,
        "intent": decision.intent,
        "routing_reasoning": decision.reasoning,
        "from_cache": False,
        **result,
    }
    semantic_cache.store(resolved_query, decision.intent, response)
    return response


# --- Streaming, additive -----------------------------------------------------
# Only historical/comparative have a long-form generation step worth streaming
# token-by-token; chemical_property/calculation are a short deterministic
# string built from a tool call, with nothing to stream. Diagram generation
# runs after the text is fully streamed (using chunks already in memory) and
# is attached to the single terminal "done" event -- an SVG can't be usefully
# streamed token-by-token, and this keeps the answer's perceived latency
# unaffected at the cost of a small delay before "done" arrives.

def _handle_historical_stream(query: str):
    for kind, payload in generate_with_crag_stream(query):
        if kind == "delta":
            yield ("delta", payload)
        else:
            diagram = _safe_incident_diagram(query, payload.get("used_chunks", []))
            yield ("done", {
                "answer": payload["answer"],
                "data": {
                    "citations": payload["citations"],
                    "retrieved_chunks": payload["retrieved_chunks"],
                    "crag_insufficient": payload["crag_insufficient"],
                    "crag_rewritten_query": payload["crag_rewritten_query"],
                    "source": payload["source"],
                    "confidence": payload["confidence"],
                    "trace": payload["trace"],
                    "faithfulness": payload.get("faithfulness"),
                    "diagram": diagram,
                },
            })


def _handle_comparative_stream(query: str, sub_queries: list[str]):
    if len(sub_queries) < 2:
        yield from _handle_historical_stream(query)
        return

    merged_chunks: dict[str, dict] = {}
    any_insufficient = False
    trace = []
    per_incident_chunks: list[tuple[str, list[dict]]] = []
    for sub_query in sub_queries:
        crag_result = retrieve_with_crag(sub_query)
        any_insufficient = any_insufficient or crag_result["insufficient"]
        for chunk in crag_result["used_chunks"]:
            merged_chunks[chunk["chunk_id"]] = chunk
        trace.append({"sub_query": sub_query, "attempts": crag_result["trace"]})
        per_incident_chunks.append((sub_query, crag_result["used_chunks"]))

    if not merged_chunks:
        web_results = None
        try:
            web_results = web_search(query)
        except WebSearchUnavailable:
            pass

        if web_results:
            for kind, payload in generate_from_web_stream(query, web_results):
                if kind == "delta":
                    yield ("delta", payload)
                else:
                    yield ("done", {
                        "answer": payload["answer"],
                        "data": {
                            "citations": payload["citations"], "retrieved_chunks": [],
                            "crag_insufficient": True, "sub_queries": sub_queries,
                            "source": "web", "confidence": 0.0, "trace": trace,
                            "faithfulness": payload.get("faithfulness"), "diagram": None,
                        },
                    })
            return

        yield ("done", {
            "answer": (
                "None of the incidents in this comparison had sufficient grounded evidence in the "
                "retrieved report excerpts, even after per-incident retrieval and query rewriting."
            ),
            "data": {
                "citations": [], "retrieved_chunks": [], "crag_insufficient": True,
                "sub_queries": sub_queries, "source": "insufficient", "confidence": 0.0,
                "trace": trace, "faithfulness": None, "diagram": None,
            },
        })
        return

    used_chunks = list(merged_chunks.values())
    for kind, payload in generate_from_hits_stream(query, used_chunks):
        if kind == "delta":
            yield ("delta", payload)
        else:
            diagram = _comparative_diagram(per_incident_chunks)
            yield ("done", {
                "answer": payload["answer"],
                "data": {
                    "citations": payload["citations"],
                    "retrieved_chunks": payload["retrieved_chunks"],
                    "crag_insufficient": any_insufficient,
                    "sub_queries": sub_queries,
                    "source": "internal",
                    "confidence": _mean_rerank_score(used_chunks),
                    "trace": trace,
                    "faithfulness": payload.get("faithfulness"),
                    "diagram": diagram,
                },
            })


def _handle_general_knowledge_stream(query: str):
    for kind, payload in generate_general_knowledge_stream(query):
        if kind == "delta":
            yield ("delta", payload)
        else:
            diagram_svg = _safe_concept_diagram(query, payload["answer"])
            yield ("done", {
                "answer": payload["answer"],
                "data": {
                    "citations": payload["citations"], "source": "general_knowledge",
                    "diagram_svg": diagram_svg,
                },
            })


def stream_ask(query: str, history: list[dict] | None = None):
    """Mirrors ask()'s dispatch, first yielding exactly one ("routing", {intent,
    reasoning}) tuple as soon as intent classification completes (so a caller
    can show a tool-specific loading state before any answer text exists),
    then ("delta", text) chunks while the final answer streams in, then
    exactly one ("done", response) tuple with the same envelope shape ask()
    returns.
    """
    resolved_query = reformulate_query(query, history or [])
    decision = classify_intent(resolved_query)
    logger.info("routed intent=%s reasoning=%s", decision.intent, decision.reasoning)
    yield ("routing", {"intent": decision.intent, "reasoning": decision.reasoning})

    cached = semantic_cache.get_cached(resolved_query, decision.intent)
    if cached is not None:
        logger.info("served from semantic cache")
        yield ("done", {
            **cached,
            "query": query,
            "resolved_query": resolved_query if resolved_query != query else None,
            "from_cache": True,
        })
        return

    start = time.perf_counter()
    if decision.intent == "historical":
        stream = _handle_historical_stream(resolved_query)
    elif decision.intent == "comparative":
        stream = _handle_comparative_stream(resolved_query, decision.sub_queries)
    elif decision.intent == "chemical_property":
        result = _handle_chemical_property(decision.chemical_name)
        stream = None
    elif decision.intent == "calculation":
        result = _handle_calculation(resolved_query)
        result["answer"] += f" {PE_DISCLAIMER}"
        stream = None
    elif decision.intent == "general_knowledge":
        stream = _handle_general_knowledge_stream(resolved_query)
    else:
        raise ValueError(f"Unhandled intent: {decision.intent}")

    if stream is not None:
        result = None
        for kind, payload in stream:
            if kind == "delta":
                yield ("delta", payload)
            else:
                result = payload
        if decision.intent == "general_knowledge":
            result["answer"] += f" {GENERAL_KNOWLEDGE_DISCLAIMER}"
    logger.info("intent=%s handled in %.2fs", decision.intent, time.perf_counter() - start)

    response = {
        "query": query,
        "resolved_query": resolved_query if resolved_query != query else None,
        "intent": decision.intent,
        "routing_reasoning": decision.reasoning,
        "from_cache": False,
        **result,
    }
    semantic_cache.store(resolved_query, decision.intent, response)
    yield ("done", response)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = " ".join(sys.argv[1:]) or "What is the molecular weight of anhydrous ammonia?"
    print(json.dumps(ask(query), indent=2))
