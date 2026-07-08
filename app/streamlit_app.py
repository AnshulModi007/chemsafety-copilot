"""Streamlit frontend for ChemSafety Copilot. Talks to the FastAPI backend over
HTTP (BACKEND_URL env var, default http://localhost:8000) rather than importing
the agent in-process -- this is a separate deployable service, matching the
brief's "FastAPI backend + Streamlit frontend" architecture.

Run locally with: streamlit run app/streamlit_app.py
(with the backend already running: uvicorn app.main:app)
"""
import json
import os
import re
from typing import Iterator

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

PE_DISCLAIMER = (
    "Ask about past CSB chemical incidents, look up a chemical's properties, or size a "
    "relief valve. This reflects historical findings and reference data, not a stamped "
    "engineering judgment -- consult a licensed Professional Engineer for any real design "
    "or safety decision."
)

# Single source of truth for each tool's badge color, icon, label, and
# per-tool loading text -- drives the sidebar legend, the answer badge, and
# the loading placeholder shown as soon as the "routing" SSE event names the
# intent (before any answer text has streamed in).
INTENT_META = {
    "historical": {
        "icon": "📜", "label": "Historical RAG", "color": "#8B5CF6",
        "loading": "Searching CSB incident reports…",
    },
    "chemical_property": {
        "icon": "🧪", "label": "PubChem lookup", "color": "#14B8A6",
        "loading": "Looking up chemical properties on PubChem…",
    },
    "calculation": {
        "icon": "🧮", "label": "PSV sizing", "color": "#22C55E",
        "loading": "Sizing relief valve (API 520)…",
    },
    "comparative": {
        "icon": "⚖️", "label": "Comparative CRAG", "color": "#F59E0B",
        "loading": "Retrieving and comparing incidents…",
    },
    "general_knowledge": {
        "icon": "💡", "label": "General knowledge", "color": "#3B82F6",
        "loading": "Answering from general chemical-engineering knowledge…",
    },
}

EXAMPLES_BY_INTENT = {
    "historical": [
        "What caused the ammonium nitrate explosion at West Fertilizer?",
    ],
    "chemical_property": [
        "What is the molecular weight of chlorine?",
    ],
    "calculation": [
        "Size a relief valve for a mass flow of 5000 lb/hr, molecular weight 44, "
        "relieving temperature 200F, set pressure 150 psig.",
    ],
    "comparative": [
        "Compare the root causes of the West Fertilizer explosion and another "
        "ammonium nitrate incident.",
    ],
    "general_knowledge": [
        "What is a tray tower, and what's the preferred design formula?",
        "What is mass transfer?",
    ],
}

DIAGRAM_KIND_LABEL = {
    "bowtie": "Bowtie risk diagram",
    "causal_chain": "Causal chain",
    "side_by_side": "Incident comparison",
}

MAX_HISTORY_TURNS = 6
MAX_RECENT_QUERIES = 5

# Inline citation tags ([[report:id:page]] / [[web:title|url]]) the streaming
# backend emits mid-answer, stripped for the live "typing" preview -- the
# final structured render pulls these back out as a proper citations table
# instead (see src/generation/generate.py's _extract_*_citations).
_CITE_TAG_RE = re.compile(r"\[\[(?:report|web):[^\]]*\]\]")

st.set_page_config(page_title="ChemSafety Copilot", page_icon="⚗️")

st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0.9rem 0;
    }
    [data-testid="stChatMessageContent"] p {
        line-height: 1.7;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 1.25rem;
    }
    .diagram-wrap {
        max-width: 100%;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    .diagram-wrap svg {
        width: 100%;
        height: auto;
        display: block;
    }
    .copy-btn {
        font-size: 0.8em;
        padding: 0.2rem 0.7rem;
        border-radius: 0.4rem;
        border: 1px solid rgba(140, 140, 140, 0.4);
        background: transparent;
        color: inherit;
        cursor: pointer;
        margin-top: 0.35rem;
    }
    .copy-btn:hover {
        border-color: rgba(140, 140, 140, 0.8);
    }
    .metric-card {
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        border: 1px solid;
    }
    .metric-card .metric-label {
        font-size: 0.8em;
        opacity: 0.75;
    }
    .metric-card .metric-value {
        font-size: 1.3em;
        font-weight: 700;
    }
    /* Stack st.columns vertically on narrow / mobile viewports rather than
       squeezing them into unreadably thin slivers. */
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_answer_text(text: str) -> None:
    st.write(text)


def _display_text(accumulated: str) -> str:
    clean = _CITE_TAG_RE.sub("", accumulated)
    # Hold back a citation tag that's still streaming in (opened but not yet
    # closed) rather than flash its raw brackets for a moment.
    tail_open = accumulated.rfind("[[")
    tail_close = accumulated.rfind("]]")
    if tail_open > tail_close:
        clean = _CITE_TAG_RE.sub("", accumulated[:tail_open])
    return clean


def _quiet_label(text: str) -> str:
    """A muted, borderless inline label -- a quiet aside rather than a loud
    pill badge, so it reads as a subtitle instead of UI chrome competing
    with the answer."""
    return f'<span style="color:rgba(140,140,140,0.95);font-size:0.85em;">{text}</span>'


def _badge_html(intent: str | None) -> str:
    """Colored-dot tool badge: purple=historical, teal=chemical_property,
    green=calculation, amber=comparative -- keyed off the `intent` field the
    backend already returns, so no schema change was needed."""
    meta = INTENT_META.get(intent, {"icon": "🤖", "label": intent or "unknown", "color": "#888888"})
    return (
        '<span style="display:inline-flex;align-items:center;gap:0.4rem;'
        'font-size:0.85em;opacity:0.92;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{meta["color"]};'
        'display:inline-block;"></span>'
        f'{meta["icon"]} {meta["label"]}</span>'
    )


def _js_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("`", "\\`").replace("</script", "<\\/script")


def copy_button(text: str, label: str = "Copy") -> None:
    """Client-side copy-to-clipboard button (no server round-trip) -- safe to
    place on every structured card since it's plain HTML/JS, not a stateful
    Streamlit widget, so it never collides with other widgets' keys."""
    st.markdown(
        f'<button class="copy-btn" onclick="navigator.clipboard.writeText(`{_js_escape(text)}`)">'
        f"📋 {label}</button>",
        unsafe_allow_html=True,
    )


def render_diagram(svg: str | None, filename: str, key: str, caption: str | None = None) -> None:
    """Render an auto-generated SVG diagram inline plus a download button.
    A no-op if svg is None -- diagram generation fails soft (see
    src/visualization), so "no diagram for this answer" is an expected,
    silent outcome, not an error state."""
    if not svg:
        return
    if caption:
        st.caption(caption)
    # st.markdown(unsafe_allow_html=True) runs raw HTML through a CommonMark
    # HTML-block parser first, and a blank line terminates an HTML block --
    # anything after the first blank line in the SVG would silently vanish
    # (observed with the PSV schematic's multi-line template). Strip blank
    # lines defensively here so no diagram generator has to remember this.
    svg = "\n".join(line for line in svg.splitlines() if line.strip())
    st.markdown(f'<div class="diagram-wrap">{svg}</div>', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download SVG", data=svg, file_name=filename, mime="image/svg+xml",
        key=key, use_container_width=False,
    )


def render_citations(citations: list[dict] | None) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        if citations and "url" in citations[0]:
            st.dataframe(
                [{"Title": c["title"], "URL": c["url"]} for c in citations],
                hide_index=True, use_container_width=True,
            )
        else:
            st.dataframe(
                [{"Report": c["report_id"], "Page": c["page"]} for c in citations],
                hide_index=True, use_container_width=True,
            )


def render_historical(result: dict, key: str) -> None:
    data = result.get("data", {})
    if data.get("crag_insufficient") and data.get("source") != "web":
        st.warning(result["answer"], icon="⚠️")
    else:
        render_answer_text(result["answer"])
    copy_button(result["answer"], label="Copy answer")

    if data.get("crag_rewritten_query"):
        st.caption(
            f"Initial retrieval was weak -- retried with rewritten query: "
            f"*{data['crag_rewritten_query']}*"
        )
    if data.get("sub_queries"):
        with st.expander("Sub-questions used for retrieval"):
            for q in data["sub_queries"]:
                st.markdown(f"- {q}")

    diagram = data.get("diagram")
    if diagram:
        render_diagram(
            diagram["svg"], filename=f"{diagram['kind']}_{key}.svg", key=f"dl_diagram_{key}",
            caption=DIAGRAM_KIND_LABEL.get(diagram["kind"], "Incident diagram"),
        )
    render_citations(data.get("citations"))


def render_chemical_property(result: dict, key: str) -> None:
    data = result.get("data", {})
    if data.get("pubchem_unavailable"):
        st.warning(result["answer"], icon="⚠️")
        return
    render_answer_text(result["answer"])

    if data.get("molecular_formula"):
        cols = st.columns(4)
        cols[0].metric("Molecular weight", data.get("molecular_weight") or "—")
        cols[1].metric("Formula", data.get("molecular_formula") or "—")
        cols[2].metric("XLogP", data.get("xlogp") or "—")
        cols[3].metric("TPSA", data.get("tpsa") or "—")
        st.caption(f"[PubChem CID {data['cid']}]({data['pubchem_url']})")
        summary = (
            f"{data.get('iupac_name') or data.get('query')}\n"
            f"Formula: {data.get('molecular_formula')}\n"
            f"Molecular weight: {data.get('molecular_weight')}\n"
            f"SMILES: {data.get('canonical_smiles')}\n"
            f"XLogP: {data.get('xlogp')}  TPSA: {data.get('tpsa')}\n"
            f"PubChem: {data.get('pubchem_url')}"
        )
        copy_button(summary, label="Copy properties")

    render_diagram(
        data.get("ghs_diagram_svg"), filename=f"ghs_{key}.svg", key=f"dl_ghs_{key}",
        caption="GHS hazard pictograms" if data.get("ghs_diagram_svg") else None,
    )


def _render_attempt(attempt: dict) -> None:
    st.markdown(
        f"**Attempt {attempt['attempt']}** — retrieval: `{attempt['retrieval_method']}` "
        f"— path: `{attempt['path']}`"
    )
    if attempt.get("expansion_queries"):
        st.caption("Expansion queries: " + "; ".join(attempt["expansion_queries"]))
    if attempt.get("hyde_passage"):
        st.caption(f"HyDE passage: *{attempt['hyde_passage']}*")
    rows = [
        {
            "Report": c["report_title"],
            "Section": c["section"],
            "Rerank score": round(c["rerank_score"], 3) if c.get("rerank_score") is not None else None,
            "Verdict": c.get("verdict") or "—",
            "Reason": c.get("reason") or "—",
        }
        for c in attempt.get("chunks", [])
    ]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def render_trace(trace: list[dict] | None) -> None:
    if not trace:
        return
    with st.expander("🔍 Under the hood"):
        if trace and "sub_query" in trace[0]:
            for i, entry in enumerate(trace):
                st.markdown(f"**Sub-question:** {entry['sub_query']}")
                for attempt in entry["attempts"]:
                    _render_attempt(attempt)
                if i < len(trace) - 1:
                    st.divider()
        else:
            for attempt in trace:
                _render_attempt(attempt)


def render_faithfulness(faithfulness: dict | None) -> None:
    if not faithfulness or faithfulness.get("faithful", True):
        return
    claims = faithfulness.get("unsupported_claims") or []
    msg = "Possible unsupported claim(s) detected in this answer -- verify against the sources before relying on it."
    if claims:
        msg += "\n" + "\n".join(f"- {c}" for c in claims)
    st.warning(msg, icon="🔎")


def render_calculation(result: dict, key: str) -> None:
    data = result.get("data", {})
    if data.get("invalid_input"):
        st.warning(result["answer"], icon="⚠️")
        return
    render_answer_text(result["answer"])

    if "required_area_in2" in data:
        inputs = data.get("inputs", {})
        orifice = data.get("recommended_orifice")
        area = data["required_area_in2"]
        orifice_label = (
            f"{orifice['designation']} ({orifice['area_in2']} in²)"
            if orifice else "none standard -- use multiple valves"
        )

        st.markdown("###### PSV sizing summary")
        rows = [
            {"Parameter": "Mass flow", "Value": f"{inputs.get('mass_flow_lb_hr'):,.1f} lb/hr"},
            {"Parameter": "Molecular weight", "Value": f"{inputs.get('molecular_weight'):g} lb/lbmol"},
            {"Parameter": "Relieving temperature", "Value": f"{inputs.get('relieving_temp_rankine'):.1f} °R"},
            {"Parameter": "Set pressure", "Value": f"{inputs.get('set_pressure_psig'):g} psig"},
            {"Parameter": "k (Cp/Cv)", "Value": f"{inputs.get('k'):g}"},
            {"Parameter": "Compressibility Z", "Value": f"{inputs.get('compressibility_z'):g}"},
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

        cols = st.columns(2)
        cols[0].markdown(
            '<div class="metric-card" style="border-color:#22C55E;background:rgba(34,197,94,0.12);">'
            '<div class="metric-label">Required effective area</div>'
            f'<div class="metric-value" style="color:#22C55E;">{area:.4f} in²</div></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            '<div class="metric-card" style="border-color:rgba(140,140,140,0.4);">'
            '<div class="metric-label">Recommended API 526 orifice</div>'
            f'<div class="metric-value">{orifice_label}</div></div>',
            unsafe_allow_html=True,
        )

        for w in data.get("warnings") or []:
            st.info(w, icon="ℹ️")

        summary = (
            "PSV Sizing Summary\n"
            + "\n".join(f"{r['Parameter']}: {r['Value']}" for r in rows)
            + f"\nRequired effective area: {area:.4f} in^2"
            + f"\nRecommended orifice: {orifice_label}"
        )
        copy_button(summary, label="Copy summary")

        with st.expander("Inputs & intermediate values"):
            st.json({"inputs": data.get("inputs"), "intermediate": data.get("intermediate")})

        render_diagram(
            data.get("diagram_svg"), filename=f"psv_schematic_{key}.svg", key=f"dl_psv_{key}",
            caption="PSV cross-section schematic (illustrative, scaled to orifice)",
        )
    elif data.get("missing_required_fields"):
        with st.expander("Missing fields"):
            for f in data["missing_required_fields"]:
                st.markdown(f"- {f}")


def render_general_knowledge(result: dict, key: str) -> None:
    data = result.get("data", {})
    render_answer_text(result["answer"])
    copy_button(result["answer"], label="Copy answer")
    render_diagram(
        data.get("diagram_svg"), filename=f"concept_{key}.svg", key=f"dl_concept_{key}",
        caption="Concept diagram" if data.get("diagram_svg") else None,
    )


RENDERERS = {
    "historical": render_historical,
    "comparative": render_historical,
    "chemical_property": render_chemical_property,
    "calculation": render_calculation,
    "general_knowledge": render_general_knowledge,
}


def render_assistant_turn(result: dict, key: str) -> None:
    intent = result.get("intent")
    data = result.get("data", {})

    label_parts = [_badge_html(intent)]
    if data.get("source") == "web":
        label_parts.append(_quiet_label("🌐 web search"))
    if result.get("from_cache"):
        label_parts.append(_quiet_label("⚡ cached"))
    st.markdown(" · ".join(label_parts), unsafe_allow_html=True)

    if result.get("resolved_query"):
        st.caption(f"Understood as: *{result['resolved_query']}*")

    renderer = RENDERERS.get(intent)
    if renderer:
        renderer(result, key)
    else:
        render_answer_text(result["answer"])

    render_faithfulness(data.get("faithfulness"))
    render_trace(data.get("trace"))

    footer = []
    if result.get("routing_reasoning"):
        footer.append(result["routing_reasoning"])
    confidence = data.get("confidence")
    if confidence is not None and intent in ("historical", "comparative") and data.get("source") == "internal":
        footer.append(f"retrieval confidence {confidence:.0%}")
    if footer:
        st.caption(" — ".join(footer))


def submit_feedback(turn: dict, rating: str) -> None:
    result = turn["result"]
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "query": result.get("query", ""),
                "resolved_query": result.get("resolved_query"),
                "intent": result.get("intent"),
                "answer": result.get("answer", ""),
                "rating": rating,
            },
            timeout=10,
        )
    except requests.RequestException:
        pass  # best-effort -- a logging failure shouldn't block the UI
    turn["feedback"] = rating


def render_feedback(turn: dict, key: str) -> None:
    if turn.get("feedback"):
        st.caption(f"Thanks for the feedback! ({'👍' if turn['feedback'] == 'up' else '👎'})")
        return
    cols = st.columns([1, 1, 10])
    if cols[0].button("👍", key=f"fb_up_{key}"):
        submit_feedback(turn, "up")
        st.rerun()
    if cols[1].button("👎", key=f"fb_down_{key}"):
        submit_feedback(turn, "down")
        st.rerun()


def _backend_history(max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    out = []
    for turn in st.session_state.history[-max_turns:]:
        if turn["role"] == "user":
            out.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "assistant" and "result" in turn:
            out.append({"role": "assistant", "content": turn["result"]["answer"]})
    return out


def stream_backend(query: str, history: list[dict]) -> Iterator[dict]:
    with requests.post(
        f"{BACKEND_URL}/ask/stream", json={"query": query, "history": history},
        stream=True, timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            yield json.loads(line[len("data: "):])


if "history" not in st.session_state:
    st.session_state.history = []
if "disclaimer_expanded" not in st.session_state:
    # Expanded by default the first time a session opens; collapses after the
    # first question is answered and stays collapsed for the rest of the
    # session so it doesn't compete with the growing chat log.
    st.session_state.disclaimer_expanded = True

st.title("⚗️ ChemSafety Copilot")
with st.expander("⚠️ Engineering disclaimer", expanded=st.session_state.disclaimer_expanded):
    st.write(PE_DISCLAIMER)

with st.sidebar:
    if st.button("🆕 New chat", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.session_state.disclaimer_expanded = True
        st.rerun()

    st.divider()
    st.markdown("**Tools**")
    for intent, meta in INTENT_META.items():
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0;font-size:0.9em;">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{meta["color"]};'
            'display:inline-block;flex-shrink:0;"></span>'
            f'<span>{meta["icon"]} {meta["label"]}</span></div>',
            unsafe_allow_html=True,
        )

    recent_questions = [t["content"] for t in st.session_state.history if t["role"] == "user"]
    recent_questions = recent_questions[-MAX_RECENT_QUERIES:][::-1]
    if recent_questions:
        st.divider()
        st.markdown("**Recent questions**")
        for i, q in enumerate(recent_questions):
            short = q if len(q) <= 60 else q[:57] + "..."
            if st.button(short, key=f"recent_{i}", use_container_width=True):
                st.session_state.pending_query = q

    st.divider()
    st.markdown("**Try asking**")
    for intent, meta in INTENT_META.items():
        examples = EXAMPLES_BY_INTENT.get(intent, [])
        if not examples:
            continue
        with st.expander(f"{meta['icon']} {meta['label']}"):
            for q in examples:
                if st.button(q, key=f"ex_{intent}_{q}", use_container_width=True):
                    st.session_state.pending_query = q

    st.divider()
    st.caption(
        "Routes each question to one of five tools: historical RAG+CRAG over CSB "
        "reports, live PubChem lookups, API 520 relief-valve sizing, per-incident "
        "comparative retrieval, or a general chemical-engineering knowledge answer "
        "for concept questions that aren't tied to a specific chemical or incident. "
        "Falls back to a live web search when the CSB corpus has no confident answer, "
        "remembers recent turns for follow-up questions, and streams answers as "
        "they're generated."
    )

for idx, turn in enumerate(st.session_state.history):
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.write(turn["content"])
    else:
        with st.chat_message("assistant", avatar="⚗️"):
            if "error" in turn:
                st.error(turn["error"])
            else:
                render_assistant_turn(turn["result"], key=str(idx))
                render_feedback(turn, key=str(idx))

query = st.chat_input("Ask a question...")
if not query and st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")

if query:
    history_for_backend = _backend_history()
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant", avatar="⚗️"):
        placeholder = st.empty()
        placeholder.markdown(_quiet_label("Thinking…"), unsafe_allow_html=True)
        accumulated = ""
        final_result = None
        error = None
        try:
            for event in stream_backend(query, history_for_backend):
                if event["type"] == "routing":
                    meta = INTENT_META.get(event.get("intent"), {})
                    loading_text = meta.get("loading", "Working…")
                    placeholder.markdown(
                        _badge_html(event.get("intent")) + "&nbsp;&nbsp;" + _quiet_label(loading_text),
                        unsafe_allow_html=True,
                    )
                elif event["type"] == "delta":
                    accumulated += event["text"]
                    placeholder.write(_display_text(accumulated) + "▌")
                elif event["type"] == "done":
                    event.pop("type")
                    final_result = event
                elif event["type"] == "error":
                    error = event.get("detail", "unknown error")
        except requests.RequestException as e:
            detail = None
            if e.response is not None:
                try:
                    detail = e.response.json().get("detail")
                except ValueError:
                    pass
            error = detail or str(e)

        placeholder.empty()
        if error:
            st.error(f"Couldn't get an answer: {error}")
            st.session_state.history.append({"role": "assistant", "error": error})
        elif final_result:
            new_key = f"new_{len(st.session_state.history)}"
            render_assistant_turn(final_result, key=new_key)
            new_turn = {"role": "assistant", "result": final_result}
            st.session_state.history.append(new_turn)
            render_feedback(new_turn, key=new_key)
        else:
            st.error("No response received from backend.")

    st.session_state.disclaimer_expanded = False
