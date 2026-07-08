"""Extract an incident's causal chain (precondition -> escalation ->
critical event -> consequence) from the same CSB report chunks already
retrieved to answer a historical question, then render it as a color-coded
SVG flowchart -- root cause and downstream effects made visible as a
sequence, not just buried in prose.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Literal

from groq import Groq
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL  # noqa: E402

_client = Groq()

CAUSAL_CHAIN_PROMPT = """Given excerpts from a U.S. Chemical Safety Board (CSB) incident investigation \
report, extract the causal chain of the incident as a short ordered sequence of stages, from root/\
contributing cause through to consequences.

Each stage has a "kind" (one of: "precondition", "escalation", "critical_event", "consequence") and a \
short "label" (under 12 words, plain language).

- "precondition": an underlying condition or root cause that existed before the incident (e.g. \
"corroded pipe wall", "inadequate mechanical integrity program")
- "escalation": an event that worsened or propagated the situation (e.g. "hydrocarbon vapor cloud formed")
- "critical_event": the pivotal event itself (e.g. "vapor cloud ignited", "vessel ruptured") -- there \
should be exactly one
- "consequence": an outcome/result (e.g. "15 fatalities", "release of toxic gas")

Only extract stages the excerpts actually support -- if the excerpts don't contain enough detail for a \
clear chain, return fewer stages rather than guessing. Return between 0 and 7 stages total, in \
chronological order.

Respond with ONLY a JSON object:
{"stages": [{"kind": "precondition"|"escalation"|"critical_event"|"consequence", "label": "<short label>"}, ...]}
"""


class CausalStage(BaseModel):
    kind: Literal["precondition", "escalation", "critical_event", "consequence"]
    label: str


class CausalChain(BaseModel):
    stages: list[CausalStage] = []


def extract_causal_chain(query: str, hits: list[dict]) -> list[dict]:
    """Best-effort structured extraction of an incident's causal chain.

    Args:
        query: the user's (resolved) question, for context.
        hits: the retrieval chunks already used to generate the answer --
            reused here rather than re-retrieving, so this costs one extra
            fast-model call and nothing else.

    Returns:
        A list of {"kind", "label"} dicts in chronological order, or an
        empty list on any failure -- this feeds an optional diagram, not
        the core answer, so it must fail soft rather than raise.
    """
    if not hits:
        return []
    # Use the small child chunk (not parent_text) and cap to the top 3 hits --
    # this call goes to GROQ_FAST_MODEL, which on the free tier has a much
    # lower tokens-per-minute cap than the main model generate.py uses
    # parent_text with; the wider context blew straight through it (observed:
    # a 5-hit parent_text request was rejected as too large for the fast
    # model's rate limit).
    context = "\n\n---\n\n".join(h["text"][:2000] for h in hits[:3])
    try:
        response = _client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": CAUSAL_CHAIN_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nReport excerpts:\n\n{context}"},
            ],
            response_format={"type": "json_object"},
        )
        chain = CausalChain.model_validate_json(response.choices[0].message.content)
        return [s.model_dump() for s in chain.stages]
    except Exception:
        return []


_KIND_COLOR = {
    "precondition": "#D9A441",
    "escalation": "#D9A441",
    "critical_event": "#C0392B",
    "consequence": "#888888",
}
_KIND_LABEL = {
    "precondition": "Precondition",
    "escalation": "Escalation",
    "critical_event": "Critical Event",
    "consequence": "Consequence",
}


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def generate_causal_chain_svg(stages: list[dict], title: str | None = None) -> str | None:
    """Render a vertical, color-coded causal-chain flowchart as an SVG string.

    Args:
        stages: output of extract_causal_chain (list of {"kind", "label"}).
        title: optional heading drawn above the chain (e.g. the incident name).

    Returns:
        An SVG document string, or None if there are fewer than 2 stages --
        a single box isn't a "chain" worth drawing; callers should treat
        None as "no diagram for this answer".
    """
    if len(stages) < 2:
        return None

    box_w, box_h, gap = 460, 64, 34
    margin_top = 50 if title else 20
    total_w = box_w + 40
    total_h = margin_top + len(stages) * box_h + (len(stages) - 1) * gap + 20

    parts = [f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
    if title:
        parts.append(
            f'<text x="{total_w / 2}" y="26" text-anchor="middle" font-size="14" '
            f'font-weight="600" fill="currentColor">{_xml_escape(title)}</text>'
        )

    y = margin_top
    for i, stage in enumerate(stages):
        color = _KIND_COLOR.get(stage["kind"], "#888888")
        kind_label = _KIND_LABEL.get(stage["kind"], stage["kind"])
        lines = textwrap.wrap(stage["label"], width=46)[:2]

        parts.append(
            f'<rect x="20" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="{color}" opacity="0.16" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(f'<text x="34" y="{y + 18}" font-size="10" font-weight="600" fill="{color}">{kind_label.upper()}</text>')
        for li, line in enumerate(lines):
            parts.append(f'<text x="34" y="{y + 36 + li * 14}" font-size="12" fill="currentColor">{_xml_escape(line)}</text>')

        if i < len(stages) - 1:
            cx = 20 + box_w / 2
            parts.append(
                f'<line x1="{cx}" y1="{y + box_h}" x2="{cx}" y2="{y + box_h + gap - 6}" '
                f'stroke="currentColor" stroke-width="2" marker-end="url(#chain-arrow)"/>'
            )
        y += box_h + gap

    parts.append(
        '<defs><marker id="chain-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/></marker></defs>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def generate_comparison_svg(named_chains: list[tuple[str, list[dict]]]) -> str | None:
    """Render up to 3 incident causal chains side by side in one SVG, for a
    comparative query -- one column per incident, so the two (or three)
    sequences can be visually scanned against each other.

    Args:
        named_chains: [(incident_name, stages), ...] -- stages as returned
            by extract_causal_chain.

    Returns:
        An SVG string, or None if fewer than 2 chains have at least 2
        stages each (nothing meaningful to compare).
    """
    usable = [(name, stages) for name, stages in named_chains if len(stages) >= 2][:3]
    if len(usable) < 2:
        return None

    col_w, box_h, gap = 340, 56, 24
    max_rows = max(len(stages) for _, stages in usable)
    col_h = 50 + max_rows * box_h + (max_rows - 1) * gap + 20
    total_w = col_w * len(usable)
    inner_w = col_w - 30

    parts = [f'<svg viewBox="0 0 {total_w} {col_h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
    for ci, (name, stages) in enumerate(usable):
        parts.append(f'<g transform="translate({ci * col_w},0)">')
        parts.append(
            f'<text x="{col_w / 2}" y="20" text-anchor="middle" font-size="12" '
            f'font-weight="600" fill="currentColor">{_xml_escape(name)}</text>'
        )
        y = 40
        for i, stage in enumerate(stages):
            color = _KIND_COLOR.get(stage["kind"], "#888888")
            kind_label = _KIND_LABEL.get(stage["kind"], stage["kind"])
            lines = textwrap.wrap(stage["label"], width=32)[:2]
            parts.append(
                f'<rect x="15" y="{y}" width="{inner_w}" height="{box_h}" rx="6" '
                f'fill="{color}" opacity="0.16" stroke="{color}" stroke-width="1.5"/>'
            )
            parts.append(f'<text x="26" y="{y + 14}" font-size="8.5" font-weight="600" fill="{color}">{kind_label.upper()}</text>')
            for li, line in enumerate(lines):
                parts.append(f'<text x="26" y="{y + 30 + li * 12}" font-size="10.5" fill="currentColor">{_xml_escape(line)}</text>')
            if i < len(stages) - 1:
                cx = 15 + inner_w / 2
                parts.append(
                    f'<line x1="{cx}" y1="{y + box_h}" x2="{cx}" y2="{y + box_h + gap - 4}" '
                    f'stroke="currentColor" stroke-width="1.5" marker-end="url(#chain-arrow)"/>'
                )
            y += box_h + gap
        parts.append("</g>")

    parts.append(
        '<defs><marker id="chain-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/></marker></defs>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    demo_stages = [
        {"kind": "precondition", "label": "Corroded pipe wall from years of sulfidation"},
        {"kind": "escalation", "label": "Hydrocarbon vapor cloud formed after pipe rupture"},
        {"kind": "critical_event", "label": "Vapor cloud ignited"},
        {"kind": "consequence", "label": "Fire spread to adjacent units, 3 injuries"},
    ]
    print(generate_causal_chain_svg(demo_stages, title="Demo Incident"))
