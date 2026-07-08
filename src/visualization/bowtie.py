"""Bowtie risk diagram: threats (left) -> critical event (center) ->
consequences (right), with barriers/safeguards marked as tick marks on the
connecting lines. Only some CSB reports describe an incident with a clean
enough threat/barrier/consequence structure to draw this meaningfully --
when extraction can't find that shape, get_incident_diagram() falls back to
the simpler causal-chain flowchart instead of forcing a bowtie that would
misrepresent the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

from groq import Groq
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL  # noqa: E402
from src.visualization.causal_chain import extract_causal_chain, generate_causal_chain_svg  # noqa: E402

_client = Groq()

BOWTIE_PROMPT = """Given excerpts from a U.S. Chemical Safety Board (CSB) incident investigation \
report, determine whether the incident has a clear "bowtie" risk structure: one central critical \
event, multiple distinct threats/causes that could lead to it, and multiple distinct consequences \
that followed it. Also note any barriers/safeguards mentioned (that were missing, failed, or present).

If the excerpts support this structure clearly, extract it. If they don't (e.g. only one threat, or \
no clear separation between cause and effect), set "has_structure" to false and leave the other \
fields empty -- do not force a structure the excerpts don't actually support.

Respond with ONLY a JSON object:
{"has_structure": true|false, "critical_event": "<short label, or empty>", \
"threats": ["<short threat label>", ...up to 4], "consequences": ["<short consequence label>", ...up to 4], \
"barriers": ["<short safeguard/barrier label>", ...up to 6]}
"""


class Bowtie(BaseModel):
    has_structure: bool = False
    critical_event: str = ""
    threats: list[str] = []
    consequences: list[str] = []
    barriers: list[str] = []


def extract_bowtie(query: str, hits: list[dict]) -> dict | None:
    """Best-effort bowtie structure extraction from already-retrieved chunks.

    Returns:
        A dict with keys {critical_event, threats, consequences, barriers}
        if the report clearly supports a bowtie (>=2 threats, >=1
        consequence, a named critical event), else None.
    """
    if not hits:
        return None
    context = "\n\n---\n\n".join(h["text"][:2000] for h in hits[:3])
    try:
        response = _client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": BOWTIE_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nReport excerpts:\n\n{context}"},
            ],
            response_format={"type": "json_object"},
        )
        bowtie = Bowtie.model_validate_json(response.choices[0].message.content)
    except Exception:
        return None

    if not bowtie.has_structure or not bowtie.critical_event:
        return None
    if len(bowtie.threats) < 2 or len(bowtie.consequences) < 1:
        return None
    return bowtie.model_dump(exclude={"has_structure"})


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def _wrap_box(x: float, y: float, w: float, h: float, text: str, color: str, anchor: str) -> str:
    import textwrap
    lines = textwrap.wrap(text, width=20)[:3]
    parts = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
             f'fill="{color}" opacity="0.16" stroke="{color}" stroke-width="1.5"/>']
    line_y = y + h / 2 - (len(lines) - 1) * 7 + 4
    for line in lines:
        parts.append(f'<text x="{x + w / 2:.1f}" y="{line_y:.1f}" font-size="10.5" text-anchor="{anchor}" '
                      f'fill="currentColor">{_xml_escape(line)}</text>')
        line_y += 14
    return "".join(parts)


def generate_bowtie_svg(bowtie: dict, title: str | None = None) -> str:
    """Render a threat / critical-event / consequence bowtie diagram as SVG."""
    threats, consequences, barriers = bowtie["threats"], bowtie["consequences"], bowtie["barriers"]
    n_left, n_right = len(threats), len(consequences)
    box_w, box_h = 160.0, 44.0
    row_gap = 16.0
    total_h = max(n_left, n_right) * (box_h + row_gap) + 80
    cx = 350.0
    cy = total_h / 2
    left_x, right_x = 20.0, 700 - 20 - box_w

    parts = [f'<svg viewBox="0 0 700 {total_h:.0f}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
    if title:
        parts.append(f'<text x="350" y="24" text-anchor="middle" font-size="14" font-weight="600" '
                      f'fill="currentColor">{_xml_escape(title)}</text>')

    # critical event, center
    event_w, event_h = 130.0, 60.0
    parts.append(_wrap_box(cx - event_w / 2, cy - event_h / 2, event_w, event_h,
                            bowtie["critical_event"], "#C0392B", "middle"))

    barrier_i = 0

    def _connector(x1: float, y1: float, x2: float, y2: float) -> str:
        nonlocal barrier_i
        seg = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>'
        if barrier_i < len(barriers):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            seg += (f'<line x1="{mx - 6:.1f}" y1="{my - 8:.1f}" x2="{mx + 6:.1f}" y2="{my + 8:.1f}" '
                    f'stroke="#3B7DBF" stroke-width="3"/>')
            barrier_i += 1
        return seg

    left_start_y = cy - (n_left * (box_h + row_gap) - row_gap) / 2
    for i, threat in enumerate(threats):
        y = left_start_y + i * (box_h + row_gap)
        parts.append(_wrap_box(left_x, y, box_w, box_h, threat, "#D9A441", "middle"))
        parts.append(_connector(left_x + box_w, y + box_h / 2, cx - event_w / 2, cy))

    right_start_y = cy - (n_right * (box_h + row_gap) - row_gap) / 2
    for i, consequence in enumerate(consequences):
        y = right_start_y + i * (box_h + row_gap)
        parts.append(_wrap_box(right_x, y, box_w, box_h, consequence, "#888888", "middle"))
        parts.append(_connector(cx + event_w / 2, cy, right_x, y + box_h / 2))

    remaining = barriers[barrier_i:]
    if remaining:
        parts.append(f'<text x="350" y="{total_h - 12:.0f}" text-anchor="middle" font-size="10" '
                      f'fill="currentColor" opacity="0.7">Also noted: {_xml_escape(", ".join(remaining))}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def get_incident_diagram(query: str, hits: list[dict], title: str | None = None) -> dict | None:
    """Best available incident diagram for a historical/comparative answer:
    tries a bowtie first (richer, but needs a clear threat/consequence
    structure), falls back to a causal-chain flowchart, and returns None if
    neither can be built from what was actually retrieved.

    Returns:
        {"kind": "bowtie"|"causal_chain", "svg": "<svg ...>"} or None.
    """
    bowtie = extract_bowtie(query, hits)
    if bowtie:
        return {"kind": "bowtie", "svg": generate_bowtie_svg(bowtie, title)}

    stages = extract_causal_chain(query, hits)
    svg = generate_causal_chain_svg(stages, title)
    if svg:
        return {"kind": "causal_chain", "svg": svg}

    return None


if __name__ == "__main__":
    import xml.etree.ElementTree as ET

    demo = {
        "critical_event": "Vapor cloud ignited",
        "threats": ["Corroded pipe wall", "Inadequate inspection program", "No isolation valve"],
        "consequences": ["Fire spread to unit", "3 injuries", "$50M property damage"],
        "barriers": ["Gas detection system", "Emergency shutdown"],
    }
    svg = generate_bowtie_svg(demo, title="Demo Bowtie")
    ET.fromstring(svg)
    print("valid, length:", len(svg))
