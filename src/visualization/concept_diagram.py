"""Generic process/concept diagram for general_knowledge answers. Unlike the
PSV schematic, incident bowtie/causal-chain, or GHS pictograms -- each of
which has one fixed shape -- a general chemical-engineering question could be
about literally any concept (a tray tower, a heat exchanger, a distillation
column, an activation-energy curve, ...), so there's no single template to
draw from.

Same division of labor as causal_chain.py though: an LLM call decides
whether a diagram would actually help THIS answer and, if so, extracts an
ordered list of labeled components; a plain Python function then draws them
as a neutral vertical flow diagram. The model never touches SVG/XML, and a
failed or negative extraction just means "no diagram", not an error.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from groq import Groq
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL  # noqa: E402

_client = Groq()

CONCEPT_DIAGRAM_PROMPT = """You decide whether THIS SPECIFIC question warrants an accompanying \
schematic diagram, and if so extract its labeled components in order.

Set "diagram_needed" to true ONLY if the question itself is fundamentally asking to explain a physical \
structure or layout -- e.g. "what is a tray tower", "how does a heat exchanger work", "explain the \
parts of a distillation column", or an explicit ask for a diagram/picture/sketch/illustration/schematic.

Set "diagram_needed" to FALSE for every other question, even about the same equipment -- this is the \
common mistake to avoid: the topic being "diagrammable" equipment does NOT mean every question about it \
needs a diagram. Say false for questions about a narrower attribute or follow-up detail: cost, \
materials of construction, operating conditions, efficiency, sizing, comparisons between two designs, \
troubleshooting/failure modes, or "why"/"when" questions that don't ask to explain the layout itself. \
Also say false for purely conceptual/definitional questions with no physical layout at all (e.g. \
"what is mass transfer", "what is activation energy", "what is the ideal gas law").

If diagram_needed is true, extract an ordered list of 2 to 8 labeled components representing the \
physical layout or process flow, each with a short optional note (e.g. for a tray tower: \
"Reflux condenser", "Tray N (liquid inlet)", "Tray N-1 (liquid & vapor)", ..., "Reboiler"). Order them \
the way they naturally appear top-to-bottom or in process-flow order.

Respond with ONLY a JSON object:
{"diagram_needed": true|false, "title": "<short diagram title, or empty>", \
"components": [{"label": "<component name>", "note": "<short note, or empty>"}, ...]}
"""


class ConceptComponent(BaseModel):
    label: str
    note: str = ""


class ConceptDiagram(BaseModel):
    diagram_needed: bool = False
    title: str = ""
    components: list[ConceptComponent] = []


def extract_concept_diagram(query: str, answer: str) -> dict | None:
    """Best-effort decision + extraction: does this general-knowledge answer
    warrant a diagram, and if so what are its labeled components?

    Args:
        query: the user's question.
        answer: the already-generated general-knowledge answer text, used as
            the source of truth for what components to draw (rather than
            re-deriving them from the question alone).

    Returns:
        {"title", "components": [{"label", "note"}, ...]} if a diagram is
        warranted and has >=2 components, else None -- callers should treat
        None as "no diagram for this answer", not an error. Never raises:
        a failed extraction call fails soft to None, since a diagram is a
        nice-to-have on top of the text answer, not the answer itself.
    """
    try:
        response = _client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": CONCEPT_DIAGRAM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nAnswer given:\n{answer[:2000]}"},
            ],
            response_format={"type": "json_object"},
        )
        parsed = ConceptDiagram.model_validate_json(response.choices[0].message.content)
    except Exception:
        return None

    if not parsed.diagram_needed or len(parsed.components) < 2:
        return None
    return {"title": parsed.title, "components": [c.model_dump() for c in parsed.components]}


_BOX_COLOR = "#3B82F6"  # matches the general_knowledge badge color in the UI


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def generate_concept_diagram_svg(diagram: dict) -> str | None:
    """Render a vertical, single-color flow diagram of labeled components
    (e.g. the trays of a tray tower, top to bottom).

    Args:
        diagram: output of extract_concept_diagram (has "title", "components").

    Returns:
        An SVG document string, or None if there are fewer than 2 components
        -- a single box isn't a diagram worth drawing.
    """
    components = diagram.get("components", [])
    if len(components) < 2:
        return None

    title = diagram.get("title") or ""
    box_w, box_h, gap = 400, 60, 30
    margin_top = 50 if title else 20
    total_w = box_w + 40
    total_h = margin_top + len(components) * box_h + (len(components) - 1) * gap + 20

    parts = [f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
    if title:
        parts.append(
            f'<text x="{total_w / 2}" y="26" text-anchor="middle" font-size="14" '
            f'font-weight="600" fill="currentColor">{_xml_escape(title)}</text>'
        )

    y = margin_top
    for i, comp in enumerate(components):
        label_lines = textwrap.wrap(comp.get("label", ""), width=40)[:2]
        note = comp.get("note") or ""

        parts.append(
            f'<rect x="20" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="{_BOX_COLOR}" opacity="0.14" stroke="{_BOX_COLOR}" stroke-width="2"/>'
        )
        text_y = y + 24 if note else y + box_h / 2 + 5
        for li, line in enumerate(label_lines):
            parts.append(
                f'<text x="{20 + box_w / 2}" y="{text_y + li * 16}" text-anchor="middle" '
                f'font-size="13" font-weight="600" fill="currentColor">{_xml_escape(line)}</text>'
            )
        if note:
            note_y = text_y + len(label_lines) * 16 + 4
            parts.append(
                f'<text x="{20 + box_w / 2}" y="{note_y}" text-anchor="middle" '
                f'font-size="10.5" fill="currentColor" opacity="0.75">{_xml_escape(note)}</text>'
            )

        if i < len(components) - 1:
            cx = 20 + box_w / 2
            parts.append(
                f'<line x1="{cx}" y1="{y + box_h}" x2="{cx}" y2="{y + box_h + gap - 6}" '
                f'stroke="currentColor" stroke-width="2" marker-end="url(#concept-arrow)"/>'
            )
        y += box_h + gap

    parts.append(
        '<defs><marker id="concept-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/></marker></defs>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    import xml.etree.ElementTree as ET

    demo = {
        "title": "Tray Tower (Distillation Column)",
        "components": [
            {"label": "Reflux condenser", "note": "condenses overhead vapor"},
            {"label": "Tray N", "note": "liquid inlet"},
            {"label": "Tray N-1", "note": "liquid & vapor contact"},
            {"label": "Reboiler", "note": "vaporizes bottoms liquid"},
        ],
    }
    svg = generate_concept_diagram_svg(demo)
    ET.fromstring(svg)
    print("valid, length:", len(svg))
