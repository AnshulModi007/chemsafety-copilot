"""Deterministic GHS hazard-class pictograms rendered as simple SVG shapes --
not the official GHS pictogram artwork (redrawing that precisely isn't the
goal), but simplified geometric stand-ins with a text label so it's clear
at a glance which hazard classes apply. Classification is a pure lookup from
the H-code number already in each hazard statement returned by
src.tools.pubchem.get_compound_properties -- no LLM call needed.
"""
from __future__ import annotations

import re

_H_CODE_RE = re.compile(r"^H(\d{3})")

_CATEGORY_LABEL = {
    "explosive": "Explosive",
    "flammable": "Flammable",
    "oxidizer": "Oxidizer",
    "gas_cylinder": "Gas under pressure",
    "corrosive": "Corrosive",
    "toxic": "Acute toxicity",
    "irritant": "Irritant",
    "health_hazard": "Health hazard",
    "environment": "Aquatic hazard",
}

_CATEGORY_COLOR = {
    "explosive": "#C0392B",
    "flammable": "#E07B39",
    "oxidizer": "#D9A441",
    "gas_cylinder": "#5B8FB9",
    "corrosive": "#7F8C8D",
    "toxic": "#8E2DE2",
    "irritant": "#B8860B",
    "health_hazard": "#8E2DE2",
    "environment": "#2E8B57",
}

# Simplified geometric icon per hazard class -- "{c}" is filled in with the
# category's color. Coordinates are relative to a translated <g> origin.
_ICON_SHAPE = {
    "explosive": '<path d="M0,-18 L5,-5 L18,-8 L8,3 L14,16 L0,8 L-14,16 L-8,3 L-18,-8 L-5,-5 Z" fill="{c}"/>',
    "flammable": '<path d="M0,-18 C10,-6 10,6 0,18 C-6,10 -3,4 0,-2 C-3,4 -8,8 -8,2 C-8,-8 -4,-14 0,-18 Z" fill="{c}"/>',
    "oxidizer": '<circle cx="0" cy="4" r="12" fill="none" stroke="{c}" stroke-width="2.5"/>'
                '<path d="M0,-18 C4,-10 4,-4 0,2 C-4,-4 -4,-10 0,-18 Z" fill="{c}"/>',
    "gas_cylinder": '<rect x="-8" y="-16" width="16" height="32" rx="6" fill="none" stroke="{c}" stroke-width="2.5"/>'
                    '<line x1="-8" y1="-8" x2="8" y2="-8" stroke="{c}" stroke-width="2"/>',
    "corrosive": '<path d="M-10,-16 L-4,-16 L-6,2 L-2,2 L-2,16 L-10,16 Z" fill="{c}"/>'
                 '<path d="M6,-16 L12,-16 L10,6 L14,6 L14,16 L6,16 Z" fill="{c}"/>'
                 '<line x1="-14" y1="10" x2="16" y2="10" stroke="{c}" stroke-width="2"/>',
    "toxic": '<circle cx="0" cy="-2" r="13" fill="none" stroke="{c}" stroke-width="2.5"/>'
             '<circle cx="-5" cy="-4" r="2" fill="{c}"/><circle cx="5" cy="-4" r="2" fill="{c}"/>'
             '<path d="M-5,4 L5,4" stroke="{c}" stroke-width="2"/>'
             '<path d="M-10,14 L-2,10 M2,10 L10,14 M-10,18 L-2,12 M2,12 L10,18" stroke="{c}" stroke-width="1.5"/>',
    "irritant": '<path d="M0,-18 L18,14 L-18,14 Z" fill="none" stroke="{c}" stroke-width="2.5"/>'
                '<line x1="0" y1="-6" x2="0" y2="4" stroke="{c}" stroke-width="2.5"/>'
                '<circle cx="0" cy="9" r="1.6" fill="{c}"/>',
    "health_hazard": '<circle cx="0" cy="0" r="14" fill="none" stroke="{c}" stroke-width="2.5"/>'
                      '<path d="M0,-9 L2,-2 L9,0 L2,2 L0,9 L-2,2 L-9,0 L-2,-2 Z" fill="{c}"/>',
    "environment": '<path d="M-14,10 C-6,-6 6,-6 14,10" fill="none" stroke="{c}" stroke-width="2"/>'
                   '<path d="M-10,10 L-4,-4 L2,10 Z" fill="{c}"/>',
}


def _classify(h_code: str) -> str | None:
    """Map a single hazard-statement string's H-code number to a GHS hazard
    class, per the standard H2xx (physical) / H3xx (health) / H4xx
    (environmental) numbering. Returns None for an unrecognized/missing code."""
    m = _H_CODE_RE.match(h_code.strip())
    if not m:
        return None
    num = int(m.group(1))
    if 200 <= num <= 211:
        return "explosive"
    if num in (220, 221, 222, 223, 224, 225, 226, 227, 228, 242, 250, 251, 252, 260, 261):
        return "flammable"
    if 270 <= num <= 272:
        return "oxidizer"
    if 280 <= num <= 281:
        return "gas_cylinder"
    if num in (290, 314, 318):
        return "corrosive"
    if num in (300, 301, 310, 311, 330, 331):
        return "toxic"
    if num in (302, 303, 312, 313, 315, 317, 319, 320, 332, 333, 335, 336):
        return "irritant"
    if num in (304, 334, 340, 341, 350, 351, 360, 361, 362, 370, 371, 372, 373):
        return "health_hazard"
    if num in (400, 410, 411, 412, 413):
        return "environment"
    return None


def hazard_categories(hazard_statements: list[str]) -> list[str]:
    """Ordered, deduplicated GHS hazard-class categories present in a
    compound's H-code hazard statements (as returned by
    src.tools.pubchem.get_compound_properties's ghs_hazard_statements)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for statement in hazard_statements:
        cat = _classify(statement)
        if cat and cat not in seen:
            seen.add(cat)
            ordered.append(cat)
    return ordered


def generate_ghs_svg(hazard_statements: list[str]) -> str | None:
    """Render a row of simplified GHS hazard-class icons.

    Args:
        hazard_statements: the ghs_hazard_statements list from PubChem
            (e.g. ["H270: May cause or intensify fire; oxidizer", ...]).

    Returns:
        An SVG string, or None if no recognizable H-code is present (e.g.
        PubChem has no GHS classification for this compound) -- callers
        should treat None as "no pictogram row for this answer".
    """
    categories = hazard_categories(hazard_statements)
    if not categories:
        return None

    icon_size = 90
    total_w = len(categories) * icon_size
    parts = [f'<svg viewBox="0 0 {total_w} {icon_size}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
    for i, cat in enumerate(categories):
        cx = i * icon_size + icon_size / 2
        color = _CATEGORY_COLOR.get(cat, "#888888")
        shape = _ICON_SHAPE.get(cat, "").format(c=color)
        parts.append(f'<g transform="translate({cx},32)">{shape}</g>')
        parts.append(
            f'<text x="{cx}" y="70" text-anchor="middle" font-size="10" '
            f'fill="currentColor">{_CATEGORY_LABEL.get(cat, cat)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    import xml.etree.ElementTree as ET

    demo_statements = [
        "H270: May cause or intensify fire; oxidizer",
        "H314: Causes severe skin burns and eye damage",
        "H331: Toxic if inhaled",
        "H400: Very toxic to aquatic life",
    ]
    svg = generate_ghs_svg(demo_statements)
    ET.fromstring(svg)
    print("categories:", hazard_categories(demo_statements))
    print("valid, length:", len(svg))
