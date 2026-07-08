"""Diagram-generation tests: every SVG generator (PSV schematic, causal chain,
comparison, bowtie, GHS pictograms, general-knowledge concept diagrams) must
produce well-formed XML for valid inputs, and the documented "no diagram"
fallbacks (None) for inputs too thin to draw meaningfully. Only the
pure-render functions are exercised here -- the LLM-based extraction
functions (extract_causal_chain, extract_bowtie, extract_concept_diagram)
require a live Groq call and are out of scope for this offline suite."""
import xml.etree.ElementTree as ET

from src.visualization.bowtie import generate_bowtie_svg
from src.visualization.causal_chain import generate_causal_chain_svg, generate_comparison_svg
from src.visualization.concept_diagram import generate_concept_diagram_svg
from src.visualization.ghs_pictograms import generate_ghs_svg, hazard_categories
from src.visualization.psv_schematic import generate_psv_svg

STAGES_A = [
    {"kind": "precondition", "label": "Corroded pipe wall"},
    {"kind": "escalation", "label": "Vapor cloud formed"},
    {"kind": "critical_event", "label": "Vapor cloud ignited"},
    {"kind": "consequence", "label": "3 injuries"},
]
STAGES_B = [
    {"kind": "precondition", "label": "Inadequate inspection program"},
    {"kind": "critical_event", "label": "Vessel ruptured"},
]


def _assert_valid_svg(svg: str) -> None:
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")


def test_psv_schematic_valid_with_orifice():
    svg = generate_psv_svg(
        {"mass_flow_lb_hr": 5000, "molecular_weight": 44, "relieving_temp_rankine": 600, "set_pressure_psig": 250},
        {"designation": "L", "area_in2": 2.853},
    )
    _assert_valid_svg(svg)


def test_psv_schematic_valid_without_orifice():
    svg = generate_psv_svg(
        {"mass_flow_lb_hr": 5_000_000, "molecular_weight": 44, "relieving_temp_rankine": 600, "set_pressure_psig": 250},
        None,
    )
    _assert_valid_svg(svg)


def test_causal_chain_svg_requires_at_least_two_stages():
    assert generate_causal_chain_svg([]) is None
    assert generate_causal_chain_svg(STAGES_A[:1]) is None
    svg = generate_causal_chain_svg(STAGES_A, title="Demo Incident")
    _assert_valid_svg(svg)
    assert "Demo Incident" in svg


def test_comparison_svg_requires_at_least_two_usable_chains():
    assert generate_comparison_svg([("Only one", STAGES_A)]) is None
    assert generate_comparison_svg([("A", STAGES_A[:1]), ("B", STAGES_B[:1])]) is None
    svg = generate_comparison_svg([("Incident A", STAGES_A), ("Incident B", STAGES_B)])
    _assert_valid_svg(svg)
    assert "Incident A" in svg and "Incident B" in svg


def test_bowtie_svg_valid():
    bowtie = {
        "critical_event": "Vapor cloud ignited",
        "threats": ["Corroded pipe", "No isolation valve"],
        "consequences": ["Fire spread", "3 injuries"],
        "barriers": ["Gas detection"],
    }
    svg = generate_bowtie_svg(bowtie, title="Demo Bowtie")
    _assert_valid_svg(svg)


def test_ghs_pictograms_recognized_codes():
    statements = [
        "H270: May cause or intensify fire; oxidizer",
        "H314: Causes severe skin burns and eye damage",
        "H331: Toxic if inhaled",
        "H400: Very toxic to aquatic life",
    ]
    assert hazard_categories(statements) == ["oxidizer", "corrosive", "toxic", "environment"]
    svg = generate_ghs_svg(statements)
    _assert_valid_svg(svg)


def test_ghs_pictograms_no_recognized_codes_returns_none():
    assert generate_ghs_svg(["not a hazard code"]) is None
    assert generate_ghs_svg([]) is None


def test_concept_diagram_svg_valid_with_title_and_notes():
    diagram = {
        "title": "Tray Tower",
        "components": [
            {"label": "Reflux condenser", "note": "condenses overhead vapor"},
            {"label": "Tray N", "note": "liquid inlet"},
            {"label": "Reboiler", "note": "vaporizes bottoms liquid"},
        ],
    }
    svg = generate_concept_diagram_svg(diagram)
    _assert_valid_svg(svg)
    assert "Tray Tower" in svg and "Reflux condenser" in svg


def test_concept_diagram_svg_requires_at_least_two_components():
    assert generate_concept_diagram_svg({"title": "", "components": []}) is None
    assert generate_concept_diagram_svg({"title": "", "components": [{"label": "Only one", "note": ""}]}) is None
