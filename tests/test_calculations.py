"""PSV sizing correctness tests: the API 520 Part I coefficient C and the
required-area equation are checked against known reference values, plus
API 526 orifice selection and the input-validation / edge-case warnings
added during the quality pass."""
import re

import pytest

from src.tools import calculations


def test_coefficient_c_matches_api_520_table_k_1_4():
    # API 520 Part I publishes C = 356 for k = 1.4 (diatomic ideal gas) --
    # the most commonly cited value in the standard's C-vs-k table.
    assert calculations._coefficient_c(1.4) == pytest.approx(356.0, abs=0.5)


def test_coefficient_c_matches_api_520_table_k_1_13():
    # k = 1.13 (propane) -> C ~= 330 per the same table.
    assert calculations._coefficient_c(1.13) == pytest.approx(330.0, abs=1.0)


def test_size_psv_vapor_worked_example():
    # 50,000 lb/hr propane vapor (M=44.1, k=1.13), relieving at 660 R (200 F),
    # 200 psig set pressure, 10% overpressure, Z=1.0, Kd=0.975, Kb=Kc=1.0.
    result = calculations.size_psv_vapor(
        mass_flow_lb_hr=50000,
        molecular_weight=44.1,
        relieving_temp_rankine=660,
        set_pressure_psig=200,
        k=1.13,
        compressibility_z=1.0,
        overpressure_fraction=0.10,
    )
    assert result["intermediate"]["p1_psia"] == pytest.approx(234.7, abs=0.01)
    assert result["required_area_in2"] == pytest.approx(2.5616, abs=0.001)
    assert result["recommended_orifice"] == {"designation": "L", "area_in2": 2.853}
    assert result["warnings"] == []
    assert "Professional Engineer" in result["disclaimer"]


def test_recommend_orifice_exact_boundary_and_none():
    assert calculations.recommend_orifice(0.503) == {"designation": "G", "area_in2": 0.503}
    assert calculations.recommend_orifice(0.5031) == {"designation": "H", "area_in2": 0.785}
    assert calculations.recommend_orifice(100.0) is None  # bigger than largest letter (T)


@pytest.mark.parametrize(
    "kwargs, message_fragment",
    [
        (dict(mass_flow_lb_hr=-1), "mass_flow_lb_hr"),
        (dict(mass_flow_lb_hr=0), "mass_flow_lb_hr"),
        (dict(molecular_weight=0), "molecular_weight"),
        (dict(relieving_temp_rankine=0), "relieving_temp_rankine"),
        (dict(k=1.0), "k (specific heat ratio)"),
        (dict(k=0.9), "k (specific heat ratio)"),
        (dict(compressibility_z=0), "compressibility_z"),
        (dict(kd=0), "kd"),
        (dict(kb=1.5), "kb"),
        (dict(kc=-0.1), "kc"),
        (dict(overpressure_fraction=-0.1), "overpressure_fraction"),
        (dict(set_pressure_psig=-20, atmospheric_psia=14.7), "set_pressure_psig"),
    ],
)
def test_size_psv_vapor_rejects_invalid_inputs(kwargs, message_fragment):
    base = dict(
        mass_flow_lb_hr=5000, molecular_weight=44, relieving_temp_rankine=600,
        set_pressure_psig=250, k=1.3,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=re.escape(message_fragment)):
        calculations.size_psv_vapor(**base)


def test_size_psv_vapor_warns_on_low_flow():
    result = calculations.size_psv_vapor(
        mass_flow_lb_hr=0.5, molecular_weight=44, relieving_temp_rankine=600, set_pressure_psig=250,
    )
    assert any("under 1 lb/hr" in w for w in result["warnings"])


def test_size_psv_vapor_warns_when_exceeds_largest_orifice():
    result = calculations.size_psv_vapor(
        mass_flow_lb_hr=5_000_000, molecular_weight=44, relieving_temp_rankine=600, set_pressure_psig=250,
    )
    assert result["recommended_orifice"] is None
    assert any("exceeds the largest standard" in w for w in result["warnings"])


def test_size_psv_vapor_warns_on_non_ideal_compressibility():
    result = calculations.size_psv_vapor(
        mass_flow_lb_hr=5000, molecular_weight=44, relieving_temp_rankine=600,
        set_pressure_psig=250, compressibility_z=0.2,
    )
    assert any("supercritical" in w or "non-ideal" in w for w in result["warnings"])
