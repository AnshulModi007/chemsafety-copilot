"""Engineering calculation tool: pressure relief valve (PSV) sizing for vapor/gas
service per API 520 Part I -- the worked example named explicitly in the project
brief. Formula and constants below (imperial units, as API 520 defines them):

    A = W / (C * Kd * P1 * Kb * Kc) * sqrt(T * Z / M)

    A   required effective discharge area, in^2
    W   relieving mass flow rate, lb/hr
    C   coefficient dependent on the ideal-gas specific heat ratio k
    Kd  effective coefficient of discharge (0.975 for preliminary sizing)
    P1  upstream relieving pressure = set pressure * (1 + overpressure) + 14.7, psia
    Kb  backpressure correction (1.0 for conventional valves / negligible backpressure)
    Kc  combination correction (0.9 with a rupture disk upstream, else 1.0)
    T   relieving temperature, degrees Rankine
    Z   compressibility factor at relieving conditions
    M   molecular weight, lb/lbmol

This sizes the valve; it does not replace a stamped calculation from a PE.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402 (kept for path-consistency with sibling tools)

# API 526 standard orifice designations, effective area in^2 (smallest-first)
STANDARD_ORIFICES = [
    ("D", 0.110), ("E", 0.196), ("F", 0.307), ("G", 0.503), ("H", 0.785),
    ("J", 1.287), ("K", 1.838), ("L", 2.853), ("M", 3.60), ("N", 4.34),
    ("P", 6.38), ("Q", 11.05), ("R", 16.0), ("T", 26.0),
]


def _coefficient_c(k: float) -> float:
    """API 520 Eq. for C as a function of the ideal-gas specific heat ratio k."""
    return 520.0 * math.sqrt(k * (2.0 / (k + 1.0)) ** ((k + 1.0) / (k - 1.0)))


def recommend_orifice(required_area_in2: float) -> dict | None:
    """Smallest standard API 526 orifice whose area covers the requirement.

    Args:
        required_area_in2: required effective discharge area, in^2.

    Returns:
        {"designation", "area_in2"} for the smallest orifice at or above the
        requirement, or None if it exceeds the largest standard letter (T) --
        callers should treat None as "needs a custom orifice or multiple
        valves in parallel", not an error.
    """
    for letter, area in STANDARD_ORIFICES:
        if area >= required_area_in2:
            return {"designation": letter, "area_in2": area}
    return None  # larger than the largest standard letter -- needs a custom/multiple valves


def size_psv_vapor(
    mass_flow_lb_hr: float,
    molecular_weight: float,
    relieving_temp_rankine: float,
    set_pressure_psig: float,
    k: float = 1.4,
    compressibility_z: float = 1.0,
    overpressure_fraction: float = 0.10,
    kd: float = 0.975,
    kb: float = 1.0,
    kc: float = 1.0,
    atmospheric_psia: float = 14.7,
) -> dict:
    """Size a conventional spring-loaded PSV for vapor/gas relief per API 520 Part I.

    Defaults (k=1.4 diatomic ideal gas, Z=1.0, 10% overpressure, Kb=Kc=1.0) are
    the standard textbook starting point when the specific chemical's real
    properties aren't known -- callers should override with PubChem/process
    data when available and note in the result which inputs were assumed.

    Raises:
        ValueError: for physically invalid inputs (non-positive flow, molecular
            weight, temperature, or compressibility; k <= 1.0; a set pressure
            that would leave zero or negative absolute upstream pressure; a
            correction factor outside its valid (0, 1] range). This equation
            has no meaningful answer for such inputs, so it fails loudly
            rather than returning a silently-wrong area.
    """
    if k <= 1.0:
        raise ValueError("k (specific heat ratio) must be > 1.0")
    if mass_flow_lb_hr <= 0:
        raise ValueError("mass_flow_lb_hr must be > 0")
    if molecular_weight <= 0:
        raise ValueError("molecular_weight must be > 0")
    if relieving_temp_rankine <= 0:
        raise ValueError("relieving_temp_rankine must be > 0 (absolute temperature)")
    if compressibility_z <= 0:
        raise ValueError("compressibility_z must be > 0")
    if not (0 < kd <= 1.0):
        raise ValueError("kd must be in (0, 1]")
    if not (0 < kb <= 1.0):
        raise ValueError("kb must be in (0, 1]")
    if not (0 < kc <= 1.0):
        raise ValueError("kc must be in (0, 1]")
    if overpressure_fraction < 0:
        raise ValueError("overpressure_fraction must be >= 0")

    p1_psia = set_pressure_psig * (1.0 + overpressure_fraction) + atmospheric_psia
    if p1_psia <= 0:
        raise ValueError(
            "set_pressure_psig gives a non-positive absolute upstream pressure -- "
            "check units (this equation expects gauge pressure in psig)"
        )
    c = _coefficient_c(k)

    required_area_in2 = (
        mass_flow_lb_hr / (c * kd * p1_psia * kb * kc)
        * math.sqrt(relieving_temp_rankine * compressibility_z / molecular_weight)
    )

    orifice = recommend_orifice(required_area_in2)

    warnings: list[str] = []
    if orifice is None:
        warnings.append(
            f"Required area ({required_area_in2:.3f} in^2) exceeds the largest standard API 526 "
            "orifice (T, 26.0 in^2) -- this needs a custom orifice or multiple valves in parallel."
        )
    if mass_flow_lb_hr < 1.0:
        warnings.append(
            "Relieving rate is under 1 lb/hr -- verify this isn't a unit-conversion error before sizing."
        )
    if not (0.5 <= compressibility_z <= 1.1):
        warnings.append(
            f"Compressibility factor Z={compressibility_z:g} is well outside the range typical of "
            "near-ideal gas relief. Highly non-ideal or supercritical fluids are not correctly sized "
            "by this ideal/real-gas equation -- use a rigorous two-phase/flashing method "
            "(e.g. API 520 Part I Annex C, or DIERS methodology) instead."
        )

    return {
        "inputs": {
            "mass_flow_lb_hr": mass_flow_lb_hr,
            "molecular_weight": molecular_weight,
            "relieving_temp_rankine": relieving_temp_rankine,
            "set_pressure_psig": set_pressure_psig,
            "k": k,
            "compressibility_z": compressibility_z,
            "overpressure_fraction": overpressure_fraction,
            "kd": kd,
            "kb": kb,
            "kc": kc,
        },
        "intermediate": {
            "p1_psia": p1_psia,
            "coefficient_c": c,
        },
        "required_area_in2": required_area_in2,
        "recommended_orifice": orifice,
        "warnings": warnings,
        "disclaimer": (
            "Preliminary sizing only, per API 520 Part I vapor/gas relief equation. "
            "Final PSV selection, backpressure/Kb verification, and stamped calculations "
            "must be completed and approved by a licensed Professional Engineer."
        ),
    }


if __name__ == "__main__":
    # Worked example: ~5000 lb/hr of a k=1.3, MW=44 vapor at 600 R, 250 psig set point.
    result = size_psv_vapor(
        mass_flow_lb_hr=5000, molecular_weight=44, relieving_temp_rankine=600,
        set_pressure_psig=250, k=1.3,
    )
    import json
    print(json.dumps(result, indent=2))
