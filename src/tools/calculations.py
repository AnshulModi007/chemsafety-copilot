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
    """
    if k <= 1.0:
        raise ValueError("k (specific heat ratio) must be > 1.0")

    p1_psia = set_pressure_psig * (1.0 + overpressure_fraction) + atmospheric_psia
    c = _coefficient_c(k)

    required_area_in2 = (
        mass_flow_lb_hr / (c * kd * p1_psia * kb * kc)
        * math.sqrt(relieving_temp_rankine * compressibility_z / molecular_weight)
    )

    orifice = recommend_orifice(required_area_in2)

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
