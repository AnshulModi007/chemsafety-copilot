"""Parametric SVG schematic of a spring-loaded pressure relief valve (PSV)
cross-section -- illustrative only, not a certified engineering drawing.
The nozzle/disc throat visually scales with the recommended API 526 orifice
area (bigger orifice -> visibly wider throat in the same body silhouette),
matching how a real PSV product line varies trim size within a common body.
"""
from __future__ import annotations

import math

# API 526 "M" (~3.60 in^2) as the 1.0x visual scale point -- keeps the
# smallest (D, 0.110 in^2) and largest (T, 26.0 in^2) standard orifices from
# producing an illegibly narrow or a wildly oversized throat.
_REFERENCE_AREA_IN2 = 3.60
_MIN_SCALE = 0.55
_MAX_SCALE = 1.6


def _orifice_scale(area_in2: float | None) -> float:
    """Visual scale factor for the nozzle/disc throat width.

    Scales with diameter (sqrt of area), not area directly -- scaling
    directly by area would make a 10x-area orifice look absurdly (100x)
    wider instead of the ~3x wider it actually is.
    """
    if not area_in2:
        return 1.0
    raw = math.sqrt(area_in2 / _REFERENCE_AREA_IN2)
    return max(_MIN_SCALE, min(_MAX_SCALE, raw))


def generate_psv_svg(inputs: dict, orifice: dict | None) -> str:
    """Render a labeled PSV cross-section schematic as a self-contained SVG.

    Args:
        inputs: the "inputs" dict from calculations.size_psv_vapor's result
            (mass_flow_lb_hr, molecular_weight, relieving_temp_rankine,
            set_pressure_psig, ...) -- used only for the caption text.
        orifice: the "recommended_orifice" dict ({"designation", "area_in2"})
            or None if no standard orifice was large enough for the required
            area (a custom/multiple-valve case).

    Returns:
        An SVG document string, ready to render inline or offer as a download.
    """
    designation = orifice["designation"] if orifice else "custom"
    area_in2 = orifice["area_in2"] if orifice else None
    scale = _orifice_scale(area_in2)

    cx = 150.0
    body_top, body_bottom = 140.0, 300.0
    body_half_w = 55.0
    bonnet_top, bonnet_bottom = 55.0, 140.0
    bonnet_half_w = 40.0
    throat_half_w = 26.0 * scale
    disc_y = 205.0
    nozzle_top_y = 230.0
    nozzle_bottom_y = 300.0
    inlet_y = 340.0
    caption = f'Orifice {designation}' + (f' ({area_in2:g} in²)' if area_in2 else '')

    spring_coils = 5
    spring_top_y, spring_bottom_y = bonnet_top + 12, disc_y - 10
    spring_step = (spring_bottom_y - spring_top_y) / (spring_coils * 2)
    spring_points = []
    for i in range(spring_coils * 2 + 1):
        x = cx + (18 if i % 2 == 0 else -18)
        y = spring_top_y + i * spring_step
        spring_points.append(f"{x:.1f},{y:.1f}")
    spring_path = "M " + " L ".join(spring_points)

    # No blank lines in this template, on purpose: st.markdown(unsafe_allow_html=True)
    # runs raw HTML through a CommonMark HTML-block parser first, and a blank line
    # terminates an HTML block -- everything after the first blank line used to get
    # silently dropped (only the title text, before the blank line, ever rendered).
    return f'''<svg viewBox="0 0 300 420" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <text x="150" y="24" text-anchor="middle" font-size="14" font-weight="600" fill="currentColor">PSV Cross-Section (illustrative)</text>
  <text x="150" y="42" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">{caption}</text>
  <!-- bonnet -->
  <path d="M {cx - bonnet_half_w},{bonnet_bottom} L {cx - bonnet_half_w},{bonnet_top + 20} Q {cx - bonnet_half_w},{bonnet_top} {cx},{bonnet_top} Q {cx + bonnet_half_w},{bonnet_top} {cx + bonnet_half_w},{bonnet_top + 20} L {cx + bonnet_half_w},{bonnet_bottom} Z"
        fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="{cx + bonnet_half_w + 8:.0f}" y="{(bonnet_top + bonnet_bottom) / 2:.0f}" font-size="11" fill="currentColor">bonnet</text>
  <!-- spring -->
  <path d="{spring_path}" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.8"/>
  <text x="{cx - bonnet_half_w - 8:.0f}" y="{(spring_top_y + spring_bottom_y) / 2:.0f}" font-size="11" fill="currentColor" text-anchor="end">spring</text>
  <!-- body -->
  <path d="M {cx - body_half_w},{body_top} L {cx - body_half_w},{body_bottom} L {cx - throat_half_w - 10},{nozzle_bottom_y} L {cx - throat_half_w - 10},{nozzle_top_y} L {cx - throat_half_w},{disc_y + 15} L {cx + throat_half_w},{disc_y + 15} L {cx + throat_half_w + 10},{nozzle_top_y} L {cx + throat_half_w + 10},{nozzle_bottom_y} L {cx + body_half_w},{body_bottom} L {cx + body_half_w},{body_top} Z"
        fill="none" stroke="currentColor" stroke-width="2"/>
  <!-- disc -->
  <rect x="{cx - throat_half_w - 6:.1f}" y="{disc_y - 6:.1f}" width="{2 * throat_half_w + 12:.1f}" height="10" fill="currentColor" opacity="0.6"/>
  <text x="{cx + body_half_w + 8:.0f}" y="{disc_y:.0f}" font-size="11" fill="currentColor">disc</text>
  <!-- nozzle seat / throat (scales with orifice area) -->
  <text x="{cx + body_half_w + 8:.0f}" y="{(nozzle_top_y + nozzle_bottom_y) / 2:.0f}" font-size="11" fill="currentColor">nozzle seat</text>
  <!-- outlet, discharging to the side -->
  <path d="M {cx + body_half_w},{(body_top + body_bottom) / 2:.0f} L {cx + body_half_w + 45},{(body_top + body_bottom) / 2:.0f}"
        stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="{cx + body_half_w + 20:.0f}" y="{(body_top + body_bottom) / 2 - 8:.0f}" font-size="11" fill="currentColor">OUTLET</text>
  <!-- inlet, flow entering from below -->
  <line x1="{cx}" y1="{inlet_y}" x2="{cx}" y2="{nozzle_bottom_y}" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="{cx}" y="{inlet_y + 16:.0f}" text-anchor="middle" font-size="11" fill="currentColor">INLET</text>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/>
    </marker>
  </defs>
</svg>'''


if __name__ == "__main__":
    svg = generate_psv_svg({"mass_flow_lb_hr": 5000}, {"designation": "G", "area_in2": 0.503})
    print(svg)
