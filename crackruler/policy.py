"""Exposure-dependent crack-width guideline and decision policy.

Source of the width table: ACI 224R-01 (Reapproved 2008), "Control of Cracking in
Concrete Structures", Table 4.1 "Guide to reasonable crack widths, reinforced concrete
under service loads", as reproduced with ACI permission in ASPIRE, Winter 2018, p. 42.

The table is a *design guideline for new construction*, not an acceptance criterion for
condition assessment. ACI notes that a portion of cracks will exceed these values and that
with time a significant portion can. CrackRuler therefore uses it only to route a finding
(log / monitor / engineer review); it never certifies a structure as safe.
"""
from __future__ import annotations

ACI_224R_TABLE_4_1_MM = {
    "dry_air": ("Dry air or protective membrane", 0.41),
    "humid": ("Humidity, moist air, soil", 0.30),
    "deicing": ("Deicing chemicals", 0.18),
    "seawater": ("Seawater and seawater spray, wetting and drying", 0.15),
    "water_retaining": ("Water-retaining structures (excluding nonpressure pipes)", 0.10),
}

ACTIONS = {
    "no_crack": "No crack found. Archive the record.",
    "log": "Crack below the exposure guideline. Log it and re-inspect on the normal cycle.",
    "monitor": "Crack close to the exposure guideline. Mark it and re-measure within 3 months.",
    "engineer_review": "Crack wider than the exposure guideline. A structural engineer must review it before any decision.",
    "retake": "Evidence is not good enough to measure. Retake the photo.",
    "human_review": "The system is not confident. A person must look at this image.",
}


def guideline_mm(exposure: str) -> float:
    return ACI_224R_TABLE_4_1_MM[exposure][1]


def decide(width_p95_mm: float | None, width_unc_mm: float, exposure: str, calibrated: bool) -> dict:
    """Map a measured width (95th percentile along the crack) to an action.

    The band [limit - unc, limit + unc] is treated as "too close to call" -> monitor or review.
    """
    limit = guideline_mm(exposure)
    if width_p95_mm is None:
        return {"action": "no_crack", "limit_mm": limit, "reason": "no crack segments survived verification"}
    if not calibrated:
        return {"action": "human_review", "limit_mm": limit,
                "reason": "no scale reference (ArUco marker) in view, widths are in pixels only"}
    if width_p95_mm + width_unc_mm < limit * 0.8:
        a = "log"
    elif width_p95_mm - width_unc_mm > limit:
        a = "engineer_review"
    else:
        a = "monitor"
    return {"action": a, "limit_mm": limit,
            "reason": f"p95 width {width_p95_mm:.2f} mm (+/- {width_unc_mm:.2f}) vs guideline {limit:.2f} mm ({ACI_224R_TABLE_4_1_MM[exposure][0]})"}
