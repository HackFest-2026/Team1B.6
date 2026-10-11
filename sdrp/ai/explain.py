"""Explainable alerts (XAI) for conjunction risk.

For each conjunction we answer "why did this alert fire?" with an attribution that
operators can audit, in the spirit of SHAP but computed exactly on the physics model:

* one-factor-at-a-time attribution in log10(Pc): each driver (miss distance, radial
  separation, position uncertainty, drag-driven uncertainty, TLE age, hard-body radius)
  is replaced by its catalog-typical value and the change in Pc is recorded; positive
  values mean the driver *raises* the risk relative to a typical conjunction;
* a counterfactual: the miss distance at which the alert would clear;
* a telemetry-style readout ([ALERT] / [DRIVERS] / [DIAGNOSTIC] / [ACTION]).
"""
from __future__ import annotations

import math

import numpy as np

from ..astro.pc import encounter_plane, pc_foster

FLOOR = 1e-10      # Pc below this is reported as 'negligible'


def _lg(p):
    return math.log10(max(p, FLOOR))


def typical_values(events_meta):
    """Reference values = medians over the conjunctions that carry non-negligible risk
    (Pc >= 1e-7), so attributions compare an alert with other risky encounters."""
    risky = [e for e in events_meta if e.get("pc", 0) >= 1e-7]
    if len(risky) >= 10:
        events_meta = risky
    miss = [e["miss_km"] for e in events_meta]
    hbr = [e["hbr_km"] for e in events_meta]
    sig = [e["sigma_major_km"] for e in events_meta if e.get("sigma_major_km")]
    age = [e["age_days"] for e in events_meta if e.get("age_days") is not None]
    return {"miss_km": float(np.median(miss)) if miss else 5.0,
            "hbr_km": float(np.median(hbr)) if hbr else 0.01,
            "sigma_major_km": float(np.median(sig)) if sig else 1.0,
            "age_days": float(np.median(age)) if age else 1.0}


def explain(dr, dv, C, hbr_km, typical, cov_terms=None, red=1e-4, yellow=1e-5, maneuver=None,
            names=("primary", "secondary")):
    """cov_terms: optional dict with alternative combined covariances:
       {'no_drag': C_without_drag_growth, 'fresh': C_with_age=typical}"""
    dr = np.asarray(dr, float); dv = np.asarray(dv, float); C = np.asarray(C, float)
    pc0, info = pc_foster(dr, dv, C, hbr_km)
    lg0 = _lg(pc0)
    B = encounter_plane(dr, dv)
    miss = float(np.linalg.norm(dr))
    contrib = []

    # 1. miss distance (scale relative vector to the typical miss)
    k = typical["miss_km"] / max(miss, 1e-6)
    p = pc_foster(dr * k, dv, C, hbr_km)[0]
    contrib.append(("Miss distance", lg0 - _lg(p), f"{miss * 1000:.0f} m vs typical {typical['miss_km'] * 1000:.0f} m"))

    # 2. radial separation (radial component replaced by typical share of the miss)
    rhat = None
    try:
        rhat = np.asarray(cov_terms.get("radial_dir")) if cov_terms else None
    except Exception:
        rhat = None
    if rhat is not None and rhat.shape == (3,):
        rad = float(np.dot(dr, rhat))
        dr2 = dr - rad * rhat + np.sign(rad or 1) * max(abs(rad), typical["miss_km"] * 0.4) * rhat
        p = pc_foster(dr2, dv, C, hbr_km)[0]
        contrib.append(("Radial separation", lg0 - _lg(p), f"{abs(rad) * 1000:.0f} m radial"))

    # 3. position uncertainty (scale covariance so the major axis is typical)
    s = typical["sigma_major_km"] / max(info["sigma_major_km"], 1e-9)
    p = pc_foster(dr, dv, C * s * s, hbr_km)[0]
    contrib.append(("Position uncertainty", lg0 - _lg(p),
                    f"σ {info['sigma_major_km'] * 1000:.0f}/{info['sigma_minor_km'] * 1000:.0f} m vs typical {typical['sigma_major_km'] * 1000:.0f} m"))

    # 4. drag-driven uncertainty (AI density model) and 5. TLE age
    if cov_terms and cov_terms.get("no_drag") is not None:
        p = pc_foster(dr, dv, np.asarray(cov_terms["no_drag"]), hbr_km)[0]
        contrib.append(("Drag uncertainty", lg0 - _lg(p), cov_terms.get("drag_note", "")))
    if cov_terms and cov_terms.get("fresh") is not None:
        p = pc_foster(dr, dv, np.asarray(cov_terms["fresh"]), hbr_km)[0]
        contrib.append(("TLE age", lg0 - _lg(p), cov_terms.get("age_note", "")))

    # 6. hard-body radius
    p = pc_foster(dr, dv, C, typical["hbr_km"])[0]
    contrib.append(("Hard-body radius", lg0 - _lg(p), f"{hbr_km * 1000:.1f} m vs typical {typical['hbr_km'] * 1000:.1f} m"))

    total = sum(abs(c[1]) for c in contrib) or 1.0
    drivers = [{"factor": f, "delta_log10_pc": float(d), "share": float(abs(d) / total),
                "direction": "raises risk" if d > 0.05 else "lowers risk" if d < -0.05 else "neutral",
                "detail": det} for f, d, det in contrib]
    drivers.sort(key=lambda d: -abs(d["delta_log10_pc"]))

    # counterfactual: miss distance at which the alert would clear (Pc < yellow)
    clear_km = None
    if pc0 >= yellow:
        lo, hi = 1.0, 1.0
        while pc_foster(dr * hi, dv, C, hbr_km)[0] >= yellow and hi < 1e4:
            hi *= 2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if pc_foster(dr * mid, dv, C, hbr_km)[0] >= yellow:
                lo = mid
            else:
                hi = mid
        clear_km = miss * hi

    risk = "RED" if pc0 >= red else "YELLOW" if pc0 >= yellow else "GREEN"
    top = [d for d in drivers if d["direction"] == "raises risk"][:2]
    lines = [f"[ALERT] {risk}: Pc {pc0:.1e} "
             f"({'≥' if pc0 >= yellow else '<'} {'1e-4 red' if pc0 >= red else '1e-5 yellow'} threshold)."]
    if top:
        lines.append("[DRIVERS] " + "; ".join(f"{d['factor']} {d['delta_log10_pc']:+.1f} decades ({d['detail']})" for d in top) + ".")
    diag = cov_terms.get("diagnostic") if cov_terms else None
    if diag:
        lines.append(f"[DIAGNOSTIC] {diag}")
    if clear_km:
        lines.append(f"[COUNTERFACTUAL] Alert clears if the miss distance grows to {clear_km * 1000:,.0f} m.")
    if maneuver and maneuver.get("best"):
        b = maneuver["best"]
        lines.append(f"[ACTION] Recommend {b['dv_ms']:.3f} m/s {b['direction']} burn {b['lead_time_h']:.1f} h before TCA "
                     f"→ Pc {b['pc_after']:.0e}.")
    elif pc0 >= red:
        lines.append("[ACTION] Primary cannot maneuver: request additional tracking of both objects.")
    else:
        lines.append("[ACTION] Monitor; re-assess after next tracking update.")
    return {"pc": pc0, "drivers": drivers, "clear_miss_km": clear_km, "readout": lines}
