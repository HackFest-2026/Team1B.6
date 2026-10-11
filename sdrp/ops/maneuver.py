"""Collision-avoidance maneuver recommendation (minimum delta-v, propellant-aware).

For each candidate burn time (k half-revolutions before TCA, never in the past) and
direction (+/- along-track, +/- radial, +/- cross-track) we use Clohessy-Wiltshire
relative-motion dynamics to map an impulse into a displacement of the primary at TCA, then
bisect on the impulse magnitude until Pc drops below the target. Every option is then costed
with the burn needed to return to the nominal orbital slot afterwards and checked against the
spacecraft's usable propellant; the cheapest option that fits wins. If none fits, fall-back
actions are proposed (avoid and drift, accept more risk, ask the other operator, wait for
fresher tracking).
Along-track burns ~half an orbit or more before TCA are usually most efficient (secular
along-track drift grows as 3*dv*t), which the search discovers automatically.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from ..astro.pc import pc_foster
from ..astro.propagate import MU, ric_frame


def cw_displacement(n, t, dv_ric):
    """Position change in RIC (km) at time t (s) after an impulse dv_ric (km/s)."""
    vx, vy, vz = dv_ric
    nt = n * t
    x = math.sin(nt) / n * vx + 2.0 / n * (1 - math.cos(nt)) * vy
    y = 2.0 / n * (math.cos(nt) - 1) * vx + (4 * math.sin(nt) - 3 * nt) / n * vy
    z = math.sin(nt) / n * vz
    return np.array([x, y, z])


def _search(ev, C, hbr_km, now, target_pc, max_dv_ms, recovery_days):
    """All burn options that bring Pc below ``target_pc``, each costed with its recovery burn."""
    r1, v1, r2, v2 = (np.asarray(ev[k]) for k in ("r1", "v1", "r2", "v2"))
    dr0 = r2 - r1
    dv_rel = v2 - v1
    a = 1.0 / (2 / np.linalg.norm(r1) - np.dot(v1, v1) / MU)
    n = math.sqrt(MU / a**3)
    period = 2 * math.pi / n
    M = ric_frame(r1, v1)                  # rows R, I, C (at TCA)
    time_left = (ev["tca"] - now).total_seconds()
    dirs = {"+along-track": (0, 1, 0), "-along-track": (0, -1, 0), "+radial": (1, 0, 0),
            "-radial": (-1, 0, 0), "+cross-track": (0, 0, 1), "-cross-track": (0, 0, -1)}
    tr = recovery_days * 86400.0
    options = []
    for half_revs in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12):
        t = half_revs * period / 2
        if t > time_left - 600:            # need 10 min to upload commands
            continue
        for name, d in dirs.items():
            d = np.array(d, float)

            def pc_for(dv_kms):
                dpos_ric = cw_displacement(n, t, d * dv_kms)
                new_dr = dr0 - M.T @ dpos_ric          # primary moves -> relative vector changes
                return pc_foster(new_dr, dv_rel, C, hbr_km)[0], new_dr

            hi = max_dv_ms / 1000.0
            if pc_for(hi)[0] > target_pc:
                continue
            lo = 0.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if pc_for(mid)[0] > target_pc:
                    lo = mid
                else:
                    hi = mid
            pc_new, new_dr = pc_for(hi)
            off = cw_displacement(n, t, d * hi)        # R, I, C offset from the nominal slot at TCA
            # recovery: cancel the burn (dv) and, for along-track burns, drift the secular
            # offset 3*dv_along*t back within the recovery window (a start and a stop burn)
            rec = hi + 2.0 * float(abs(d[1])) * hi * t / tr
            options.append({"burn_time": (ev["tca"] - timedelta(seconds=t)).isoformat(),
                            "lead_time_h": t / 3600.0, "revs_before_tca": half_revs / 2,
                            "direction": name, "dv_ms": hi * 1000.0, "recovery_dv_ms": rec * 1000.0,
                            "total_dv_ms": (hi + rec) * 1000.0, "pc_after": pc_new,
                            "miss_after_km": float(np.linalg.norm(new_dr)),
                            "slot_offset_km": float(np.linalg.norm(off)),
                            "drift_km_per_day": float(3.0 * float(abs(d[1])) * hi * 86400.0),
                            "_d": d, "_dv": hi, "_t": t})
    return options, n


def _path(opt, n, recovery_days, steps=40):
    """Offset from the nominal slot (km, R/I/C) from the burn to the end of the recovery."""
    d, dv, t = opt["_d"], opt["_dv"], opt["_t"]
    tr = recovery_days * 86400.0
    pts = []
    for k in range(steps + 1):
        tau = t * k / steps
        r, i, c = cw_displacement(n, tau, d * dv)
        pts.append({"t_h": (tau - t) / 3600.0, "radial_km": float(r), "along_km": float(i), "cross_km": float(c)})
    r0, i0, c0 = (pts[-1][k] for k in ("radial_km", "along_km", "cross_km"))
    for k in range(1, 13):                     # recovery leg: the offset is drifted back to zero
        f = 1.0 - k / 12.0
        pts.append({"t_h": k / 12.0 * tr / 3600.0, "radial_km": r0 * f, "along_km": i0 * f, "cross_km": c0 * f})
    return pts


def fuel_verdict(pc0, best, fuel_ms, sk_ms_per_year=None):
    """Is the burn worth its propellant? A spent satellite should not burn its disposal reserve."""
    if best is None or fuel_ms is None:
        return {"verdict": "NO_BURN", "fraction_used": None, "maneuvers_affordable": 0}
    frac = best["total_dv_ms"] / max(fuel_ms, 1e-9)
    afford = int(fuel_ms // max(best["total_dv_ms"], 1e-9))
    if frac <= 0.25:
        v, why = "WORTH_IT", "Uses a small share of the usable propellant."
    elif frac <= 0.6:
        v, why = "COSTLY", "Uses a large share of the usable propellant; fewer burns will be possible afterwards."
    else:
        v, why = "VERY_COSTLY", "Would use most of the usable propellant; only justified for a very high-risk encounter."
    out = {"verdict": v, "reason": why, "fraction_used": frac, "maneuvers_affordable": afford,
           "left_after_ms": max(fuel_ms - best["total_dv_ms"], 0.0)}
    if sk_ms_per_year:
        out["station_keeping_days"] = best["total_dv_ms"] / sk_ms_per_year * 365.0
    return out


def recommend(ev, C, hbr_km, now: datetime, target_pc=1e-5, max_dv_ms=5.0, fuel_ms=None,
              recovery_days=2.0, secondary_maneuverable=False, relax_pc=None, age_days=None,
              sk_ms_per_year=None):
    """Cheapest avoidance maneuver, its recovery to the nominal slot, and fall-backs.

    fuel_ms: usable propellant (m/s), or None for unlimited. relax_pc: the highest residual Pc the
    operator would accept if the target cannot be afforded.
    """
    r1, v1, r2, v2 = (np.asarray(ev[k]) for k in ("r1", "v1", "r2", "v2"))
    pc0 = pc_foster(r2 - r1, v2 - v1, C, hbr_km)[0]
    # allow the search up to the full cap so affordable-but-one-way options are still found
    options, n = _search(ev, C, hbr_km, now, target_pc, max_dv_ms, recovery_days)
    for o in options:
        o["returns_to_slot"] = bool(fuel_ms is None or o["total_dv_ms"] <= fuel_ms)
        o["affordable_one_way"] = bool(fuel_ms is None or o["dv_ms"] <= fuel_ms)
    options.sort(key=lambda o: o["total_dv_ms"])
    full = [o for o in options if o["returns_to_slot"]]
    oneway = [o for o in options if o["affordable_one_way"]]
    best = full[0] if full else (min(oneway, key=lambda o: o["dv_ms"]) if oneway else None)

    # naive reference: radial burn a quarter rev before TCA (typical late "reactive" maneuver)
    late = sorted((o for o in options if o["revs_before_tca"] == 0.25), key=lambda o: o["dv_ms"])
    naive = next((o for o in late if o["direction"].endswith("radial")), None) or (late[0] if late else None)
    eff = None
    if best and naive and naive["dv_ms"] > 0:
        eff = 1 - best["dv_ms"] / naive["dv_ms"]

    if pc0 < target_pc:
        status = "NO_MANEUVER_NEEDED"
    elif full:
        status = "RETURNS_TO_SLOT"
    elif oneway:
        status = "AVOIDS_BUT_DRIFTS"
    else:
        status = "NOT_FEASIBLE"

    alts = []
    if status == "AVOIDS_BUT_DRIFTS":
        alts.append({"action": "Avoid and accept a new orbital slot",
                     "detail": f"A {best['dv_ms']:.3f} m/s burn avoids the encounter, but the {best['total_dv_ms']:.3f} m/s round trip "
                               f"exceeds the {fuel_ms:.3f} m/s available. The satellite ends {best['slot_offset_km']:.1f} km off "
                               f"its slot, drifting {best['drift_km_per_day']:.0f} km/day until it is re-phased or retired."})
    if status in ("AVOIDS_BUT_DRIFTS", "NOT_FEASIBLE"):
        if relax_pc and pc0 > relax_pc:
            ro, _ = _search(ev, C, hbr_km, now, relax_pc, max_dv_ms, recovery_days)
            ro = [o for o in ro if fuel_ms is None or o["total_dv_ms"] <= fuel_ms]
            if ro:
                o = min(ro, key=lambda o: o["total_dv_ms"])
                alts.append({"action": "Accept a higher residual risk",
                             "detail": f"A {o['direction']} burn of {o['dv_ms']:.3f} m/s ({o['total_dv_ms']:.3f} m/s with the return) "
                                       f"cuts Pc from {pc0:.1e} to {o['pc_after']:.1e}, below {relax_pc:.0e}, and fits the propellant."})
        if secondary_maneuverable:
            alts.append({"action": "Ask the other operator to maneuver",
                         "detail": "The secondary is an active payload; share the avoidance burn through operator coordination."})
        if age_days is not None and age_days > 1.0:
            alts.append({"action": "Get fresher tracking first",
                         "detail": f"The orbit data is {age_days:.1f} days old; a new tracking update shrinks the uncertainty and often lowers Pc without any burn."})
        alts.append({"action": "Re-evaluate at the next update",
                     "detail": "Keep monitoring; the latest burn that still works is in the options table."})
        if fuel_ms is not None and fuel_ms <= 0:
            alts.append({"action": "No usable propellant",
                         "detail": "Nothing is left above the disposal reserve. Passive monitoring and notifying the other operator are the only options."})

    res = {"pc_before": pc0, "target_pc": target_pc, "best": best, "naive_reference": naive,
           "dv_saving_vs_naive": eff, "feasible": best is not None, "status": status,
           "fuel": {**fuel_verdict(pc0, best, fuel_ms, sk_ms_per_year), "available_ms": fuel_ms, "avoid_dv_ms": best["dv_ms"] if best else None,
                    "recovery_dv_ms": best["recovery_dv_ms"] if best else None,
                    "total_dv_ms": best["total_dv_ms"] if best else None,
                    "remaining_after_ms": (fuel_ms - best["total_dv_ms"]) if (best and fuel_ms is not None) else fuel_ms,
                    "recovery_days": recovery_days},
           "alternatives": alts,
           "path": _path(best, n, recovery_days) if best else [],
           "options": options[:12]}
    for o in options:
        for k in ("_d", "_dv", "_t"):
            o.pop(k, None)
    return res
