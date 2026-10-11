"""Satellite life-cycle stage and propellant (delta-v) budget.

A satellite is only a *satellite* while it can still be commanded. Once its design life is
over, or its propellant is down to the disposal reserve, it can no longer dodge anything: it
is a derelict and, for collision purposes, behaves like debris.

Neither TLEs nor the public catalogs say how much propellant a satellite has left, so the budget
is an *assumption*: a total delta-v budget by orbit regime, scaled by how much of the design life
has been used (launch year comes from the international designator in TLE line 1), minus a
reserve that is kept for end-of-life disposal. Replace it with operator data by putting
``{"<norad>": {"remaining_dv_ms": 12.5}}`` into ``data/fuel.json``.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime

from .. import config

# regime -> (design life [years], total delta-v budget [m/s], disposal reserve fraction,
#            routine station-keeping [m/s per year])
REGIME = {
    "LEO": (7.0, 250.0, 0.20, 12.0),
    "MEO": (12.0, 600.0, 0.20, 15.0),
    "GEO": (15.0, 1500.0, 0.20, 50.0),
}
# propellant assumptions used to turn delta-v into litres (rocket equation)
ISP_S = 220.0            # hydrazine monopropellant
PROP_DENSITY = 1.004     # kg per litre
DEFAULT_MASS_KG = 500.0
STAGES = ("commissioning", "operational", "late life", "end of life", "decaying", "inactive")
_overrides = None


def _fuel_overrides():
    global _overrides
    if _overrides is None:
        p = config.DATA_DIR / "fuel.json"
        try:
            _overrides = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            _overrides = {}
    return _overrides


def launch_year(obj):
    """Launch year from the international designator (TLE line 1, columns 10-11)."""
    try:
        yy = int(obj.line1[9:11])
    except ValueError:
        return None
    return 2000 + yy if yy < 57 else 1900 + yy


def _regime(obj):
    alt = 0.5 * (obj.perigee_km + obj.apogee_km)
    return "GEO" if abs(alt - 35786) < 2000 else "LEO" if alt < 2000 else "MEO"


def tank_litres(total_dv_ms, mass_kg=DEFAULT_MASS_KG):
    """Size of the full tank that provides ``total_dv_ms`` (Tsiolkovsky rocket equation)."""
    return mass_kg * (1.0 - math.exp(-total_dv_ms / (ISP_S * 9.80665))) / PROP_DENSITY


def propellant_litres(dv_ms, mass_kg=DEFAULT_MASS_KG, total_dv_ms=None):
    """Litres for a delta-v, as its share of the full tank (100 % = the tank at launch)."""
    if total_dv_ms:
        return tank_litres(total_dv_ms, mass_kg) * dv_ms / total_dv_ms
    return tank_litres(dv_ms, mass_kg)


def _now_year(now: datetime):
    return now.year + now.timetuple().tm_yday / 366.0


def lifecycle(obj, now: datetime, reentry_days: float | None = None):
    """Life-cycle stage, usable avoidance delta-v and end-of-life date of an object."""
    reg = _regime(obj)
    if obj.obj_type != "PAYLOAD":
        return {"stage": "inactive (no propulsion)", "stage_key": "inactive", "maneuverable": False,
                "age_years": None, "usable_dv_ms": 0.0, "remaining_dv_ms": 0.0, "assumed": False, "regime": reg,
                "note": "Rocket bodies and fragments cannot maneuver."}
    life, total, reserve_f, sk_year = REGIME[reg]
    ly = launch_year(obj)
    age = max(_now_year(now) - ly, 0.0) if ly else None
    used = min(age / life, 1.0) if age is not None else 0.5
    remaining = total * (1.0 - used)
    reserve = total * reserve_f
    usable = max(remaining - reserve, 0.0) if used < 1.0 else 0.0
    assumed = True
    ov = _fuel_overrides().get(str(obj.norad))
    if ov and ov.get("remaining_dv_ms") is not None:
        remaining = float(ov["remaining_dv_ms"]); usable = max(remaining - reserve, 0.0); assumed = False
    decaying = reentry_days is not None and reentry_days < 60
    if decaying:
        key, stage = "decaying", "decaying / re-entry phase"
    elif age is not None and age < 0.5:
        key, stage = "commissioning", "commissioning / early operations"
    elif used >= 1.0 or usable <= 0:
        key, stage = "end of life", "end of life: spent, cannot maneuver"
    elif used >= 0.6:
        key, stage = "late life", "late life (conserving propellant)"
    else:
        key, stage = "operational", "operational (station-keeping)"
    can = usable > 0 and key not in ("decaying", "end of life")
    eol = None if ly is None else ly + life
    return {"stage": stage, "stage_key": key, "maneuverable": can,
            "age_years": None if age is None else round(age, 1), "launch_year": ly, "design_life_years": life,
            "life_used_pct": round(used * 100, 0), "total_dv_ms": total,
            "remaining_dv_ms": round(remaining, 2), "reserve_dv_ms": round(reserve, 2),
            "used_dv_ms": round(total - remaining, 2), "reserve_held_dv_ms": round(min(reserve, remaining), 2),
            "reserve_pct": round(reserve_f * 100), "tank_litres": round(tank_litres(total, float(obj.mass_kg or DEFAULT_MASS_KG)), 1),
            "usable_dv_ms": round(usable if can else 0.0, 2), "sk_dv_ms_per_year": sk_year,
            "mass_kg": float(obj.mass_kg or DEFAULT_MASS_KG), "isp_s": ISP_S, "prop_density": PROP_DENSITY,
            "eol_year": eol, "years_to_eol": None if eol is None else round(eol - _now_year(now), 1),
            "assumed": assumed, "regime": reg,
            "note": ("Past its design life: it can no longer be commanded to avoid anything, so for collision "
                     "purposes it is debris." if key == "end of life" else "")}


def fleet(objs, now: datetime, reentry_days: dict | None = None, red_norads=()):
    """Life-cycle picture of every payload in the catalog."""
    reentry_days = reentry_days or {}
    rows, by_reg = [], {}
    for o in objs:
        if o.obj_type != "PAYLOAD":
            continue
        lc = lifecycle(o, now, reentry_days.get(o.norad))
        by_reg.setdefault(lc["regime"], Counter())[lc["stage_key"]] += 1
        rows.append((o, lc))
    stages = Counter(lc["stage_key"] for _, lc in rows)
    watch = [(o, lc) for o, lc in rows if lc["stage_key"] in ("end of life", "late life")
             or (lc["years_to_eol"] is not None and lc["years_to_eol"] < 1.0)]
    watch.sort(key=lambda x: (x[1]["years_to_eol"] is None, x[1]["years_to_eol"] if x[1]["years_to_eol"] is not None else 0))
    return {
        "payloads": len(rows), "stages": {k: stages.get(k, 0) for k in STAGES if k != "inactive"},
        "by_regime": {r: {k: c.get(k, 0) for k in STAGES if k != "inactive"} for r, c in sorted(by_reg.items())},
        "spent_in_orbit": stages.get("end of life", 0),
        "design": {r: {"design_life_years": v[0], "total_dv_ms": v[1], "reserve_pct": round(v[2] * 100),
                       "sk_dv_ms_per_year": v[3]} for r, v in REGIME.items()},
        "watchlist": [{"norad": o.norad, "name": o.name, "regime": lc["regime"], "stage": lc["stage_key"],
                       "launch_year": lc["launch_year"], "age_years": lc["age_years"],
                       "design_life_years": lc["design_life_years"], "eol_year": lc["eol_year"],
                       "years_to_eol": lc["years_to_eol"], "usable_dv_ms": lc["usable_dv_ms"],
                       "usable_litres": round(propellant_litres(lc["usable_dv_ms"], lc["mass_kg"], lc["total_dv_ms"]), 2) if lc["usable_dv_ms"] else 0.0,
                       "has_red_alert": o.norad in red_norads} for o, lc in watch[:60]],
        "assumption": "Design life and propellant are regime-level assumptions scaled by launch year; "
                      "supply operator figures in data/fuel.json to replace them."}
