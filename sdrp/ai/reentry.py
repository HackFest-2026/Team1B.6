"""Re-entry prediction with uncertainty (Monte Carlo over AI density + space weather + B*).

Orbit decay is integrated in mean semi-major axis / perigee / apogee using
da/dt = -rho * B * sqrt(mu * a), with rho from the static model times the ML correction and
a per-sample density error drawn from the ML quantile spread. Output: re-entry window
(P10/P50/P90), the latitude band at risk and an indicative casualty-expectation score.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from ..astro.atmosphere import static_density
from ..astro.propagate import MU, RE, SpaceObject

MU_M = MU * 1e9
# average population density (people / km^2) under ground tracks of a given inclination
_POP = [(10, 20.0), (30, 22.0), (45, 18.0), (55, 15.0), (70, 13.0), (90, 11.0)]


def _pop_density(inc):
    inc = min(abs(inc), 180 - abs(inc)) if inc > 90 else abs(inc)
    for lim, d in _POP:
        if inc <= lim:
            return d
    return 11.0


def predict_reentry(obj: SpaceObject, now: datetime, density_model=None, sw=None,
                    n_mc=300, max_days=400, rng=None):
    rng = rng or np.random.default_rng(obj.norad)
    hp0 = obj.perigee_km
    ha0 = obj.apogee_km
    if hp0 > 600:
        return None
    # propagate the mean elements from TLE epoch to now first (included in the integration)
    B0 = 2.2 * obj.am_ratio
    B = B0 * rng.lognormal(0, 0.30, n_mc)                    # B* / OD uncertainty
    rel_sig = density_model.relative_sigma(hp0, sw) if density_model else 0.35
    fac = density_model.correction_factor(max(hp0, 150)) if density_model and density_model._table else 1.0
    dens_mult = fac * np.exp(rng.normal(0, rel_sig, n_mc))   # persistent density error
    sw_mult = rng.lognormal(0, 0.15, n_mc)                   # driver forecast error
    hp = np.full(n_mc, hp0); ha = np.full(n_mc, ha0)
    t = np.zeros(n_mc)
    alive = np.ones(n_mc, bool)
    t_start = (now - obj.epoch).total_seconds() / 86400.0
    t[:] = 0.0
    tre = np.full(n_mc, np.nan)
    elapsed = 0.0
    horizon = max_days + max(t_start, 0)
    while alive.any() and elapsed < horizon:
        a = RE + (hp + ha) / 2
        h_eff = hp + 0.15 * (ha - hp)
        rho = static_density(h_eff) * dens_mult * sw_mult
        dadt = -rho * B * np.sqrt(MU_M * a * 1000) / 1000.0 * 86400.0   # km/day
        tau = np.where(dadt < 0, (hp - 100) / np.maximum(-dadt, 1e-9), 1e9)
        dt = float(np.clip(np.min(tau[alive]) * 0.05, 1e-3, 2.0))
        da = dadt * dt
        ecc_like = (ha - hp) > 15
        ha = np.where(alive, ha + np.where(ecc_like, 1.8 * da, da), ha)
        hp = np.where(alive, hp + np.where(ecc_like, 0.2 * da, da), hp)
        ha = np.maximum(ha, hp)
        elapsed += dt
        down = alive & (hp < 120)
        tre[down] = elapsed
        alive &= ~down
    valid = ~np.isnan(tre)
    frac = float(valid.mean())
    days_from_now = tre - t_start
    res = {"norad": obj.norad, "name": obj.name, "type": obj.obj_type,
           "perigee_km": round(hp0, 1), "apogee_km": round(ha0, 1),
           "inc_deg": round(obj.inc_deg, 2), "prob_reentry_in_window": frac,
           "window_days": max_days}
    if frac >= 0.1:
        q = np.nanpercentile(days_from_now, [10, 50, 90])
        q = np.maximum(q, 0)
        res.update({
            "p10": (now + timedelta(days=float(q[0]))).isoformat(),
            "p50": (now + timedelta(days=float(q[1]))).isoformat(),
            "p90": (now + timedelta(days=float(q[2]))).isoformat(),
            "days_p50": float(q[1]), "window_width_days": float(q[2] - q[0]),
        })
    # indicative casualty expectation (NASA-STD-8719.14 style, simplified)
    mass = obj.est_mass()
    surv_frac = {"ROCKET BODY": 0.3, "PAYLOAD": 0.2, "DEBRIS": 0.05}.get(obj.obj_type, 0.1)
    n_frag = max(1.0, mass * surv_frac / 15.0)
    a_frag = 0.25 * (mass * surv_frac / n_frag) ** (2 / 3) * 0.05
    a_cas = n_frag * (math.sqrt(max(a_frag, 0)) + 0.6) ** 2      # m^2 (0.6 m = human radius)
    ec = a_cas * 1e-6 * _pop_density(obj.inc_deg)
    res.update({"mass_kg": round(mass, 1), "casualty_area_m2": round(a_cas, 2),
                "casualty_expectation": ec, "exceeds_1e-4": ec > 1e-4,
                "lat_band_deg": round(min(obj.inc_deg, 180 - obj.inc_deg), 1)})
    return res


def reentry_watchlist(catalog, now, density_model=None, sw=None, max_perigee_km=330, limit=40):
    cands = [o for o in catalog.objects if o.perigee_km < max_perigee_km and o.apogee_km < 2000]
    cands.sort(key=lambda o: o.perigee_km)
    out = []
    for o in cands[:limit]:
        r = predict_reentry(o, now, density_model, sw)
        if r:
            out.append(r)
    out.sort(key=lambda r: r.get("days_p50", 1e9))
    return out
