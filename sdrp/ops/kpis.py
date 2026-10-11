"""KPI computations: propagation error vs horizon, covariance realism, CDM hit/miss,
lead time. Results feed the dashboard KPI panel."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..astro.propagate import SpaceObject, dt_to_jd, ric_frame

HORIZONS_D = (1, 2, 3, 5, 7)


def propagation_errors(histories, cov_model=None, sw=None, max_objects=120):
    """TLE-vs-TLE test: propagate TLE_k to the epoch of TLE_{k+h}, compare positions (RIC).
    Pairs straddling a labelled maneuver/breakup are excluded."""
    per_h = {h: [] for h in HORIZONS_D}
    nees = {h: [] for h in HORIZONS_D}
    comp_ratio = {h: [] for h in HORIZONS_D}
    for hist in histories[:max_objects]:
        sats, ep, lab = hist["satrecs"], hist["epochs"], hist.get("labels") or [0] * len(hist["satrecs"])
        cum = np.cumsum([1 if l else 0 for l in lab])
        for k in range(0, len(sats) - 1, 2):
            for h in HORIZONS_D:
                j = k + h
                if j >= len(sats) or cum[j] != cum[k]:
                    continue
                jd, fr = dt_to_jd(ep[j])
                e1, r1, v1 = sats[k].sgp4(jd, fr)
                e2, r2, v2 = sats[j].sgp4(jd, fr)
                if e1 or e2:
                    continue
                M = ric_frame(r2, v2)
                d = M @ (np.array(r1) - np.array(r2))
                per_h[h].append(d)
                if cov_model is not None:
                    tmp = SpaceObject(norad=hist["norad"], name="", line1="", line2="", sat=sats[k],
                                      obj_type=hist.get("type", "PAYLOAD"))
                    s = cov_model.sigma_ric(tmp, h, sw)
                    # the reference TLE also carries error: combine both
                    s0 = cov_model.sigma_ric(tmp, 0.0, sw)
                    st = np.sqrt(s**2 + s0**2)
                    nees[h].append(float(np.sum((d / st) ** 2)))
                    comp_ratio[h].append(d**2 / st**2)
    out = {"horizons_days": list(HORIZONS_D), "median_km": [], "p90_km": [], "robust_sigma_ric_km": [],
           "samples": []}
    for h in HORIZONS_D:
        D = np.array(per_h[h]) if per_h[h] else np.zeros((0, 3))
        tot = np.linalg.norm(D, axis=1) if len(D) else np.array([np.nan])
        out["median_km"].append(float(np.nanmedian(tot)))
        out["p90_km"].append(float(np.nanpercentile(tot, 90)))
        out["robust_sigma_ric_km"].append((1.4826 * np.median(np.abs(D), 0)).tolist() if len(D) else [None] * 3)
        out["samples"].append(int(len(D)))
    if cov_model is not None:
        out["median_nees"] = [float(np.median(nees[h])) if nees[h] else None for h in HORIZONS_D]   # ideal 2.37 (chi2, 3 dof)
        out["chi2_95_containment"] = [float(np.mean(np.array(nees[h]) < 7.815)) if nees[h] else None
                                      for h in HORIZONS_D]
        allr = np.vstack([np.array(comp_ratio[h]) for h in HORIZONS_D if comp_ratio[h]])
        out["realism_scale_ric"] = np.sqrt(np.median(allr, 0) / 0.455).tolist()   # median chi2(1)=0.455
        out["nees_q95"] = [float(np.percentile(nees[h], 95)) if nees[h] else None for h in HORIZONS_D]
    return out


def fit_realism(histories, cov_model, sw):
    """Learn per-axis covariance scale factors so normalised errors follow chi-square."""
    before = propagation_errors(histories, cov_model, sw)
    k = before.get("realism_scale_ric", [1, 1, 1])
    cov_model.realism = {"R": float(np.clip(k[0], 0.2, 20)), "I": float(np.clip(k[1], 0.2, 20)),
                         "C": float(np.clip(k[2], 0.2, 20))}
    after = propagation_errors(histories, cov_model, sw)
    # The median fit leaves heavy tails outside the chi-square bound (TLE errors are not Gaussian).
    # Widen all axes until the 95th-percentile NEES of the worst horizon sits at the chi2(3) 95 % point.
    for _ in range(3):
        q = [x for x in after.get("nees_q95", []) if x]
        if not q:
            break
        g = float(np.sqrt(max(q) / 7.815))
        if abs(g - 1) < 0.03:
            break
        cov_model.realism = {a: float(np.clip(v * g, 0.2, 60)) for a, v in cov_model.realism.items()}
        after = propagation_errors(histories, cov_model, sw)
    return before, after


def _latest_per_encounter(cdms, gap_s=600):
    """Space-Track publishes many CDM versions per encounter (and one per object order).
    Keep only the most recent message for each pair + TCA."""
    best = {}
    for c in cdms:
        key = (min(c["sat1"], c["sat2"]), max(c["sat1"], c["sat2"]))
        tc = datetime.fromisoformat(c["tca"])
        for (k2, t2) in list(best):
            if k2 == key and abs((t2 - tc).total_seconds()) < gap_s:
                key_t = t2
                break
        else:
            key_t = tc
        cur = best.get((key, key_t))
        if cur is None or str(c.get("created") or "") > str(cur.get("created") or ""):
            best[(key, key_t)] = c
    return list(best.values())


def compare_with_cdms(events, catalog, cdms, horizon_start: datetime, horizon_h: float, tol_s=300):
    """Hit/miss of our screening against reference CDMs (Space-Track, or synthetic truth)."""
    end = horizon_start + timedelta(hours=horizon_h)
    cdms = _latest_per_encounter(cdms)
    in_cat = [c for c in cdms if catalog.get(c["sat1"]) is not None and catalog.get(c["sat2"]) is not None
              and horizon_start <= datetime.fromisoformat(c["tca"]) <= end]
    by_pair = {}
    for e in events:
        a, b = catalog.objects[e["i"]].norad, catalog.objects[e["j"]].norad
        by_pair.setdefault((min(a, b), max(a, b)), []).append(e)
    hits, tca_err, miss_err, lead = 0, [], [], []
    for c in in_cat:
        key = (min(c["sat1"], c["sat2"]), max(c["sat1"], c["sat2"]))
        tc = datetime.fromisoformat(c["tca"])
        m = [e for e in by_pair.get(key, []) if abs((e["tca"] - tc).total_seconds()) < tol_s]
        if m:
            hits += 1
            e = min(m, key=lambda e: abs((e["tca"] - tc).total_seconds()))
            tca_err.append(abs((e["tca"] - tc).total_seconds()))
            miss_err.append(abs(e["miss_km"] - c["min_rng_km"]) * 1000)
            lead.append((tc - horizon_start).total_seconds() / 3600)
    n = len(in_cat)
    return {"reference_cdms": n, "hits": hits, "misses": n - hits,
            "hit_rate": hits / n if n else None,
            "median_tca_error_s": float(np.median(tca_err)) if tca_err else None,
            "median_miss_distance_error_m": float(np.median(miss_err)) if miss_err else None,
            "median_lead_time_h": float(np.median(lead)) if lead else None}
