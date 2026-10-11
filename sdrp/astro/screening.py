"""All-vs-all conjunction screening.

Pipeline
1. (optional) apogee/perigee filter  -> candidate pair graph (also used by the GNN screener)
2. time-stepped spatial search: KD-tree (CPU) or blocked brute force (GPU via CuPy)
   with a search radius that guarantees no encounter is missed between steps
   (R = threshold + v_rel_max * step / 2), followed by a linear-motion TCA estimate
3. event clustering + exact TCA refinement with golden-section search on SGP4
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta

import numpy as np
from scipy.spatial import cKDTree

from .. import config
from .propagate import Catalog, dt_to_jd, ric_frame, time_grid

log = logging.getLogger("sdrp.screen")
V_REL_MAX = 15.6   # km/s, upper bound for LEO head-on encounters


def _get_xp():
    mode = str(config.USE_GPU).lower()
    if mode in ("off", "0", "false"):
        return np, False
    try:
        import cupy as cp  # noqa
        cp.zeros(1)
        return cp, True
    except Exception:
        if mode in ("on", "1", "true"):
            log.warning("GPU requested but CuPy is not available; falling back to CPU")
        return np, False


def apogee_perigee_pairs(catalog: Catalog, pad_km: float):
    """Pairs whose altitude shells overlap (classic Hoots filter, first stage)."""
    per = np.array([o.perigee_km for o in catalog.objects])
    apo = np.array([o.apogee_km for o in catalog.objects])
    order = np.argsort(per)
    pairs = []
    # sweep: object j can only overlap i if per_j <= apo_i + pad
    per_s = per[order]
    for k, i in enumerate(order):
        hi = np.searchsorted(per_s, apo[i] + pad_km, side="right")
        jj = order[k + 1:hi]
        if len(jj) == 0:
            continue
        ok = np.maximum(per[i], per[jj]) - np.minimum(apo[i], apo[jj]) <= pad_km
        jj = jj[ok]
        pairs.append(np.stack([np.minimum(i, jj), np.maximum(i, jj)], 1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(pairs).astype(np.int64)


def _pairs_gpu(xp, P, R, block=4096):
    """Blocked all-vs-all distance test on the GPU (or numpy for testing)."""
    Pd = xp.asarray(P)
    n = Pd.shape[0]
    out = []
    for a in range(0, n, block):
        A = Pd[a:a + block]
        for b in range(a, n, block):
            B = Pd[b:b + block]
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
            ii, jj = xp.nonzero(d2 < R * R)
            ii = ii + a; jj = jj + b
            keep = ii < jj
            if int(keep.sum()):
                out.append(xp.stack([ii[keep], jj[keep]], 1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    res = xp.concatenate(out)
    return res.get() if hasattr(res, "get") else np.asarray(res)


def screen(catalog: Catalog, start: datetime, horizon_h=None, step_s=None, threshold_km=None,
           allowed_pairs: set | None = None, chunk=90, force_backend=None):
    """Return (events, stats). ``allowed_pairs`` (set of (i,j)) restricts which pairs are
    refined - used to evaluate the AI pre-filter."""
    horizon_h = horizon_h or config.SCREEN_HORIZON_H
    step_s = step_s or config.SCREEN_STEP_S
    thr = threshold_km or config.SCREEN_THRESHOLD_KM
    xp, gpu = _get_xp()
    if force_backend == "cpu":
        xp, gpu = np, False
    elif force_backend == "bruteforce":
        xp, gpu = np, True
    R = thr + V_REL_MAX * step_s / 2.0
    jd, fr, secs = time_grid(start, horizon_h, step_s)
    N = len(catalog)
    t0 = time.perf_counter()
    cand = []                 # (i, j, t_sec, dmin)
    n_close_checks = 0
    for c0 in range(0, len(secs), chunk):
        r, v, _ = catalog.propagate(jd[c0:c0 + chunk], fr[c0:c0 + chunk])
        for k in range(r.shape[1]):
            P = r[:, k, :]
            ok = np.isfinite(P[:, 0])
            idx = np.nonzero(ok)[0]
            if gpu:
                pr = _pairs_gpu(xp, P[ok], R)
            else:
                pr = cKDTree(P[ok]).query_pairs(R, output_type="ndarray")
            if len(pr) == 0:
                continue
            i, j = idx[pr[:, 0]], idx[pr[:, 1]]
            n_close_checks += len(i)
            dr = r[j, k] - r[i, k]
            dv = v[j, k] - v[i, k]
            vv = np.einsum("ij,ij->i", dv, dv)
            tau = np.clip(-np.einsum("ij,ij->i", dr, dv) / np.maximum(vv, 1e-12), -step_s / 2, step_s / 2)
            dmin = np.linalg.norm(dr + dv * tau[:, None], axis=1)
            sel = dmin < thr * 1.5 + 0.5
            for a, b, ta, dm in zip(i[sel], j[sel], tau[sel], dmin[sel]):
                cand.append((int(a), int(b), float(secs[c0 + k] + ta), float(dm)))
    t_search = time.perf_counter() - t0

    # group into events per pair
    cand.sort()
    groups, cur = [], []
    for c in cand:
        if cur and (c[0], c[1]) == (cur[-1][0], cur[-1][1]) and c[2] - cur[-1][2] < 600:
            cur.append(c)
        else:
            if cur:
                groups.append(cur)
            cur = [c]
    if cur:
        groups.append(cur)

    events = []
    t1 = time.perf_counter()
    for g in groups:
        i, j = g[0][0], g[0][1]
        if allowed_pairs is not None and (i, j) not in allowed_pairs:
            continue
        best = min(g, key=lambda c: c[3])
        ev = refine(catalog, i, j, start + timedelta(seconds=best[2]), span_s=step_s)
        if ev and ev["miss_km"] <= thr:
            events.append(ev)
    t_refine = time.perf_counter() - t1
    total_pairs = N * (N - 1) / 2 * len(secs)
    stats = {
        "objects": N, "time_steps": len(secs), "step_s": step_s, "horizon_h": horizon_h,
        "threshold_km": thr, "search_radius_km": R, "backend": "GPU (CuPy)" if gpu and xp is not np else
        ("brute-force" if gpu else "CPU KD-tree"), "pair_time_checks": total_pairs,
        "close_pairs_examined": n_close_checks, "candidates": len(cand), "events": len(events),
        "search_s": round(t_search, 3), "refine_s": round(t_refine, 3),
        "throughput_pairs_per_s": total_pairs / max(t_search, 1e-9),
        "unique_pairs_per_s": N * (N - 1) / 2 / max(t_search, 1e-9) * len(secs) / len(secs),
    }
    events.sort(key=lambda e: e["miss_km"])
    return events, stats


def refine(catalog: Catalog, i, j, t_guess: datetime, span_s=30.0):
    a, b = catalog.objects[i].sat, catalog.objects[j].sat

    def state(ts):
        jd, fr = dt_to_jd(t_guess + timedelta(seconds=ts))
        e1, r1, v1 = a.sgp4(jd, fr)
        e2, r2, v2 = b.sgp4(jd, fr)
        if e1 or e2:
            return None
        return np.array(r1), np.array(v1), np.array(r2), np.array(v2)

    def d(ts):
        s = state(ts)
        return 1e9 if s is None else float(np.linalg.norm(s[0] - s[2]))

    lo, hi = -span_s, span_s
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = d(x1), d(x2)
    for _ in range(48):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = d(x1)
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = d(x2)
    ts = (lo + hi) / 2
    s = state(ts)
    if s is None:
        return None
    r1, v1, r2, v2 = s
    dr, dv = r2 - r1, v2 - v1
    M = ric_frame(r1, v1)
    ric = M @ dr
    tca = t_guess + timedelta(seconds=ts)
    return {"i": i, "j": j, "tca": tca, "miss_km": float(np.linalg.norm(dr)),
            "rel_speed_kms": float(np.linalg.norm(dv)), "r1": r1, "v1": v1, "r2": r2, "v2": v2,
            "miss_ric_km": ric.tolist()}


def screen_pairs(catalog: Catalog, pairs: np.ndarray, start: datetime, horizon_h, step_s,
                 threshold_km, chunk=60, refine_events=True):
    """Pairwise screening restricted to an explicit candidate list (used after the
    graph/AI pre-filter). Cost scales with len(pairs) instead of N^2."""
    jd, fr, secs = time_grid(start, horizon_h, step_s)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    t0 = time.perf_counter()
    best = {}
    if len(pairs):
        objs = np.unique(pairs)
        remap = {o: k for k, o in enumerate(objs)}
        pi = np.array([remap[a] for a in pairs[:, 0]]); pj = np.array([remap[b] for b in pairs[:, 1]])
        for c0 in range(0, len(secs), chunk):
            r, v, _ = catalog.propagate(jd[c0:c0 + chunk], fr[c0:c0 + chunk], idx=objs)
            for k in range(r.shape[1]):
                dr = r[pj, k] - r[pi, k]
                d = np.sqrt(np.einsum("ij,ij->i", dr, dr))
                near = np.nonzero(d < threshold_km + V_REL_MAX * step_s / 2)[0]
                if len(near) == 0:
                    continue
                dv = v[pj[near], k] - v[pi[near], k]
                drn = dr[near]
                vv = np.einsum("ij,ij->i", dv, dv)
                tau = np.clip(-np.einsum("ij,ij->i", drn, dv) / np.maximum(vv, 1e-12), -step_s / 2, step_s / 2)
                dmin = np.linalg.norm(drn + dv * tau[:, None], axis=1)
                for n_, ta, dm in zip(near[dmin < threshold_km * 1.5 + 0.5], tau[dmin < threshold_km * 1.5 + 0.5],
                                      dmin[dmin < threshold_km * 1.5 + 0.5]):
                    key = (int(pairs[n_, 0]), int(pairs[n_, 1]), int((secs[c0 + k] + ta) // 900))
                    if key not in best or dm < best[key][1]:
                        best[key] = (float(secs[c0 + k] + ta), float(dm))
    elapsed = time.perf_counter() - t0
    events = []
    if refine_events:
        for (i, j, _), (ts, _) in best.items():
            ev = refine(catalog, i, j, start + timedelta(seconds=ts), span_s=step_s)
            if ev and ev["miss_km"] <= threshold_km:
                events.append(ev)
    # de-duplicate events of the same pair within 10 minutes
    events.sort(key=lambda e: (e["i"], e["j"], e["tca"]))
    dedup = []
    for e in events:
        if dedup and (dedup[-1]["i"], dedup[-1]["j"]) == (e["i"], e["j"]) and \
                abs((e["tca"] - dedup[-1]["tca"]).total_seconds()) < 600:
            if e["miss_km"] < dedup[-1]["miss_km"]:
                dedup[-1] = e
            continue
        dedup.append(e)
    stats = {"pairs": int(len(pairs)), "time_steps": len(secs), "search_s": round(elapsed, 3),
             "pair_steps_per_s": len(pairs) * len(secs) / max(elapsed, 1e-9), "events": len(dedup)}
    return dedup, stats


def screen_subset(catalog: Catalog, subset_idx, start: datetime, horizon_h, step_s, threshold_km,
                  chunk=90):
    """One-vs-all screening for objects with freshly updated TLEs (incremental update)."""
    R = threshold_km + V_REL_MAX * step_s / 2.0
    jd, fr, secs = time_grid(start, horizon_h, step_s)
    subset_idx = np.asarray(subset_idx)
    t0 = time.perf_counter()
    cand = {}
    for c0 in range(0, len(secs), chunk):
        r, v, _ = catalog.propagate(jd[c0:c0 + chunk], fr[c0:c0 + chunk])
        for k in range(r.shape[1]):
            P = r[:, k, :]
            ok = np.isfinite(P[:, 0])
            idx = np.nonzero(ok)[0]
            sub = subset_idx[ok[subset_idx]]
            if len(sub) == 0:
                continue
            sdm = cKDTree(P[sub]).sparse_distance_matrix(cKDTree(P[ok]), R, output_type="ndarray")
            if len(sdm) == 0:
                continue
            a = sub[sdm["i"]]; b = idx[sdm["j"]]
            m = a != b
            a, b = a[m], b[m]
            dr = r[b, k] - r[a, k]; dv = v[b, k] - v[a, k]
            vv = np.einsum("ij,ij->i", dv, dv)
            tau = np.clip(-np.einsum("ij,ij->i", dr, dv) / np.maximum(vv, 1e-12), -step_s / 2, step_s / 2)
            dm = np.linalg.norm(dr + dv * tau[:, None], axis=1)
            for aa, bb, ta, d_ in zip(a[dm < threshold_km * 1.5 + 0.5], b[dm < threshold_km * 1.5 + 0.5],
                                      tau[dm < threshold_km * 1.5 + 0.5], dm[dm < threshold_km * 1.5 + 0.5]):
                key = (int(min(aa, bb)), int(max(aa, bb)), int((secs[c0 + k] + ta) // 900))
                if key not in cand or d_ < cand[key][1]:
                    cand[key] = (float(secs[c0 + k] + ta), float(d_))
    events = []
    for (i, j, _), (ts, _) in cand.items():
        ev = refine(catalog, int(i), int(j), start + timedelta(seconds=ts), span_s=step_s)
        if ev and ev["miss_km"] <= threshold_km:
            events.append(ev)
    return events, round(time.perf_counter() - t0, 3)
