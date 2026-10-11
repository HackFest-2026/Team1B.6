"""Anomaly detection on TLE time series: maneuvers and fragmentation/breakup signatures,
plus fragmentation-cloud discovery in the catalog.

Per consecutive TLE pair we build physics-normalised features (mean-motion change relative
to the object's own drag trend, inclination jump, RAAN residual vs J2 prediction,
eccentricity jump, B* ratio). An Isolation Forest scores them unsupervised; a robust
z-score gate makes detections explainable.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from ..astro.propagate import MU, RE

FEATS = ["dn_resid_z", "di_deg", "draan_resid_deg", "de", "log_bstar_ratio", "da_km"]


def _transition_features(hist):
    sats, ep = hist["satrecs"], hist["epochs"]
    rows = []
    n = np.array([s.no_kozai for s in sats])
    dn_all = np.diff(n) / np.maximum(np.array([(ep[k + 1] - ep[k]).total_seconds() / 86400 for k in range(len(ep) - 1)]), 1e-3)
    for k in range(1, len(sats)):
        a, b = sats[k - 1], sats[k]
        dt_min = (ep[k] - ep[k - 1]).total_seconds() / 60.0
        prev = dn_all[max(0, k - 8):k - 1]
        med = np.median(prev) if len(prev) >= 3 else np.median(dn_all)
        mad = np.median(np.abs(prev - med)) * 1.4826 if len(prev) >= 3 else np.std(dn_all)
        z = (dn_all[k - 1] - med) / max(mad, 1e-9, abs(med) * 0.05)
        draan_pred = a.nodedot * dt_min
        draan = (b.nodeo - a.nodeo + math.pi) % (2 * math.pi) - math.pi
        aa = (MU / (a.no_kozai / 60) ** 2) ** (1 / 3); ab = (MU / (b.no_kozai / 60) ** 2) ** (1 / 3)
        rows.append([z, math.degrees(b.inclo - a.inclo), math.degrees(draan - draan_pred),
                     b.ecco - a.ecco,
                     math.log((abs(b.bstar) + 1e-7) / (abs(a.bstar) + 1e-7)), ab - aa])
    return np.array(rows)


# noise floors per feature (sigma of the OD noise), used when an object's own MAD is tiny
_FLOORS = np.array([1.0, 1e-3, 3e-3, 4e-5, 0.35, 0.02])


def _per_object_z(F):
    """Robust z-score of each transition against the object's *own* history."""
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826
    return (F - med) / np.maximum(mad, _FLOORS)


def reference_labels(hist, da_km=1.5, di_deg=0.02):
    """Independent rule-based reference for live data, where no ground truth exists.

    Flags a TLE transition when the semi-major axis steps by more than ``da_km`` against the
    object's own median step (net of normal decay), or the inclination jumps by more than
    ``di_deg`` -- fixed physical thresholds, unrelated to the ML detector's z-scores."""
    sats = hist["satrecs"]
    a = np.array([(MU / (s.no_kozai / 60) ** 2) ** (1 / 3) for s in sats])
    inc = np.array([s.inclo for s in sats])
    da = np.diff(a)
    base = np.median(da) if len(da) else 0.0
    lab = [0]
    for k in range(len(da)):
        hit = abs(da[k] - base) > da_km or abs(math.degrees(inc[k + 1] - inc[k])) > di_deg
        lab.append(1 if hit else 0)
    return lab


def detect(histories, contamination=0.02, seed=0, z_gate=6.0):
    zs, raw, meta = [], [], []
    label_source = "ground truth"
    for h in histories:
        if not h.get("labels"):
            h["labels"] = reference_labels(h)
            label_source = "rule-based reference"
        if len(h["satrecs"]) < 8:
            continue
        F = _transition_features(h)
        zs.append(_per_object_z(F)); raw.append(F)
        labels = h.get("labels") or [0] * len(h["satrecs"])
        for k in range(1, len(h["satrecs"])):
            meta.append((h["norad"], h.get("name", ""), h["epochs"][k], labels[k]))
    if not zs:
        return [], {}
    Z = np.vstack(zs); X = np.vstack(raw)
    # common-mode removal: space weather moves every object's drag on the same day
    days = np.array([m[2].date().toordinal() for m in meta])
    for col in (0, 5):
        for d in np.unique(days):
            sel = days == d
            if sel.sum() >= 10:
                Z[sel, col] -= np.median(Z[sel, col])
    Zt = np.sign(Z) * np.log1p(np.abs(Z))
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(Zt)
    score = -iso.score_samples(Zt)
    gate = (np.abs(Z[:, 1:4]).max(1) > z_gate) | (np.abs(Z[:, 5]) > z_gate) | (Z[:, 4] > z_gate * 0.7)
    det = gate & (score >= np.quantile(score, 1 - max(contamination * 3, 0.05)))
    events = []
    for k in np.nonzero(det)[0]:
        f, z = X[k], Z[k]
        if z[4] > z_gate * 0.7:
            kind = "fragmentation / area-to-mass change"
        elif abs(z[1]) > z_gate:
            kind = "plane-change maneuver"
        elif z[5] > z_gate:
            kind = "orbit-raise maneuver"
        elif z[5] < -z_gate:
            kind = "orbit-lowering maneuver / decay anomaly"
        else:
            if np.abs(z[1:]).max() < z_gate * 1.5:
                continue
            kind = "unexplained orbit change"
        events.append({"norad": meta[k][0], "name": meta[k][1], "epoch": meta[k][2].isoformat(),
                       "score": float(score[k]), "kind": kind,
                       "features": {n: float(v) for n, v in zip(FEATS, f)},
                       "zscores": {n: float(v) for n, v in zip(FEATS, z)}})
    # space-weather filter: many objects decaying faster on the same day = storm, not a maneuver
    by_day = {}
    for ev in events:
        if ev["kind"].startswith("orbit-lowering"):
            by_day.setdefault(ev["epoch"][:10], []).append(ev)
    n_obj = max(len(zs), 1)
    for day, evs in by_day.items():
        if len(evs) >= max(4, 0.03 * n_obj):
            for ev in evs:
                ev["kind"] = "drag surge (geomagnetic storm)"
    for ev in events:
        if ev["kind"].startswith("orbit-lowering"):
            ev["kind"] = "decay anomaly (drag)"
        ev["category"] = ("maneuver" if "maneuver" in ev["kind"] else
                          "fragmentation" if "fragmentation" in ev["kind"] else "space-weather/drag"
                          if ("drag" in ev["kind"]) else "other")
    events.sort(key=lambda e: -e["score"])
    labels = np.array([m[3] for m in meta])
    actionable = [e for e in events if e["category"] in ("maneuver", "fragmentation", "other")]
    metrics = {"transitions": int(len(X)), "detections": len(actionable),
               "drag_or_storm_flags": len(events) - len(actionable), "objects": len(zs),
               "label_source": label_source}
    if labels.any():
        # event-level scoring: a detection within 0..2 days after a true event is a hit
        truth = [(m[0], m[2]) for m, l in zip(meta, labels) if l > 0]
        hit = set(); fp = 0
        # a maneuver reference can only confirm maneuver detections
        scored = actionable if label_source == "ground truth" else [e for e in actionable if e["category"] == "maneuver"]
        for e in scored:
            ep = datetime.fromisoformat(e["epoch"])
            match = [t for t in truth if t[0] == e["norad"] and 0 <= (ep - t[1]).days <= 2]
            if match:
                hit.add(match[0])
            else:
                fp += 1
        tp = len(hit); fn = len(truth) - tp
        metrics.update({"labelled_events": len(truth), "tp": tp, "fp": fp, "fn": fn,
                        "precision": tp / max(tp + fp, 1), "recall": tp / max(len(truth), 1),
                        "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    return events, metrics


def detect_debris_clouds(catalog, eps=0.35, min_samples=12):
    """DBSCAN in (altitude, inclination, eccentricity) space to find fragmentation families."""
    objs = [o for o in catalog.objects if o.obj_type in ("DEBRIS", "UNKNOWN") and o.apogee_km < 2500]
    if len(objs) < min_samples:
        return []
    X = np.array([[(o.sma_km - RE) / 60.0, o.inc_deg / 0.6, o.ecc / 0.01] for o in objs])
    lab = DBSCAN(eps=eps * 3, min_samples=min_samples).fit_predict(X)
    out = []
    for c in sorted(set(lab) - {-1}):
        mem = [objs[i] for i in np.nonzero(lab == c)[0]]
        names = {}
        for o in mem:
            key = o.name.split(" DEB")[0]
            names[key] = names.get(key, 0) + 1
        parent = max(names, key=names.get)
        out.append({"cluster": int(c), "likely_parent": parent, "members": len(mem),
                    "mean_alt_km": float(np.mean([o.sma_km - RE for o in mem])),
                    "inc_deg": float(np.mean([o.inc_deg for o in mem])),
                    "alt_spread_km": float(np.std([o.sma_km - RE for o in mem])),
                    "sample_norads": [o.norad for o in mem[:20]]})
    out.sort(key=lambda c: -c["members"])
    return out
