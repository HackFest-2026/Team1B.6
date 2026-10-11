"""Pc calibration study + learned Pc calibrator.

Monte-Carlo "encounter population" experiment (the standard way Pc calibration is assessed,
since real collisions are too rare to score a model on):

* true miss vectors are drawn in the encounter plane, the operator's estimate is the truth
  plus an error drawn from the TRUE covariance, while the operator computes Pc with the
  MODEL covariance (which is mis-scaled by the factor measured from TLE residuals);
* outcome = 1 if the true miss < hard-body radius.
* the true/model covariance ratio depends on TLE age (model under-estimates for stale TLEs,
  over-estimates for fresh ones), which a learned calibrator can exploit.

We report Brier score / Brier skill, log loss, reliability curve, and false-alert / missed-
detection rates at the red threshold for: raw Pc, Pc with realism-scaled covariance, and an
an ML-calibrated Pc: logistic calibration on [logit(Pc), log TLE age, log sigma] learned on a
training half (the "AI" calibrator), evaluated on the held-out half.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..astro.pc import pc_foster


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, bins=None):
    bins = bins if bins is not None else np.array([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 1.0001])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            out.append({"bin": f"{lo:g}-{hi:g}", "mean_pred": float(p[m].mean()),
                        "obs_freq": float(y[m].mean()), "n": int(m.sum())})
    return out


def _cal_features(pc, age_d, sigma_km):
    pc = np.clip(np.asarray(pc, float), 1e-12, 1 - 1e-9)
    lg = np.log(pc / (1 - pc))
    la = np.log(np.maximum(np.asarray(age_d, float), 0.05))
    ls = np.log(np.maximum(np.asarray(sigma_km, float), 1e-4))
    return np.column_stack([lg, la, ls, lg * la])


class PcCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, max_iter=2000)

    def fit(self, pc, age, sig, y):
        self.lr.fit(_cal_features(pc, age, sig), y)
        return self

    def predict(self, pc, age, sig):
        p = self.lr.predict_proba(_cal_features(pc, age, sig))[:, 1]
        return np.where(np.asarray(pc) <= 1e-12, 0.0, p)


def tune_threshold(p, y, far_target):
    """Alert threshold with the fewest missed collisions while false alerts stay <= far_target."""
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        return 1e-4
    k = int(np.ceil((1.0 - far_target) * len(neg))) - 1
    return float(neg[min(max(k, 0), len(neg) - 1)]) * 1.0000001


def operating_curve(p, y, points=40):
    """False-alert rate vs missed-collision rate as the alert threshold sweeps from loose to strict."""
    thr = np.unique(np.quantile(p, np.linspace(0.0, 1.0, points * 4)))
    thr = thr[np.linspace(0, len(thr) - 1, min(points, len(thr))).astype(int)]
    neg, pos = max((y == 0).sum(), 1), max((y == 1).sum(), 1)
    return [{"threshold": float(t), "false_alert_rate": float(((p >= t) & (y == 0)).sum() / neg),
             "missed_rate": float(((p < t) & (y == 1)).sum() / pos)} for t in thr]


def tracking_sweep(realism_scale, seed, base, scales=(0.5, 0.25, 0.1), n=6000, far_target=0.10):
    """How the false-alert rate falls as tracking improves (smaller position uncertainty).

    For each level the alert threshold is tuned on one half so that at most 5 % of real
    collisions are missed, then the false-alert rate is read off the other half.
    """
    out = [_sweep_row(1.0, base)]
    for s in scales:
        r, _ = run_study(realism_scale, n=n, seed=seed + 1, sigma_scale=s, far_target=far_target, sweep=False)
        out.append(_sweep_row(s, r))
    return out


def _sweep_row(s, r):
    c = r["ai_at_5pct_missed"]
    return {"sigma_scale": s, "false_alert_rate": c["false_alert_rate"], "missed_rate": c["missed_detection_rate"],
            "alert_precision": c["alerts_that_were_collisions"], "collision_rate": r["raw"]["observed_rate"]}


def run_study(realism_scale=1.0, n=12000, seed=0, red=1e-4, far_target=0.10, sigma_scale=1.0, sweep=True):
    rng = np.random.default_rng(seed)
    dv = np.array([0.0, 0.0, 10.0])
    pc_raw, pc_scaled, y = np.zeros(n), np.zeros(n), np.zeros(n)
    ages, sigs = np.zeros(n), np.zeros(n)
    for k in range(n):
        sx0, sy0 = np.exp(rng.uniform(np.log(0.01), np.log(0.12), 2))
        sx, sy = sx0 * sigma_scale, sy0 * sigma_scale          # better tracking = smaller errors, same encounters
        rot = rng.uniform(0, np.pi)
        c, s = np.cos(rot), np.sin(rot)
        Rm = np.array([[c, -s], [s, c]])
        P_model = Rm @ np.diag([sx**2, sy**2]) @ Rm.T
        age = rng.uniform(0.2, 5.0)
        k_true = realism_scale * (0.6 + 0.3 * age) * rng.lognormal(0, 0.1)   # true / model sigma
        P_true = P_model * k_true**2
        hbr = rng.uniform(0.008, 0.04)
        rad = 3.0 * np.sqrt(sx0**2 + sy0**2)
        r_ = rad * np.sqrt(rng.uniform()); th = rng.uniform(0, 2 * np.pi)
        m_true = np.array([r_ * np.cos(th), r_ * np.sin(th)])
        m_est = m_true + rng.multivariate_normal([0, 0], P_true)
        y[k] = float(np.hypot(*m_true) < hbr)
        dr = np.array([m_est[0], m_est[1], 0.0])
        C3 = np.zeros((3, 3)); C3[:2, :2] = P_model; C3[2, 2] = 1.0
        pc_raw[k] = pc_foster(dr, dv, C3, hbr)[0]
        C3s = C3.copy(); C3s[:2, :2] *= realism_scale**2
        pc_scaled[k] = pc_foster(dr, dv, C3s, hbr)[0]
        ages[k] = age; sigs[k] = max(sx, sy) * realism_scale
    half = n // 2
    cal = PcCalibrator().fit(pc_scaled[:half], ages[:half], sigs[:half], y[:half])
    pc_ml = cal.predict(pc_scaled[half:], ages[half:], sigs[half:])
    te = slice(half, None)
    yt = y[te]
    base = yt.mean()

    def summary(p, thr=None):
        bs = _brier(p, yt)
        thr = red if thr is None else thr
        alerts = p >= thr
        return {"brier": bs, "brier_skill_vs_climatology": 1 - bs / max(_brier(np.full_like(p, base), yt), 1e-12),
                "log_loss": _logloss(p, yt), "alert_threshold": float(thr), "mean_pc": float(p.mean()), "observed_rate": float(base),
                "false_alert_rate": float((alerts & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
                "missed_detection_rate": float((~alerts & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                "alerts_that_were_collisions": float((alerts & (yt == 1)).sum() / max(alerts.sum(), 1)),
                "reliability": reliability(p, yt)}

    p_train = cal.predict(pc_scaled[:half], ages[:half], sigs[:half])
    # threshold that misses at most 5 % of real collisions on the training half
    pos = np.sort(p_train[y[:half] == 1])
    thr5 = float(pos[int(0.05 * len(pos))]) if len(pos) else red
    res = {"n_test": int(len(yt)), "collisions_in_test": int(yt.sum()), "realism_scale": realism_scale,
           "raw": summary(pc_raw[te]), "realism_scaled": summary(pc_scaled[te]),
           "ai_calibrated": summary(pc_ml),
           "ai_tuned": summary(pc_ml, tune_threshold(p_train, y[:half], far_target)),
           "ai_at_5pct_missed": summary(pc_ml, thr5),
           "far_target": far_target,
           # a model that never raises an alert is right this often, which is why plain accuracy misleads here
           "accuracy_of_never_alerting": float(1 - base)}
    if sweep:
        res["operating_curve"] = operating_curve(pc_ml, yt)
        res["tracking_sweep"] = tracking_sweep(realism_scale, seed, res, far_target=far_target)
    return res, cal
