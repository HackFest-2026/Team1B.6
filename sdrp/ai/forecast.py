"""Option C - forecast how an encounter estimate will change.

Task: at every orbital update (CDM) of a conjunction, predict whether the NEXT update will
change the estimate substantially:

    |miss_{k+1} - miss_k| > max(200 m, 25 % of miss_k)   or   |TCA_{k+1} - TCA_k| > 1 s

This is a genuine forecasting task: each training example uses only information available
at update k (current estimate, reported uncertainty, changes since earlier updates, space-
weather state and forecast, object properties), and validation is strictly time-ordered:
encounters are split by TCA date (train on the earliest 60 %, calibrate on the next 20 %,
test on the latest 20 %), so no encounter and no future information leaks into training.

Data sources
* Space-Track ``cdm_public`` history (multiple CDMs per event, grouped by object pair and
  TCA) when credentials are configured;
* otherwise a physics-based simulator of successive orbit-determination updates: estimate
  errors follow the covariance model (TLE-class errors + drag-driven along-track growth),
  are correlated between updates (shared tracking data), react to geomagnetic storms that
  the density model under-predicts, and include occasional primary maneuvers.

The model is gradient boosting on lagged sequence features (missing lags are native NaNs),
compared with two baselines: climatology (base rate) and a physics heuristic (probability of
change from the ratio of reported uncertainty to miss distance).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..astro.atmosphere import static_density

MISS_ABS_KM = 0.2
MISS_REL = 0.25
TCA_S = 1.0

FEATURES = [
    "tau_days", "miss_km", "log_miss", "rel_speed_kms", "sigma_major_km", "sigma_minor_km",
    "mahal_miss", "log_pc", "update_idx", "dmiss_prev_km", "dmiss_prev_rel", "dtca_prev_s",
    "dsigma_prev_rel", "dmiss_cum_km", "miss_volatility", "dt_prev_days", "am_secondary",
    "alt_km", "drag_sens", "f107", "ap", "ap_forecast_next", "ap_jump", "primary_maneuverable",
    "secondary_debris", "rcs_secondary",
]


def drag_sensitivity(am_ratio, alt_km):
    return float(am_ratio * 2.2 * static_density(max(alt_km, 150)) / static_density(600) * 0.02)


def change_label(m0, m1, t0, t1):
    return int(abs(m1 - m0) > max(MISS_ABS_KM, MISS_REL * m0) or abs(t1 - t0) > TCA_S)


# ----------------------------------------------------------------------------- simulator
def simulate_sequences(n_enc=2500, start=None, days=360, seed=0, sw_series=None):
    """Physics-based synthetic encounter-update histories (see module doc)."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2025, 10, 1, tzinfo=timezone.utc)
    sw = {}
    if sw_series:
        sw = {d["date"]: d for d in sw_series}
    seqs = []
    for e in range(n_enc):
        tca = start + timedelta(days=float(rng.uniform(8, days)))
        # objects
        alt = float(rng.uniform(380, 950))
        sec_type = rng.choice(["DEBRIS", "PAYLOAD", "ROCKET BODY"], p=[0.65, 0.2, 0.15])
        am = float(np.clip(rng.lognormal(math.log(0.06 if sec_type == "DEBRIS" else 0.015), 0.7), 0.003, 1.5))
        rcs = float(np.clip(rng.lognormal(math.log(0.05 if sec_type == "DEBRIS" else 4.0), 0.9), 0.002, 30))
        prim_man = bool(rng.random() < 0.7)
        v_rel = float(rng.uniform(1.0, 15.0))
        cross = 2 * math.asin(min(v_rel / 15.2, 0.999))          # crossing angle from v_rel
        par = abs(math.cos(cross / 2))                            # along-track -> TCA timing share
        perp = math.sin(cross / 2) + 0.15                         # along-track -> in-plane share
        obs = float(rng.lognormal(0, 0.5))                        # tracking density (obs/day, rel.)
        realism = float(rng.lognormal(0, 0.25))                   # true / reported sigma
        rho = float(static_density(alt))
        drag_sens = drag_sensitivity(am, alt)                     # dimensionless drag sensitivity
        # truth
        r_true = 5.0 * math.sqrt(rng.uniform())
        th = rng.uniform(0, 2 * math.pi)
        m_true = np.array([r_true * math.cos(th), r_true * math.sin(th)])
        # update schedule
        taus = []
        t = float(rng.uniform(5.0, 7.0))
        while t > 0.15:
            taus.append(t)
            t -= float(rng.uniform(0.3, 1.4)) * (0.6 if t < 2 else 1.0)
        z = rng.normal(0, 1, 3)           # correlated OD error state (R, I, C)
        bias_I = 0.0                       # storm / maneuver induced along-track bias (km)
        man_shift = np.zeros(2)
        prev_date = None
        updates = []
        for k, tau in enumerate(taus):
            when = tca - timedelta(days=tau)
            d = sw.get(when.date().isoformat())
            if d is None:
                f107 = 130 + 40 * math.sin(2 * math.pi * when.timetuple().tm_yday / 27)
                ap = float(np.clip(rng.lognormal(2.0, 0.6), 2, 300))
                if rng.random() < 0.04:
                    ap = float(rng.uniform(60, 250))
            else:
                f107, ap = d["f107"], d["ap"]
            ap_next = float(np.clip(ap * rng.lognormal(0, 0.4) + (rng.uniform(60, 200) if rng.random() < 0.04 else 0), 2, 300))
            # storm effect between updates -> along-track bias the density model under-predicts
            if k > 0:
                storm = max(0.0, ap - 40) / 100.0
                dt = taus[k - 1] - tau
                bias_I += storm * drag_sens * 40.0 * tau * dt * rng.uniform(0.5, 1.5)
                if prim_man and rng.random() < 0.04:
                    man_shift += rng.normal(0, 1.5, 2)
            # error magnitudes (km) at this time-to-TCA
            sig_R = (0.08 + 0.03 * tau) / math.sqrt(obs)
            sig_C = sig_R
            sig_I = (0.3 + 0.12 * tau) / math.sqrt(obs) + 0.75 * drag_sens * 9.0 * tau**2
            rho_c = 0.55
            z = rho_c * z + math.sqrt(1 - rho_c**2) * rng.normal(0, 1, 3)
            eR, eI, eC = z * np.array([sig_R, sig_I, sig_C]) * realism
            eI += bias_I
            plane_err = np.array([eR, eI * perp + eC * 0.5])
            m_est = m_true + plane_err + man_shift
            tca_err_s = (eI * par) / v_rel
            sig_major = math.hypot(sig_I * perp, sig_C * 0.5)
            sig_minor = sig_R
            miss = float(np.linalg.norm(m_est))
            mahal = math.hypot(m_est[0] / sig_minor, m_est[1] / max(sig_major, 1e-6))
            pc_proxy = math.exp(-0.5 * mahal**2) * (0.01**2) / (2 * sig_major * sig_minor) if mahal < 38 else 0.0
            updates.append({"time": when, "tau_days": tau, "miss_km": miss, "tca_offset_s": tca_err_s,
                            "rel_speed_kms": v_rel, "sigma_major_km": sig_major, "sigma_minor_km": sig_minor,
                            "pc": min(pc_proxy, 1.0), "f107": f107, "ap": ap, "ap_forecast_next": ap_next})
            prev_date = when
        seqs.append({"id": f"SYN-{e}", "tca": tca, "alt_km": alt, "am_secondary": am, "rcs_secondary": rcs,
                     "drag_sens": drag_sens, "primary_maneuverable": int(prim_man),
                     "secondary_debris": int(sec_type == "DEBRIS"), "updates": updates, "source": "synthetic"})
    return seqs


# ----------------------------------------------------------------------------- real CDM history
def sequences_from_cdm_history(rows, sat_meta=None):
    """Group Space-Track cdm_public rows into per-encounter update sequences."""
    sat_meta = sat_meta or {}
    groups = {}
    for r in rows:
        try:
            tca = datetime.fromisoformat(r["TCA"].replace(" ", "T").split(".")[0]).replace(tzinfo=timezone.utc)
            created = datetime.fromisoformat(r["CREATED"].replace(" ", "T").split(".")[0]).replace(tzinfo=timezone.utc)
            a, b = int(r["SAT_1_ID"]), int(r["SAT_2_ID"])
        except Exception:
            continue
        key = (min(a, b), max(a, b), int(tca.timestamp() // 600))
        groups.setdefault(key, []).append((created, tca, r))
    seqs = []
    for (a, b, _), lst in groups.items():
        lst.sort(key=lambda x: x[0])
        if len(lst) < 3:
            continue
        tca0 = lst[-1][1]
        ups = []
        for created, tca, r in lst:
            tau = (tca - created).total_seconds() / 86400
            if tau < 0:
                continue
            ups.append({"time": created, "tau_days": tau, "miss_km": float(r.get("MIN_RNG") or 0) / 1000.0,
                        "tca_offset_s": (tca - tca0).total_seconds(), "rel_speed_kms": np.nan,
                        "sigma_major_km": np.nan, "sigma_minor_km": np.nan, "pc": float(r.get("PC") or 0),
                        "f107": np.nan, "ap": np.nan, "ap_forecast_next": np.nan})
        r0 = lst[-1][2]
        deb = int("DEB" in str(r0.get("SAT2_OBJECT_TYPE", r0.get("SAT_2_NAME", ""))))
        seqs.append({"id": f"CDM-{a}-{b}-{tca0:%Y%m%d%H%M}", "tca": tca0, "alt_km": np.nan, "am_secondary": np.nan,
                     "rcs_secondary": np.nan, "drag_sens": np.nan, "primary_maneuverable": np.nan,
                     "secondary_debris": deb, "updates": ups, "source": "space-track"})
    return seqs


# ----------------------------------------------------------------------------- features
def features_at(seq, k):
    u = seq["updates"]
    cur = u[k]
    f = dict.fromkeys(FEATURES, np.nan)
    f.update({"tau_days": cur["tau_days"], "miss_km": cur["miss_km"], "log_miss": math.log10(max(cur["miss_km"], 1e-3)),
              "rel_speed_kms": cur["rel_speed_kms"], "sigma_major_km": cur["sigma_major_km"],
              "sigma_minor_km": cur["sigma_minor_km"], "update_idx": k,
              "log_pc": math.log10(max(cur["pc"], 1e-15)) if cur["pc"] == cur["pc"] else np.nan,
              "f107": cur["f107"], "ap": cur["ap"], "ap_forecast_next": cur["ap_forecast_next"],
              "am_secondary": seq["am_secondary"], "alt_km": seq["alt_km"], "drag_sens": seq["drag_sens"],
              "primary_maneuverable": seq["primary_maneuverable"], "secondary_debris": seq["secondary_debris"],
              "rcs_secondary": seq["rcs_secondary"]})
    if cur["sigma_major_km"] == cur["sigma_major_km"] and cur["sigma_major_km"]:
        f["mahal_miss"] = cur["miss_km"] / math.hypot(cur["sigma_major_km"], cur["sigma_minor_km"])
    if cur["ap_forecast_next"] == cur["ap_forecast_next"] and cur["ap"] == cur["ap"]:
        f["ap_jump"] = cur["ap_forecast_next"] - cur["ap"]
    if k > 0:
        p = u[k - 1]
        f["dmiss_prev_km"] = cur["miss_km"] - p["miss_km"]
        f["dmiss_prev_rel"] = f["dmiss_prev_km"] / max(p["miss_km"], 0.05)
        f["dtca_prev_s"] = cur["tca_offset_s"] - p["tca_offset_s"]
        f["dt_prev_days"] = p["tau_days"] - cur["tau_days"]
        if cur["sigma_major_km"] == cur["sigma_major_km"] and p["sigma_major_km"]:
            f["dsigma_prev_rel"] = cur["sigma_major_km"] / p["sigma_major_km"] - 1
        f["dmiss_cum_km"] = cur["miss_km"] - u[0]["miss_km"]
        f["miss_volatility"] = float(np.std([x["miss_km"] for x in u[:k + 1]]))
    return f


def build_dataset(seqs):
    X, y, groups, times, meta = [], [], [], [], []
    for gi, s in enumerate(seqs):
        u = s["updates"]
        for k in range(len(u) - 1):
            f = features_at(s, k)
            X.append([f[n] for n in FEATURES])
            y.append(change_label(u[k]["miss_km"], u[k + 1]["miss_km"], u[k]["tca_offset_s"], u[k + 1]["tca_offset_s"]))
            groups.append(gi); times.append(s["tca"]); meta.append((s["id"], u[k]["tau_days"]))
    return np.array(X, float), np.array(y, int), np.array(groups), np.array(times), meta


def _physics_baseline(X):
    """P(change) heuristic: large uncertainty relative to the miss distance => estimate will move."""
    i_m = FEATURES.index("miss_km"); i_s = FEATURES.index("sigma_major_km"); i_t = FEATURES.index("tau_days")
    s = np.where(np.isnan(X[:, i_s]), 0.3 + 0.15 * X[:, i_t], X[:, i_s])
    ratio = s / np.maximum(X[:, i_m] * MISS_REL, MISS_ABS_KM)
    return 1 - np.exp(-0.5 * ratio**2)


# ----------------------------------------------------------------------------- model
class ChangeForecaster:
    def __init__(self, seed=0):
        self.seed = seed
        self.model = None
        self.cal = None
        self.metrics = {}

    def fit_evaluate(self, seqs):
        X, y, g, t, meta = build_dataset(seqs)
        order = np.argsort([s["tca"] for s in seqs])
        n = len(seqs)
        tr_ids = set(order[: int(0.6 * n)]); va_ids = set(order[int(0.6 * n): int(0.8 * n)])
        tr = np.array([gi in tr_ids for gi in g]); va = np.array([gi in va_ids for gi in g]); te = ~(tr | va)
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                    min_samples_leaf=30, l2_regularization=1.0, random_state=self.seed)
        self.model.fit(X[tr], y[tr])
        self.cal = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(
            self.model.predict_proba(X[va])[:, 1], y[va])
        p_te = self.predict_proba(X[te])
        base = float(y[tr].mean())
        p_clim = np.full(te.sum(), base)
        p_phys_raw = _physics_baseline(X)
        lr = LogisticRegression().fit(p_phys_raw[va, None], y[va])       # calibrate the heuristic fairly
        p_phys = lr.predict_proba(p_phys_raw[te, None])[:, 1]
        yt = y[te]

        def scores(p):
            return {"auc": float(roc_auc_score(yt, p)), "avg_precision": float(average_precision_score(yt, p)),
                    "brier": float(brier_score_loss(yt, p))}
        taus = np.array([m[1] for m in meta])[te]
        by_tau = []
        for lo, hi in [(0, 1), (1, 2), (2, 3.5), (3.5, 8)]:
            m = (taus >= lo) & (taus < hi)
            if m.sum() > 30 and 0 < yt[m].sum() < m.sum():
                by_tau.append({"tau_bin_days": f"{lo}-{hi}", "n": int(m.sum()), "change_rate": float(yt[m].mean()),
                               "auc_model": float(roc_auc_score(yt[m], p_te[m])),
                               "auc_physics": float(roc_auc_score(yt[m], p_phys[m]))})
        # operating point: flag "likely to change" at p >= 0.5
        flag = p_te >= 0.5
        tp = int((flag & (yt == 1)).sum()); fp = int((flag & (yt == 0)).sum()); fn = int((~flag & (yt == 1)).sum())
        rel = []
        for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
            m = (p_te >= lo) & (p_te < hi)
            if m.sum() >= 20:
                rel.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "mean_pred": float(p_te[m].mean()), "obs": float(yt[m].mean()), "n": int(m.sum())})
        imp = self._importance(X[te], yt)
        self.metrics = {
            "source": seqs[0]["source"] if seqs else None, "encounters": n, "updates": int(len(y)),
            "train_updates": int(tr.sum()), "calib_updates": int(va.sum()), "test_updates": int(te.sum()),
            "split": "time-ordered by TCA: train 60% oldest, calibrate next 20%, test newest 20%",
            "change_definition": f"next update moves miss by > max({MISS_ABS_KM * 1000:.0f} m, {MISS_REL:.0%}) or TCA by > {TCA_S:.0f} s",
            "base_rate": base, "model": scores(p_te), "baseline_climatology": scores(p_clim),
            "baseline_physics": scores(p_phys), "by_time_to_tca": by_tau, "reliability": rel,
            "at_p50": {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)},
            "top_features": imp,
            "test_window": f"{min(t[te]):%Y-%m-%d} → {max(t[te]):%Y-%m-%d}" if te.any() else None,
        }
        return self

    def _importance(self, X, y, n=6):
        """Permutation importance on the test set (AUC drop)."""
        rng = np.random.default_rng(self.seed)
        base = roc_auc_score(y, self.predict_proba(X))
        out = []
        for j, name in enumerate(FEATURES):
            if np.all(np.isnan(X[:, j])):
                continue
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            out.append((name, base - roc_auc_score(y, self.predict_proba(Xp))))
        out.sort(key=lambda x: -x[1])
        return [{"feature": a, "auc_drop": float(b)} for a, b in out[:n]]

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        return self.cal.predict(p) if self.cal is not None else p

    def score_current(self, enc):
        """enc: dict with the current-estimate fields of a live conjunction (first update)."""
        seq = {"am_secondary": enc.get("am_secondary", np.nan), "alt_km": enc.get("alt_km", np.nan),
               "drag_sens": enc.get("drag_sens", np.nan), "primary_maneuverable": enc.get("primary_maneuverable", np.nan),
               "secondary_debris": enc.get("secondary_debris", np.nan), "rcs_secondary": enc.get("rcs_secondary", np.nan),
               "updates": [{k: enc.get(k, np.nan) for k in ("tau_days", "miss_km", "tca_offset_s", "rel_speed_kms",
                                                             "sigma_major_km", "sigma_minor_km", "pc", "f107", "ap",
                                                             "ap_forecast_next")}]}
        seq["updates"][0]["tca_offset_s"] = 0.0
        f = features_at(seq, 0)
        p = float(self.predict_proba(np.array([[f[n] for n in FEATURES]], float))[0])
        level = "HIGH" if p >= 0.6 else "MEDIUM" if p >= 0.3 else "LOW"
        reasons = []
        if f["mahal_miss"] == f["mahal_miss"] and f["mahal_miss"] < 1.5:
            reasons.append("uncertainty is large compared with the miss distance")
        if f["tau_days"] > 2:
            reasons.append(f"{f['tau_days']:.1f} days to TCA: errors still shrink with new tracking")
        if f["drag_sens"] == f["drag_sens"] and f["drag_sens"] > 0.02:
            reasons.append("secondary is drag-sensitive (high area-to-mass, low altitude)")
        if f["ap_jump"] == f["ap_jump"] and f["ap_jump"] > 30:
            reasons.append("geomagnetic activity forecast to rise")
        if not reasons:
            reasons.append("small uncertainty relative to the miss distance; estimate looks settled")
        return {"p_change_next_update": p, "level": level, "reasons": reasons[:3]}
