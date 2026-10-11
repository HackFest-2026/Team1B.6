"""ML thermospheric-density / drag-uncertainty model.

Learns a correction  log(rho_true / rho_static)  as a function of altitude and the
space-weather drivers (F10.7 daily, F10.7 81-day mean, Ap, day-of-year) using gradient-
boosted quantile regression (10/50/90 %). The median gives the corrected density, the
quantile spread gives a *calibrated* drag uncertainty that feeds the covariance model,
re-entry Monte Carlo and Pc.

Training targets come from ``astro.atmosphere.reference_density`` (semi-empirical surrogate
of the real thermosphere). To train on measured data, pass arrays of observed densities
(e.g. GRACE-FO / Swarm / CHAMP accelerometer-derived) to ``DensityModel.fit_arrays``.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..astro.atmosphere import reference_density, static_density


def _features(h, f107, f107a, ap, doy):
    h = np.atleast_1d(np.asarray(h, float))
    n = len(h)
    b = lambda x: np.broadcast_to(np.asarray(x, float), (n,))
    doy = b(doy)
    return np.column_stack([h, b(f107), b(f107a), b(ap),
                            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                            np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)])


class DensityModel:
    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(self, seed=0):
        self.seed = seed
        self.models = {}
        self.metrics = {}
        self._table = None

    # ------------------------------------------------------------------ training
    def build_training_set(self, sw_series, n_per_day=40, rng=None):
        rng = rng or np.random.default_rng(self.seed)
        rows, ys, dates = [], [], []
        for d in sw_series:
            if "PRED" in d.get("source", ""):
                continue
            doy = datetime.fromisoformat(d["date"]).timetuple().tm_yday
            h = rng.uniform(150, 1000, n_per_day)
            rho = reference_density(h, d["f107"], d["f107a"], d["ap"], doy, rng, noise=0.10)
            rows.append(_features(h, d["f107"], d["f107a"], d["ap"], doy))
            ys.append(np.log(rho / static_density(h)))
            dates += [d["date"]] * n_per_day
        return np.vstack(rows), np.concatenate(ys), np.array(dates)

    def fit(self, sw_series, holdout_days=180):
        X, y, dates = self.build_training_set(sw_series)
        cut = sorted(set(dates))[-holdout_days] if len(set(dates)) > holdout_days * 2 else None
        tr = dates < cut if cut else np.ones(len(y), bool)
        te = ~tr if cut else np.zeros(len(y), bool)
        return self.fit_arrays(X[tr], y[tr], X[te] if te.any() else None, y[te] if te.any() else None)

    def fit_arrays(self, X, y, Xte=None, yte=None):
        for q in self.QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250,
                                              learning_rate=0.08, max_leaf_nodes=31,
                                              random_state=self.seed)
            m.fit(X, y)
            self.models[q] = m
        if Xte is not None and len(Xte):
            p50 = self.models[0.5].predict(Xte)
            p10 = self.models[0.1].predict(Xte)
            p90 = self.models[0.9].predict(Xte)
            rms_static = float(np.sqrt(np.mean(yte**2)))
            rms_ml = float(np.sqrt(np.mean((yte - p50) ** 2)))
            self.metrics = {
                "test_samples": int(len(yte)),
                "static_model_rms_log_error": rms_static,
                "ml_model_rms_log_error": rms_ml,
                "static_model_density_error_pct": float((math.exp(rms_static) - 1) * 100),
                "ml_model_density_error_pct": float((math.exp(rms_ml) - 1) * 100),
                "error_reduction_pct": float((1 - rms_ml / rms_static) * 100),
                "interval_80_coverage": float(np.mean((yte >= p10) & (yte <= p90))),
            }
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, h, sw, doy=None):
        doy = doy or datetime.fromisoformat(sw["date"]).timetuple().tm_yday
        X = _features(h, sw["f107"], sw["f107a"], sw["ap"], doy)
        rho0 = static_density(np.atleast_1d(h))
        q = {k: rho0 * np.exp(m.predict(X)) for k, m in self.models.items()}
        return q[0.1], q[0.5], q[0.9]

    def prepare(self, sw):
        """Cache a relative-sigma lookup table for the current space weather."""
        hs = np.arange(100, 1600, 10.0)
        lo, med, hi = self.predict(hs, sw)
        rel = (np.log(hi) - np.log(lo)) / 2.563      # 1-sigma of log-density
        self._table = (hs, rel, med / static_density(hs))
        return self

    def relative_sigma(self, h_km, sw=None):
        if self._table is None and sw is not None:
            self.prepare(sw)
        if self._table is None:
            return 0.25
        hs, rel, _ = self._table
        # add a forecast-uncertainty floor (drivers are themselves uncertain)
        return float(np.sqrt(np.interp(h_km, hs, rel) ** 2 + 0.12**2))

    def correction_factor(self, h_km):
        hs, _, fac = self._table
        return float(np.interp(h_km, hs, fac))

    def profile(self, sw):
        hs = np.arange(150, 1001, 25.0)
        lo, med, hi = self.predict(hs, sw)
        return {"alt_km": hs.tolist(), "static": static_density(hs).tolist(),
                "ml_p10": lo.tolist(), "ml_p50": med.tolist(), "ml_p90": hi.tolist()}
