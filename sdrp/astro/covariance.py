"""Covariance model for TLE-based states, in the RIC frame.

sigma(t) = sigma0(type, size) + linear growth (radial/cross) + along-track growth that is
quadratic in time and driven by *drag uncertainty*. The drag (density) uncertainty comes
from the AI density model's predictive quantiles, and a learned realism factor (fitted on
TLE-vs-TLE residuals, see ops/kpis.py) rescales the result so covariances are realistic.
"""
from __future__ import annotations

import math

import numpy as np

from .propagate import MU, RE, SpaceObject, ric_frame

# epoch 1-sigma (km) for (R, I, C), by object class - typical published TLE accuracy figures
BASE_SIGMA = {
    "PAYLOAD": (0.08, 0.25, 0.08),
    "ROCKET BODY": (0.10, 0.35, 0.10),
    "DEBRIS": (0.15, 0.60, 0.15),
    "UNKNOWN": (0.20, 0.80, 0.20),
}
GROWTH_PER_DAY = (0.03, 0.10, 0.03)   # linear growth (km/day)


class CovarianceModel:
    def __init__(self, density_model=None, realism=None):
        self.density_model = density_model     # ai.density.DensityModel or None
        self.realism = realism or {"R": 1.0, "I": 1.0, "C": 1.0}

    def drag_sigma_rel(self, h_km, sw):
        if self.density_model is not None and sw is not None:
            return float(self.density_model.relative_sigma(h_km, sw))
        return 0.25

    def sigma_ric(self, obj: SpaceObject, age_days: float, sw=None, drag=True):
        b = np.array(BASE_SIGMA.get(obj.obj_type, BASE_SIGMA["UNKNOWN"]), float)
        if obj.obj_type == "DEBRIS" and obj.rcs_m2:
            b *= float(np.clip((0.1 / max(obj.rcs_m2, 1e-3)) ** 0.25, 0.7, 2.5))
        age = abs(age_days)
        g = np.array(GROWTH_PER_DAY) * age
        # drag-induced along-track error: ds = 0.75 * a * n * v * B * (drho) * t^2
        h = max(obj.perigee_km, 120.0)
        a_m = obj.sma_km * 1000.0
        if h < 1500 and drag:
            from .atmosphere import static_density
            rho = float(static_density(h))
            n = math.sqrt(MU / obj.sma_km**3)               # rad/s
            v = math.sqrt(MU / obj.sma_km) * 1000.0         # m/s
            B = 2.2 * obj.am_ratio
            t = age * 86400.0
            ds = 0.75 * a_m * n * v * B * rho * self.drag_sigma_rel(h, sw) * t * t   # m
            g[1] += ds / 1000.0
        s = b + g
        s *= np.array([self.realism["R"], self.realism["I"], self.realism["C"]])
        return s

    def cov_eci(self, obj: SpaceObject, r, v, age_days, sw=None, drag=True):
        s = self.sigma_ric(obj, age_days, sw, drag)
        M = ric_frame(r, v)
        return M.T @ np.diag(s**2) @ M, s
