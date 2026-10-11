"""Atmospheric density models used for drag, decay and re-entry.

* ``static_density``  - Vallado exponential model (mean solar activity). This is the
  "physics baseline" the AI model corrects.
* ``reference_density`` - a semi-empirical surrogate of the true thermosphere that responds
  to F10.7 (daily + 81-day mean), Ap and season. It is used as the *ground truth* for
  training/validating the ML drag model when no accelerometer-derived densities
  (GRACE/CHAMP/Swarm) are supplied. Replace ``reference_density`` with real
  density observations to retrain on measured data.
"""
import numpy as np

# h0 (km), rho0 (kg/m^3), H (km)  -- Vallado, "Fundamentals of Astrodynamics", Table 8-4
_TABLE = np.array([
    (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554),
    (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799),
    (90, 3.396e-6, 5.382), (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
])


def static_density(h_km):
    h = np.clip(np.asarray(h_km, float), 0, 2500)
    i = np.clip(np.searchsorted(_TABLE[:, 0], h, side="right") - 1, 0, len(_TABLE) - 1)
    h0, r0, H = _TABLE[i, 0], _TABLE[i, 1], _TABLE[i, 2]
    return r0 * np.exp(-(h - h0) / H)


def solar_sensitivity(h_km):
    """How strongly log-density responds to solar/geomagnetic drivers at a given altitude."""
    return 0.3 + 1.7 * np.clip((np.asarray(h_km, float) - 120.0) / 380.0, 0.0, 1.2)


def reference_density(h_km, f107, f107a, ap, doy, rng=None, noise=0.10):
    k = solar_sensitivity(h_km)
    log_s = k * ((np.asarray(f107a) - 150.0) / 100.0 * 1.0 +
                 (np.asarray(f107) - np.asarray(f107a)) / 100.0 * 0.4 +
                 0.010 * (np.asarray(ap) - 15.0) +
                 0.12 * np.sin(4 * np.pi * (np.asarray(doy) - 100) / 365.25))
    rho = static_density(h_km) * np.exp(log_s)
    if rng is not None and noise > 0:
        rho = rho * np.exp(rng.normal(0, noise, np.shape(rho)))
    return rho
