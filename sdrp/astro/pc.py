"""Probability of collision (Pc).

* ``pc_foster``   2-D short-encounter Pc (Foster/Alfriend-Akella): integrate the combined
                  position Gaussian, projected on the encounter plane, over the hard-body disk.
* ``pc_montecarlo`` brute-force sampling, used to validate the analytic value.
"""
from __future__ import annotations

import numpy as np

_GL_R = np.polynomial.legendre.leggauss(48)
_N_THETA = 96


def encounter_plane(dr, dv):
    """Orthonormal basis (2x3) of the plane perpendicular to the relative velocity."""
    u = dv / np.linalg.norm(dv)
    x = dr - np.dot(dr, u) * u
    if np.linalg.norm(x) < 1e-12:
        x = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(u, x)
    return np.vstack([x, y])


def pc_foster(dr, dv, C, hbr_km):
    """dr, dv: relative position/velocity (km, km/s) at TCA; C: combined 3x3 cov (km^2)."""
    dr = np.asarray(dr, float); dv = np.asarray(dv, float)
    if not np.all(np.isfinite(dr)) or not np.all(np.isfinite(C)) or np.linalg.norm(dv) < 1e-6:
        # co-moving objects have no defined encounter plane: not a short-term conjunction
        return 0.0, {"miss_2d_km": [float(np.linalg.norm(dr)) if np.all(np.isfinite(dr)) else 0.0, 0.0],
                     "sigma_major_km": 0.0, "sigma_minor_km": 0.0}
    Bm = encounter_plane(dr, dv)
    m = Bm @ dr
    P = Bm @ C @ Bm.T
    P = 0.5 * (P + P.T) + np.eye(2) * 1e-12
    Pinv = np.linalg.inv(P)
    det = np.linalg.det(P)
    # polar Gauss-Legendre quadrature over the disk of radius hbr
    xr, wr = _GL_R
    rr = 0.5 * hbr_km * (xr + 1)
    wr = 0.5 * hbr_km * wr
    th = np.linspace(0, 2 * np.pi, _N_THETA, endpoint=False)
    R, T = np.meshgrid(rr, th, indexing="ij")
    X = np.stack([R * np.cos(T), R * np.sin(T)], -1) - m
    q = np.einsum("...i,ij,...j->...", X, Pinv, X)
    f = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
    pc = float(np.sum(f * R * wr[:, None]) * (2 * np.pi / _N_THETA))
    sig = np.sqrt(np.linalg.eigvalsh(P))
    return min(max(pc, 0.0), 1.0), {"miss_2d_km": m.tolist(), "sigma_major_km": float(sig[-1]),
                                    "sigma_minor_km": float(sig[0])}


def pc_max(dr, dv, C, hbr_km):
    """Maximum Pc over covariance scale factors (Alfano 'dilution' check)."""
    best = (0.0, 1.0)
    for k in np.logspace(-1.5, 1.5, 19):
        p, _ = pc_foster(dr, dv, C * k * k, hbr_km)
        if p > best[0]:
            best = (p, k)
    return best


def pc_montecarlo(dr, dv, C, hbr_km, n=200_000, rng=None):
    rng = rng or np.random.default_rng(0)
    Bm = encounter_plane(np.asarray(dr, float), np.asarray(dv, float))
    m = Bm @ dr
    P = Bm @ C @ Bm.T
    s = rng.multivariate_normal(m, P, size=n)
    hits = np.count_nonzero(np.hypot(s[:, 0], s[:, 1]) < hbr_km)
    p = hits / n
    return p, 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n)
