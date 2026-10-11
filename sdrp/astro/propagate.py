"""SGP4/SDP4 propagation helpers (vectorised with sgp4.api.SatrecArray).

All positions are in the TEME frame (km, km/s) unless stated otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, SatrecArray, WGS72, jday

MU = 398600.4418          # km^3/s^2
RE = 6378.137             # km
J2 = 1.08262668e-3
OMEGA_E = 7.2921150e-5    # rad/s
MIN_PER_DAY = 1440.0


# ----------------------------------------------------------------------------- time
def dt_to_jd(dt: datetime):
    dt = dt.astimezone(timezone.utc)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                  dt.second + dt.microsecond * 1e-6)
    return jd, fr


def jd_to_dt(jd: float, fr: float = 0.0) -> datetime:
    return datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=(jd - 2451545.0) + fr)


def time_grid(start: datetime, hours: float, step_s: float):
    """Return (jd, fr) arrays for a uniform grid plus the list of seconds offsets."""
    n = int(round(hours * 3600 / step_s)) + 1
    secs = np.arange(n) * step_s
    jd0, fr0 = dt_to_jd(start)
    fr = fr0 + secs / 86400.0
    jd = np.full(n, jd0, dtype=float)
    return jd, fr, secs


def gmst(jd_ut1: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (rad), IAU-82 (as used by SGP4/TEME)."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    g = (-6.2e-6 * t**3 + 0.093104 * t**2 + (876600.0 * 3600 + 8640184.812866) * t + 67310.54841)
    g = np.radians((g % 86400.0) / 240.0)
    return g % (2 * np.pi)


def teme_to_ecef(r: np.ndarray, jd_total: np.ndarray) -> np.ndarray:
    """Rotate TEME position(s) to an Earth-fixed frame (polar motion ignored).
    r: (..., T, 3) with jd_total shape (T,)"""
    th = gmst(np.asarray(jd_total))
    c, s = np.cos(th), np.sin(th)
    x = c * r[..., 0] + s * r[..., 1]
    y = -s * r[..., 0] + c * r[..., 1]
    return np.stack([x, y, r[..., 2]], axis=-1)


def ecef_to_geodetic(r: np.ndarray):
    """Spherical-earth approx: returns lat (deg), lon (deg), alt (km)."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    rn = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(np.clip(z / np.maximum(rn, 1e-9), -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, rn - RE


# ----------------------------------------------------------------------------- catalog
@dataclass
class SpaceObject:
    norad: int
    name: str
    line1: str
    line2: str
    obj_type: str = "UNKNOWN"          # PAYLOAD | ROCKET BODY | DEBRIS | UNKNOWN
    rcs_m2: float | None = None         # radar cross section
    mass_kg: float | None = None
    area_m2: float | None = None
    country: str = ""
    source: str = "synthetic"
    sat: Satrec = field(default=None, repr=False)

    def __post_init__(self):
        if self.sat is None:
            self.sat = Satrec.twoline2rv(self.line1, self.line2, WGS72)

    # ---- mean elements
    @property
    def n_rad_min(self):
        return self.sat.no_kozai

    @property
    def period_min(self):
        return 2 * math.pi / self.sat.no_kozai

    @property
    def sma_km(self):
        n = self.sat.no_kozai / 60.0  # rad/s
        return (MU / n**2) ** (1 / 3)

    @property
    def ecc(self):
        return self.sat.ecco

    @property
    def inc_deg(self):
        return math.degrees(self.sat.inclo)

    @property
    def raan_deg(self):
        return math.degrees(self.sat.nodeo)

    @property
    def argp_deg(self):
        return math.degrees(self.sat.argpo)

    @property
    def perigee_km(self):
        return self.sma_km * (1 - self.ecc) - RE

    @property
    def apogee_km(self):
        return self.sma_km * (1 + self.ecc) - RE

    @property
    def bstar(self):
        return self.sat.bstar

    @property
    def epoch(self) -> datetime:
        return jd_to_dt(self.sat.jdsatepoch, self.sat.jdsatepochF)

    @property
    def regime(self):
        p, a = self.perigee_km, self.apogee_km
        if a < 2000:
            return "LEO"
        if 34000 < p < 37600 and a < 37600:
            return "GEO"
        if p < 2000:
            return "HEO"
        return "MEO"

    @property
    def am_ratio(self):
        """Area-to-mass ratio (m^2/kg). Uses physical data if known, otherwise B* inversion."""
        if self.area_m2 and self.mass_kg:
            return self.area_m2 / self.mass_kg
        # B* = rho0 * Cd * A/m / 2 ; rho0 = 0.15696615 kg/m^2/ER ; Cd ~ 2.2
        b = 2.0 * abs(self.bstar) / 0.15696615
        return max(b / 2.2, 1e-4)

    def est_mass(self):
        if self.mass_kg:
            return self.mass_kg
        rcs = self.rcs_m2 or {"PAYLOAD": 4.0, "ROCKET BODY": 10.0}.get(self.obj_type, 0.05)
        # crude areal-density relation: m ~ 60 * RCS^1.3 for intact, small debris ~ 1 kg/0.01 m^2
        return float(max(0.1, 60.0 * rcs ** 1.3))

    def est_radius_m(self):
        """Circumscribing-sphere radius used for the hard-body radius. RCS/area understate
        the span of solar arrays, so intact objects get operational minimums (CARA-style)."""
        if self.area_m2:
            r = math.sqrt(self.area_m2 / math.pi)
        else:
            rcs = self.rcs_m2 or {"PAYLOAD": 4.0, "ROCKET BODY": 10.0}.get(self.obj_type, 0.05)
            r = max(0.05, math.sqrt(rcs / math.pi))
        floor = {"PAYLOAD": 5.0, "ROCKET BODY": 3.0}.get(self.obj_type, 0.0)
        return max(r * (2.5 if self.obj_type == "PAYLOAD" else 1.0), floor)

    def summary(self):
        return {
            "norad": self.norad, "name": self.name, "type": self.obj_type,
            "regime": self.regime, "perigee_km": round(self.perigee_km, 1),
            "apogee_km": round(self.apogee_km, 1), "inc_deg": round(self.inc_deg, 2),
            "ecc": round(self.ecc, 5), "period_min": round(self.period_min, 2),
            "bstar": self.bstar, "rcs_m2": self.rcs_m2, "mass_kg": round(self.est_mass(), 1),
            "am_ratio": round(self.am_ratio, 5), "epoch": self.epoch.isoformat(),
            "source": self.source, "country": self.country,
        }


class Catalog:
    def __init__(self, objects: list[SpaceObject]):
        # de-duplicate by NORAD id (keep newest epoch)
        best = {}
        for o in objects:
            if o.sat.error:
                continue
            if o.norad not in best or o.sat.jdsatepoch + o.sat.jdsatepochF > \
                    best[o.norad].sat.jdsatepoch + best[o.norad].sat.jdsatepochF:
                best[o.norad] = o
        self.objects = list(best.values())
        self.index = {o.norad: i for i, o in enumerate(self.objects)}
        self._arr = SatrecArray([o.sat for o in self.objects]) if self.objects else None

    def __len__(self):
        return len(self.objects)

    def get(self, norad):
        i = self.index.get(int(norad))
        return None if i is None else self.objects[i]

    def propagate(self, jd: np.ndarray, fr: np.ndarray, idx=None):
        """Return r, v arrays (N, T, 3) and error codes (N, T)."""
        if idx is None:
            arr = self._arr
        else:
            arr = SatrecArray([self.objects[i].sat for i in idx])
        e, r, v = arr.sgp4(np.asarray(jd, float), np.asarray(fr, float))
        r[e != 0] = np.nan
        v[e != 0] = np.nan
        return r, v, e

    def elements_table(self):
        return np.array([[o.sma_km, o.ecc, o.inc_deg, o.raan_deg, o.argp_deg,
                          o.perigee_km, o.apogee_km, o.bstar] for o in self.objects])


def propagate_one(obj: SpaceObject, when: datetime):
    jd, fr = dt_to_jd(when)
    e, r, v = obj.sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return np.array(r), np.array(v)


def ric_frame(r, v):
    """Rotation matrix whose rows are the Radial, In-track, Cross-track unit vectors."""
    r = np.asarray(r, float); v = np.asarray(v, float)
    R = r / np.linalg.norm(r)
    C = np.cross(r, v); C /= np.linalg.norm(C)
    I = np.cross(C, R)
    return np.vstack([R, I, C])
