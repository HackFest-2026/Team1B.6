"""Offline / reproducible data generation.

Used automatically when live sources (CelesTrak, Space-Track, NOAA, DISCOS) cannot be
reached, and for building labelled training/validation sets (injected conjunctions,
maneuvers, fragmentations) that real public data does not label.

Everything produced here is tagged ``source="synthetic"`` so the dashboard can show it.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.exporter import export_tle

from ..astro.propagate import (MU, RE, OMEGA_E, SpaceObject, dt_to_jd, gmst)

EPOCH_1950 = datetime(1949, 12, 31, tzinfo=timezone.utc)


# ----------------------------------------------------------------------------- TLE building
def make_satrec(norad, epoch: datetime, n_rad_min, ecc, inc, raan, argp, M, bstar=1e-4):
    s = Satrec()
    ep = (epoch - EPOCH_1950).total_seconds() / 86400.0
    s.sgp4init(WGS72, "i", int(norad), ep, float(bstar), 0.0, 0.0, float(ecc), float(argp) % (2 * math.pi),
               min(max(float(inc), 0.0), math.pi), float(M) % (2 * math.pi), float(n_rad_min),
               float(raan) % (2 * math.pi))
    return s


def _tle_checksum(line):
    return sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line[:68]) % 10


def _with_launch_year(l1, years, norad):
    """Write a launch year into the international designator (TLE line 1, columns 10-17).

    The year is drawn from ``years=(first, last)`` with a generator seeded by the NORAD id, so the
    catalog's orbits (and the main random stream) are unchanged.
    """
    r = np.random.default_rng(int(norad))
    yy = int(r.integers(years[0], years[1] + 1)) % 100
    des = f"{yy:02d}{int(r.integers(1, 200)):03d}A  "
    l1 = l1[:9] + des + l1[17:68]
    return l1 + str(_tle_checksum(l1))


def satrec_to_object(s, name, obj_type="DEBRIS", rcs=None, mass=None, area=None,
                     country="", source="synthetic", launch_years=None):
    l1, l2 = export_tle(s)
    if launch_years:
        l1 = _with_launch_year(l1, launch_years, s.satnum)
    return SpaceObject(norad=int(s.satnum), name=name, line1=l1, line2=l2, obj_type=obj_type,
                       rcs_m2=rcs, mass_kg=mass, area_m2=area, country=country, source=source)


def alt_to_n(alt_km, ecc=0.0):
    a = RE + alt_km
    return math.sqrt(MU / a**3) * 60.0  # rad/min


def rv2coe(r, v):
    r = np.asarray(r, float); v = np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v); hn = np.linalg.norm(h)
    k = np.array([0, 0, 1.0])
    nvec = np.cross(k, h); nn = np.linalg.norm(nvec)
    evec = ((vn**2 - MU / rn) * r - np.dot(r, v) * v) / MU
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / rn - vn**2 / MU)
    inc = math.acos(np.clip(h[2] / hn, -1, 1))
    raan = math.atan2(nvec[1], nvec[0]) if nn > 1e-9 else 0.0
    if e > 1e-8 and nn > 1e-9:
        argp = math.acos(np.clip(np.dot(nvec, evec) / (nn * e), -1, 1))
        if evec[2] < 0:
            argp = 2 * math.pi - argp
        nu = math.acos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:
        argp = 0.0
        u = math.acos(np.clip(np.dot(nvec, r) / (nn * rn), -1, 1)) if nn > 1e-9 else 0.0
        if r[2] < 0:
            u = 2 * math.pi - u
        nu = u
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    M = E - e * math.sin(E)
    return a, e, inc, raan % (2 * math.pi), argp % (2 * math.pi), M % (2 * math.pi)


def _state(s, when):
    jd, fr = dt_to_jd(when)
    e, r, v = s.sgp4(jd, fr)
    return e, np.array(r), np.array(v)


def _fit_once(norad, epoch, t_target, r_t, v_t, bstar, iters, e_init=None, argp_shift=0.0):
    a, e, inc, raan, argp, M = rv2coe(r_t, v_t)
    if e_init is not None:
        # move perigee while keeping the argument of latitude (argp+M) consistent
        e = max(e, e_init)
        argp = argp + argp_shift
        M = M - argp_shift
    e = max(e, 2e-4)
    n = math.sqrt(MU / a**3) * 60.0
    dt_min = (t_target - epoch).total_seconds() / 60.0
    p = a * (1 - e * e)
    fac = 1.5 * 1.08262668e-3 * (RE / p) ** 2 * n
    raan0 = raan + fac * math.cos(inc) * dt_min
    argp0 = argp - fac * (2 - 2.5 * math.sin(inc) ** 2) * dt_min
    x = np.array([n, e, inc, raan0, argp0, M - n * dt_min])
    target = np.concatenate([r_t, np.asarray(v_t) * 1000.0])
    best = (np.inf, x.copy())
    for _ in range(iters):
        s = make_satrec(norad, epoch, *x, bstar=bstar)
        err, r, v = _state(s, t_target)
        if err:
            x[1] = max(2e-4, x[1] * 0.5)
            continue
        f = np.concatenate([r, v * 1000.0]) - target
        res = np.linalg.norm(f[:3]) + np.linalg.norm(f[3:]) * 1e-3
        if res < best[0]:
            best = (res, x.copy())
        if np.linalg.norm(f[:3]) < 1e-4 and np.linalg.norm(f[3:]) < 1e-3:
            break
        J = np.zeros((6, 6))
        h = np.array([1e-9, 1e-7, 1e-7, 1e-7, 1e-6, 1e-7])
        base = np.concatenate([r, v * 1000.0])
        for k in range(6):
            xp = x.copy(); xp[k] += h[k]
            _, rp, vp = _state(make_satrec(norad, epoch, *xp, bstar=bstar), t_target)
            J[:, k] = (np.concatenate([rp, vp * 1000.0]) - base) / h[k]
        dx = np.linalg.lstsq(J, -f, rcond=None)[0]
        lim = np.array([2e-4, 2e-3, 5e-3, 5e-3, 0.5, 0.5])
        x = x + np.clip(dx, -lim, lim)
        x[1] = min(max(x[1], 1e-5), 0.9)
    return best


def fit_tle_to_state(norad, epoch: datetime, t_target: datetime, r_t, v_t, bstar=1e-4, iters=30):
    """Find SGP4 mean elements at ``epoch`` whose propagation hits (r_t, v_t) at ``t_target``.

    Newton iteration with a finite-difference Jacobian and multi-start for near-circular
    orbits. Typical convergence: < 1 m.
    """
    best = _fit_once(norad, epoch, t_target, r_t, v_t, bstar, iters)
    for shift in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        if best[0] < 1e-3:
            break
        cand = _fit_once(norad, epoch, t_target, r_t, v_t, bstar, iters, 1e-3, shift)
        if cand[0] < best[0]:
            best = cand
    return make_satrec(norad, epoch, *best[1], bstar=bstar)


# ----------------------------------------------------------------------------- populations
def _shell(rng, norad0, name, n_planes, per_plane, alt, inc, epoch, obj_type="PAYLOAD",
           rcs=10.0, mass=300.0, area=12.0, bstar=1.5e-4, raan0=0.0, phase=1, alt_jit=0.0,
           raan_span=360.0, launch_years=None):
    out = []
    k = 0
    for p in range(n_planes):
        raan = math.radians(raan0 + raan_span * p / n_planes)
        for j in range(per_plane):
            M = 2 * math.pi * j / per_plane + 2 * math.pi * phase * p / (n_planes * per_plane)
            h = alt + rng.normal(0, alt_jit)
            argp = rng.uniform(0, 2 * math.pi)      # keep argument of latitude = M (phasing)
            s = make_satrec(norad0 + k, epoch, alt_to_n(h), 1e-4 + abs(rng.normal(0, 1e-4)),
                            math.radians(inc), raan, argp, M - argp,
                            bstar * rng.lognormal(0, 0.3))
            out.append(satrec_to_object(s, f"{name}-{k + 1:04d}", obj_type, rcs, mass, area, launch_years=launch_years))
            k += 1
    return out


def _cloud(rng, norad0, name, n, alt, inc, epoch, spread_km, ecc_max, raan_c, raan_spread,
           bstar_med, obj_type="DEBRIS"):
    out = []
    for k in range(n):
        a = RE + alt + rng.normal(0, spread_km)
        e = abs(rng.normal(0, ecc_max / 2))
        e = min(e, max(0.0, 1 - (RE + 160) / a) * 0.95)
        e = max(e, 1e-4)
        rcs = float(np.clip(rng.lognormal(math.log(0.03), 1.1), 0.002, 2.0))
        am = float(np.clip(rng.lognormal(math.log(0.08), 0.8), 0.005, 3.0))
        s = make_satrec(norad0 + k, epoch, math.sqrt(MU / a**3) * 60.0, e,
                        math.radians(inc + rng.normal(0, 0.35)),
                        math.radians(raan_c + rng.normal(0, raan_spread)),
                        rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                        bstar_med * am / 0.08 * rng.lognormal(0, 0.2))
        out.append(satrec_to_object(s, f"{name} DEB {k + 1:04d}", obj_type, rcs=rcs,
                                    area=rcs, mass=rcs / am))
    return out


def build_catalog(now: datetime, seed=42, scale=1.0, n_injected=14):
    """Return (objects, injected_events). ``scale`` multiplies population sizes."""
    rng = np.random.default_rng(seed)
    ep = now - timedelta(hours=6)
    sz = lambda x: max(1, int(round(x * scale)))
    objs: list[SpaceObject] = []
    objs += _shell(rng, 70000, "STARLINK-SYN", sz(24), 22, 550, 53.0, ep, rcs=12, mass=800,
                   area=30, bstar=2.0e-4, phase=11, launch_years=(2020, 2026))
    objs += _shell(rng, 71000, "STARLINK-SYN-B", sz(12), 20, 560, 97.6, ep, rcs=12, mass=800,
                   area=30, bstar=2.0e-4, phase=3, raan0=17, launch_years=(2021, 2026))
    objs += _shell(rng, 72000, "ONEWEB-SYN", sz(12), 16, 1200, 87.9, ep, rcs=6, mass=150,
                   area=6, bstar=1e-5, phase=1, launch_years=(2019, 2023))
    objs += _shell(rng, 73000, "IRIDIUM-NEXT-SYN", 6, 11, 780, 86.4, ep, rcs=8, mass=860,
                   area=12, bstar=3e-5, phase=2, raan0=5, raan_span=180.0, launch_years=(2017, 2019))
    objs += _shell(rng, 73500, "GPS-SYN", 6, 5, 20180, 55.0, ep, rcs=20, mass=2000,
                   area=20, bstar=0.0, phase=1, launch_years=(2004, 2024))
    # sun-synchronous EO payloads, random planes
    for k in range(sz(220)):
        h = rng.uniform(480, 830)
        s = make_satrec(74000 + k, ep, alt_to_n(h), abs(rng.normal(1e-3, 5e-4)) + 1e-4,
                        math.radians(97.4 + (h - 500) / 330 * 1.4 + rng.normal(0, 0.1)),
                        rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                        rng.uniform(0, 2 * math.pi), 6e-5 * rng.lognormal(0, 0.5))
        objs.append(satrec_to_object(s, f"EO-SAT-SYN {k + 1:03d}", "PAYLOAD",
                                     rcs=float(rng.uniform(0.5, 6)), mass=float(rng.uniform(50, 1500)),
                                     launch_years=(2012, 2025)))
    # rocket bodies (LEO and GTO)
    for k in range(sz(140)):
        h = rng.uniform(600, 1000); inc = float(rng.choice([71.0, 74.0, 82.9, 98.5, 51.6]))
        s = make_satrec(75000 + k, ep, alt_to_n(h), abs(rng.normal(4e-3, 3e-3)) + 2e-4,
                        math.radians(inc + rng.normal(0, 0.2)), rng.uniform(0, 2 * math.pi),
                        rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                        5e-5 * rng.lognormal(0, 0.5))
        objs.append(satrec_to_object(s, f"R/B-SYN {k + 1:03d}", "ROCKET BODY",
                                     rcs=float(rng.uniform(4, 15)), mass=float(rng.uniform(1000, 9000))))
    for k in range(sz(30)):
        hp, ha = rng.uniform(200, 600), rng.uniform(30000, 36000)
        a = RE + (hp + ha) / 2
        e = (ha - hp) / (2 * a)
        s = make_satrec(75500 + k, ep, math.sqrt(MU / a**3) * 60, e, math.radians(rng.uniform(5, 28)),
                        rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                        rng.uniform(0, 2 * math.pi), 1e-4)
        objs.append(satrec_to_object(s, f"GTO R/B-SYN {k + 1:02d}", "ROCKET BODY", rcs=12, mass=3000))
    # GEO belt
    for k in range(sz(110)):
        s = make_satrec(76000 + k, ep, alt_to_n(35786) * (1 + rng.normal(0, 3e-4)),
                        abs(rng.normal(2e-4, 1e-4)), math.radians(abs(rng.normal(0.05, 1.5))),
                        rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                        rng.uniform(0, 2 * math.pi), 0.0)
        objs.append(satrec_to_object(s, f"GEO-SAT-SYN {k + 1:03d}", "PAYLOAD", rcs=25, mass=4000, launch_years=(2000, 2024)))
    # historical fragmentation clouds (orbit families approximating the real clouds)
    objs += _cloud(rng, 80000, "FENGYUN 1C", sz(420), 850, 98.7, ep, 120, 0.03, 110, 25, 1.2e-4)
    objs += _cloud(rng, 81000, "COSMOS 2251", sz(260), 780, 74.0, ep, 90, 0.02, 200, 20, 1.5e-4)
    objs += _cloud(rng, 82000, "IRIDIUM 33", sz(110), 775, 86.4, ep, 80, 0.015, 40, 10, 1.5e-4)
    objs += _cloud(rng, 83000, "COSMOS 1408", sz(140), 420, 82.6, ep, 40, 0.008, 300, 15, 4e-4)
    # decaying objects -> re-entry candidates
    for k in range(sz(30)):
        hp = rng.uniform(150, 290); ha = hp + rng.uniform(0, 120)
        a = RE + (hp + ha) / 2; e = (ha - hp) / (2 * a) + 1e-4
        typ = str(rng.choice(["ROCKET BODY", "DEBRIS", "PAYLOAD"], p=[0.3, 0.5, 0.2]))
        rcs = {"ROCKET BODY": 8.0, "DEBRIS": 0.1, "PAYLOAD": 2.0}[typ] * rng.lognormal(0, 0.4)
        s = make_satrec(84000 + k, ep, math.sqrt(MU / a**3) * 60, e,
                        math.radians(float(rng.choice([51.6, 41.5, 97.5, 28.5, 65.0]))),
                        rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                        rng.uniform(0, 2 * math.pi), 3e-4 * rng.lognormal(0, 0.5))
        objs.append(satrec_to_object(s, f"DECAYING-SYN {k + 1:02d}", typ, rcs=float(rcs),
                                     mass=float({"ROCKET BODY": 1500, "DEBRIS": 5,
                                                 "PAYLOAD": 400}[typ] * rng.lognormal(0, 0.3))))

    events = inject_conjunctions(objs, now, rng, n_injected)
    return objs, events


def inject_conjunctions(objs, now, rng, n_events, horizon_h=22.0):
    """Create debris objects that pass close to selected LEO payloads at known times.
    Gives the screening/Pc pipeline realistic, high-risk events with known ground truth."""
    leo_payloads = [o for o in objs if o.obj_type == "PAYLOAD" and o.apogee_km < 1300]
    events = []
    base = 86000
    for k in range(n_events):
        prim = leo_payloads[int(rng.integers(len(leo_payloads)))]
        tca = now + timedelta(hours=float(rng.uniform(1.5, horizon_h)))
        jd, fr = dt_to_jd(tca)
        e, r, v = prim.sat.sgp4(jd, fr)
        if e:
            continue
        r, v = np.array(r), np.array(v)
        rhat = r / np.linalg.norm(r)
        ang = math.radians(rng.uniform(25, 170))            # crossing angle
        # Rodrigues rotation of v about the radial direction
        v2 = (v * math.cos(ang) + np.cross(rhat, v) * math.sin(ang)
              + rhat * np.dot(rhat, v) * (1 - math.cos(ang)))
        v2 = v2 * (1 + rng.normal(0, 0.003)) + rhat * rng.normal(0, 0.02)
        # miss vector in the encounter plane (perpendicular to relative velocity)
        dv = v2 - v
        u = dv / np.linalg.norm(dv)
        p1 = np.cross(u, rhat); p1 /= np.linalg.norm(p1)
        p2 = np.cross(u, p1)
        miss = float(np.exp(rng.uniform(math.log(0.015), math.log(4.0))))   # km
        phi = rng.uniform(0, 2 * math.pi)
        r2 = r + miss * (math.cos(phi) * p1 + math.sin(phi) * p2)
        age = timedelta(hours=float(rng.uniform(4, 40)))
        norad = base + k
        s = fit_tle_to_state(norad, now - age, tca, r2, v2, bstar=float(1.5e-4 * rng.lognormal(0, 0.4)))
        rcs = float(np.clip(rng.lognormal(math.log(0.05), 1.0), 0.003, 3))
        obj = satrec_to_object(s, f"DEBRIS-SYN {norad}", "DEBRIS", rcs=rcs, area=rcs,
                               mass=rcs / 0.05)
        # verify through the exported (rounded) TLE; recompute the true miss distance
        miss_true, tca_true = _min_distance(prim, obj, tca)
        if abs(miss_true - miss) > 0.25 + 0.2 * miss:
            continue
        miss, tca = miss_true, tca_true
        objs.append(obj)
        events.append({"primary": prim.norad, "secondary": norad, "tca": tca.isoformat(),
                       "true_miss_km": miss})
    return events


def _min_distance(a, b, t0, span_s=120.0):
    """Exact closest approach of two SpaceObjects near t0 (golden-section on SGP4)."""
    def d(ts):
        jd, fr = dt_to_jd(t0 + timedelta(seconds=ts))
        _, r1, _ = a.sat.sgp4(jd, fr); _, r2, _ = b.sat.sgp4(jd, fr)
        return float(np.linalg.norm(np.subtract(r1, r2)))
    lo, hi = -span_s, span_s
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = d(x1), d(x2)
    for _ in range(60):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = d(x1)
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = d(x2)
    ts = (lo + hi) / 2
    return d(ts), t0 + timedelta(seconds=ts)


# ----------------------------------------------------------------------------- Iridium-Cosmos
def iridium_cosmos_scenario(rng=None, od_sigma_ric_km=(0.08, 0.35, 0.08)):
    """Reconstruct the 10 Feb 2009 Iridium 33 / Cosmos 2251 collision geometry.

    Collision: 2009-02-10 16:55:59.8 UTC, ~789 km altitude over northern Siberia
    (72.5 N, 97.9 E), relative speed ~11.7 km/s. Iridium 33 i=86.4 deg, Cosmos 2251 i=74.0 deg.
    The 'truth' orbits collide exactly; the 'catalog' TLEs carry realistic orbit-determination
    errors, as the operators' TLE-based screening did (SOCRATES predicted ~584 m that week).
    """
    rng = rng or np.random.default_rng(2012)   # gives a ~0.4 km predicted miss, comparable to SOCRATES
    tca = datetime(2009, 2, 10, 16, 55, 59, 800000, tzinfo=timezone.utc)
    lat, lon, alt = math.radians(72.5), math.radians(97.9), 789.0
    jd, fr = dt_to_jd(tca)
    th = float(gmst(np.array([jd + fr]))[0])
    rr = RE + alt
    lam = lon + th   # inertial longitude
    r = rr * np.array([math.cos(lat) * math.cos(lam), math.cos(lat) * math.sin(lam), math.sin(lat)])
    east = np.array([-math.sin(lam), math.cos(lam), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lam), -math.sin(lat) * math.sin(lam), math.cos(lat)])
    vc = math.sqrt(MU / rr)

    def vel(inc_deg, descending):
        saz = math.cos(math.radians(inc_deg)) / math.cos(lat)
        az = math.asin(np.clip(saz, -1, 1))
        if descending:
            az = math.pi - az
        return vc * (math.sin(az) * east + math.cos(az) * north)

    v_iri = vel(86.4, True)
    v_cos = vel(74.0, False)
    truth_iri = fit_tle_to_state(24946, tca - timedelta(hours=20), tca, r, v_iri, bstar=2.5e-5)
    truth_cos = fit_tle_to_state(22675, tca - timedelta(hours=14), tca, r, v_cos, bstar=6.0e-5)

    def noisy(norad, s_truth, v_ref, epoch):
        jd_, fr_ = dt_to_jd(tca)
        _, rt, vt = s_truth.sgp4(jd_, fr_)
        rt, vt = np.array(rt), np.array(vt)
        R = rt / np.linalg.norm(rt); C = np.cross(rt, vt); C /= np.linalg.norm(C); I = np.cross(C, R)
        d = rng.normal(0, 1, 3) * np.array(od_sigma_ric_km)
        return fit_tle_to_state(norad, epoch, tca, rt + d[0] * R + d[1] * I + d[2] * C, vt,
                                bstar=s_truth.bstar)

    cat_iri = noisy(24946, truth_iri, v_iri, tca - timedelta(hours=20))
    cat_cos = noisy(22675, truth_cos, v_cos, tca - timedelta(hours=14))
    o1 = satrec_to_object(cat_iri, "IRIDIUM 33", "PAYLOAD", rcs=4.9, mass=560, area=4.0,
                          country="US", source="reconstruction")
    o2 = satrec_to_object(cat_cos, "COSMOS 2251", "PAYLOAD", rcs=2.9, mass=900, area=3.0,
                          country="CIS", source="reconstruction")
    return {"tca": tca, "objects": [o1, o2], "truth": [truth_iri, truth_cos],
            "location": {"lat": 72.5, "lon": 97.9, "alt_km": 789.0}}


# ----------------------------------------------------------------------------- space weather
def synthetic_space_weather(start: datetime, end: datetime, seed=7):
    """Daily F10.7 / Ap series shaped like solar cycles 24-25 (peak ~late 2024)."""
    rng = np.random.default_rng(seed)
    days = (end - start).days + 1
    out = []
    t0 = datetime(2019, 12, 1, tzinfo=timezone.utc)   # cycle 25 minimum
    rot = rng.uniform(0, 2 * math.pi)
    ap_state = 0.0
    for d in range(days):
        day = start + timedelta(days=d)
        yrs = (day - t0).days / 365.25
        cyc = (yrs % 11.0) / 11.0
        base = 68 + 120 * math.sin(math.pi * cyc) ** 2.2
        f = base + 0.18 * base * math.sin(2 * math.pi * d / 27.0 + rot) + rng.normal(0, 6)
        ap_state = 0.7 * ap_state + rng.normal(0, 0.45)
        ap = float(np.clip(math.exp(1.9 + ap_state) * (1 + 0.004 * (base - 70)), 2, 300))
        if rng.random() < 0.01:
            ap = float(rng.uniform(80, 250))   # geomagnetic storm
        out.append({"date": day.date().isoformat(), "f107": float(max(63, f)), "ap": ap})
    f = np.array([o["f107"] for o in out])
    for i, o in enumerate(out):
        lo, hi = max(0, i - 40), min(len(out), i + 41)
        o["f107a"] = float(f[lo:hi].mean())
        o["kp"] = float(np.clip(math.log(max(o["ap"], 1)) / math.log(400) * 9, 0, 9))
        o["source"] = "synthetic"
    return out


# ----------------------------------------------------------------------------- TLE histories
def _sgp4_ndot_per_bstar(norad, ep, n, e, i, node, argp, M, t_min=1440.0):
    s1 = make_satrec(norad, ep, n, e, i, node, argp, M, bstar=1e-4)
    s0 = make_satrec(norad, ep, n, e, i, node, argp, M, bstar=0.0)
    when = ep + timedelta(minutes=t_min)
    _, r1, v1 = _state(s1, when)
    _, r0, _ = _state(s0, when)
    a = np.linalg.norm(r0)
    ang = math.atan2(np.dot(np.cross(r0, r1), np.cross(r0, v1)) / max(np.linalg.norm(np.cross(r0, v1)), 1e-9),
                     np.dot(r0, r1))
    return max(2 * ang / t_min**2 / 1e-4, 0.0)



def tle_histories(objs, sw_by_date, end: datetime, days=60, n_objects=160, seed=11,
                  p_maneuver=0.25, p_frag=0.05):
    """Generate daily TLE histories for LEO objects with known injected events.

    Truth evolution: mean elements advance with SGP4's own secular J2 rates, while the
    mean motion decays according to the *reference* (space-weather-driven) atmosphere,
    which SGP4's static drag model does not know. Reported B* includes OD noise.
    Returns list of {norad, epochs[], satrecs[], labels[]} (label: 0 normal, 1 maneuver,
    2 fragmentation/breakup signature).
    """
    from ..astro.atmosphere import reference_density
    rng = np.random.default_rng(seed)
    pool = [o for o in objs if o.regime == "LEO" and 420 < o.perigee_km < 1000 and o.ecc < 0.02]
    pick = [pool[i] for i in rng.choice(len(pool), min(n_objects, len(pool)), replace=False)]
    out = []
    for o in pick:
        s0 = o.sat
        n, e, i, node, argp, M = s0.no_kozai, s0.ecco, s0.inclo, s0.nodeo, s0.argpo, s0.mo
        B = o.am_ratio * 2.2                        # Cd*A/m  (m^2/kg)
        B_prev = B
        start = end - timedelta(days=days)
        epochs, sats, labels = [], [], []
        # payloads maneuver (and rarely break up); debris / rocket bodies can only fragment
        if o.obj_type == "PAYLOAD":
            u = rng.random()
            ev_kind = 1 if u < p_maneuver else 2 if u < p_maneuver + p_frag * 0.5 else 0
        else:
            ev_kind = 2 if rng.random() < p_frag else 0
        ev_day = int(rng.integers(15, days - 5)) if ev_kind else -1
        # SGP4's own drag response per unit B* (the OD process fits B* so SGP4 reproduces
        # the observed decay; the fitted B* therefore lags real density changes)
        k_sgp4 = _sgp4_ndot_per_bstar(o.norad, start, n, e, i, node, argp, M)
        ndot_hist = []
        for d in range(days):
            ep = start + timedelta(days=d)
            sw = sw_by_date.get(ep.date().isoformat()) or {"f107": 120, "f107a": 120, "ap": 12}
            a_km = (MU / (n / 60) ** 2) ** (1 / 3)
            h = a_km - RE
            if h < 180:          # decayed - end of this object's history
                break
            rho = float(reference_density(h, sw["f107"], sw["f107a"], sw["ap"],
                                          ep.timetuple().tm_yday, rng, 0.08))
            v = math.sqrt(MU / a_km) * 1000
            ndot = 1.5 * (n / 60) * rho * B * v * 3600.0   # rad/min^2
            lab = 0
            B_prev = B
            if d == ev_day:
                lab = ev_kind
                if ev_kind == 1:    # maneuver: orbit raise or plane change
                    if rng.random() < 0.7:
                        n *= 1 - rng.uniform(4e-4, 2e-3)
                    else:
                        i += math.radians(rng.uniform(0.02, 0.1)) * rng.choice([-1, 1])
                else:               # breakup: area-to-mass jump and small dv
                    B *= rng.uniform(4, 12)
                    n *= 1 + rng.normal(0, 4e-4)
                    e = min(0.05, e + abs(rng.normal(0, 1e-3)))
            tmp = make_satrec(o.norad, ep, n, e, i, node, argp, M)
            ndot_hist.append(ndot)
            ndot_fit = float(np.mean(ndot_hist[-3:]))
            bstar_rep = max(ndot_fit / k_sgp4, 1e-7) * rng.lognormal(0, 0.10) if k_sgp4 > 0 else 1e-5
            if lab == 2:
                bstar_rep *= B / max(B_prev, 1e-9)     # the breakup's new area-to-mass shows up at once
            s = make_satrec(o.norad, ep, n * (1 + rng.normal(0, 1e-6)), max(1e-5, e + rng.normal(0, 2e-5)),
                            i + rng.normal(0, 1e-5), node + rng.normal(0, 2e-5), argp, M + rng.normal(0, 2e-5),
                            bstar_rep)
            epochs.append(ep); sats.append(s); labels.append(lab)
            dt = 1440.0
            M = (M + tmp.mdot * dt + 0.5 * ndot * dt * dt) % (2 * math.pi)
            node = (node + tmp.nodedot * dt) % (2 * math.pi)
            argp = (argp + tmp.argpdot * dt) % (2 * math.pi)
            n = n + ndot * dt
        out.append({"norad": o.norad, "name": o.name, "type": o.obj_type, "epochs": epochs,
                    "satrecs": sats, "labels": labels, "source": "synthetic"})
    return out
