"""End-to-end batch pipeline: ingest -> AI models -> screening -> Pc -> alerts -> KPIs.

Run with ``python run.py pipeline``. Designed to be scheduled (cron / cloud batch); the
web server can also trigger it. Results are written to outputs/state.json (dashboard) and
outputs/state.pkl (full state for the API).
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config
from .ai.anomaly import detect, detect_debris_clouds
from .ai.calibration import run_study
from .ai.density import DensityModel
from .ai.explain import explain, typical_values
from .ai.forecast import ChangeForecaster, drag_sensitivity, sequences_from_cdm_history, simulate_sequences
from .ai.gnn_screen import GNNScreener
from .ai.reentry import reentry_watchlist
from .astro.covariance import CovarianceModel
from .astro.pc import pc_foster, pc_max
from .astro.propagate import Catalog, SpaceObject, ecef_to_geodetic, teme_to_ecef, dt_to_jd
from .astro.screening import apogee_perigee_pairs, screen, screen_pairs, screen_subset
from .data import sources, synthetic
from .ops.kpis import compare_with_cdms, fit_realism
from .ops.lifecycle import fleet as lifecycle_fleet, lifecycle
from .ops.maneuver import recommend

log = logging.getLogger("sdrp.pipeline")


class Progress:
    def __init__(self, cb=None):
        self.cb = cb
        self.t0 = time.perf_counter()
        self.timings = {}
        self._last = self.t0

    def step(self, name, pct):
        now = time.perf_counter()
        log.info("[%3d%%] %s (%.1fs)", pct, name, now - self.t0)
        if self.cb:
            self.cb(name, pct)

    def lap(self, key):
        now = time.perf_counter()
        self.timings[key] = round(now - self._last, 2)
        self._last = now


# ----------------------------------------------------------------------------- data
def load_data(now, mode, prog):
    prov = {"mode": mode}
    st = sources.SpaceTrack()
    cdms = []
    objs, injected = [], []
    if mode != "offline":
        try:
            objs = sources.fetch_celestrak()
        except Exception as e:
            log.warning("CelesTrak unavailable: %s", e)
        if len(objs) >= 100:
            prov["catalog"] = "CelesTrak GP (live)"
            sc = {}
            try:
                sc = sources.fetch_satcat()
                for o in objs:
                    m = sc.get(o.norad)
                    if m:
                        o.obj_type, o.rcs_m2, o.country = m["type"], m["rcs"], m["country"]
                prov["physical"] = "CelesTrak SATCAT (type, RCS)"
            except Exception as e:
                log.warning("SATCAT unavailable: %s", e)
            # cap size: keep crewed stations + decaying objects, then a stratified sample
            uniq = {o.norad: o for o in objs if not o.sat.error}
            objs = list(uniq.values())
            if len(objs) > config.MAX_OBJECTS:
                objs = _stratified_cap(objs, config.MAX_OBJECTS, config.RANDOM_SEED)
            # make sure every object named in a live CDM is in the screened catalog
            if st.enabled:
                try:
                    cdms = st.cdms()
                    have = {o.norad for o in objs}
                    need = {c[k] for c in cdms for k in ("sat1", "sat2")} - have
                    extra = [o for o in st.gp_tles(need) if not o.sat.error]
                    for o in extra:
                        m = sc.get(o.norad) if sc else None
                        if m:
                            o.obj_type, o.rcs_m2, o.country = m["type"], m["rcs"], m["country"]
                    objs += extra
                    prov["cdm"] = f"Space-Track cdm_public ({len(cdms)} messages)"
                except Exception as e:
                    log.warning("Space-Track CDMs failed: %s", e)
            if config.DISCOS_TOKEN:
                d = sources.fetch_discos([o.norad for o in objs])
                for o in objs:
                    if o.norad in d:
                        o.mass_kg = d[o.norad].get("mass") or o.mass_kg
                        o.area_m2 = d[o.norad].get("area") or o.area_m2
                prov["physical"] = prov.get("physical", "") + f" + ESA DISCOS ({len(d)} objects)"
        elif mode == "live":
            raise RuntimeError("Live mode requested but CelesTrak could not be reached")
    if len(objs) < 100:
        objs, injected = synthetic.build_catalog(now, seed=config.RANDOM_SEED)
        prov["catalog"] = "Offline synthetic catalog (realistic orbit families + injected encounters)"
        prov["physical"] = "synthetic RCS / mass"
    prog.step(f"catalog loaded: {len(objs)} objects", 8)

    sw, sw_now = [], {}
    if mode != "offline":
        try:
            sw = sources.fetch_space_weather()
            prov["space_weather"] = "CelesTrak SW (F10.7, Ap)"
        except Exception as e:
            log.warning("space weather unavailable: %s", e)
        sw_now = sources.fetch_noaa_now()
        if sw_now:
            prov["space_weather"] = prov.get("space_weather", "") + " + NOAA SWPC real-time"
    if len(sw) < 400:
        sw = synthetic.synthetic_space_weather(now - timedelta(days=5 * 365), now + timedelta(days=60))
        prov["space_weather"] = "synthetic solar-cycle series (offline)"

    histories = []
    return objs, injected, sw, sw_now, cdms, histories, prov, st


def _stratified_cap(objs, cap, seed):
    rng = np.random.default_rng(seed)
    must = [o for o in objs if any(k in o.name for k in ("ISS", "CSS", "TIANHE")) or o.perigee_km < 350]
    must = must[: cap // 5]
    must_ids = {o.norad for o in must}
    rest = [o for o in objs if o.norad not in must_ids]
    quota = {"DEBRIS": 0.45, "PAYLOAD": 0.43, "ROCKET BODY": 0.08, "UNKNOWN": 0.04}
    left = cap - len(must)
    out = list(must)
    pools = {t: [o for o in rest if o.obj_type == t] for t in quota}
    for t, q in quota.items():
        pool = pools[t]
        k = min(len(pool), int(left * q))
        out += [pool[i] for i in rng.choice(len(pool), k, replace=False)] if k else []
    chosen = {o.norad for o in out}
    spare = [o for o in rest if o.norad not in chosen]
    if len(out) < cap and spare:
        out += [spare[i] for i in rng.choice(len(spare), min(cap - len(out), len(spare)), replace=False)]
    return out[:cap]


def _sw_for(sw_by_date, day):
    key = day.date().isoformat()
    if key in sw_by_date:
        return sw_by_date[key]
    keys = sorted(sw_by_date)
    return sw_by_date[keys[-1]] if key > keys[-1] else sw_by_date[keys[0]]


def _histories_from_spacetrack(st, catalog, n=80):
    leo = [o for o in catalog.objects if o.regime == "LEO" and o.perigee_km > 300][:n]
    rows = st.gp_history([o.norad for o in leo])
    by = {}
    for r in rows:
        try:
            o = SpaceObject(norad=int(r["NORAD_CAT_ID"]), name=r.get("OBJECT_NAME", ""),
                            line1=r["TLE_LINE1"], line2=r["TLE_LINE2"],
                            obj_type={"PAYLOAD": "PAYLOAD", "ROCKET BODY": "ROCKET BODY",
                                      "DEBRIS": "DEBRIS"}.get(r.get("OBJECT_TYPE"), "UNKNOWN"))
            by.setdefault(o.norad, []).append(o)
        except Exception:
            continue
    out = []
    for norad, lst in by.items():
        lst.sort(key=lambda o: o.epoch)
        # one per ~day
        keep, last = [], None
        for o in lst:
            if last is None or (o.epoch - last).total_seconds() > 18 * 3600:
                keep.append(o); last = o.epoch
        if len(keep) >= 10:
            out.append({"norad": norad, "name": keep[0].name, "type": keep[0].obj_type,
                        "epochs": [o.epoch for o in keep], "satrecs": [o.sat for o in keep],
                        "labels": None, "source": "space-track"})
    return out


# ----------------------------------------------------------------------------- main
def run(mode=None, horizon_h=None, step_s=None, threshold_km=None, fast=False, progress_cb=None):
    mode = mode or config.DATA_MODE
    horizon_h = horizon_h or (12.0 if fast else config.SCREEN_HORIZON_H)
    step_s = step_s or (30.0 if fast else config.SCREEN_STEP_S)
    thr = threshold_km or config.SCREEN_THRESHOLD_KM
    prog = Progress(progress_cb)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    prog.step("starting", 1)

    objs, injected, sw, sw_now, cdms, histories, prov, st = load_data(now, mode, prog)
    catalog = Catalog(objs)
    sw_by_date = {d["date"]: d for d in sw}
    sw_today = dict(_sw_for(sw_by_date, now))
    if sw_now.get("f107"):
        sw_today["f107"] = sw_now["f107"]
    if sw_now.get("kp") is not None:
        sw_today["kp"] = sw_now["kp"]
    prog.lap("ingest")

    # ---- AI density / drag model
    prog.step("training ML density / drag-uncertainty model", 12)
    dm = DensityModel(seed=config.RANDOM_SEED).fit(sw)
    dm.prepare(sw_today)
    prog.lap("density_model")

    # ---- TLE histories
    if st.enabled and mode != "offline":
        try:
            histories = _histories_from_spacetrack(st, catalog)
            prov["history"] = f"Space-Track gp_history ({len(histories)} objects)"
        except Exception as e:
            log.warning("gp_history failed: %s", e)
    if not histories:
        histories = synthetic.tle_histories(catalog.objects, sw_by_date, now)
        prov["history"] = "synthetic 60-day TLE histories with injected maneuvers/breakups"

    # ---- covariance realism
    prog.step("fitting covariance realism on TLE residuals", 18)
    cov = CovarianceModel(density_model=dm)
    prop_before, prop_after = fit_realism(histories, cov, sw_today)
    prog.lap("covariance_realism")

    # ---- Pc calibration study
    prog.step("Pc calibration study (Brier score)", 24)
    k_scale = float(np.exp(np.mean(np.log(prop_before.get("realism_scale_ric", [1, 1, 1])))))
    k_scale = float(np.clip(k_scale, 0.3, 5))
    calib, iso = run_study(realism_scale=k_scale, n=10000, seed=config.RANDOM_SEED, red=config.PC_RED,
                           far_target=config.FAR_TARGET)
    prog.lap("pc_calibration")

    # ---- screening
    prog.step(f"all-vs-all screening: {len(catalog)} objects, {horizon_h:.0f} h", 30)
    events, sstats = screen(catalog, now, horizon_h, step_s, thr)
    prog.lap("screening")

    # ---- Pc for every event
    events = [e for e in events if e["rel_speed_kms"] > 1e-3]   # drop co-moving pairs
    prog.step(f"computing Pc for {len(events)} conjunctions", 50)
    out_events = []
    for k, e in enumerate(events):
        a, b = catalog.objects[e["i"]], catalog.objects[e["j"]]
        swd = _sw_for(sw_by_date, e["tca"])
        age_a = (e["tca"] - a.epoch).total_seconds() / 86400
        age_b = (e["tca"] - b.epoch).total_seconds() / 86400
        Ca, sa = cov.cov_eci(a, e["r1"], e["v1"], age_a, swd)
        Cb, sb = cov.cov_eci(b, e["r2"], e["v2"], age_b, swd)
        C = Ca + Cb
        hbr = max(a.est_radius_m() + b.est_radius_m(), 2.0) / 1000.0
        dr = e["r2"] - e["r1"]; dv = e["v2"] - e["v1"]
        pc, info = pc_foster(dr, dv, C, hbr)
        pmax, pscale = pc_max(dr, dv, C, hbr) if e["miss_km"] < 3 else (pc, 1.0)
        pc_cal = float(iso.predict([pc], [max(abs(age_a), abs(age_b))], [info["sigma_major_km"]])[0]) if pc > 0 else 0.0
        jd, fr = dt_to_jd(e["tca"])
        lat, lon, alt = ecef_to_geodetic(teme_to_ecef(np.asarray(e["r1"])[None, :], np.array([jd + fr]))[0])
        risk = "red" if pc >= config.PC_RED else "yellow" if pc >= config.PC_YELLOW else "green"
        m_km = e["miss_km"]
        miss_class = ("HIGH" if m_km <= config.MISS_HIGH_KM else "MEDIUM" if m_km <= config.MISS_MEDIUM_KM
                      else "LOW" if m_km <= config.MISS_LOW_KM else "NONE")
        regime = "GEO" if abs(alt - 35786) < 1500 else "LEO" if alt < 2000 else "MEO/HEO"
        # primary = the maneuverable / larger object
        swap = (b.obj_type == "PAYLOAD" and a.obj_type != "PAYLOAD")
        p, s_ = (b, a) if swap else (a, b)
        lc_p = lifecycle(p, now)
        e.update({"C": C, "pc_scale": float(pscale), "hbr_km": hbr, "swap": swap, "age_a": age_a, "age_b": age_b, "swd": swd})
        out_events.append({
            "id": k, "primary": p.norad, "primary_name": p.name, "primary_type": p.obj_type,
            "secondary": s_.norad, "secondary_name": s_.name, "secondary_type": s_.obj_type,
            "tca": e["tca"].isoformat(), "lead_time_h": (e["tca"] - now).total_seconds() / 3600,
            "miss_km": e["miss_km"], "miss_ric_km": e["miss_ric_km"], "rel_speed_kms": e["rel_speed_kms"],
            "pc": pc, "pc_max": pmax, "pc_calibrated": pc_cal, "risk": risk, "miss_class": miss_class, "regime": regime, "hbr_m": hbr * 1000,
            "sigma_primary_ric_km": (sb if swap else sa).tolist(),
            "sigma_secondary_ric_km": (sa if swap else sb).tolist(),
            "sigma_major_km": info["sigma_major_km"], "sigma_minor_km": info["sigma_minor_km"],
            "miss_2d_km": info["miss_2d_km"],
            "tca_lat": float(lat), "tca_lon": float(lon), "tca_alt_km": float(alt),
            "maneuverable": lc_p["maneuverable"],
            "primary_lifecycle": lc_p,
            "age_days": max(abs(age_a), abs(age_b)),
        })
    order = sorted(range(len(out_events)), key=lambda i: -out_events[i]["pc"])
    out_events = [out_events[i] for i in order]
    events = [events[i] for i in order]
    for k, (oe, e) in enumerate(zip(out_events, events)):
        oe["id"] = k
    prog.lap("pc")

    # ---- maneuver recommendations
    prog.step("maneuver recommendations for red alerts", 58)
    man = {}
    for oe, e in zip(out_events, events):
        if oe["risk"] != "red" or not oe["maneuverable"]:
            continue
        ev = dict(e)
        if e["swap"]:
            ev.update({"r1": e["r2"], "v1": e["v2"], "r2": e["r1"], "v2": e["v1"]})
        prim_obj = catalog.objects[e["j"]] if e["swap"] else catalog.objects[e["i"]]
        sec_obj = catalog.objects[e["i"]] if e["swap"] else catalog.objects[e["j"]]
        man[oe["id"]] = recommend(ev, e["C"], e["hbr_km"], now, target_pc=config.PC_RED / 10,
                                  fuel_ms=oe["primary_lifecycle"]["usable_dv_ms"],
                                  secondary_maneuverable=sec_obj.obj_type == "PAYLOAD",
                                  relax_pc=config.PC_RED / 2, age_days=oe["age_days"],
                                  sk_ms_per_year=oe["primary_lifecycle"].get("sk_dv_ms_per_year"))
        if len(man) >= 25:
            break
    # efficiency benchmark: with no maneuverable red alert, cost the highest-Pc maneuverable encounters
    bench_dvs = []
    if not man:
        for oe, e in zip(out_events, events):
            if not oe["maneuverable"] or oe["pc"] <= 0:
                continue
            ev = dict(e)
            if e["swap"]:
                ev.update({"r1": e["r2"], "v1": e["v2"], "r2": e["r1"], "v2": e["v1"]})
            sec_obj = catalog.objects[e["i"]] if e["swap"] else catalog.objects[e["j"]]
            # benchmark: cost of cutting each encounter's own Pc ten-fold (capped at the red-alert goal)
            rb = recommend(ev, e["C"], e["hbr_km"], now, target_pc=min(config.PC_RED / 10, oe["pc"] / 10),
                           fuel_ms=oe["primary_lifecycle"]["usable_dv_ms"],
                           secondary_maneuverable=sec_obj.obj_type == "PAYLOAD",
                           relax_pc=config.PC_RED / 2, age_days=oe["age_days"],
                           sk_ms_per_year=oe["primary_lifecycle"].get("sk_dv_ms_per_year"))
            if rb.get("best"):
                bench_dvs.append(rb["best"]["dv_ms"])
            if len(bench_dvs) >= 10:
                break
    prog.lap("maneuvers")

    # ---- what happens after the burn: new encounters on the way back, re-optimised against the fuel left
    prog.step("re-encounter outlook after each maneuver", 60)
    edges = apogee_perigee_pairs(catalog, thr)
    for oe, e in zip(out_events, events):
        m = man.get(oe["id"])
        if m and m.get("best"):
            try:
                m["followup"] = followup(m, e, catalog, cov, edges, sw_by_date, thr)
            except Exception as ex:
                log.warning("follow-up outlook failed for conjunction %s: %s", oe["id"], ex)
    prog.lap("followup")

    # ---- explainable alerts (XAI)
    prog.step("explaining alerts (driver attribution)", 61)
    typ = typical_values([{**oe, "hbr_km": oe["hbr_m"] / 1000} for oe in out_events])
    n_exp = 0
    for oe, e in zip(out_events, events):
        if n_exp >= 300 or (oe["risk"] == "green" and oe["pc"] < 1e-7):
            continue
        a, b = catalog.objects[e["i"]], catalog.objects[e["j"]]
        Ca_nd, _ = cov.cov_eci(a, e["r1"], e["v1"], e["age_a"], e["swd"], drag=False)
        Cb_nd, _ = cov.cov_eci(b, e["r2"], e["v2"], e["age_b"], e["swd"], drag=False)
        Ca_f, _ = cov.cov_eci(a, e["r1"], e["v1"], typ["age_days"], e["swd"])
        Cb_f, _ = cov.cov_eci(b, e["r2"], e["v2"], typ["age_days"], e["swd"])
        sa = cov.sigma_ric(a, e["age_a"], e["swd"]); sa0 = cov.sigma_ric(a, e["age_a"], e["swd"], drag=False)
        sb = cov.sigma_ric(b, e["age_b"], e["swd"]); sb0 = cov.sigma_ric(b, e["age_b"], e["swd"], drag=False)
        worst, ws, ws0, wage = (b, sb, sb0, e["age_b"]) if sb[1] >= sa[1] else (a, sa, sa0, e["age_a"])
        drag_pct = max(0.0, 1 - (ws0[1] / max(ws[1], 1e-9)) ** 2) * 100
        diag = (f"{worst.name} along-track σ {ws[1]:.2f} km after a {abs(wage):.1f}-day-old TLE; "
                f"{drag_pct:.0f}% of that variance comes from drag uncertainty "
                f"(AI density model ±{dm.relative_sigma(max(worst.perigee_km, 150), e['swd']) * 100:.0f}% at "
                f"{worst.perigee_km:.0f} km, F10.7 {e['swd']['f107']:.0f}).")
        ex = explain(e["r2"] - e["r1"], e["v2"] - e["v1"], e["C"], e["hbr_km"], typ,
                     cov_terms={"no_drag": Ca_nd + Cb_nd, "fresh": Ca_f + Cb_f,
                                "radial_dir": e["r1"] / np.linalg.norm(e["r1"]),
                                "drag_note": f"{drag_pct:.0f}% of {worst.name} along-track variance",
                                "age_note": f"TLE ages {abs(e['age_a']):.1f} / {abs(e['age_b']):.1f} d vs typical {typ['age_days']:.1f} d",
                                "diagnostic": diag},
                     red=config.PC_RED, yellow=config.PC_YELLOW, maneuver=man.get(oe["id"]))
        oe["explanation"] = {k: v for k, v in ex.items() if k != "pc"}
        n_exp += 1
    prog.lap("explain")

    # ---- forecast whether each encounter estimate will change at the next update
    prog.step("training update-change forecaster (Option C)", 62)
    seqs = []
    if st.enabled and mode != "offline":
        try:
            seqs = sequences_from_cdm_history(st.cdm_history())
        except Exception as ex:
            log.warning("CDM history unavailable: %s", ex)
    fc = None
    if len(seqs) >= 300:
        try:
            fc = ChangeForecaster(seed=config.RANDOM_SEED).fit_evaluate(seqs)
            prov["update_history"] = f"Space-Track cdm_public history ({len(seqs)} encounters)"
        except Exception as ex:
            log.warning("forecaster on real CDM history failed (%s); using simulated sequences", ex)
    if fc is None:
        seqs = simulate_sequences(n_enc=4000, start=now - timedelta(days=400), days=390,
                                  seed=config.RANDOM_SEED, sw_series=sw)
        prov["update_history"] = "simulated orbit-determination update sequences (physics-based)"
        fc = ChangeForecaster(seed=config.RANDOM_SEED).fit_evaluate(seqs)
    for oe, e in zip(out_events, events):
        a, b = catalog.objects[e["i"]], catalog.objects[e["j"]]
        sec = a if e["swap"] else b
        prim = b if e["swap"] else a
        swd = e["swd"]
        nxt = sw_by_date.get((e["tca"] - timedelta(days=max(oe["lead_time_h"] / 24 - 1, 0))).date().isoformat(), swd)
        oe["change_forecast"] = fc.score_current({
            "tau_days": max(oe["lead_time_h"], 0) / 24, "miss_km": oe["miss_km"], "rel_speed_kms": oe["rel_speed_kms"],
            "sigma_major_km": oe["sigma_major_km"], "sigma_minor_km": oe["sigma_minor_km"], "pc": oe["pc"],
            "f107": swd["f107"], "ap": swd["ap"], "ap_forecast_next": nxt["ap"], "am_secondary": sec.am_ratio,
            "alt_km": sec.perigee_km, "drag_sens": drag_sensitivity(sec.am_ratio, sec.perigee_km),
            "primary_maneuverable": int(prim.obj_type == "PAYLOAD"), "secondary_debris": int(sec.obj_type == "DEBRIS"),
            "rcs_secondary": sec.rcs_m2 or np.nan})
    prog.lap("change_forecast")

    # ---- GNN pre-screener
    prog.step("graph-learning screener: training on previous 24 h", 64)
    gnn = GNNScreener(seed=config.RANDOM_SEED)
    gnn_bench = {}
    try:
        tr_events, _ = screen(catalog, now - timedelta(hours=horizon_h), horizon_h, step_s, thr)
        gnn.fit(catalog, edges, {(e["i"], e["j"]) for e in tr_events})
        if gnn.clf is not None:
            keep, _ = gnn.evaluate(catalog, edges, {(e["i"], e["j"]) for e in events})
            # measured pairwise-screening speed-up on a 2-hour window
            bh = 2.0
            _, s_all = screen_pairs(catalog, edges, now, bh, step_s, thr, refine_events=False)
            _, s_ai = screen_pairs(catalog, edges[keep], now, bh, step_s, thr, refine_events=False)
            gnn_bench = {"all_pairs_s": s_all["search_s"], "ai_shortlist_s": s_ai["search_s"],
                         "speedup": s_all["search_s"] / max(s_ai["search_s"], 1e-6),
                         "all_vs_all_pairs": len(catalog) * (len(catalog) - 1) // 2,
                         "apogee_perigee_edges": int(len(edges)), "ai_shortlist_edges": int(keep.sum())}
    except Exception as ex:
        log.exception("GNN screener failed: %s", ex)
        gnn.metrics = {"status": f"failed: {ex}"}
    prog.lap("gnn")

    # ---- anomalies
    prog.step("anomaly detection (maneuvers / fragmentation)", 76)
    anomalies, an_metrics = detect(histories)
    clouds = detect_debris_clouds(catalog)
    prog.lap("anomaly")

    # ---- re-entry
    prog.step("re-entry predictions", 82)
    reentries = reentry_watchlist(catalog, now, dm, sw_today)
    prog.lap("reentry")

    # ---- life-cycle of the whole fleet: who can still maneuver, who is debris in all but name
    red_primaries = {oe["primary"] for oe in out_events if oe["risk"] == "red"}
    life_fleet = lifecycle_fleet(catalog.objects, now, {r["norad"]: r.get("days_p50") for r in reentries
                                                        if r.get("days_p50") is not None}, red_primaries)
    prog.lap("lifecycle")

    # ---- validation: Iridium-Cosmos
    prog.step("historical validation: Iridium 33 / Cosmos 2251", 88)
    val = validate_iridium_cosmos(cov, dm, sw_by_date)
    prog.lap("validation")

    # ---- CDM comparison
    if not cdms and injected:
        cdms = [{"cdm_id": f"SYN-{k}", "tca": e["tca"], "min_rng_km": e["true_miss_km"], "pc": None,
                 "sat1": e["primary"], "sat2": e["secondary"]} for k, e in enumerate(injected)]
        prov["cdm"] = "synthetic reference CDMs (injected ground-truth encounters)"
    cdm_kpi = compare_with_cdms(events, catalog, cdms, now, horizon_h) if cdms else \
        {"reference_cdms": 0, "note": "add Space-Track credentials to compare with real CDMs"}

    # ---- incremental update latency
    prog.step("incremental update latency test", 93)
    rng = np.random.default_rng(1)
    sub = rng.choice(len(catalog), min(20, len(catalog)), replace=False)
    _, latency = screen_subset(catalog, sub, now, horizon_h, step_s, thr)
    prog.lap("incremental")

    # ---- KPIs
    reds = [e for e in out_events if e["risk"] == "red"]
    yellows = [e for e in out_events if e["risk"] == "yellow"]
    savings = [m["dv_saving_vs_naive"] for m in man.values() if m.get("dv_saving_vs_naive") is not None]
    dvs = [m["best"]["dv_ms"] for m in man.values() if m.get("best")]
    kpis = {
        "pc_brier_raw": calib["raw"]["brier"],
        "pc_brier_calibrated": calib["ai_calibrated"]["brier"],
        "pc_brier_skill_calibrated": calib["ai_calibrated"]["brier_skill_vs_climatology"],
        "false_alert_rate": calib["ai_tuned"]["false_alert_rate"],
        "false_alert_rate_policy": calib["ai_calibrated"]["false_alert_rate"],
        "false_alert_rate_raw": calib["raw"]["false_alert_rate"],
        "false_alert_rate_at_5pct_missed": calib["ai_at_5pct_missed"]["false_alert_rate"],
        "accuracy_of_never_alerting": calib["accuracy_of_never_alerting"],
        "alert_precision": calib["ai_tuned"]["alerts_that_were_collisions"],
        "missed_detection_rate": calib["ai_tuned"]["missed_detection_rate"],
        "missed_detection_rate_policy": calib["ai_calibrated"]["missed_detection_rate"],
        "alert_threshold_tuned": calib["ai_tuned"]["alert_threshold"],
        "cdm_hit_rate": cdm_kpi.get("hit_rate"),
        "median_alert_lead_time_h": float(np.median([e["lead_time_h"] for e in reds + yellows])) if reds + yellows else None,
        # no alerts: lead time at which the screening predicted the real (reference) CDM encounters
        "median_cdm_lead_time_h": cdm_kpi.get("median_lead_time_h"),
        "prop_error_median_km": dict(zip(map(str, prop_before["horizons_days"]), prop_before["median_km"])),
        "covariance_nees_before": prop_before.get("median_nees"),
        "covariance_chi2_containment_before": prop_before.get("chi2_95_containment"),
        "covariance_chi2_containment_after": prop_after.get("chi2_95_containment"),
        "covariance_nees_after": prop_after.get("median_nees"),
        "screening_throughput_pairs_per_s": sstats["throughput_pairs_per_s"],
        "screening_time_s": sstats["search_s"] + sstats["refine_s"],
        "maneuver_mean_dv_ms": float(np.mean(dvs)) if dvs else None,
        "maneuver_bench_dv_ms": float(np.mean(bench_dvs)) if bench_dvs else None,
        "maneuver_bench_n": len(bench_dvs),
        "maneuver_dv_saving_vs_naive": float(np.mean(savings)) if savings else None,
        "catalog_objects": len(catalog),
        "update_latency_s": latency,
        "density_error_reduction_pct": dm.metrics.get("error_reduction_pct"),
        "gnn_recall": gnn.metrics.get("test_recall"),
        "gnn_pruned_fraction": gnn.metrics.get("test_pruned_fraction"),
        "anomaly_f1": an_metrics.get("f1"),
        "change_forecast_auc": fc.metrics["model"]["auc"],
        "change_forecast_auc_physics": fc.metrics["baseline_physics"]["auc"],
        "change_forecast_brier": fc.metrics["model"]["brier"],
    }
    prog.step("saving results", 97)
    state = {
        "generated": now.isoformat(), "provenance": prov, "config": {
            "horizon_h": horizon_h, "step_s": step_s, "threshold_km": thr, "pc_red": config.PC_RED, "far_target": config.FAR_TARGET, "miss_high_km": config.MISS_HIGH_KM,
            "miss_medium_km": config.MISS_MEDIUM_KM, "miss_low_km": config.MISS_LOW_KM,
            "pc_yellow": config.PC_YELLOW},
        "space_weather": {"today": sw_today, "noaa_now": sw_now,
                          "series": [d for d in sw if d["date"] >= (now - timedelta(days=120)).date().isoformat()]},
        "summary": {"objects": len(catalog), "conjunctions": len(out_events), "red": len(reds),
                    "yellow": len(yellows), "reentries_30d": sum(1 for r in reentries if r.get("days_p50", 1e9) < 30),
                    "anomalies": an_metrics.get("detections", 0), "debris_clouds": len(clouds),
                    "spent_satellites": life_fleet["spent_in_orbit"]},
        "screening": sstats, "conjunctions": out_events, "maneuvers": {str(k): v for k, v in man.items()},
        "density_model": {"metrics": dm.metrics, "profile": dm.profile(sw_today)},
        "propagation": {"before": prop_before, "after": prop_after, "realism": cov.realism},
        "change_forecast": fc.metrics,
        "calibration": calib, "lifecycle": life_fleet, "gnn": {"metrics": gnn.metrics, "benchmark": gnn_bench},
        "anomalies": {"events": anomalies[:60], "metrics": an_metrics, "clouds": clouds},
        "reentry": reentries, "validation": val, "cdm_comparison": cdm_kpi, "kpis": kpis,
        "timings_s": prog.timings,
    }
    with open(config.OUTPUT_DIR / "state.json", "w") as f:
        json.dump(_clean(state), f, indent=1, default=str)
    with open(config.OUTPUT_DIR / "state.pkl", "wb") as f:
        pickle.dump({"state": state, "tles": [(o.norad, o.name, o.line1, o.line2, o.obj_type, o.rcs_m2,
                                                o.mass_kg, o.area_m2, o.country, o.source) for o in catalog.objects],
                     "events_raw": events, "cov_realism": cov.realism}, f)
    prog.step("done", 100)
    return state


def followup(man, e, catalog, cov, edges, sw_by_date, thr, days=7.0):
    """Encounters the maneuvered satellite faces after it is back on its slot.

    The burn and the return to the slot are over at ``start``. From there the satellite is screened
    for ``days`` more days; every new red encounter is re-optimised with only the propellant that
    is left, so the chain shows when the usable fuel (above the disposal reserve) runs out.
    """
    pi = e["j"] if e["swap"] else e["i"]
    best, fuel = man["best"], man["fuel"]
    start = e["tca"] + timedelta(days=fuel["recovery_days"])
    mine = edges[(edges[:, 0] == pi) | (edges[:, 1] == pi)]
    evs, _ = screen_pairs(catalog, mine, start, days * 24.0, 30.0, thr)
    left = fuel["remaining_after_ms"]
    first_secondary = catalog.objects[e["i"] if e["swap"] else e["j"]].norad
    rows, chain = [], []
    for ev in sorted(evs, key=lambda x: x["tca"]):
        if ev["rel_speed_kms"] < 1e-3:
            continue
        a, b = catalog.objects[ev["i"]], catalog.objects[ev["j"]]
        swd = _sw_for(sw_by_date, ev["tca"])
        age_a = (ev["tca"] - a.epoch).total_seconds() / 86400
        age_b = (ev["tca"] - b.epoch).total_seconds() / 86400
        Ca, _ = cov.cov_eci(a, ev["r1"], ev["v1"], age_a, swd)
        Cb, _ = cov.cov_eci(b, ev["r2"], ev["v2"], age_b, swd)
        C = Ca + Cb
        hbr = max(a.est_radius_m() + b.est_radius_m(), 2.0) / 1000.0
        pc, _ = pc_foster(ev["r2"] - ev["r1"], ev["v2"] - ev["v1"], C, hbr)
        risk = "red" if pc >= config.PC_RED else "yellow" if pc >= config.PC_YELLOW else "green"
        other = b if ev["i"] == pi else a
        row = {"tca": ev["tca"].isoformat(), "days_after_return": (ev["tca"] - start).total_seconds() / 86400,
               "secondary": other.norad, "secondary_name": other.name, "secondary_type": other.obj_type,
               "same_object": other.norad == first_secondary,
               "miss_km": ev["miss_km"], "pc": pc, "risk": risk, "action": None}
        if risk == "red":
            o = dict(ev)
            if ev["i"] != pi:
                o.update({"r1": ev["r2"], "v1": ev["v2"], "r2": ev["r1"], "v2": ev["v1"]})
            m2 = recommend(o, C, hbr, start, target_pc=config.PC_RED / 10, fuel_ms=left,
                           secondary_maneuverable=other.obj_type == "PAYLOAD", relax_pc=config.PC_RED / 2,
                           age_days=max(abs(age_a), abs(age_b)))
            b2 = m2.get("best")
            if b2 and b2["total_dv_ms"] <= left:
                row["action"] = {"status": m2["status"], "total_dv_ms": b2["total_dv_ms"], "fuel_before_ms": left,
                                 "fuel_after_ms": left - b2["total_dv_ms"], "direction": b2["direction"],
                                 "pc_after": b2["pc_after"]}
                left -= b2["total_dv_ms"]
            else:
                row["action"] = {"status": "OUT_OF_FUEL", "total_dv_ms": b2["total_dv_ms"] if b2 else None,
                                 "fuel_before_ms": left, "fuel_after_ms": left}
            chain.append(row)
        rows.append(row)
    # a co-orbiting neighbour re-approaches every orbit: show each object once, at its worst pass
    worst = {}
    for r in rows:
        k = r["secondary"]
        if k not in worst or r["pc"] > worst[k]["pc"]:
            worst[k] = r
    rows = sorted(worst.values(), key=lambda r: -r["pc"])
    return {"window_start": start.isoformat(), "window_days": days, "encounters": len(rows),
            "red": sum(r["risk"] == "red" for r in rows), "yellow": sum(r["risk"] == "yellow" for r in rows),
            "same_object_returns": sum(r["same_object"] for r in rows),
            "fuel_after_first_ms": fuel["remaining_after_ms"], "fuel_after_all_ms": left,
            "top": rows[:6], "chain": sorted(chain, key=lambda r: r["tca"])}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, datetime):
        return x.isoformat()
    return x


def validate_iridium_cosmos(cov, dm, sw_by_date):
    sc = synthetic.iridium_cosmos_scenario()
    tca = sc["tca"]
    cat = Catalog(sc["objects"])
    evs, _ = screen(cat, tca - timedelta(hours=12), 24, 10, 10)
    if not evs:
        return {"status": "not detected"}
    e = evs[0]
    a, b = cat.objects[e["i"]], cat.objects[e["j"]]
    hist_sw = {"date": "2009-02-10", "f107": 70.0, "f107a": 70.5, "ap": 5.0}   # deep solar minimum
    Ca, _ = cov.cov_eci(a, e["r1"], e["v1"], (e["tca"] - a.epoch).total_seconds() / 86400, hist_sw)
    Cb, _ = cov.cov_eci(b, e["r2"], e["v2"], (e["tca"] - b.epoch).total_seconds() / 86400, hist_sw)
    hbr = (a.est_radius_m() + b.est_radius_m()) / 1000
    dr, dv = e["r2"] - e["r1"], e["v2"] - e["v1"]
    pc, _ = pc_foster(dr, dv, Ca + Cb, hbr)
    pmax, k = pc_max(dr, dv, Ca + Cb, hbr)
    jd, fr = dt_to_jd(e["tca"])
    lat, lon, alt = ecef_to_geodetic(teme_to_ecef(np.asarray(e["r1"])[None, :], np.array([jd + fr]))[0])
    return {"event": "Iridium 33 / Cosmos 2251 collision, 2009-02-10 16:56 UTC",
            "actual_outcome": "COLLISION (both destroyed, >2,000 tracked fragments)",
            "actual_location": sc["location"],
            "predicted_tca": e["tca"].isoformat(), "tca_error_s": (e["tca"] - tca).total_seconds(),
            "predicted_miss_km": e["miss_km"], "rel_speed_kms": e["rel_speed_kms"],
            "predicted_location": {"lat": float(lat), "lon": float(lon), "alt_km": float(alt)},
            "pc": pc, "pc_max": pmax, "would_alert_red": pc >= config.PC_RED or pmax >= config.PC_RED,
            "historical_note": "CelesTrak SOCRATES ranked this pair with a predicted miss of ~584 m; it was "
                               "not in the top-10 list. TLE-only screening without realistic covariance "
                               "under-ranked it.",
            "data": "reconstructed geometry (public collision facts) with TLE-level orbit errors"}
