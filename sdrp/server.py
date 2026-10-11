"""FastAPI backend + static web dashboard (no Streamlit)."""
from __future__ import annotations

import json
import logging
import math
import pickle
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .astro.propagate import (Catalog, SpaceObject, dt_to_jd, ecef_to_geodetic, teme_to_ecef,
                              time_grid)
from .astro.screening import screen_subset

log = logging.getLogger("sdrp.server")
WEB = config.ROOT / "web"

app = FastAPI(title="Space Debris Risk Prediction (AI)", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.catalog = None
        self.events_raw = []
        self.status = {"running": False, "step": "idle", "pct": 0, "error": None, "started": None,
                       "finished": None}

    def load(self):
        p = config.OUTPUT_DIR / "state.pkl"
        if not p.exists():
            return False
        with open(p, "rb") as f:
            d = pickle.load(f)
        objs = [SpaceObject(norad=t[0], name=t[1], line1=t[2], line2=t[3], obj_type=t[4], rcs_m2=t[5],
                            mass_kg=t[6], area_m2=t[7], country=t[8], source=t[9]) for t in d["tles"]]
        with open(config.OUTPUT_DIR / "state.json") as f:
            st = json.load(f)
        with self.lock:
            self.state = st
            self.catalog = Catalog(objs)
            self.events_raw = d["events_raw"]
        return True


S = Store()


def _run_pipeline(kwargs):
    from .pipeline import run
    S.status.update(running=True, error=None, started=datetime.now(timezone.utc).isoformat(), pct=0)

    def cb(step, pct):
        S.status.update(step=step, pct=pct)
    try:
        run(progress_cb=cb, **{k: v for k, v in kwargs.items() if v is not None})
        S.load()
        S.status.update(finished=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        log.exception("pipeline failed")
        S.status.update(error=str(e))
    finally:
        S.status.update(running=False)


def start_run(**kwargs):
    if S.status["running"]:
        return False
    threading.Thread(target=_run_pipeline, args=(kwargs,), daemon=True).start()
    return True


def _need_state():
    if S.state is None:
        raise HTTPException(503, "No results yet - the pipeline is running. Watch /api/status.")


# ----------------------------------------------------------------------------- pages
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


# ----------------------------------------------------------------------------- API
@app.get("/api/status")
def status():
    return {**S.status, "has_results": S.state is not None,
            "generated": S.state["generated"] if S.state else None}


@app.post("/api/run")
def run_now(fast: bool = True, mode: str | None = None):
    ok = start_run(fast=fast, mode=mode)
    return {"started": ok, "status": S.status}


@app.get("/api/summary")
def summary():
    _need_state()
    st = S.state
    keys = ["generated", "provenance", "config", "summary", "kpis", "screening", "timings_s",
            "cdm_comparison", "validation"]
    return {k: st.get(k) for k in keys}


@app.get("/api/state/{section}")
def section(section: str):
    _need_state()
    if section not in S.state:
        raise HTTPException(404, f"unknown section {section}")
    return JSONResponse(S.state[section])


@app.get("/api/tles")
def tles():
    _need_state()
    return [{"norad": o.norad, "name": o.name, "type": o.obj_type, "l1": o.line1, "l2": o.line2,
             "regime": o.regime} for o in S.catalog.objects]


@app.get("/api/catalog")
def catalog(q: str = "", type: str = "", regime: str = "", limit: int = 300):
    _need_state()
    out = []
    ql = q.lower()
    for o in S.catalog.objects:
        if ql and ql not in o.name.lower() and ql != str(o.norad):
            continue
        if type and o.obj_type != type:
            continue
        if regime and o.regime != regime:
            continue
        out.append(o.summary())
        if len(out) >= limit:
            break
    return {"total": len(S.catalog), "items": out}


@app.get("/api/object/{norad}")
def obj(norad: int):
    _need_state()
    o = S.catalog.get(norad)
    if o is None:
        raise HTTPException(404, "object not in catalog")
    now = datetime.now(timezone.utc)
    jd, fr, secs = time_grid(now, o.period_min / 60.0 * 1.02, max(o.period_min * 60 / 240, 10))
    r, v, e = S.catalog.propagate(jd, fr, idx=[S.catalog.index[o.norad]])
    lat, lon, alt = ecef_to_geodetic(teme_to_ecef(r[0], jd + fr))
    conj = [c for c in S.state["conjunctions"] if norad in (c["primary"], c["secondary"])][:50]
    return {"summary": o.summary(), "line1": o.line1, "line2": o.line2,
            "ground_track": {"lat": np.nan_to_num(lat).tolist(), "lon": np.nan_to_num(lon).tolist(),
                             "alt": np.nan_to_num(alt).tolist()},
            "conjunctions": conj,
            "reentry": next((x for x in S.state["reentry"] if x["norad"] == norad), None),
            "anomalies": [a for a in S.state["anomalies"]["events"] if a["norad"] == norad]}


@app.get("/api/conjunctions")
def conjunctions(risk: str = "", min_pc: float = 0.0, q: str = "", limit: int = 500):
    _need_state()
    ql = q.lower()
    out = [c for c in S.state["conjunctions"]
           if (not risk or c["risk"] == risk) and (c["pc"] or 0.0) >= min_pc and
           (not ql or ql in c["primary_name"].lower() or ql in c["secondary_name"].lower()
            or ql in (str(c["primary"]), str(c["secondary"])))]
    return {"total": len(S.state["conjunctions"]), "items": out[:limit]}


@app.get("/api/conjunction/{cid}")
def conjunction(cid: int, target_pc: float = Query(None)):
    _need_state()
    if cid < 0 or cid >= len(S.state["conjunctions"]):
        raise HTTPException(404, "unknown conjunction")
    c = S.state["conjunctions"][cid]
    e = S.events_raw[cid]
    C, hbr = np.asarray(e["C"]), e["hbr_km"]
    dr, dv = e["r2"] - e["r1"], e["v2"] - e["v1"]
    # encounter-plane picture
    from .astro.pc import encounter_plane
    B = encounter_plane(dr, dv)
    P = B @ C @ B.T
    w, V = np.linalg.eigh(P)
    plane = {"miss": (B @ dr).tolist(), "sig": np.sqrt(np.maximum(w, 0)).tolist(),
             "angle_deg": float(math.degrees(math.atan2(V[1, 1], V[0, 1]))), "hbr_km": hbr}
    # relative range timeline around TCA
    a, b = S.catalog.objects[e["i"]], S.catalog.objects[e["j"]]
    ts = np.linspace(-300, 300, 121)
    rng_km = []
    for t in ts:
        jd, fr = dt_to_jd(e["tca"] + timedelta(seconds=float(t)))
        _, r1, _ = a.sat.sgp4(jd, fr); _, r2, _ = b.sat.sgp4(jd, fr)
        rng_km.append(float(np.linalg.norm(np.subtract(r2, r1))))
    man = S.state["maneuvers"].get(str(cid))
    if (man is None or target_pc is not None) and c["maneuverable"] and (c["pc"] or 0) > 1e-7:
        from .ops.maneuver import recommend
        ev = dict(e)
        if e.get("swap"):
            ev.update({"r1": e["r2"], "v1": e["v2"], "r2": e["r1"], "v2": e["v1"]})
        man = recommend(ev, C, hbr, datetime.now(timezone.utc),
                        target_pc=target_pc or min(config.PC_RED / 10, (c["pc"] or 1e-6) / 10),
                        fuel_ms=(c.get("primary_lifecycle") or {}).get("usable_dv_ms"),
                        secondary_maneuverable=c["secondary_type"] == "PAYLOAD",
                        relax_pc=config.PC_RED / 2, age_days=c.get("age_days"),
                        sk_ms_per_year=(c.get("primary_lifecycle") or {}).get("sk_dv_ms_per_year"))
    return {"conjunction": c, "encounter_plane": plane, "range_timeline": {"t_s": ts.tolist(), "range_km": rng_km},
            "maneuver": man,
            "objects": {"primary": S.catalog.get(c["primary"]).summary(),
                        "secondary": S.catalog.get(c["secondary"]).summary()}}


@app.get("/api/alerts")
def alerts():
    _need_state()
    items = [c for c in S.state["conjunctions"] if c["risk"] in ("red", "yellow")]
    re_ = [r for r in S.state["reentry"] if r.get("days_p50", 1e9) < 3]
    return {"generated": S.state["generated"], "conjunctions": items[:50], "reentry_72h": re_}


@app.post("/api/screen_object/{norad}")
def screen_object(norad: int, horizon_h: float = 24.0):
    """Incremental re-screen of one object against the whole catalog (e.g. after a new TLE)."""
    _need_state()
    i = S.catalog.index.get(norad)
    if i is None:
        raise HTTPException(404, "object not in catalog")
    now = datetime.now(timezone.utc)
    evs, latency = screen_subset(S.catalog, [i], now, horizon_h, config.SCREEN_STEP_S,
                                 config.SCREEN_THRESHOLD_KM)
    out = []
    for e in sorted(evs, key=lambda e: e["miss_km"])[:30]:
        other = S.catalog.objects[e["j"] if e["i"] == i else e["i"]]
        out.append({"other": other.norad, "other_name": other.name, "tca": e["tca"].isoformat(),
                    "miss_km": e["miss_km"], "rel_speed_kms": e["rel_speed_kms"]})
    return {"norad": norad, "latency_s": latency, "events": out}


def _scheduler(hours, run_kwargs):
    while True:
        time.sleep(hours * 3600)
        log.info("scheduled pipeline run")
        start_run(**(run_kwargs or {}))


def start(host="127.0.0.1", port=8000, initial_run=True, run_kwargs=None, refresh_hours=0):
    import uvicorn
    have = S.load()
    stale = False
    if have:
        age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(S.state["generated"])).total_seconds() / 3600
        stale = age_h > 6
    # saved results were produced in another data mode (e.g. live data shown after switching to offline)
    want_mode = (run_kwargs or {}).get("mode") or config.DATA_MODE
    saved_mode = (S.state.get("provenance") or {}).get("mode") if have else None
    mode_changed = have and saved_mode != want_mode
    if initial_run and (not have or stale or mode_changed):
        log.info("%s - starting the pipeline in the background",
                 f"Saved results used mode '{saved_mode}' but mode '{want_mode}' is requested" if mode_changed
                 else "Results are older than 6 h" if stale else "No previous results")
        start_run(**(run_kwargs or {}))
    if refresh_hours and refresh_hours > 0:
        threading.Thread(target=_scheduler, args=(refresh_hours, run_kwargs), daemon=True).start()
    print(f"\n  Dashboard:  http://{host}:{port}\n  API docs:   http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
