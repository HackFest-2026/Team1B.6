"""Fast self-tests for the core algorithms (run: python run.py test  or  pytest -q)."""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdrp.astro.pc import pc_foster, pc_montecarlo                       # noqa: E402
from sdrp.astro.propagate import Catalog, SpaceObject, dt_to_jd          # noqa: E402
from sdrp.astro.screening import screen, screen_pairs, apogee_perigee_pairs  # noqa: E402
from sdrp.data import sources, synthetic                                 # noqa: E402
from sdrp.ops.maneuver import cw_displacement                            # noqa: E402

ISS = ("ISS (ZARYA)",
       "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
       "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482")
NOW = datetime(2026, 10, 9, 6, 0, tzinfo=timezone.utc)


def test_tle_parsing_3le_and_2le():
    objs = sources.parse_tle_text("\n".join(ISS) + "\n" + "\n".join(ISS[1:]))
    assert len(objs) == 2 and objs[0].norad == 25544 and objs[0].name.startswith("ISS")
    assert 400 < objs[0].perigee_km < 430 and abs(objs[0].inc_deg - 51.64) < 0.01


def test_pc_matches_monte_carlo():
    dr = np.array([0.1, 0.05, 0.0]); dv = np.array([0, 0, 10.0]); C = np.diag([0.04, 0.01, 0.25])
    pc, _ = pc_foster(dr, dv, C, 0.02)
    mc, ci = pc_montecarlo(dr, dv, C, 0.02, n=600_000)
    assert abs(pc - mc) < 3 * ci + 1e-4


def test_tle_fit_reaches_target():
    objs, _ = synthetic.build_catalog(NOW, scale=0.05, n_injected=0)
    o = objs[0]
    jd, fr = dt_to_jd(NOW + timedelta(hours=5))
    _, r, v = o.sat.sgp4(jd, fr)
    s = synthetic.fit_tle_to_state(99001, NOW - timedelta(hours=10), NOW + timedelta(hours=5),
                                   np.array(r), np.array(v), bstar=1e-4)
    _, r2, _ = s.sgp4(jd, fr)
    assert np.linalg.norm(np.subtract(r, r2)) < 0.01


def test_screening_finds_injected_encounters():
    objs, inj = synthetic.build_catalog(NOW, scale=0.15, n_injected=5)
    cat = Catalog(objs)
    events, st = screen(cat, NOW, 24, 30, 5)
    found = {(cat.objects[e["i"]].norad, cat.objects[e["j"]].norad) for e in events}
    found |= {(b, a) for a, b in found}
    hits = sum((e["primary"], e["secondary"]) in found for e in inj)
    assert hits == len(inj)
    # pairwise screening over the apogee/perigee graph must agree
    edges = apogee_perigee_pairs(cat, 5)
    ev2, _ = screen_pairs(cat, edges, NOW, 24, 30, 5)
    assert {(e["i"], e["j"]) for e in events} <= {(e["i"], e["j"]) for e in ev2} | {(e["j"], e["i"]) for e in ev2}


def test_cw_along_track_drift():
    n = 0.00113
    d = cw_displacement(n, 2 * np.pi / n, np.array([0, 1e-4, 0]))   # 0.1 m/s along-track, one rev
    assert abs(d[1] - (-3 * 2 * np.pi / n * 1e-4)) < 1e-3          # secular drift -3*dv*t


def test_iridium_cosmos_reconstruction():
    sc = synthetic.iridium_cosmos_scenario()
    jd, fr = dt_to_jd(sc["tca"])
    r = [np.array(s.sgp4(jd, fr)[1]) for s in sc["truth"]]
    assert np.linalg.norm(r[0] - r[1]) < 0.01                       # truth orbits collide


def test_space_weather_csv_parser(tmp_path, monkeypatch):
    csv_txt = ("DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
               "F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n"
               "2026-10-01,2600,1,20,20,23,27,17,13,10,7,137,7,7,9,12,6,5,4,3,7,0.4,2,120,165.2,163.0,OBS,158.1,152.0,156.0,150.0\n")
    monkeypatch.setattr(sources, "_cached_get", lambda *a, **k: csv_txt)
    sw = sources.fetch_space_weather()
    assert sw[0]["f107"] == 165.2 and sw[0]["f107a"] == 158.1 and sw[0]["ap"] == 7


def test_live_code_path_with_mocked_sources(monkeypatch):
    """Exercise the 'live' ingest branch offline by serving synthetic TLEs as CelesTrak text."""
    from sdrp import pipeline
    objs, _ = synthetic.build_catalog(NOW, scale=0.2, n_injected=0)
    txt = "\n".join(f"{o.name}\n{o.line1}\n{o.line2}" for o in objs)
    monkeypatch.setattr(sources, "fetch_celestrak", lambda groups=None: sources.parse_tle_text(txt))
    monkeypatch.setattr(sources, "fetch_satcat", lambda: {o.norad: {"type": o.obj_type, "rcs": o.rcs_m2, "country": "XX"} for o in objs})
    monkeypatch.setattr(sources, "fetch_space_weather", lambda: [])
    monkeypatch.setattr(sources, "fetch_noaa_now", lambda: {"f107": 150.0, "kp": 3.0, "source": "NOAA SWPC"})
    monkeypatch.setattr(pipeline.config, "MAX_OBJECTS", 300)

    class P:
        def step(self, *a): pass
    got, inj, sw, sw_now, cdms, hist, prov, st = pipeline.load_data(NOW, "auto", P())
    assert prov["catalog"].startswith("CelesTrak") and len(got) == 300 and not inj
    assert sw_now["kp"] == 3.0 and len(sw) > 400


def test_explanation_ranks_miss_distance_for_close_pass():
    from sdrp.ai.explain import explain
    dr = np.array([0.05, 0.02, 0.0]); dv = np.array([0, 0, 10.0]); C = np.diag([0.04, 0.09, 0.25])
    typ = {"miss_km": 2.0, "hbr_km": 0.01, "sigma_major_km": 0.3, "age_days": 1.0}
    ex = explain(dr, dv, C, 0.01, typ)
    assert ex["drivers"][0]["factor"] == "Miss distance" and ex["drivers"][0]["delta_log10_pc"] > 0
    assert ex["readout"][0].startswith("[ALERT]") and ex["clear_miss_km"] > 0.05


def test_option_c_forecaster_beats_climatology_time_ordered():
    from sdrp.ai.forecast import ChangeForecaster, simulate_sequences
    seqs = simulate_sequences(n_enc=900, seed=3)
    fc = ChangeForecaster().fit_evaluate(seqs)
    m = fc.metrics
    assert m["model"]["auc"] > 0.75 and m["model"]["brier"] < m["baseline_climatology"]["brier"]
    r = fc.score_current({"tau_days": 3.0, "miss_km": 0.3, "sigma_major_km": 2.0, "sigma_minor_km": 0.1, "pc": 1e-5,
                          "rel_speed_kms": 10.0})
    assert 0 <= r["p_change_next_update"] <= 1 and r["level"] in ("LOW", "MEDIUM", "HIGH")


def _payload(line1):
    return SpaceObject(norad=25544, name="TEST-SAT", line1=line1, line2=ISS[2], obj_type="PAYLOAD")


def test_lifecycle_spent_satellite_cannot_maneuver_and_young_one_can():
    from sdrp.ops.lifecycle import lifecycle
    old = lifecycle(_payload(ISS[1]), NOW)                                    # launched 1998, design life 7 y
    assert old["stage_key"] == "end of life" and not old["maneuverable"] and old["usable_dv_ms"] == 0.0
    young_l1 = synthetic._with_launch_year(ISS[1], (2025, 2025), 25544)
    young = lifecycle(_payload(young_l1), NOW)
    assert young["maneuverable"] and young["usable_dv_ms"] > 0 and young["years_to_eol"] > 0
    rb = SpaceObject(norad=1, name="R/B", line1=ISS[1], line2=ISS[2], obj_type="ROCKET BODY")
    assert not lifecycle(rb, NOW)["maneuverable"]


def test_lifecycle_reserve_is_never_spendable():
    from sdrp.ops.lifecycle import lifecycle, REGIME
    for yr in range(2018, 2027):
        lc = lifecycle(_payload(synthetic._with_launch_year(ISS[1], (yr, yr), 25544)), NOW)
        assert lc["remaining_dv_ms"] - lc["usable_dv_ms"] >= min(lc["reserve_dv_ms"], lc["remaining_dv_ms"]) - 1e-6
        assert lc["usable_dv_ms"] <= REGIME[lc["regime"]][1]


def test_synthetic_launch_year_keeps_a_valid_tle():
    l1 = synthetic._with_launch_year(ISS[1], (2020, 2024), 12345)
    assert len(l1) == 69 and int(l1[-1]) == synthetic._tle_checksum(l1)
    assert 2020 <= 2000 + int(l1[9:11]) <= 2024


def test_maneuver_respects_propellant_and_reports_cost():
    from sdrp.ops.maneuver import recommend
    from sdrp.astro.pc import pc_foster as pcf
    objs, _ = synthetic.build_catalog(NOW, scale=0.15, n_injected=3)
    cat = Catalog(objs)
    events, _ = screen(cat, NOW, 24, 30, 5)
    ev = min(events, key=lambda e: e["miss_km"])
    C = np.diag([0.03, 0.05, 0.03])
    pc0 = pcf(ev["r2"] - ev["r1"], ev["v2"] - ev["v1"], C, 0.02)[0]
    assert pc0 > 0
    rich = recommend(ev, C, 0.02, NOW, target_pc=pc0 / 10, fuel_ms=100.0, sk_ms_per_year=12.0)
    assert rich["status"] == "RETURNS_TO_SLOT" and rich["fuel"]["verdict"] in ("WORTH_IT", "COSTLY", "VERY_COSTLY")
    assert rich["fuel"]["maneuvers_affordable"] >= 1 and rich["fuel"]["station_keeping_days"] > 0
    need = rich["best"]["total_dv_ms"]
    poor = recommend(ev, C, 0.02, NOW, target_pc=pc0 / 10, fuel_ms=need * 0.5)
    assert poor["status"] != "RETURNS_TO_SLOT"
    empty = recommend(ev, C, 0.02, NOW, target_pc=pc0 / 10, fuel_ms=0.0)
    assert empty["best"] is None and empty["status"] == "NOT_FEASIBLE"


def test_alert_threshold_tuning_and_tracking_sweep():
    from sdrp.ai.calibration import tune_threshold, run_study
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=4000) < 0.05).astype(float)
    p = np.clip(0.2 * y + rng.uniform(0, 0.6, 4000), 0, 1)
    thr = tune_threshold(p, y, 0.10)
    assert ((p >= thr) & (y == 0)).sum() / (y == 0).sum() <= 0.10
    res, _ = run_study(1.0, n=1500, seed=1, sweep=False)
    assert 0.9 < res["accuracy_of_never_alerting"] < 1.0                      # why plain accuracy misleads
    good, _ = run_study(1.0, n=1500, seed=1, sigma_scale=0.1, sweep=False)
    assert good["ai_at_5pct_missed"]["false_alert_rate"] < res["ai_at_5pct_missed"]["false_alert_rate"]
