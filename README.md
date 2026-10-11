# ORBITAL RISK — Space Debris Risk Prediction with AI (Problem 1C)

A working end-to-end prototype that predicts **collision risk** and **re-entry risk** from space debris
using orbital catalogs, space-weather drivers and machine learning, served through a custom web
dashboard (FastAPI + HTML/JS — **no Streamlit**).

```
TLE catalog ──► SGP4/SDP4 ──► all-vs-all screening ──► Pc (Foster 2-D) ──► alerts ──► min-Δv maneuver
     │              ▲              ▲ GNN pre-screener      ▲ AI covariance          ▲
space weather ──► ML density / drag-uncertainty model ─────┘ + learned Pc calibrator │
TLE history ──► anomaly detection (maneuvers, breakups) · re-entry Monte-Carlo · KPIs ┘
```

---

## 1. Quick start (2 minutes)

**Requirements:** Python 3.10+ (3.11/3.12 recommended), ~1 GB RAM, internet optional.

### Option A — one command
| OS | Command |
|---|---|
| macOS / Linux | `./run.sh` |
| Windows | double-click `run.bat` (or run it in a terminal) |

This creates a virtual environment, installs the dependencies and starts the dashboard.

### Option B — manual
```bash
cd space-debris-ai
python -m venv .venv
# macOS/Linux:  source .venv/bin/activate      Windows:  .venv\Scripts\activate
pip install -r requirements.txt
python run.py serve
```

Then open **http://127.0.0.1:8000** in a browser.

On the first start the pipeline runs automatically in the background (progress bar at the top of
the page, **≈1–3 min** depending on your CPU). The dashboard fills in when it finishes. Click
**Run pipeline** at any time to refresh.

### Option C — Docker
```bash
docker build -t orbital-risk .
docker run -p 8000:8000 orbital-risk          # refreshes data/alerts every 6 h
```

### Other commands
```bash
python run.py pipeline                 # batch run only (results -> outputs/state.json)
python run.py pipeline --fast          # quicker: 12 h horizon, 30 s step
python run.py serve --mode offline     # force the bundled offline data (no internet needed)
python run.py serve --mode live        # require live CelesTrak/NOAA data
python run.py serve --refresh-hours 6  # continuous alerting: re-run every 6 h
python run.py fetch                    # just download + cache the live datasets
python run.py test                     # 10 self-tests of the core algorithms
```
API documentation (Swagger) is at **http://127.0.0.1:8000/docs**.

---

## 2. Datasets used

| Problem-statement dataset | How the prototype uses it | Access |
|---|---|---|
| **CelesTrak** GP element sets (TLE) | Catalog: stations, Starlink, OneWeb, Iridium-NEXT, GPS, weather and the **Cosmos 2251, Iridium 33, Fengyun-1C, Cosmos 1408 debris clouds** | free, automatic |
| **CelesTrak SATCAT** | object type (payload / rocket body / debris) and **RCS** | free, automatic |
| **Space-Track.org / 18th–19th SDS** | public **Conjunction Data Messages** (hit/miss KPI) and **60-day TLE history** (anomaly + propagation-error KPIs) | free account → `.env` |
| **ESA DISCOS** | mass and cross-section (area-to-mass ratio for drag and casualty risk) | free token → `.env` |
| **NOAA SWPC** F10.7 & Kp | real-time drivers shown in the dashboard and fed to the drag model | free, automatic |
| CelesTrak space weather (F10.7, Ap, 81-day mean, 5 yrs + forecasts) | training data for the ML density/drag model | free, automatic |
| Historical events (**Iridium 33 – Cosmos 2251**, fragmentations) | validation case and fragmentation-cloud detection | built in |

**Data modes** (`--mode` or `SDRP_DATA_MODE`):

* `auto` (default) — downloads live data; anything unreachable falls back to the bundled offline data.
* `live` — fails if CelesTrak is unreachable.
* `offline` — fully reproducible, no internet. Uses a generated catalog (~2,500 objects) built from
  the real orbit families above, plus labelled ground truth that public data does not provide:
  injected close encounters (“synthetic CDMs”), 60-day TLE histories with injected maneuvers and
  breakups, and a solar-cycle space-weather series.

Every source actually used in a run is shown as a badge under the header (yellow = offline/synthetic).

To enable Space-Track and DISCOS, copy `.env.example` to `.env` and fill in the credentials.

---

## 3. What is implemented (mapping to the problem statement)

| Requirement | Implementation | Code |
|---|---|---|
| SGP4/SDP4 propagation | `sgp4` (Vallado C++ core) vectorised with `SatrecArray`; browser uses `satellite.js` for the live 3-D globe | `astro/propagate.py` |
| Orbit determination / TLE fitting | Newton least-squares fit of SGP4 mean elements to a state vector (used for scenario reconstruction and synthetic data) | `data/synthetic.py` |
| Covariance handling | RIC covariance = TLE-class epoch σ + linear growth + **drag-driven along-track growth** (from the AI density uncertainty), with **learned realism scale factors** fitted on TLE-vs-TLE residuals (χ² test) | `astro/covariance.py`, `ops/kpis.py` |
| ML for drag / density uncertainty | Gradient-boosted **quantile regression** (P10/P50/P90) of log(ρ/ρ_static) vs altitude, F10.7, F10.7₈₁, Ap, season | `ai/density.py` |
| Probabilistic Pc | Foster/Alfriend 2-D Pc (Gauss–Legendre quadrature), Monte-Carlo validation, Pc-max (covariance dilution), **ML-calibrated Pc** | `astro/pc.py`, `ai/calibration.py` |
| Anomaly detection (maneuvers / fragmentation) | per-object robust z-scores of TLE transitions, space-weather common-mode removal, **Isolation Forest**; **DBSCAN** discovers fragmentation clouds | `ai/anomaly.py` |
| GNN-based conjunction screening | graph = apogee/perigee-overlap pairs; GraphSAGE-style message passing + learned edge readout with MOID features; trained on yesterday, predicts tomorrow’s risky pairs | `ai/gnn_screen.py` |
| Uncertainty quantification | quantile density model, covariance realism, Pc-max, Monte-Carlo re-entry windows (P10/P50/P90) | several |
| GPU/HPC all-vs-all screening | KD-tree spatial index (CPU) or blocked brute force on **GPU via CuPy** (`SDRP_USE_GPU=on`, `pip install -r requirements-gpu.txt`) | `astro/screening.py` |
| Cloud batch scheduling | `python run.py pipeline` is a stateless batch job (cron / Cloud Run job / AWS Batch); Dockerfile; `--refresh-hours` in-process scheduler | `pipeline.py`, `Dockerfile` |
| Risk dashboard | 7-tab web app: 3-D globe, alerts, conjunction drill-down, re-entry, space weather, anomalies, KPIs, catalog | `web/` |
| Conjunction alerting | red (Pc ≥ 1e-4) / yellow (Pc ≥ 1e-5) list, toast + browser notifications, `/api/alerts` | `server.py`, `web/app.js` |
| Explainable alerts (XAI) | exact per-driver attribution in log10(Pc) (miss distance, radial separation, position uncertainty, drag uncertainty, TLE age, hard-body radius), counterfactual miss distance, and an `[ALERT]/[DRIVERS]/[DIAGNOSTIC]/[COUNTERFACTUAL]/[ACTION]` readout | `ai/explain.py` |
| **Option C — forecast estimate changes** | at each orbital update, predicts whether the next update will move the miss distance by > max(200 m, 25 %) or the TCA by > 1 s. Gradient boosting on lagged update features; strict time-ordered split (train oldest 60 % of encounters, calibrate 20 %, test newest 20 %); compared with climatology and a calibrated physics heuristic. Trains on Space-Track `cdm_public` version histories when credentials are set, otherwise on physics-based simulated OD update sequences | `ai/forecast.py` |
| Maneuver recommendation | Clohessy–Wiltshire search over burn time × direction, bisection on Δv to reach target Pc; reports **min Δv**, the return-to-slot burn, a fuel-cost verdict and how many such burns the satellite can still afford | `ops/maneuver.py` |
| **Satellite life cycle** | stage (commissioning → operational → late life → end of life → decaying) from launch year, design life and an assumed propellant budget. A satellite whose usable propellant is gone (design life over, or only the disposal reserve left) **cannot maneuver and is treated as debris** | `ops/lifecycle.py` |
| **Re-encounter outlook** | after each avoidance burn the satellite is re-screened for 7 days; new red encounters are re-optimised with only the propellant left, until the usable fuel runs out | `pipeline.py` (`followup`) |
| Re-entry risk | Monte-Carlo decay integration (AI density + F10.7 forecast + B* uncertainty) → re-entry window, latitude band, casualty expectation (NASA 1e-4 rule) | `ai/reentry.py` |

---

## 4. KPIs reported (KPIs & Validation tab)

| KPI | How it is measured |
|---|---|
| Pc accuracy & calibration (**Brier score**) | encounter-population study; raw vs ML-calibrated Pc, reliability diagram, skill vs climatology |
| Conjunction **hit/miss rate vs CDMs** and **lead time** | match screening events with Space-Track CDMs (or the injected ground-truth CDMs offline) by pair and TCA (±120 s); median lead time of alerts |
| **Propagation position error** (km) at 1–7 days; **covariance realism** | TLE-vs-TLE residuals in RIC; χ² 95 % containment and median NEES before/after learned realism scaling |
| **False-alert rate** and **missed-collision rate** | tuned alert threshold on the encounter-population study; the trade-off curve and the effect of better tracking are shown on the Metrics tab. Plain accuracy is not reported because a system that never alerts scores ~96 % |
| **Maneuver efficiency** (min Δv) | mean recommended Δv and saving vs a late radial burn |
| **Catalog coverage** and **update latency** | objects screened; time to re-screen updated objects one-vs-all over the full horizon |
| Extra AI KPIs | density-model error reduction, GNN recall / pairs pruned / speed-up, anomaly-detection F1, Option C AUC / Brier vs baselines |

Typical offline run (2,530 objects, 24 h, 20 s step, 2-core laptop): ~1.2 billion pair·steps/s,
100 % hit rate on reference CDMs, propagation error ≈0.8 / 3.4 / 17 km at 1 / 3 / 7 days,
χ² containment at 7 days 40 % → 67 % with learned covariance realism, density error −80 %,
GNN recall ≈95 % while pruning ≈85 % of candidate pairs (6× faster pairwise screening),
anomaly F1 ≈0.9, min-Δv maneuvers ≈75 % cheaper than a late burn, and the reconstructed
**Iridium 33 / Cosmos 2251** encounter flagged **red** (Pc ≈ 5e-4).

---

## 5. Dashboard tour (demo script)

1. **Overview** — live 3-D globe (drag to rotate, scroll to zoom; time speed selector). Pulsing
   red/yellow dots are predicted conjunction points; double-click one to open it. Right: alert list
   and re-entries within 72 h.
2. **Conjunctions** — sortable screening results. Selecting an event shows the encounter-plane
   covariance ellipses vs the hard-body radius, range around TCA, a **“Why this alert?”** explanation panel,
   the **recommended minimum-Δv maneuver** with its fuel cost, and the **re-encounter outlook** (will it meet debris again, and can it still afford to dodge?).
   A spent satellite shows why it cannot maneuver instead.
2b. **Life cycle** — every payload by stage and orbit, and the satellites near or past end of life.
3. **Re-entry** — P10–P90 re-entry windows and casualty expectation (red bars exceed 1e-4).
4. **Drag & Space Weather** — F10.7/Ap drivers and the ML density correction with its uncertainty band.
5. **Anomalies** — detected maneuvers/breakups and the fragmentation clouds found by DBSCAN.
6. **Metrics** — the KPIs above, the false-alert explanation and charts; model internals are folded away under “Model details”.
7. **Catalog** — search any object, show its orbit, or click **re-screen** to run an incremental
   one-vs-all screening (simulates a fresh observation arriving).

---

## 6. REST API (selection)

| Endpoint | Description |
|---|---|
| `GET /api/status` · `POST /api/run?fast=true` | pipeline status / start a run |
| `GET /api/summary` | provenance, summary counts, KPIs, validation |
| `GET /api/conjunctions?risk=red&min_pc=1e-6` | screening results |
| `GET /api/conjunction/{id}?target_pc=1e-5` | geometry, Pc sensitivity, maneuver options |
| `GET /api/alerts` | red/yellow conjunctions + re-entries < 72 h |
| `GET /api/object/{norad}` · `POST /api/screen_object/{norad}` | object details · incremental screening |
| `GET /api/state/{reentry, anomalies, space_weather, density_model, calibration, gnn, propagation}` | module outputs |
| `GET /api/tles` · `GET /api/catalog?q=` | catalog |

---

## 7. Configuration (environment variables or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SDRP_DATA_MODE` | `auto` | `auto` / `live` / `offline` |
| `SDRP_MAX_OBJECTS` | 3000 | cap on live catalog size (stratified sample) |
| `SDRP_HORIZON_H` / `SDRP_STEP_S` / `SDRP_THRESHOLD_KM` | 24 / 20 / 10 | screening window, step, miss threshold |
| `SDRP_PC_RED` / `SDRP_PC_YELLOW` | 1e-4 / 1e-5 | alert thresholds |
| `SDRP_USE_GPU` | `auto` | CuPy GPU screening |
| `SPACETRACK_USER` / `SPACETRACK_PASS` | — | CDMs + TLE history |
| `DISCOS_TOKEN` | — | ESA DISCOS physical properties |
| `SDRP_CELESTRAK_GROUPS` | see `config.py` | which CelesTrak groups to download |

Larger runs: `SDRP_MAX_OBJECTS=10000 python run.py pipeline` (KD-tree screening scales ~N log N per step).

---

## 8. Project structure
```
space-debris-ai/
├── run.py                 CLI: serve | pipeline | fetch | test
├── run.sh / run.bat       one-command launchers
├── requirements.txt       Dockerfile   .env.example
├── sdrp/
│   ├── config.py
│   ├── pipeline.py        batch orchestration -> outputs/state.json
│   ├── server.py          FastAPI REST API + static dashboard
│   ├── data/   sources.py (CelesTrak, SATCAT, NOAA, Space-Track, DISCOS)  synthetic.py (offline data, TLE fitting, Iridium-Cosmos)
│   ├── astro/  propagate.py screening.py pc.py covariance.py atmosphere.py
│   ├── ai/     density.py gnn_screen.py anomaly.py calibration.py reentry.py explain.py forecast.py
│   └── ops/    maneuver.py lifecycle.py kpis.py
├── web/        index.html app.js styles.css vendor/ (three.js, Chart.js, satellite.js, Natural Earth – bundled, works offline)
└── tests/      test_core.py
```

---

## 9. Honest notes & limitations

* **Offline ground truth is synthetic by design.** Real collisions are too rare to score Pc models,
  and public data does not label maneuvers. The prototype therefore injects labelled encounters,
  maneuvers and breakups. Everything synthetic is tagged in the UI. With a Space-Track account the
  CDM comparison and TLE histories switch to real data automatically.
* The density model’s training target is a semi-empirical thermosphere surrogate driven by real
  F10.7/Ap. To train on measured densities (GRACE-FO / Swarm / CHAMP), pass them to
  `DensityModel.fit_arrays` — the rest of the pipeline is unchanged.
* The GNN uses fixed mean-aggregation message passing with a gradient-boosted edge readout so it runs
  on any laptop; replacing it with a trainable PyTorch-Geometric GraphSAGE/GAT only changes
  `ai/gnn_screen.py`.
* **Fuel is an assumption.** Public catalogs do not publish propellant. The budget is a regime-level
  total Δv scaled by how much of the design life has passed, minus a disposal reserve that is never
  spent on avoidance. Put real figures in `data/fuel.json` (`{"<norad>": {"remaining_dv_ms": 12.5}}`).
* **False alerts cannot reach zero with public tracking data.** Position errors of hundreds of metres to
  kilometres are far larger than a satellite, so near misses and hits cannot always be told apart.
  Better tracking, not a bigger model, is what lowers the rate (see the Metrics tab).
* Maneuver planning uses linear (Clohessy–Wiltshire) relative motion; verify with high-fidelity
  propagation (Orekit / GMAT) before operational use.
* Iridium–Cosmos validation uses a reconstruction of the public collision geometry with TLE-level
  errors (archival 2009 TLEs need Space-Track).
* Not for operational flight-safety decisions.

## 10. Troubleshooting

* **Port in use** → `python run.py serve --port 8080`
* **No internet / firewall** → `python run.py serve --mode offline`
* **Slow first run** → `python run.py serve --fast` (12 h horizon, 30 s step)
* **Globe blank** → the browser needs WebGL (any modern Chrome/Edge/Firefox/Safari).
* **Stale results** → results older than 6 h are refreshed automatically on start; or click *Run pipeline*.
