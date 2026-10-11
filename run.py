#!/usr/bin/env python3
"""Space Debris Risk Prediction (AI) - command-line entry point.

  python run.py serve              # start the dashboard at http://127.0.0.1:8000 (runs the pipeline on first start)
  python run.py pipeline           # run the full batch pipeline once (results in outputs/)
  python run.py pipeline --fast    # 12 h horizon, 30 s step (quicker demo)
  python run.py fetch              # only download/cache the live datasets
  python run.py test               # quick self-test of the core algorithms

Options: --mode auto|live|offline  --horizon HOURS  --step SECONDS  --threshold KM  --port N
         --refresh-hours N (serve: re-run the pipeline every N hours = continuous alerting)
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Space Debris Risk Prediction (AI)")
    ap.add_argument("command", choices=["serve", "pipeline", "fetch", "test"], nargs="?", default="serve")
    ap.add_argument("--mode", choices=["auto", "live", "offline"], default=None)
    ap.add_argument("--horizon", type=float, default=None, help="screening horizon in hours")
    ap.add_argument("--step", type=float, default=None, help="screening time step in seconds")
    ap.add_argument("--threshold", type=float, default=None, help="miss-distance threshold in km")
    ap.add_argument("--fast", action="store_true", help="quick demo settings")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--refresh-hours", type=float, default=0, help="re-run the pipeline every N hours while serving")
    ap.add_argument("--no-initial-run", action="store_true", help="serve without running the pipeline first")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.mode:
        os.environ["SDRP_DATA_MODE"] = a.mode

    from sdrp import config
    if a.mode:
        config.DATA_MODE = a.mode

    if a.command == "pipeline":
        from sdrp.pipeline import run
        st = run(mode=a.mode, horizon_h=a.horizon, step_s=a.step, threshold_km=a.threshold, fast=a.fast)
        s = st["summary"]
        print(f"\nDone. {s['objects']} objects | {s['conjunctions']} conjunctions | "
              f"{s['red']} red, {s['yellow']} yellow alerts | results in {config.OUTPUT_DIR}")
    elif a.command == "fetch":
        from sdrp.data import sources
        objs = sources.fetch_celestrak()
        print("CelesTrak objects:", len(objs))
        try:
            print("SATCAT entries:", len(sources.fetch_satcat()))
        except Exception as e:
            print("SATCAT failed:", e)
        try:
            print("Space-weather days:", len(sources.fetch_space_weather()))
        except Exception as e:
            print("Space weather failed:", e)
        print("NOAA now:", sources.fetch_noaa_now())
        st = sources.SpaceTrack()
        if st.enabled:
            print("Space-Track CDMs:", len(st.cdms()))
        else:
            print("Space-Track: set SPACETRACK_USER / SPACETRACK_PASS to download CDMs and TLE history")
    elif a.command == "test":
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"]))
    else:
        from sdrp.server import start
        start(host=a.host or config.SERVER_HOST, port=a.port or config.SERVER_PORT,
              initial_run=not a.no_initial_run,
              run_kwargs=dict(mode=a.mode, horizon_h=a.horizon, step_s=a.step,
                              threshold_km=a.threshold, fast=a.fast),
              refresh_hours=a.refresh_hours)


if __name__ == "__main__":
    main()
