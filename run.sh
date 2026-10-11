#!/usr/bin/env bash
# One-command start (macOS / Linux): creates a virtualenv, installs deps, starts the dashboard.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  $PY -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
python run.py serve "$@"
