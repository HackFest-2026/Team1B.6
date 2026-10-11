@echo off
REM One-command start (Windows): creates a virtualenv, installs deps, starts the dashboard.
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
python run.py serve %*
