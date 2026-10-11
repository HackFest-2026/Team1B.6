"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# optional .env file (KEY=VALUE per line) for credentials such as SPACETRACK_USER
_envf = ROOT / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8-sig").splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
DATA_DIR = Path(os.getenv("SDRP_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = Path(os.getenv("SDRP_OUTPUT_DIR", ROOT / "outputs"))
MODEL_DIR = OUTPUT_DIR / "models"
for _d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name, default, cast=str):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except Exception:
        return default


# ---------------- data source selection ----------------
# "auto"  -> try live downloads (CelesTrak, NOAA, Space-Track, DISCOS), fall back to offline data
# "live"  -> live only (errors if offline)
# "offline" -> bundled/synthetic data only (always works, fully reproducible)
DATA_MODE = _env("SDRP_DATA_MODE", "auto").strip().lower()
if DATA_MODE not in ("auto", "live", "offline"):
    print(f"[config] unknown SDRP_DATA_MODE '{DATA_MODE}' - using 'auto'")
    DATA_MODE = "auto"

# CelesTrak GP groups to pull in live mode (comma separated).
CELESTRAK_GROUPS = _env(
    "SDRP_CELESTRAK_GROUPS",
    "stations,starlink,oneweb,iridium-NEXT,cosmos-2251-debris,iridium-33-debris,"
    "fengyun-1c-debris,cosmos-1408-debris,weather,gps-ops",
)
MAX_OBJECTS = _env("SDRP_MAX_OBJECTS", 3000, int)        # cap for laptop-friendly runs

# Optional credentials (only needed for CDMs / TLE history / DISCOS physical properties)
SPACETRACK_USER = _env("SPACETRACK_USER", "")
SPACETRACK_PASS = _env("SPACETRACK_PASS", "")
DISCOS_TOKEN = _env("DISCOS_TOKEN", "")

HTTP_TIMEOUT = _env("SDRP_HTTP_TIMEOUT", 20, int)

# ---------------- screening ----------------
SCREEN_HORIZON_H = _env("SDRP_HORIZON_H", 24.0, float)   # look-ahead window
SCREEN_STEP_S = _env("SDRP_STEP_S", 20.0, float)         # coarse time step
SCREEN_THRESHOLD_KM = _env("SDRP_THRESHOLD_KM", 10.0, float)  # report conjunctions closer than this
HARD_BODY_RADIUS_M = _env("SDRP_HBR_M", 20.0, float)     # default combined hard-body radius
PC_RED = _env("SDRP_PC_RED", 1e-4, float)                # maneuver threshold (NASA CARA red)
PC_YELLOW = _env("SDRP_PC_YELLOW", 1e-5, float)
# miss-distance risk tiers (km): HIGH <= 0.5, MEDIUM <= 2, LOW <= 4, anything farther is not classed
FAR_TARGET = _env("SDRP_FAR_TARGET", 0.20, float)           # max false-alert rate when tuning the alert threshold
MISS_HIGH_KM = _env("SDRP_MISS_HIGH_KM", 0.5, float)
MISS_MEDIUM_KM = _env("SDRP_MISS_MEDIUM_KM", 2.0, float)
MISS_LOW_KM = _env("SDRP_MISS_LOW_KM", 4.0, float)
USE_GPU = _env("SDRP_USE_GPU", "auto")                    # auto|on|off (CuPy if available)

RANDOM_SEED = _env("SDRP_SEED", 42, int)

SERVER_HOST = _env("SDRP_HOST", "127.0.0.1")
SERVER_PORT = _env("SDRP_PORT", 8000, int)
