"""Live data connectors with on-disk caching and graceful fallback.

Sources (all public):
* CelesTrak GP element sets (TLE)            https://celestrak.org/NORAD/elements/
* CelesTrak SATCAT (object type, RCS)        https://celestrak.org/pub/satcat.csv
* CelesTrak space weather (F10.7, Kp/Ap)     https://celestrak.org/SpaceData/SW-Last5Years.csv
* NOAA SWPC real-time F10.7 & Kp             https://services.swpc.noaa.gov/
* Space-Track.org (18th/19th SDS) CDMs + TLE history  - needs free account (env vars)
* ESA DISCOS physical properties (mass, area)          - needs free API token (env var)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from .. import config
from ..astro.propagate import SpaceObject

log = logging.getLogger("sdrp.sources")
UA = {"User-Agent": "space-debris-risk-ai/1.0 (hackathon prototype)"}


def _cached_get(name, url, max_age_h=6.0, headers=None, session=None, binary=False):
    path = config.CACHE_DIR / name
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600:
        return path.read_bytes() if binary else path.read_text(encoding="utf-8", errors="replace")
    s = session or requests
    r = s.get(url, headers={**UA, **(headers or {})}, timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content if binary else r.text


def parse_tle_text(text, source="celestrak"):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name, l1, l2 = f"OBJ {lines[i][2:7]}", lines[i], lines[i + 1]; i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2 = lines[i].strip().lstrip("0 ").strip(), lines[i + 1], lines[i + 2]; i += 3
        else:
            i += 1
            continue
        try:
            norad = int(l1[2:7]) if l1[2:7].strip().isdigit() else int(l2[2:7])
            typ = ("DEBRIS" if " DEB" in name else "ROCKET BODY" if "R/B" in name else "PAYLOAD")
            out.append(SpaceObject(norad=norad, name=name, line1=l1, line2=l2, obj_type=typ,
                                   source=source))
        except Exception as e:   # malformed line
            log.debug("skip TLE %s: %s", name, e)
    return out


# ----------------------------------------------------------------------------- CelesTrak
def fetch_celestrak(groups=None):
    groups = [g.strip() for g in (groups or config.CELESTRAK_GROUPS).split(",") if g.strip()]
    objs = []
    for g in groups:
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle"
        try:
            txt = _cached_get(f"celestrak_{g}.tle", url, max_age_h=4)
            got = parse_tle_text(txt, "celestrak")
            log.info("CelesTrak %s: %d objects", g, len(got))
            objs += got
        except Exception as e:
            log.warning("CelesTrak group %s failed: %s", g, e)
    return objs


def fetch_satcat():
    """Returns {norad: {type, rcs, country, decay}}."""
    txt = _cached_get("satcat.csv", "https://celestrak.org/pub/satcat.csv", max_age_h=48)
    out = {}
    tmap = {"PAY": "PAYLOAD", "R/B": "ROCKET BODY", "DEB": "DEBRIS", "UNK": "UNKNOWN"}
    for row in csv.DictReader(io.StringIO(txt)):
        try:
            rcs = row.get("RCS") or ""
            out[int(row["NORAD_CAT_ID"])] = {
                "type": tmap.get(row.get("OBJECT_TYPE", "UNK"), "UNKNOWN"),
                "rcs": float(rcs) if rcs.strip() else None,
                "country": row.get("OWNER", ""),
            }
        except Exception:
            continue
    return out


def fetch_space_weather():
    """Daily F10.7 (obs + 81-day centred mean) and Ap since ~5 years ago, incl. predictions."""
    txt = _cached_get("SW-Last5Years.csv", "https://celestrak.org/SpaceData/SW-Last5Years.csv",
                      max_age_h=12)
    out = []
    for row in csv.DictReader(io.StringIO(txt)):
        try:
            f = float(row.get("F10.7_OBS") or row.get("F10.7_ADJ") or 0)
            fa = float(row.get("F10.7_OBS_CENTER81") or row.get("F10.7_OBS_LAST81") or f)
            ap = float(row.get("AP_AVG") or 0)
            kp_sum = float(row.get("KP_SUM") or 0) / 10.0
            if f <= 0:
                continue
            out.append({"date": row["DATE"][:10], "f107": f, "f107a": fa, "ap": ap,
                        "kp": kp_sum / 8.0, "source": "celestrak:" + (row.get("F10.7_DATA_TYPE") or "OBS")})
        except Exception:
            continue
    return out


def fetch_noaa_now():
    """Latest observed F10.7 and planetary Kp from NOAA SWPC."""
    res = {}
    try:
        f = json.loads(_cached_get("noaa_f107.json",
                                   "https://services.swpc.noaa.gov/json/f107_cm_flux.json", 3))
        last = f[-1] if isinstance(f, list) and f else {}
        res["f107"] = float(last.get("flux"))
        res["f107_time"] = last.get("time_tag")
    except Exception as e:
        log.warning("NOAA F10.7 failed: %s", e)
    try:
        k = json.loads(_cached_get("noaa_kp.json",
                                   "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", 1))
        if isinstance(k, list) and len(k) > 1:
            last = k[-1]
            if isinstance(last, dict):
                res["kp"] = float(last.get("Kp") or last.get("kp_index"))
                res["kp_time"] = last.get("time_tag")
            else:
                res["kp"] = float(last[1]); res["kp_time"] = last[0]
    except Exception as e:
        log.warning("NOAA Kp failed: %s", e)
    if res:
        res["source"] = "NOAA SWPC"
    return res


# ----------------------------------------------------------------------------- Space-Track
class SpaceTrack:
    BASE = "https://www.space-track.org"

    def __init__(self, user=None, password=None):
        self.user = user or config.SPACETRACK_USER
        self.password = password or config.SPACETRACK_PASS
        self.s = None

    @property
    def enabled(self):
        return bool(self.user and self.password)

    def login(self):
        if self.s is None:
            self.s = requests.Session()
            r = self.s.post(self.BASE + "/ajaxauth/login",
                            data={"identity": self.user, "password": self.password},
                            headers=UA, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
        return self.s

    def query(self, q, cache_name, max_age_h=2.0):
        s = self.login()
        return _cached_get(cache_name, self.BASE + "/basicspacedata/query/" + q, max_age_h,
                           session=s)

    def cdms(self, limit=300):
        """Public Conjunction Data Messages with TCA in the future."""
        q = f"class/cdm_public/TCA/%3E{datetime.now(timezone.utc):%Y-%m-%d}/orderby/PC%20desc/limit/{limit}/format/json"
        rows = json.loads(self.query(q, "spacetrack_cdm_public.json"))
        out = []
        for r in rows:
            try:
                out.append({"cdm_id": r.get("CDM_ID"), "tca": r["TCA"].replace(" ", "T") + "+00:00",
                            "min_rng_km": float(r.get("MIN_RNG") or 0) / 1000.0,
                            "pc": float(r.get("PC") or 0), "sat1": int(r["SAT_1_ID"]),
                            "sat2": int(r["SAT_2_ID"]), "sat1_name": r.get("SAT_1_NAME"),
                            "sat2_name": r.get("SAT_2_NAME"), "created": r.get("CREATED")})
            except Exception:
                continue
        return out

    def gp_tles(self, norads):
        """Latest TLEs for specific NORAD ids (used to make sure every CDM object is screened)."""
        out = []
        ids = sorted({int(n) for n in norads})
        for k in range(0, len(ids), 150):
            chunk = ",".join(map(str, ids[k:k + 150]))
            q = f"class/gp/NORAD_CAT_ID/{chunk}/orderby/NORAD_CAT_ID%20asc/format/3le"
            try:
                out += parse_tle_text(self.query(q, f"spacetrack_gp_{k}_{len(ids)}.tle", 3), "space-track")
            except Exception as e:
                log.warning("Space-Track gp query failed: %s", e)
        return out

    def cdm_history(self, days=30, limit=20000):
        """All public CDMs created in the last ``days`` (several versions per encounter)."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        q = f"class/cdm_public/CREATED/%3E{since}/orderby/CREATED%20asc/limit/{limit}/format/json"
        return json.loads(self.query(q, "spacetrack_cdm_history.json", 12))

    def gp_history(self, norads, days=60):
        ids = ",".join(str(n) for n in norads)
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        q = f"class/gp_history/NORAD_CAT_ID/{ids}/EPOCH/%3E{since}/orderby/EPOCH%20asc/format/json"
        rows = json.loads(self.query(q, f"spacetrack_hist_{abs(hash(ids)) % 10**8}.json", 24))
        return rows


# ----------------------------------------------------------------------------- ESA DISCOS
def fetch_discos(norads, token=None):
    token = token or config.DISCOS_TOKEN
    if not token:
        return {}
    out = {}
    hdr = {"Authorization": f"Bearer {token}", "DiscosWeb-Api-Version": "2"}
    norads = list(norads)
    for k in range(0, len(norads), 100):
        chunk = norads[k:k + 100]
        url = ("https://discosweb.esoc.esa.int/api/objects?filter=in(satno,(" +
               ",".join(map(str, chunk)) + "))&page[size]=100")
        try:
            r = requests.get(url, headers={**UA, **hdr}, timeout=config.HTTP_TIMEOUT)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)))
                r = requests.get(url, headers={**UA, **hdr}, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            for d in r.json().get("data", []):
                a = d.get("attributes", {})
                if a.get("satno"):
                    out[int(a["satno"])] = {"mass": a.get("mass"), "area": a.get("xSectAvg"),
                                            "shape": a.get("shape"), "class": a.get("objectClass")}
        except Exception as e:
            log.warning("DISCOS request failed: %s", e)
            break
    return out
