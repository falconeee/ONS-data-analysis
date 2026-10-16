"""Client for the ONS load API (``apicarga.ons.org.br``): verified and programmed load.

Semi-hourly values in UTC, timestamped at the END of each half-hour, by load area (SECO, S, NE, N,
SIN and UFs). One call covers at most ~3 months, so requests are split in windows and each window is
cached under ``data/raw/api_carga/<endpoint>/<area>/``. Recent windows are re-fetched because the
ONS revises them (``din_atualizacao``).
"""

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

from ons_energy.data.catalog import LoadApiSpec
from ons_energy.data.ckan import http_session
from ons_energy.utils import RAW_DIR, get_logger

log = get_logger(__name__)

API_DIR = RAW_DIR / "api_carga"
_EMPTY_VALUE = re.compile(r'(":)\s*(,|\s*})')


def _windows(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out


def _fetch_window(
    session: requests.Session, url: str, area: str, start: date, end: date
) -> pl.DataFrame:
    params = {"dat_inicio": start.isoformat(), "dat_fim": end.isoformat(), "cod_areacarga": area}
    resp = session.get(url, params=params, timeout=120)
    resp.raise_for_status()
    # Missing values come as bare `"key": ,` (invalid JSON) -> turn them into null.
    text = _EMPTY_VALUE.sub(r"\1 null\2", resp.text)
    records = json.loads(text)
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records, infer_schema_length=None)


def fetch_load_api(
    spec: LoadApiSpec,
    endpoint: str,
    area: str,
    start: date,
    end: date,
    cache_dir: Path = API_DIR,
    refresh_days: int = 15,
    pause_s: float = 0.2,
) -> pl.DataFrame:
    """Semi-hourly records for one area, from cache where possible."""
    url = f"{spec.base_url}/{spec.endpoints[endpoint]}"
    area_dir = cache_dir / endpoint / area
    area_dir.mkdir(parents=True, exist_ok=True)
    session = http_session()
    frames: list[pl.DataFrame] = []
    stale_after = date.today() - timedelta(days=refresh_days)

    for w_start, w_end in _windows(start, end, spec.max_window_days):
        path = area_dir / f"{w_start:%Y%m%d}_{w_end:%Y%m%d}.parquet"
        if path.exists() and w_end < stale_after:
            frames.append(pl.read_parquet(path))
            continue
        df = _fetch_window(session, url, area, w_start, w_end)
        df.write_parquet(path)
        frames.append(df)
        time.sleep(pause_s)

    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def ingest_load_api(spec: LoadApiSpec, end: date | None = None) -> dict[str, int]:
    """Verified + programmed load for the subsystem areas, and one year of verified load per UF."""
    from ons_energy.calendar.holidays import UF_TO_SUBSYSTEM

    end = end or date.today() + timedelta(days=1)  # programmed load is published for D+1
    start = date.fromisoformat(spec.start)
    counts: dict[str, int] = {}
    for endpoint in ("verified", "programmed"):
        for area in spec.areas:
            df = fetch_load_api(spec, endpoint, area, start, end)
            counts[f"{endpoint}/{area}"] = df.height
            log.info("load API %s/%s: %d rows", endpoint, area, df.height)

    y = spec.uf_share_year
    for uf in sorted(UF_TO_SUBSYSTEM):
        df = fetch_load_api(spec, "verified", uf, date(y, 1, 1), date(y, 12, 31))
        counts[f"verified/{uf}"] = df.height
    return counts
