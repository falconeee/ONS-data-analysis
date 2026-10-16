"""Hourly weather per subsystem from Open-Meteo (no API key).

Two sources, kept separate on purpose:

* ``archive``  - ERA5-based reanalysis. Used for training and as the *perfect-forecast* scenario
  (observed weather treated as if it were known in advance: an upper bound, not a real forecast).
* ``forecast`` - previous-runs API ``*_previous_day1``: the forecast issued the day before each
  hour (lead ~24-47 h). This is the *honest* day-ahead scenario; available from ~2022 with gaps.

City series (``resources/subsystem_states.yaml``) are combined into one series per subsystem with
weights = load share of the UFs each city represents.
"""

import time
from datetime import date

import polars as pl
import requests

from ons_energy.calendar.holidays import SUBSYSTEMS, _states_cfg, uf_load_shares
from ons_energy.data.catalog import OpenMeteoSpec
from ons_energy.data.ckan import http_session
from ons_energy.data.clean import regularize, utc_to_local_naive
from ons_energy.utils import PROCESSED_DIR, RAW_DIR, get_logger

log = get_logger(__name__)

WEATHER_DIR = RAW_DIR / "open_meteo"
SHORT = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "shortwave_radiation": "ghi"}


def city_weights() -> pl.DataFrame:
    """Columns: city, subsystem, lat, lon, weight (sums to 1 within each subsystem)."""
    shares = uf_load_shares()
    rows = []
    for city, c in _states_cfg()["cities"].items():
        w = sum(shares[c["subsystem"]].get(uf, 0.0) for uf in c["ufs"])
        rows.append((city, c["subsystem"], c["lat"], c["lon"], w))
    df = pl.DataFrame(rows, schema=["city", "subsystem", "lat", "lon", "weight"], orient="row")
    return df.with_columns(weight=pl.col("weight") / pl.col("weight").sum().over("subsystem"))


def _get(session: requests.Session, url: str, params: dict) -> dict:
    for _ in range(30):
        resp = session.get(url, params=params, timeout=180)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        reason = resp.json().get("reason", "")
        if "Daily" in reason:
            raise RuntimeError(f"Open-Meteo daily limit reached: {reason}")
        wait = 600 if "Hourly" in reason else 65
        log.warning("Open-Meteo rate limit (%s); sleeping %ds", reason or "429", wait)
        time.sleep(wait)
    raise RuntimeError("Open-Meteo: too many rate-limit retries")


def fetch_city_year(
    spec: OpenMeteoSpec, source: str, city: str, lat: float, lon: float, year: int, session: requests.Session
) -> pl.DataFrame:
    """Hourly UTC series for one city/year (cached). Columns: ts_utc, temp, rh, ghi."""
    path = WEATHER_DIR / source / f"{city}_{year}.parquet"
    today = date.today()
    if path.exists() and year < today.year:
        return pl.read_parquet(path)

    start = max(date(year, 1, 1), date.fromisoformat(spec.archive_start if source == "archive" else spec.forecast_start))
    end = min(date(year, 12, 31), today)
    suffix = "" if source == "archive" else "_previous_day1"
    hourly = [v + suffix for v in spec.variables]
    url = spec.archive_url if source == "archive" else spec.previous_runs_url
    params = {
        "latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
        "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "GMT",
    }
    payload = _get(session, url, params)["hourly"]
    df = pl.DataFrame(
        {"ts_utc": payload["time"], **{SHORT[v]: payload[v + suffix] for v in spec.variables}},
        schema_overrides={SHORT[v]: pl.Float64 for v in spec.variables},
    ).with_columns(pl.col("ts_utc").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    time.sleep(2.0)
    return df


def ingest_weather(spec: OpenMeteoSpec, end_year: int | None = None) -> None:
    session = http_session()
    end_year = end_year or date.today().year
    for source, first in (("archive", spec.archive_start), ("forecast", spec.forecast_start)):
        for city, c in _states_cfg()["cities"].items():
            for year in range(date.fromisoformat(first).year, end_year + 1):
                fetch_city_year(spec, source, city, c["lat"], c["lon"], year, session)
        log.info("Open-Meteo %s: done", source)


def build_weather_hourly() -> pl.DataFrame:
    """Load-weighted subsystem weather on the local hourly grid of the ONS files.

    Columns: subsystem, ts_local, source (archive|forecast), temp, rh, ghi, is_imputed.
    """
    weights = city_weights()
    out = []
    for source in ("archive", "forecast"):
        files = sorted((WEATHER_DIR / source).glob("*.parquet"))
        if not files:
            continue
        city = pl.concat(
            [pl.read_parquet(f).with_columns(city=pl.lit(f.stem.rsplit("_", 1)[0])) for f in files]
        ).drop_nulls(["temp"])
        sub = (
            city.join(weights.select("city", "subsystem", "weight"), on="city")
            .with_columns(ts_local=utc_to_local_naive(pl.col("ts_utc")))
            # renormalize weights over the cities with data at that hour (forecast gaps)
            .group_by("subsystem", "ts_local")
            .agg([(pl.col(v) * pl.col("weight")).sum() / pl.col("weight").sum() for v in SHORT.values()])
        )
        reg = regularize(sub, "ts_local", list(SHORT.values()), every="1h", max_gap=6)
        out.append(reg.with_columns(source=pl.lit(source)))
    df = pl.concat(out).select("subsystem", "ts_local", "source", *SHORT.values(), "is_imputed")
    df = df.filter(pl.col("subsystem").is_in(list(SUBSYSTEMS)))
    df.write_parquet(PROCESSED_DIR / "weather_hourly.parquet")
    return df
