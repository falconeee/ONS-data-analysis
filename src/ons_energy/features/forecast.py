"""Forecasting panel in Nixtla long format: ``unique_id, ds, y`` + future-known covariates.

Only *future-known* covariates are built here (calendar, curated events, methodology regimes,
weather per scenario). Lags of ``y`` are created inside the models from the history available at
each cutoff, so no feature here can leak the target.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from ons_energy.calendar.events import all_events, hourly_event_flags
from ons_energy.calendar.holidays import daily_calendar
from ons_energy.data.clean import add_utc
from ons_energy.utils import PROCESSED_DIR, get_logger

log = get_logger(__name__)

REGIMES = {"regime_2021_03": datetime(2021, 3, 1), "regime_2023_04": datetime(2023, 4, 29)}
CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
    "is_weekend", "is_national_holiday", "is_carnival", "is_ash_wednesday", "is_corpus_christi",
    "state_holiday_w", "is_bridge", "is_year_end", "worldcup_match", "is_dst", *REGIMES,
]
WEATHER_FEATURES = ["temp", "rh", "ghi"]


@dataclass
class Panel:
    df: pd.DataFrame
    futr_exog: list[str]
    weather_gap_share: float = 0.0


def calendar_features(ts: pl.Series, subsystem: str, events: pl.DataFrame | None = None) -> pl.DataFrame:
    """Future-known calendar/event covariates for hourly local timestamps of one subsystem."""
    events = all_events() if events is None else events
    start, end = ts.min().date(), ts.max().date()
    cal = daily_calendar(start, end, subsystem)
    df = pl.DataFrame({"ds": ts}).with_columns(date=pl.col("ds").dt.date())
    df = df.join(cal, on="date", how="left")

    hour = pl.col("ds").dt.hour()
    dow = pl.col("dow")
    doy = pl.col("ds").dt.ordinal_day()
    two_pi = 2 * np.pi
    matches = events.with_columns(start=pl.col("start") - pl.duration(hours=1), end=pl.col("end") + pl.duration(hours=1))
    df = df.with_columns(
        hour_sin=(two_pi * hour / 24).sin(),
        hour_cos=(two_pi * hour / 24).cos(),
        dow_sin=(two_pi * dow / 7).sin(),
        dow_cos=(two_pi * dow / 7).cos(),
        doy_sin1=(two_pi * doy / 365.25).sin(),
        doy_cos1=(two_pi * doy / 365.25).cos(),
        doy_sin2=(2 * two_pi * doy / 365.25).sin(),
        doy_cos2=(2 * two_pi * doy / 365.25).cos(),
        is_weekend=dow >= 5,
        worldcup_match=hourly_event_flags(df["ds"], subsystem, ["worldcup_match"], matches),
        **{name: pl.col("ds") >= when for name, when in REGIMES.items()},
    )
    utc = add_utc(df.select(ts_local="ds"))["ts_utc"].dt.replace_time_zone(None)
    df = df.with_columns(is_dst=pl.Series((df["ds"] - utc) == timedelta(hours=-2)))
    bools = [c for c in CALENDAR_FEATURES if df.schema[c] == pl.Boolean]
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in bools]).select("ds", *CALENDAR_FEATURES)


def weather_features(subsystem: str, scenario: str, switch_at: datetime) -> tuple[pl.DataFrame, float]:
    """Weather covariates for a scenario.

    perfect: reanalysis everywhere. honest: reanalysis before ``switch_at`` (training history) and
    the day-before forecast from then on; forecast gaps are filled with the reanalysis value 48 h
    earlier (a persistence fallback that only uses information older than the origin).
    Returns (frame, share of honest-period hours that needed the fallback).
    """
    w = pl.read_parquet(PROCESSED_DIR / "weather_hourly.parquet").filter(pl.col("subsystem") == subsystem)
    arch = w.filter(pl.col("source") == "archive").select("ts_local", *WEATHER_FEATURES).sort("ts_local")
    if scenario == "perfect":
        return arch.rename({"ts_local": "ds"}), 0.0
    fc = w.filter(pl.col("source") == "forecast").select("ts_local", *WEATHER_FEATURES)
    persist = arch.with_columns(pl.col("ts_local") + pl.duration(hours=48))
    later = (
        arch.select("ts_local").filter(pl.col("ts_local") >= switch_at)
        .join(fc, on="ts_local", how="left")
        .join(persist, on="ts_local", how="left", suffix="_p")
    )
    gap = later["temp"].is_null().mean() if later.height else 0.0
    later = later.select("ts_local", *[pl.coalesce(c, f"{c}_p").alias(c) for c in WEATHER_FEATURES])
    out = pl.concat([arch.filter(pl.col("ts_local") < switch_at), later]).sort("ts_local")
    return out.rename({"ts_local": "ds"}), float(gap or 0.0)


def build_panel(
    subsystems: list[str],
    start: date,
    end: date,
    weather: str = "none",
    weather_switch: date | None = None,
) -> Panel:
    """Hourly panel from ``start`` 00:00 to ``end`` 23:00 (so daily cutoffs land on fixed hours)."""
    load = pl.read_parquet(PROCESSED_DIR / "load_hourly.parquet")
    events = all_events()
    frames, gaps = [], []
    t0, t1 = datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()) + timedelta(hours=23)
    for sub in subsystems:
        y = (
            load.filter((pl.col("subsystem") == sub) & pl.col("ts_local").is_between(t0, t1))
            .select(unique_id=pl.lit(sub), ds="ts_local", y="load_mw")
        )
        if y["y"].null_count():
            raise ValueError(f"{sub}: {y['y'].null_count()} missing load values in [{start}, {end}]")
        df = y.join(calendar_features(y["ds"], sub, events), on="ds", how="left")
        if weather != "none":
            switch = datetime.combine(weather_switch or start, datetime.min.time())
            wf, gap = weather_features(sub, weather, switch)
            df = df.join(wf, on="ds", how="left")
            gaps.append(gap)
            missing = df.select(pl.col(WEATHER_FEATURES).null_count().sum()).item()
            if missing:
                raise ValueError(f"{sub}: weather missing for {missing} hours; re-run `ons ingest -s weather`")
        frames.append(df)
    out = pl.concat(frames).sort("unique_id", "ds")
    exog = CALENDAR_FEATURES + (WEATHER_FEATURES if weather != "none" else [])
    return Panel(out.to_pandas(), exog, float(np.mean(gaps)) if gaps else 0.0)
