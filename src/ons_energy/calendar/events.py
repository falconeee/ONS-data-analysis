"""Curated events (``resources/events.csv``) plus large load interruptions from ONS data."""

from functools import lru_cache

import polars as pl

from ons_energy.utils.io import PROCESSED_DIR, RESOURCES_DIR

ALL_SUBSYSTEMS = ["SE", "S", "NE", "N"]


@lru_cache(maxsize=1)
def load_events() -> pl.DataFrame:
    """Columns: event_id, kind, start, end (local naive datetimes), subsystems (list), description."""
    df = pl.read_csv(RESOURCES_DIR / "events.csv")
    return df.with_columns(
        start=pl.col("start_local").str.to_datetime("%Y-%m-%d %H:%M"),
        end=pl.col("end_local").str.to_datetime("%Y-%m-%d %H:%M"),
        subsystems=pl.when(pl.col("subsystems") == "SIN")
        .then(pl.lit(ALL_SUBSYSTEMS))
        .otherwise(pl.col("subsystems").str.split(";")),
    ).select("event_id", "kind", "start", "end", "subsystems", "description")


def interruption_events(min_mw: float = 1000.0) -> pl.DataFrame:
    """One event per disturbance (SIPER code) whose total interrupted load reaches ``min_mw``.

    ``subsystems`` lists every subsystem with interrupted load; end = start + mean restoration time
    (at least one hour).
    """
    path = PROCESSED_DIR / "load_interruptions.parquet"
    schema = {"event_id": pl.String, "kind": pl.String, "start": pl.Datetime("us"), "end": pl.Datetime("us"),
              "subsystems": pl.List(pl.String), "description": pl.String}
    if not path.exists():
        return pl.DataFrame(schema=schema)
    df = pl.read_parquet(path)
    agg = (
        df.group_by("disturbance_id")
        .agg(
            start=pl.col("ts_local").min(),
            mw=pl.col("interrupted_mw").sum(),
            minutes=pl.col("restoration_min").mean(),
            subsystems=pl.col("subsystem").unique().sort(),
        )
        .filter(pl.col("mw") >= min_mw)
    )
    return agg.select(
        event_id=pl.concat_str(pl.lit("int_"), pl.col("disturbance_id").str.replace_all("/", "_")),
        kind=pl.lit("interruption"),
        start=pl.col("start").cast(pl.Datetime("us")),
        end=(pl.col("start") + pl.duration(minutes=pl.col("minutes").fill_null(60).clip(60, None))).cast(pl.Datetime("us")),
        subsystems=pl.col("subsystems"),
        description=pl.format("Load interruption {} MW", pl.col("mw").round(0)),
    ).sort("start")


def all_events(min_interruption_mw: float = 1000.0) -> pl.DataFrame:
    """Curated events + large interruptions, dropping interruptions that are a curated blackout."""
    curated = load_events().with_columns(pl.col("start", "end").cast(pl.Datetime("us")))
    ints = interruption_events(min_interruption_mw)
    blackouts = curated.filter(pl.col("kind") == "blackout")["start"].to_list()
    if blackouts and ints.height:
        near = pl.any_horizontal([(pl.col("start") - b).abs() <= pl.duration(hours=3) for b in blackouts])
        ints = ints.filter(~near)
    return pl.concat([curated, ints], how="vertical_relaxed").sort("start")


def hourly_event_flags(ts: pl.Series, subsystem: str, kinds: list[str], events: pl.DataFrame | None = None) -> pl.Series:
    """Boolean flag per hour: True if any event of ``kinds`` touching ``subsystem`` covers that hour."""
    events = all_events() if events is None else events
    ev = events.filter(pl.col("kind").is_in(kinds) & pl.col("subsystems").list.contains(subsystem))
    hours = pl.DataFrame({"ts": ts})
    flag = pl.Series("flag", [False] * len(ts))
    for start, end in ev.select("start", "end").iter_rows():
        hour_start = start.replace(minute=0)
        flag = flag | ((hours["ts"] >= hour_start) & (hours["ts"] <= end))
    return flag
