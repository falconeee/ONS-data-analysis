"""Day-type labels used to segment forecast evaluation ("normal" vs. "special" days)."""

from datetime import date

import polars as pl

from ons_energy.calendar.events import all_events
from ons_energy.calendar.holidays import daily_calendar

# Highest priority first: a day gets the first label whose condition holds.
DAY_TYPES = [
    "atypical_event",
    "worldcup_match",
    "carnival",
    "national_holiday",
    "state_holiday",
    "bridge",
    "year_end",
    "normal",
]
ATYPICAL_KINDS = ["blackout", "interruption", "covid", "strike"]


def _event_dates(events: pl.DataFrame, subsystem: str, kinds: list[str]) -> set[date]:
    ev = events.filter(pl.col("kind").is_in(kinds) & pl.col("subsystems").list.contains(subsystem))
    out: set[date] = set()
    for start, end in ev.select("start", "end").iter_rows():
        out.update(pl.date_range(start.date(), end.date(), interval="1d", eager=True).to_list())
    return out


def day_types(
    start: date, end: date, subsystem: str, state_threshold: float = 0.2, events: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Columns: date, day_type. ``state_threshold`` = min share of load on a state holiday."""
    events = all_events() if events is None else events
    cal = daily_calendar(start, end, subsystem)
    atypical = _event_dates(events, subsystem, ATYPICAL_KINDS)
    matches = _event_dates(events, subsystem, ["worldcup_match"])

    label = (
        pl.when(pl.col("date").is_in(list(atypical))).then(pl.lit("atypical_event"))
        .when(pl.col("date").is_in(list(matches))).then(pl.lit("worldcup_match"))
        .when(pl.col("is_carnival") | pl.col("is_ash_wednesday")).then(pl.lit("carnival"))
        .when(pl.col("is_national_holiday") | pl.col("is_corpus_christi")).then(pl.lit("national_holiday"))
        .when(pl.col("state_holiday_w") >= state_threshold).then(pl.lit("state_holiday"))
        .when(pl.col("is_bridge")).then(pl.lit("bridge"))
        .when(pl.col("is_year_end")).then(pl.lit("year_end"))
        .otherwise(pl.lit("normal"))
    )
    return cal.select("date", day_type=label)
