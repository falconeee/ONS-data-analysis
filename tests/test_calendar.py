from datetime import date, datetime

import polars as pl

from ons_energy.calendar.holidays import UF_TO_SUBSYSTEM, daily_calendar, holiday_table
from ons_energy.calendar.segments import day_types


def _row(df: pl.DataFrame, d: date) -> dict:
    return df.filter(pl.col("date") == d).row(0, named=True)


def test_uf_mapping_covers_all_27_ufs():
    assert len(UF_TO_SUBSYSTEM) == 27
    assert UF_TO_SUBSYSTEM["MA"] == "N" and UF_TO_SUBSYSTEM["RO"] == "SE"


def test_national_carnival_and_corpus_christi_2024():
    cal = daily_calendar(date(2024, 1, 1), date(2024, 12, 31), "SE")
    assert _row(cal, date(2024, 9, 7))["is_national_holiday"]
    assert _row(cal, date(2024, 2, 13))["is_carnival"]
    assert _row(cal, date(2024, 2, 14))["is_ash_wednesday"]
    assert _row(cal, date(2024, 5, 30))["is_corpus_christi"]
    # Corpus Christi is on Thursday -> Friday is a bridge
    assert _row(cal, date(2024, 5, 31))["is_bridge"]


def test_bridge_before_tuesday_holiday():
    # 2026-04-21 (Tiradentes) is a Tuesday -> Monday 20 is a bridge
    cal = daily_calendar(date(2026, 4, 1), date(2026, 4, 30), "SE")
    assert _row(cal, date(2026, 4, 21))["is_national_holiday"]
    assert _row(cal, date(2026, 4, 20))["is_bridge"]


def test_state_holiday_weight_is_partial_and_local():
    # 09/07 Revolução Constitucionalista: SP only -> large share in SE, nothing in S.
    se_shares = {"SP": 0.45, "MG": 0.2, "RJ": 0.15, "GO": 0.1, "DF": 0.1}
    se = _row(daily_calendar(date(2024, 7, 9), date(2024, 7, 9), "SE", se_shares), date(2024, 7, 9))
    s = _row(daily_calendar(date(2024, 7, 9), date(2024, 7, 9), "S", {"PR": 0.4, "SC": 0.25, "RS": 0.35}), date(2024, 7, 9))
    assert se["state_holiday_w"] == 0.45
    assert s["state_holiday_w"] == 0.0


def test_holiday_table_categories():
    t = holiday_table([2024])
    assert set(t["category"].unique()) == {"national", "optional", "state"}
    assert t.filter((pl.col("category") == "state") & (pl.col("uf") == "SP"))["date"].to_list() == [date(2024, 7, 9)]


def test_day_types_priority():
    events = pl.DataFrame(
        {
            "event_id": ["m", "b"],
            "kind": ["worldcup_match", "blackout"],
            "start": [datetime(2022, 12, 9, 12), datetime(2023, 8, 15, 8, 30)],
            "end": [datetime(2022, 12, 9, 14), datetime(2023, 8, 15, 15)],
            "subsystems": [["SE", "S", "NE", "N"], ["NE"]],
            "description": ["", ""],
        }
    )
    dt = day_types(date(2022, 12, 1), date(2023, 8, 31), "NE", events=events)
    label = dict(dt.iter_rows())
    assert label[date(2022, 12, 9)] == "worldcup_match"
    assert label[date(2023, 8, 15)] == "atypical_event"
    assert label[date(2022, 12, 25)] == "national_holiday"
    assert label[date(2022, 12, 27)] == "year_end"
    assert label[date(2023, 2, 21)] == "carnival"
    assert label[date(2023, 3, 15)] == "normal"
    dt_s = dict(day_types(date(2023, 8, 15), date(2023, 8, 15), "S", events=events).iter_rows())
    assert dt_s[date(2023, 8, 15)] == "normal"  # blackout row only tagged NE
