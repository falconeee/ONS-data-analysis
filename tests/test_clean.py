from datetime import datetime

import polars as pl
import pytest

from ons_energy.data.clean import add_utc, read_raw, regularize, subsystem, to_datetime, to_float
from ons_energy.data.schemas import SchemaError, validate_raw

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_to_float_handles_decimal_strings_and_blanks():
    df = pl.DataFrame({"v": ["0E-8", " 5337.70000000", "", "-2.5"]})
    out = df.select(to_float("v", df.schema["v"]))["v"].to_list()
    assert out[0] == 0.0 and out[1] == pytest.approx(5337.7) and out[2] is None and out[3] == -2.5


def test_to_float_is_strict_on_garbage():
    df = pl.DataFrame({"v": ["12,5"]})  # decimal comma would silently corrupt data
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select(to_float("v", df.schema["v"]))


def test_subsystem_normalization_strips_and_maps_aliases():
    df = pl.DataFrame({"id_subsistema": ["N ", " S", "SECO", "SE", "sin"]})
    assert df.select(subsystem())["subsystem"].to_list() == ["N", "S", "SE", "SE", "SIN"]


def test_regularize_creates_missing_dst_hour_and_flags_it():
    # 2000-10-08 00:00 does not exist in Brasília (spring forward); ONS files skip it.
    ts = [datetime(2000, 10, 7, 22), datetime(2000, 10, 7, 23), datetime(2000, 10, 8, 1), datetime(2000, 10, 8, 2)]
    df = pl.DataFrame({"subsystem": "S", "ts_local": ts, "load_mw": [10.0, 12.0, 16.0, 18.0]})
    out = regularize(df, "ts_local", ["load_mw"], every="1h")
    assert out.height == 5
    row = out.filter(pl.col("ts_local") == datetime(2000, 10, 8, 0))
    assert row["is_imputed"].item() is True
    assert row["load_mw"].item() == pytest.approx(14.0)


def test_regularize_leaves_long_gaps_null():
    ts = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 2), "1h", eager=True)
    vals = [1.0] * len(ts)
    for i in range(5, 15):
        vals[i] = None
    df = pl.DataFrame({"subsystem": "SE", "ts_local": ts, "v": vals})
    out = regularize(df, "ts_local", ["v"], every="1h", max_gap=6)
    assert out["v"].null_count() == 10


def test_add_utc_spring_forward_and_regular_hours():
    ts = [datetime(2000, 10, 7, 23), datetime(2000, 10, 8, 0), datetime(2000, 10, 8, 1), datetime(2023, 1, 1, 0)]
    out = add_utc(pl.DataFrame({"ts_local": ts}))["ts_utc"].dt.replace_time_zone(None).to_list()
    assert out[0] == datetime(2000, 10, 8, 2)  # UTC-3 before the switch
    assert out[2] == datetime(2000, 10, 8, 3)  # UTC-2 after
    assert out[1] == out[2]  # non-existent hour mapped to the next instant
    assert out[3] == datetime(2023, 1, 1, 3)


def test_old_csv_fixture_parses_and_validates():
    # Real slice of CURVA_CARGA_2000.csv (Sul, around the 2000-10-08 DST switch).
    raw = read_raw(FIXTURES / "curva_carga_2000_dst_sample.csv")
    validate_raw(raw, "load_hourly", "fixture")
    df = raw.select(
        subsystem(),
        to_datetime("din_instante", raw.schema["din_instante"]).alias("ts_local"),
        to_float("val_cargaenergiahomwmed", raw.schema["val_cargaenergiahomwmed"]).alias("load_mw"),
    )
    out = regularize(df, "ts_local", ["load_mw"], every="1h")
    assert out.height == df.height + 1
    assert out["is_imputed"].sum() == 1


def test_validate_raw_detects_renamed_column():
    df = pl.DataFrame({"id_subsistema": ["S"], "din_instante": [datetime(2020, 1, 1)], "val_carga_mw": [1.0]})
    with pytest.raises(SchemaError, match="missing columns"):
        validate_raw(df, "load_hourly")
