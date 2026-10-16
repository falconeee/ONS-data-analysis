"""Build canonical Parquet tables in ``data/processed`` from the raw ONS files.

Tables
------
load_hourly          subsystem, ts_local, ts_utc, load_mw, is_imputed          (curva-carga)
generation_hourly    subsystem, ts_local, source, gen_mw                       (geracao-usina-2, aggregated)
balance_hourly       subsystem, ts_local, ts_utc, load, hydro, thermal, nuclear, wind, solar,
                     interchange, is_imputed                                   (balanco + nuclear)
hydro_daily          subsystem, date, EAR and ENA columns, is_imputed
load_interruptions   disturbance_id, ts_local, subsystem, uf, interrupted_mw, restoration_min, ens_mwh
ons_load_api_hourly  subsystem, ts_local, verified_load, verified_mmgd, programmed_load
uf_load_share        subsystem, uf, mean_load_mw, share
"""

from datetime import date

import polars as pl

from ons_energy.data.clean import (
    add_utc,
    check_subsystems,
    fill_seasonal,
    read_raw,
    regularize,
    subsystem,
    to_date,
    to_datetime,
    to_float,
    utc_to_local_naive,
)
from ons_energy.data.download import raw_files
from ons_energy.data.schemas import RAW, validate_raw
from ons_energy.utils import PROCESSED_DIR, RAW_DIR, get_logger

log = get_logger(__name__)

SOURCE_MAP = {
    "HIDROELÉTRICA": "hydro",
    "HIDROELETRICA": "hydro",
    "TÉRMICA": "thermal",
    "TERMICA": "thermal",
    "EOLIELÉTRICA": "wind",
    "EOLIELETRICA": "wind",
    "FOTOVOLTAICA": "solar",
    "NUCLEAR": "nuclear",
    "BOMBEAMENTO": "pumping",  # pumping stations (consumption), kept separate
}


def _coerced(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Strictly coerce the dictionary columns of a raw frame to their kinds."""
    exprs = []
    for col, kind in RAW[dataset].items():
        if kind == "num":
            exprs.append(to_float(col, df.schema[col]))
        elif kind == "time":
            exprs.append(to_datetime(col, df.schema[col]))
        else:
            exprs.append(pl.col(col).cast(pl.String).str.strip_chars())
    return df.select(exprs)


def _read_dataset(dataset: str) -> pl.DataFrame:
    frames = []
    for path in raw_files(dataset):
        raw = read_raw(path, columns=None)
        validate_raw(raw, dataset, path.name)
        frames.append(_coerced(raw, dataset).with_columns(_file=pl.lit(path.name)))
    return pl.concat(frames, how="vertical_relaxed")


def _write(df: pl.DataFrame, name: str) -> pl.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PROCESSED_DIR / f"{name}.parquet")
    log.info("processed/%s: %d rows, %s -> %s", name, df.height, *_span(df))
    return df


def _span(df: pl.DataFrame) -> tuple:
    col = "ts_local" if "ts_local" in df.columns else "date"
    return (df[col].min(), df[col].max()) if col in df.columns else (None, None)


def build_load_hourly() -> pl.DataFrame:
    raw = _read_dataset("load_hourly")
    df = raw.select(subsystem(), ts_local=pl.col("din_instante"), load_mw=pl.col("val_cargaenergiahomwmed"))
    check_subsystems(df)
    df = regularize(df.filter(pl.col("subsystem") != "SIN"), "ts_local", ["load_mw"], every="1h")
    df = fill_seasonal(df, ["load_mw"])
    df = add_utc(df).select("subsystem", "ts_local", "ts_utc", "load_mw", "is_imputed").sort("subsystem", "ts_local")
    return _write(df, "load_hourly")


def build_generation_hourly() -> pl.DataFrame:
    """Plant-level generation aggregated per (subsystem, hour, source), file by file."""
    frames = []
    for path in raw_files("generation_plant_hourly"):
        cols = list(RAW["generation_plant_hourly"])
        raw = read_raw(path, columns=cols)
        validate_raw(raw, "generation_plant_hourly", path.name)
        df = _coerced(raw, "generation_plant_hourly")
        agg = (
            df.with_columns(
                subsystem(),
                source=pl.col("nom_tipousina").str.to_uppercase().replace_strict(SOURCE_MAP, default=None),
            )
            .group_by("subsystem", "din_instante", "source")
            .agg(gen_mw=pl.col("val_geracao").sum(), n_plants=pl.len())
            .rename({"din_instante": "ts_local"})
        )
        unmapped = agg.filter(pl.col("source").is_null()).height
        if unmapped:
            raise ValueError(f"{path.name}: unmapped plant types {df['nom_tipousina'].unique().to_list()}")
        frames.append(agg)
    df = pl.concat(frames).group_by("subsystem", "ts_local", "source").agg(pl.col("gen_mw", "n_plants").sum())
    check_subsystems(df)
    return _write(df.sort("subsystem", "source", "ts_local"), "generation_hourly")


def build_balance_hourly(generation: pl.DataFrame | None = None) -> pl.DataFrame:
    """Hourly balance per subsystem, with nuclear split out of thermal.

    The ONS balance has no nuclear column; ``val_gertermica`` includes Angra 1/2 (checked against
    plant-level data in notebooks/01_data_eda). ``thermal`` here is thermal *excluding* nuclear.
    """
    raw = _read_dataset("balance_hourly")
    df = raw.select(
        subsystem(),
        ts_local=pl.col("din_instante"),
        load=pl.col("val_carga"),
        hydro=pl.col("val_gerhidraulica"),
        thermal_total=pl.col("val_gertermica"),
        wind=pl.col("val_gereolica"),
        solar=pl.col("val_gersolar"),
        interchange=pl.col("val_intercambio"),
    )
    check_subsystems(df)
    # Structural zeros: a source absent for a whole subsystem-year did not exist yet.
    year = pl.col("ts_local").dt.year()
    for col in ("wind", "solar"):
        all_null = pl.col(col).is_null().all().over("subsystem", year)
        df = df.with_columns(pl.when(all_null).then(0.0).otherwise(pl.col(col)).alias(col))

    generation = generation if generation is not None else pl.read_parquet(PROCESSED_DIR / "generation_hourly.parquet")
    nuclear = generation.filter(pl.col("source") == "nuclear").select("subsystem", "ts_local", nuclear="gen_mw")
    nuclear_sin = nuclear.group_by("ts_local").agg(pl.col("nuclear").sum()).with_columns(subsystem=pl.lit("SIN"))
    nuclear = pl.concat([nuclear, nuclear_sin.select(nuclear.columns)])

    values = ["load", "hydro", "thermal_total", "wind", "solar", "interchange"]
    df = regularize(df, "ts_local", values, every="1h")
    df = fill_seasonal(df, values)
    df = (
        df.join(nuclear, on=["subsystem", "ts_local"], how="left")
        .with_columns(nuclear=pl.col("nuclear").fill_null(0.0))
        .with_columns(thermal=pl.col("thermal_total") - pl.col("nuclear"))
    )
    df = add_utc(df).select(
        "subsystem", "ts_local", "ts_utc", "load", "hydro", "thermal", "nuclear", "wind", "solar",
        "interchange", "is_imputed",
    ).sort("subsystem", "ts_local")
    return _write(df, "balance_hourly")


def build_hydro_daily() -> pl.DataFrame:
    ear_raw = _read_dataset("ear_daily")
    ear = ear_raw.select(
        subsystem(),
        date=pl.col("ear_data").cast(pl.Date),
        ear_max_mwmes=pl.col("ear_max_subsistema"),
        ear_mwmes=pl.col("ear_verif_subsistema_mwmes"),
        ear_pct=pl.col("ear_verif_subsistema_percentual"),
    )
    ena_raw = _read_dataset("ena_daily")
    ena = ena_raw.select(
        subsystem(),
        date=pl.col("ena_data").cast(pl.Date),
        ena_mwmed=pl.col("ena_bruta_regiao_mwmed"),
        ena_pct_mlt=pl.col("ena_bruta_regiao_percentualmlt"),
        ena_arm_mwmed=pl.col("ena_armazenavel_regiao_mwmed"),
        ena_arm_pct_mlt=pl.col("ena_armazenavel_regiao_percentualmlt"),
    )
    for d in (ear, ena):
        check_subsystems(d)
    ear = ear.unique(["subsystem", "date"], keep="last")
    ena = ena.unique(["subsystem", "date"], keep="last")
    df = ear.join(ena, on=["subsystem", "date"], how="full", coalesce=True)
    values = [c for c in df.columns if c not in ("subsystem", "date")]
    df = regularize(df, "date", values, every="1d", max_gap=3).sort("subsystem", "date")
    return _write(df, "hydro_daily")


def build_load_interruptions() -> pl.DataFrame:
    raw = _read_dataset("load_interruptions")
    df = raw.select(
        subsystem(),
        disturbance_id=pl.col("cod_perturbacao"),
        ts_local=pl.col("din_interrupcaocarga"),
        uf=pl.col("id_estado"),
        interrupted_mw=pl.col("val_cargainterrompida_mw"),
        restoration_min=pl.col("val_tempomedio_minutos"),
        ens_mwh=pl.col("val_energianaosuprida_mwh"),
    ).sort("ts_local")
    check_subsystems(df)
    return _write(df, "load_interruptions")


_API_AREA = {"SECO": "SE", "S": "S", "NE": "NE", "N": "N", "SIN": "SIN"}


def _api_hourly(endpoint: str, area: str, value_cols: list[str]) -> pl.DataFrame:
    files = sorted((RAW_DIR / "api_carga" / endpoint / area).glob("*.parquet"))
    frames = [f for f in (pl.read_parquet(p) for p in files) if not f.is_empty()]
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="diagonal_relaxed").unique("din_referenciautc", keep="last")
    # Values are integrated over the half-hour ENDING at din_referenciautc; assign each to the
    # local hour in which the interval starts, then average the two halves.
    end = pl.col("din_referenciautc").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ", time_zone="UTC")
    local = utc_to_local_naive(end - pl.duration(minutes=30)).dt.truncate("1h")
    out = (
        df.with_columns(ts_local=local, *[pl.col(c).cast(pl.Float64) for c in value_cols])
        .group_by("ts_local")
        .agg([pl.col(c).mean() for c in value_cols])
        .with_columns(subsystem=pl.lit(_API_AREA.get(area, area)))
    )
    return regularize(out, "ts_local", value_cols, every="1h", max_gap=3).drop("is_imputed")


def build_load_api() -> pl.DataFrame:
    frames = []
    for area in _API_AREA:
        ver = _api_hourly("verified", area, ["val_cargaglobal", "val_cargammgd"])
        prog = _api_hourly("programmed", area, ["val_cargaglobalprogramada"])
        if ver.is_empty() or prog.is_empty():
            continue
        frames.append(ver.join(prog, on=["subsystem", "ts_local"], how="full", coalesce=True))
    df = pl.concat(frames, how="diagonal_relaxed").rename(
        {"val_cargaglobal": "verified_load", "val_cargammgd": "verified_mmgd", "val_cargaglobalprogramada": "programmed_load"}
    )
    return _write(df.sort("subsystem", "ts_local"), "ons_load_api_hourly")


def build_uf_load_share(year: int) -> pl.DataFrame:
    from ons_energy.calendar.holidays import UF_TO_SUBSYSTEM

    rows = []
    for uf, sub in UF_TO_SUBSYSTEM.items():
        files = sorted((RAW_DIR / "api_carga" / "verified" / uf).glob(f"{year}*.parquet"))
        frames = [f for f in (pl.read_parquet(p) for p in files) if not f.is_empty()]
        mean = pl.concat(frames, how="diagonal_relaxed")["val_cargaglobal"].cast(pl.Float64).mean() if frames else 0.0
        rows.append((sub, uf, float(mean or 0.0)))
    df = pl.DataFrame(rows, schema=["subsystem", "uf", "mean_load_mw"], orient="row").with_columns(
        share=pl.col("mean_load_mw") / pl.col("mean_load_mw").sum().over("subsystem")
    )
    return _write(df.sort("subsystem", "uf"), "uf_load_share")


def build_all(include_api: bool = True, include_weather: bool = True) -> dict[str, int]:
    from ons_energy.data.catalog import load_catalog

    out = {}
    out["load_hourly"] = build_load_hourly().height
    gen = build_generation_hourly()
    out["generation_hourly"] = gen.height
    out["balance_hourly"] = build_balance_hourly(gen).height
    out["hydro_daily"] = build_hydro_daily().height
    out["load_interruptions"] = build_load_interruptions().height
    if include_api:
        out["uf_load_share"] = build_uf_load_share(load_catalog().load_api.uf_share_year).height
        out["ons_load_api_hourly"] = build_load_api().height
    if include_weather:
        from ons_energy.weather.openmeteo import build_weather_hourly

        if (RAW_DIR / "open_meteo").exists():
            out["weather_hourly"] = build_weather_hourly().height
    (PROCESSED_DIR / "_built_at.txt").write_text(date.today().isoformat())
    return out
