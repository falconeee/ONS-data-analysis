"""Low-level cleaning helpers shared by the table builders."""

from pathlib import Path

import polars as pl

TZ = "America/Sao_Paulo"
VALID_SUBSYSTEMS = {"SE", "S", "NE", "N", "SIN"}
_SUBSYSTEM_ALIASES = {"SECO": "SE", "SUDESTE": "SE", "SUL": "S", "NORDESTE": "NE", "NORTE": "N"}


def read_raw(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Read an ONS raw file. CSVs are ``;``-separated and read as strings (coerced later)."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        return pl.read_csv(path, separator=";", infer_schema=False, encoding="utf8-lossy", columns=columns)
    raise ValueError(f"Unsupported raw format: {path}")


def to_float(name: str, dtype: pl.DataType) -> pl.Expr:
    """Strict numeric coercion: decimal strings ('0E-8', '5337.70000000') and '' -> null."""
    col = pl.col(name)
    if dtype == pl.String:
        s = col.str.strip_chars()
        return pl.when(s == "").then(None).otherwise(s).cast(pl.Float64, strict=True).alias(name)
    return col.cast(pl.Float64).alias(name)


def to_datetime(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if dtype == pl.String:
        s = col.str.strip_chars()
        return s.str.to_datetime(strict=True, time_unit="us").alias(name)
    if dtype == pl.Date:
        return col.cast(pl.Datetime("us")).alias(name)
    return col.cast(pl.Datetime("us")).alias(name)


def to_date(name: str, dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.String:
        return pl.col(name).str.strip_chars().str.slice(0, 10).str.to_date("%Y-%m-%d").alias(name)
    return pl.col(name).cast(pl.Date).alias(name)


def subsystem(name: str = "id_subsistema") -> pl.Expr:
    """Normalize from the *id* column: strip spaces, upper-case, map aliases (SECO -> SE)."""
    s = pl.col(name).cast(pl.String).str.strip_chars().str.to_uppercase()
    return s.replace(_SUBSYSTEM_ALIASES).alias("subsystem")


def check_subsystems(df: pl.DataFrame, col: str = "subsystem") -> None:
    found = set(df[col].unique().to_list())
    unknown = found - VALID_SUBSYSTEMS
    if unknown:
        raise ValueError(f"Unknown subsystem codes {unknown}")


def _null_run_length(s: pl.Expr) -> pl.Expr:
    """Length of the run of consecutive nulls each row belongs to (0 for non-null rows)."""
    is_null = s.is_null()
    run_id = (is_null != is_null.shift(1)).fill_null(True).cum_sum()
    return pl.when(is_null).then(pl.len().over(run_id)).otherwise(0)


def regularize(
    df: pl.DataFrame,
    time_col: str,
    values: list[str],
    every: str,
    key: str = "subsystem",
    max_gap: int = 6,
) -> pl.DataFrame:
    """Reindex each ``key`` group to a regular grid and fill short gaps by linear interpolation.

    Rows that were absent or had any null value get ``is_imputed = True``. Gaps longer than
    ``max_gap`` steps stay null (callers decide what to do). For hourly local-time ONS data this
    also creates the missing spring-forward DST hour.
    """
    out = []
    for k, g in df.group_by(key, maintain_order=True):
        g = g.sort(time_col).unique(time_col, keep="last", maintain_order=True)
        grid = pl.datetime_range(g[time_col].min(), g[time_col].max(), interval=every, eager=True)
        if df.schema[time_col] == pl.Date:
            grid = pl.date_range(g[time_col].min(), g[time_col].max(), interval=every, eager=True)
        full = pl.DataFrame({time_col: grid}).join(g, on=time_col, how="left").with_columns(pl.lit(k[0]).alias(key))
        imputed = pl.any_horizontal([pl.col(v).is_null() for v in values])
        filled = [
            pl.when(_null_run_length(pl.col(v)) <= max_gap).then(pl.col(v).interpolate()).otherwise(None).alias(v)
            for v in values
        ]
        out.append(full.with_columns(is_imputed=imputed).with_columns(filled))
    return pl.concat(out).select(key, time_col, *values, "is_imputed")


def add_utc(df: pl.DataFrame, local_col: str = "ts_local") -> pl.DataFrame:
    """UTC timestamp from the published local wall clock.

    The spring-forward hour (non-existent locally, imputed by :func:`regularize`) gets the UTC
    instant of the next hour; the merged fall-back hour is mapped to its first occurrence.
    """
    utc = (
        pl.col(local_col)
        .dt.replace_time_zone(TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
    )
    nxt = (pl.col(local_col) + pl.duration(hours=1)).dt.replace_time_zone(
        TZ, ambiguous="earliest", non_existent="null"
    ).dt.convert_time_zone("UTC")
    return df.with_columns(ts_utc=pl.coalesce(utc, nxt).cast(pl.Datetime("us", "UTC")))


def utc_to_local_naive(expr: pl.Expr) -> pl.Expr:
    """UTC instant -> Brasília wall clock (naive), the convention of the ONS hourly files."""
    return expr.dt.convert_time_zone(TZ).dt.replace_time_zone(None)


def fill_seasonal(df: pl.DataFrame, values: list[str], period: int = 168, key: str = "subsystem", time_col: str = "ts_local") -> pl.DataFrame:
    """Fill gaps left by :func:`regularize` (whole missing days) with the value ``period`` steps
    earlier (same hour last week), repeatedly for multi-week gaps. Expects a regular grid; marks
    filled rows ``is_imputed``."""
    df = df.sort(key, time_col)
    filled = df
    for _ in range(4):
        filled = filled.with_columns([pl.col(v).fill_null(pl.col(v).shift(period).over(key)) for v in values])
    was_null = pl.any_horizontal([df[v].is_null() for v in values])
    return filled.with_columns(is_imputed=pl.col("is_imputed") | was_null)
