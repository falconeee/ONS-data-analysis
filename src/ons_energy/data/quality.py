"""Data-quality checks behind ``notebooks/03_data_quality.ipynb``."""

import numpy as np
import pandas as pd
import polars as pl

from ons_energy.data.download import latest_entries, read_manifest
from ons_energy.utils import PROCESSED_DIR


def _read(name: str) -> pl.DataFrame:
    return pl.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def manifest_summary() -> pd.DataFrame:
    m = latest_entries(read_manifest())
    return (
        m.group_by("dataset")
        .agg(files=pl.len(), mb=(pl.col("bytes").sum() / 1e6).round(1), first_year=pl.col("year").min(),
             last_year=pl.col("year").max(), last_fetch=pl.col("fetched_at").max())
        .sort("dataset").to_pandas()
    )


def imputation_summary() -> pd.DataFrame:
    rows = []
    for name in ("load_hourly", "balance_hourly", "hydro_daily"):
        df = _read(name)
        g = df.group_by("subsystem").agg(rows=pl.len(), imputed=pl.col("is_imputed").sum())
        for r in g.iter_rows(named=True):
            rows.append({"table": name, **r, "imputed_pct": round(100 * r["imputed"] / r["rows"], 4)})
    return pd.DataFrame(rows).sort_values(["table", "subsystem"])


def dst_days() -> pd.DataFrame:
    """Days whose raw hourly file had 23 rows (spring-forward) = imputed midnight hours."""
    load = _read("load_hourly").filter((pl.col("subsystem") == "SE") & pl.col("is_imputed"))
    dst = load.filter(pl.col("ts_local").dt.hour() == 0).with_columns(day=pl.col("ts_local").dt.date())
    return dst.group_by("day").len().sort("day").to_pandas()


def balance_vs_load() -> dict[str, float]:
    a = _read("load_hourly").select("subsystem", "ts_local", "load_mw")
    b = _read("balance_hourly").select("subsystem", "ts_local", "load")
    j = a.join(b, on=["subsystem", "ts_local"]).with_columns(d=(pl.col("load_mw") - pl.col("load")).abs())
    return {"mean_abs_diff_mw": j["d"].mean(), "share_gt_1mw": float((j["d"] > 1).mean())}


def reconciliation(year: int) -> pd.DataFrame:
    """Balance vs. sum of plant-level generation, by subsystem and source (MAE in MW)."""
    gen = _read("generation_hourly").filter(pl.col("ts_local").dt.year() == year)
    g = gen.pivot(on="source", index=["subsystem", "ts_local"], values="gen_mw").fill_null(0)
    b = _read("balance_hourly").filter((pl.col("ts_local").dt.year() == year) & (pl.col("subsystem") != "SIN"))
    j = b.join(g, on=["subsystem", "ts_local"], suffix="_plant")
    rows = []
    for sub in ("SE", "S", "NE", "N"):
        x = j.filter(pl.col("subsystem") == sub)
        for src in ("hydro", "thermal", "wind", "solar", "nuclear"):
            plant = f"{src}_plant" if f"{src}_plant" in x.columns else None
            if plant is None:
                continue
            rows.append({"subsystem": sub, "source": src, "balance_mean": x[src].mean(),
                         "plants_mean": x[plant].mean(), "mae": (x[src] - x[plant]).abs().mean()})
    return pd.DataFrame(rows)


def thermal_includes_nuclear(year: int = 2025) -> dict[str, float]:
    """Compare the raw balance thermal (= thermal + nuclear here) with plant sums, both ways."""
    gen = _read("generation_hourly").filter((pl.col("ts_local").dt.year() == year) & (pl.col("subsystem") == "SE"))
    g = gen.pivot(on="source", index="ts_local", values="gen_mw").fill_null(0)
    b = _read("balance_hourly").filter((pl.col("ts_local").dt.year() == year) & (pl.col("subsystem") == "SE"))
    j = b.join(g, on="ts_local", suffix="_plant").with_columns(raw=pl.col("thermal") + pl.col("nuclear"))
    return {
        "mae_vs_thermal_plus_nuclear": (j["raw"] - j["thermal_plant"] - j["nuclear_plant"]).abs().mean(),
        "mae_vs_thermal_only": (j["raw"] - j["thermal_plant"]).abs().mean(),
    }


def api_alignment(subsystem: str = "SE", start: str = "2024-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Correlation and mean ratio of API verified load vs curva-carga for lags -2..2 h
    (checks the half-hour -> local-hour mapping)."""
    load = _read("load_hourly").filter(pl.col("subsystem") == subsystem).select("ts_local", "load_mw").to_pandas().set_index("ts_local")
    api = _read("ons_load_api_hourly").filter(pl.col("subsystem") == subsystem).select("ts_local", "verified_load").to_pandas().set_index("ts_local")
    j = load.join(api, how="inner").loc[start:end]
    rows = []
    for lag in range(-2, 3):
        s = j["verified_load"].shift(lag)
        rows.append({"lag_h": lag, "corr": j["load_mw"].corr(s), "mean_ratio": float((s / j["load_mw"]).mean())})
    return pd.DataFrame(rows)


def methodology_levels(subsystem: str = "SE") -> pd.DataFrame:
    """Monthly mean load around the breaks, plus the MMGD component reported by the API."""
    load = _read("load_hourly").filter(pl.col("subsystem") == subsystem)
    m = load.group_by_dynamic("ts_local", every="1mo").agg(load=pl.col("load_mw").mean())
    api = _read("ons_load_api_hourly").filter(pl.col("subsystem") == subsystem)
    a = api.group_by_dynamic("ts_local", every="1mo").agg(mmgd=pl.col("verified_mmgd").mean(), programmed=pl.col("programmed_load").mean())
    out = m.join(a, on="ts_local", how="left").filter(pl.col("ts_local").dt.year().is_between(2020, 2026)).to_pandas()
    return out


def programmed_vs_actual(subsystem: str = "SE", start: str = "2024-01-01") -> dict[str, float]:
    load = _read("load_hourly").filter(pl.col("subsystem") == subsystem).select("ts_local", "load_mw")
    api = _read("ons_load_api_hourly").filter(pl.col("subsystem") == subsystem).select("ts_local", "programmed_load")
    j = load.join(api, on="ts_local").drop_nulls().filter(pl.col("ts_local") >= pl.lit(start).str.to_datetime()).to_pandas()
    err = j.programmed_load - j.load_mw
    return {"bias_mw": float(err.mean()), "bias_pct": float(100 * (err / j.load_mw).mean()),
            "mape": float(100 * np.mean(np.abs(err / j.load_mw))), "n_hours": len(j)}
