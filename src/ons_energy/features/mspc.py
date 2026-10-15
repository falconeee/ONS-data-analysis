"""Monitoring matrices for Module 2 (one column per ``<subsystem>_<quantity>``).

hourly: load, hydro, thermal (ex-nuclear), wind, solar, net interchange per subsystem + nuclear (SE)
daily : daily means of the hourly variables + EAR (% of max) and ENA (% of long-term mean)

EAR/ENA are daily, so they only enter the daily model (no artificial intra-day variability).
"""

import pandas as pd
import polars as pl

from ons_energy.utils.io import PROCESSED_DIR

SUBSYSTEMS = ["SE", "S", "NE", "N"]
HOURLY_VARS = ["load", "hydro", "thermal", "wind", "solar", "interchange"]
DAILY_HYDRO = {"ear_pct": "ear", "ena_pct_mlt": "ena"}

DESCRIPTIONS = {
    "load": "load (MWmed)", "hydro": "hydro generation", "thermal": "thermal generation (ex-nuclear)",
    "wind": "wind generation", "solar": "solar generation", "interchange": "net interchange (export > 0)",
    "nuclear": "nuclear generation (Angra)", "ear": "stored energy EAR (% max)", "ena": "natural inflow ENA (% MLT)",
}


def hourly_matrix(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    bal = pl.read_parquet(PROCESSED_DIR / "balance_hourly.parquet").filter(pl.col("subsystem").is_in(SUBSYSTEMS))
    if start:
        bal = bal.filter(pl.col("ts_local") >= pl.lit(start).str.to_datetime())
    if end:
        bal = bal.filter(pl.col("ts_local") <= pl.lit(end).str.to_datetime())
    wide = bal.pivot(on="subsystem", index="ts_local", values=HOURLY_VARS)
    # polars names pivoted columns "<value>_<subsystem>"; rename to "<subsystem>_<value>"
    ren = {f"{v}_{s}": f"{s}_{v}" for v in HOURLY_VARS for s in SUBSYSTEMS}
    nuc = bal.filter(pl.col("subsystem") == "SE").select("ts_local", SE_nuclear="nuclear")
    wide = wide.rename(ren).join(nuc, on="ts_local", how="left").sort("ts_local")
    cols = [f"{s}_{v}" for s in SUBSYSTEMS for v in HOURLY_VARS] + ["SE_nuclear"]
    return wide.to_pandas().set_index("ts_local")[cols].asfreq("h")


def daily_matrix(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    hourly = hourly_matrix(start, end)
    daily = hourly.resample("D").mean()
    hydro = pl.read_parquet(PROCESSED_DIR / "hydro_daily.parquet").filter(pl.col("subsystem").is_in(SUBSYSTEMS))
    hw = hydro.pivot(on="subsystem", index="date", values=list(DAILY_HYDRO)).to_pandas()
    hw["date"] = pd.to_datetime(hw["date"])
    hw = hw.set_index("date")
    hw.columns = [f"{c.rsplit('_', 1)[1]}_{DAILY_HYDRO[c.rsplit('_', 1)[0]]}" for c in hw.columns]
    out = daily.join(hw, how="left")
    order = [f"{s}_{v}" for s in SUBSYSTEMS for v in HOURLY_VARS + list(DAILY_HYDRO.values())] + ["SE_nuclear"]
    return out[order].asfreq("D")


def variable_table(columns: list[str]) -> pd.DataFrame:
    """MSCVAE ``df_sistema``: VARIAVEL, DESC, SISTEMA (subsystem)."""
    rows = []
    for c in columns:
        sub, qty = c.split("_", 1)
        rows.append({"VARIAVEL": c, "DESC": f"{sub} {DESCRIPTIONS.get(qty, qty)}", "SISTEMA": sub})
    return pd.DataFrame(rows)


def negligible_variables(X: pd.DataFrame, min_share: float) -> list[str]:
    """Generation/interchange columns whose mean |level| is below ``min_share`` of the subsystem
    load in ``X`` (e.g. SE wind: ~0.02% of SE load in 2016-19). Their tiny reference variance makes
    any later growth look like a fault, so they are left out of the monitoring matrix."""
    out = []
    for c in X.columns:
        sub, qty = c.split("_", 1)
        if qty in ("load", "ear", "ena") or f"{sub}_load" not in X:
            continue
        if X[c].abs().mean() < min_share * X[f"{sub}_load"].mean():
            out.append(c)
    return out
