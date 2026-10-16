"""Hourly monitoring matrix for Module 2: one column per ``<subsystem>_<quantity>``.

Per subsystem: load, hydro, thermal (ex-nuclear), wind, solar and net interchange, plus SE nuclear.
"""

import pandas as pd
import polars as pl

from ons_energy.utils import PROCESSED_DIR

SUBSYSTEMS = ["SE", "S", "NE", "N"]
HOURLY_VARS = ["load", "hydro", "thermal", "wind", "solar", "interchange"]

DESCRIPTIONS = {
    "load": "load (MWmed)", "hydro": "hydro generation", "thermal": "thermal generation (ex-nuclear)",
    "wind": "wind generation", "solar": "solar generation", "interchange": "net interchange (export > 0)",
    "nuclear": "nuclear generation (Angra)",
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


def weekly_residuals(X: pd.DataFrame, n_weeks: int = 4) -> pd.DataFrame:
    """Causal deseasonalization: x_t minus the mean of the same hour in the previous ``n_weeks``
    weeks. Removes daily/weekly cycles and follows growth and level shifts within weeks, so PCA
    sees (roughly) stationary data; the first ``n_weeks`` weeks are NaN."""
    return X - sum(X.shift(168 * k) for k in range(1, n_weeks + 1)) / n_weeks


def variable_table(columns: list[str]) -> pd.DataFrame:
    """MSCVAE ``df_sistema``: VARIAVEL, DESC, SISTEMA (subsystem)."""
    rows = []
    for c in columns:
        sub, qty = c.split("_", 1)
        rows.append({"VARIAVEL": c, "DESC": f"{sub} {DESCRIPTIONS.get(qty, qty)}", "SISTEMA": sub})
    return pd.DataFrame(rows)


def negligible_variables(X: pd.DataFrame, min_share: float) -> list[str]:
    """Generation/interchange columns whose mean |level| is below ``min_share`` of the subsystem
    load in ``X`` (e.g. SE wind before 2025). Their tiny reference variance makes any later growth
    look like a fault, so they are left out of the monitoring matrix."""
    out = []
    for c in X.columns:
        sub, qty = c.split("_", 1)
        if qty == "load" or f"{sub}_load" not in X:
            continue
        if X[c].abs().mean() < min_share * X[f"{sub}_load"].mean():
            out.append(c)
    return out
