"""Module 1 figures. Every function takes DataFrames (backtests / evaluation tables) and returns the
figure, so the notebooks show them inline."""

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from ons_energy.calendar.segments import DAY_TYPES
from ons_energy.utils import PROCESSED_DIR
from ons_energy.viz.style import EVENT_FILL, INK, INK_2, MODEL_COLORS, SEQ_LIGHT, SEQ_MID, apply_style


def load_history(subsystem: str = "SE") -> plt.Figure:
    """Weekly mean load 2000-2026 with the two methodology breaks."""
    apply_style()
    load = pl.read_parquet(PROCESSED_DIR / "load_hourly.parquet").filter(pl.col("subsystem") == subsystem)
    d = load.group_by_dynamic("ts_local", every="1w").agg(pl.col("load_mw").mean()).to_pandas()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(d.ts_local, d.load_mw / 1000, color=MODEL_COLORS["ONS"], linewidth=1.4)
    ax.axvspan(datetime(2001, 6, 1), datetime(2002, 2, 28), color=EVENT_FILL, zorder=0)
    ax.axvspan(datetime(2020, 3, 16), datetime(2020, 6, 30), color=EVENT_FILL, zorder=0)
    for when, label in ((datetime(2021, 3, 1), "Mar/2021: + non-dispatched\ngeneration forecast"),
                        (datetime(2023, 4, 29), "Apr/2023: + MMGD\n(distributed solar)")):
        ax.axvline(when, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.annotate(label, (when, ax.get_ylim()[0]), xytext=(-6, 8), textcoords="offset points", ha="right",
                    fontsize=8, color=INK_2)
    ax.annotate("2001 rationing", (datetime(2001, 10, 1), d.load_mw.max() / 1000), fontsize=8, color=INK_2, ha="center")
    ax.annotate("COVID-19", (datetime(2020, 5, 1), d.load_mw.max() / 1000), fontsize=8, color=INK_2, ha="center")
    ax.set_title(f"{subsystem} hourly load, weekly mean (GW) — ONS curva-carga")
    ax.set_ylabel("GW")
    fig.tight_layout()
    return fig


def actual_vs_forecast(
    cv: pd.DataFrame,
    models: list[str],
    start: str,
    days: int = 7,
    subsystem: str = "SE",
    band_model: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Hourly actual load vs day-ahead forecasts over ``days`` days, optional 80/90% band."""
    apply_style()
    t0 = pd.Timestamp(start)
    w = cv[(cv.unique_id == subsystem) & (cv.ds >= t0) & (cv.ds < t0 + pd.Timedelta(days=days))].sort_values("ds")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if band_model and f"{band_model}-q0.05" in w:
        ax.fill_between(w.ds, w[f"{band_model}-q0.05"] / 1000, w[f"{band_model}-q0.95"] / 1000, color=SEQ_LIGHT, linewidth=0, label=f"{band_model} 90% interval")
        ax.fill_between(w.ds, w[f"{band_model}-q0.1"] / 1000, w[f"{band_model}-q0.9"] / 1000, color=SEQ_MID, alpha=0.5, linewidth=0, label=f"{band_model} 80% interval")
    ax.plot(w.ds, w.y / 1000, color=INK, linewidth=2.2, label="Actual")
    for m in models:
        ax.plot(w.ds, w[m] / 1000, color=MODEL_COLORS.get(m, INK_2), linewidth=1.5, label=m)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax.set_ylabel("GW")
    ax.set_title(title or f"{subsystem}: actual vs day-ahead forecast from {t0:%d/%m/%Y}")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    return fig


def actual_vs_forecast_daily(cv: pd.DataFrame, models: list[str], subsystem: str = "SE", title: str | None = None) -> plt.Figure:
    """Whole evaluation period: daily mean actual vs forecast (top) and daily MAPE, 7-day mean (bottom)."""
    apply_style()
    g = cv[cv.unique_id == subsystem].assign(day=lambda d: d.ds.dt.normalize())
    daily = g.groupby("day")[["y", *models]].mean() / 1000
    ape = g[models].sub(g.y, axis=0).abs().div(g.y, axis=0).mul(100).assign(day=g.day).groupby("day").mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(daily.index, daily.y, color=INK, linewidth=1.8, label="Actual")
    for m in models:
        a1.plot(daily.index, daily[m], color=MODEL_COLORS.get(m, INK_2), linewidth=1.0, alpha=0.9, label=m)
        a2.plot(ape.index, ape[m].rolling(7, min_periods=1).mean(), color=MODEL_COLORS.get(m, INK_2), linewidth=1.3)
    a1.set_ylabel("daily mean (GW)")
    a1.legend(ncol=len(models) + 1, loc="upper left")
    a1.set_title(title or f"{subsystem}: actual vs day-ahead forecast, whole evaluation period")
    a2.set_ylabel("MAPE (%), 7-day mean")
    fig.tight_layout()
    return fig


def mape_by_day_type(seg: pd.DataFrame, models: list[str], title: str) -> plt.Figure:
    """Dot + 95% CI per model and day type; n_days printed next to each day type."""
    apply_style()
    seg = seg[seg.model.isin(models)].copy()
    types = [t for t in DAY_TYPES if t in set(seg.day_type)]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(types) + 1.4))
    offs = np.linspace(-0.25, 0.25, len(models))
    for off, m in zip(offs, models):
        g = seg[seg.model == m].set_index("day_type").reindex(types)
        y = np.arange(len(types)) + off
        ax.errorbar(g.MAPE, y, xerr=[g.MAPE - g.MAPE_lo, g.MAPE_hi - g.MAPE], fmt="o", ms=6, color=MODEL_COLORS.get(m, INK_2),
                    ecolor=MODEL_COLORS.get(m, INK_2), elinewidth=1.2, capsize=0, label=m)
    n = seg.groupby("day_type")["n_days"].max().reindex(types)
    ax.set_yticks(range(len(types)), [f"{t.replace('_', ' ')}  (n={int(k)})" for t, k in n.items()])
    ax.invert_yaxis()
    ax.set_xlabel("MAPE (%) with day-bootstrap 95% CI")
    ax.set_title(title)
    ax.legend(ncol=len(models), loc="upper left", bbox_to_anchor=(0, -0.08 - 0.3 / len(types)))
    fig.tight_layout()
    return fig


def mape_by_hour(by_step: pd.DataFrame, models: list[str], title: str) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for m in models:
        g = by_step[by_step.model == m].sort_values("step")
        hour = g.step - 13  # step 13 = 00:00 of day D
        ax.plot(hour, g.MAPE, color=MODEL_COLORS.get(m, INK_2), linewidth=2)
        ax.annotate(m, (hour.iloc[-1], g.MAPE.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of target day D (forecast issued D-1 12:00)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def tft_vsn(vsn: pd.DataFrame, top: int = 12) -> plt.Figure:
    apply_style()
    fut = vsn[vsn.block == "future"]
    order = fut.groupby("variable")["weight"].mean().sort_values(ascending=False).head(top).index[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * top + 1.2))
    y = np.arange(len(order))
    for off, (seg, color) in zip((-0.18, 0.18), (("normal", MODEL_COLORS["ONS"]), ("special", MODEL_COLORS["LightGBM"]))):
        g = fut[fut.segment == seg].set_index("variable").reindex(order)
        ax.barh(y + off, g.weight, height=0.34, color=color, label=f"{seg} days")
    ax.set_yticks(y, order)
    ax.set_xlabel("Mean variable-selection weight (decoder / future inputs)")
    ax.set_title("TFT: which future inputs matter — normal vs special days")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def nbeats_decomposition(dec: pd.DataFrame) -> plt.Figure:
    apply_style()
    days = dec.drop_duplicates("date")
    pick = [days[days.day_type == "normal"].date.iloc[0], days[days.day_type != "normal"].date.iloc[0]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, d in zip(axes, pick):
        g = dec[dec.date == d].sort_values("step")
        h = g.step - 13
        ax.plot(h, g.trend / 1000, color=MODEL_COLORS["ONS"], label="trend stack")
        ax.plot(h, g.seasonality / 1000, color=MODEL_COLORS["LightGBM"], label="seasonality stack")
        ax.set_title(f"{d} ({g.day_type.iloc[0].replace('_', ' ')})", fontsize=10)
        ax.set_xlabel("Hour of day D")
    axes[0].set_ylabel("GW (component)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Interpretable N-BEATS: forecast = trend + seasonality stacks", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def weather_ablation(tables: dict[str, pd.DataFrame], model: str = "LightGBM") -> plt.Figure:
    """MAPE per subsystem of ``model`` by weather scenario (``by_subsystem`` tables, grouped bars)."""
    apply_style()
    subs = ["SE", "S", "NE", "N"]
    colors = {"none": MODEL_COLORS["SeasonalNaive"], "perfect": MODEL_COLORS["ONS"], "honest": MODEL_COLORS["TFT"]}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(subs))
    for i, (scen, tab) in enumerate(tables.items()):
        g = tab[tab.model == model].set_index("unique_id").reindex(subs)
        bars = ax.bar(x + (i - 1) * 0.27, g.MAPE, width=0.25, color=colors[scen], label=f"weather: {scen}")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, subs)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"{model} day-ahead MAPE by subsystem and weather scenario")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    return fig
