"""Module 2 figures. Every function takes the outputs of ``run_mspc`` (scores / contributions /
summaries) and returns the figure, so the notebooks show them inline."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ons_energy.viz.style import DETECTOR_COLORS, EVENT_FILL, INK, INK_2, apply_style, detector_color


def control_chart(panels: list[tuple[pd.DataFrame, str, str]], start: str, end: str, event: tuple[str, str], title: str) -> plt.Figure:
    """Small multiples of (scores, statistic, label): statistic / limit (log), limit at 1, event shaded."""
    apply_style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.1 * len(panels) + 0.6), sharex=True)
    for ax, (scores, stat, label) in zip(np.atleast_1d(axes), panels):
        s = scores.loc[start:end]
        r = (s[stat] / s[f"{stat}_lim"]).clip(lower=1e-2)
        color = detector_color(label)
        ax.axvspan(pd.Timestamp(event[0]), pd.Timestamp(event[1]), color=EVENT_FILL, zorder=0)
        ax.plot(r.index, r, color=color, linewidth=1.4)
        alarm = s["alarm"].astype(bool)
        ax.scatter(r.index[alarm], r[alarm], s=10, color=color, zorder=3, edgecolor="#fcfcfb", linewidth=0.6)
        ax.axhline(1, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yscale("log")
        ax.set_ylabel("stat / limit")
        ax.set_title(label, fontsize=10)
    np.atleast_1d(axes)[-1].annotate("control limit = 1", (r.index[0], 1), xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def contributions(items: list[tuple[pd.DataFrame, str]], event_id: str, title: str, top: int = 8) -> plt.Figure:
    """Top variable contributions to one event, one panel per (contributions, label)."""
    apply_style()
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 0.38 * top + 1.4))
    for ax, (contrib, label) in zip(np.atleast_1d(axes), items):
        c = contrib[contrib.event_id == event_id].head(top)[::-1]
        bars = ax.barh(c.variable, c.pct, color=detector_color(label), height=0.6)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK_2, padding=2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("share of contribution (%)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def daily_crisis(scores: pd.DataFrame, label: str, ear: pd.Series, start: str, end: str, crisis: tuple[str, str]) -> plt.Figure:
    """Two panels sharing time (no dual axis): statistic/limit and SE EAR."""
    apply_style()
    s = scores.loc[start:end]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax in (a1, a2):
        ax.axvspan(pd.Timestamp(crisis[0]), pd.Timestamp(crisis[1]), color=EVENT_FILL, zorder=0)
    r = s["ratio"].clip(lower=1e-2)
    a1.plot(r.index, r, color=detector_color(label), linewidth=1.3)
    a1.axhline(1, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    a1.set_yscale("log")
    a1.set_ylabel("max(T², SPE) / limit")
    a1.set_title(f"{label} — daily model", fontsize=10)
    e = ear.loc[start:end]
    a2.plot(e.index, e, color=DETECTOR_COLORS["PCA"], linewidth=1.6)
    a2.set_ylabel("SE/CO EAR (% of max)")
    a2.annotate("2021 water crisis", (pd.Timestamp(crisis[0]), e.max()), fontsize=8, color=INK_2, va="top")
    fig.suptitle("Slow anomaly: 2021 water crisis (shaded)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def detector_tradeoff(summary: pd.DataFrame, title: str) -> plt.Figure:
    """FAR vs detection rate, one labelled point per detector."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in summary.itertuples():
        ax.scatter(100 * r.far, 100 * r.detection_rate, s=70, color=detector_color(r.detector), edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.annotate(r.detector, (100 * r.far, 100 * r.detection_rate), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("False-alarm rate on normal periods (%)  ← better")
    ax.set_ylabel("Events detected (%)  ↑ better")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def far_by_year(items: list[tuple[pd.DataFrame, str]], title: str) -> plt.Figure:
    """Share of points in alarm per year, one line per (scores, label)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for s, label in items:
        y = s.groupby(s.index.year)["alarm"].mean() * 100
        ax.plot(y.index, y.values, color=detector_color(label), marker="o", ms=5, linewidth=1.8)
        ax.annotate(label, (y.index[-1], y.values[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
    ax.set_ylabel("points in alarm (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
