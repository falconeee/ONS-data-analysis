"""Shared matplotlib style for the notebook figures (light surface).

Palette: the validated reference categorical palette, used in FIXED slot order and assigned per
entity (a model / detector keeps its color in every figure). One y-axis per panel; two measures of
different scale go to separate panels. Text uses ink tokens, never series colors.
"""

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_LIGHT = "#cde2fb"  # blue ramp step 100 (bands)
SEQ_MID = "#86b6ef"  # blue ramp step 250
EVENT_FILL = "#f0efec"  # neutral gray (diverging midpoint) for event shading

# Fixed entity -> slot assignment (never by rank)
MODEL_COLORS = {
    "ONS": SERIES[0], "LightGBM": SERIES[1], "TFT": SERIES[2], "NHITS": SERIES[3],
    "NBEATSx": SERIES[4], "NBEATS_I": SERIES[5], "SeasonalNaive": SERIES[6], "ONS_adj": SERIES[7],
}
DETECTOR_COLORS = {"PCA": SERIES[0], "MSCVAE": SERIES[1]}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
    })


def detector_color(name: str) -> str:
    for key in ("MSCVAE", "PCA"):
        if name.startswith(key):
            return DETECTOR_COLORS[key]
    return INK_2

