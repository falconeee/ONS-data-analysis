"""Run one MSPC experiment (configs are built in the anomaly notebook with :func:`mspc_config`).

Pipeline: monitoring matrix -> residualizer (fitted on the reference) -> reference segments with
known events removed -> detector.fit -> detector.score on the monitored span -> persistence rule
-> event-based evaluation + contributions. Every detector goes through exactly this path.
"""

from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ons_energy.anomaly.base import Detector, persistent_alarm
from ons_energy.anomaly.deseason import make_residualizer
from ons_energy.anomaly.evaluate import evaluate_events, event_windows, normal_mask, summary
from ons_energy.calendar.events import all_events
from ons_energy.config import StrictModel
from ons_energy.features.mspc import daily_matrix, hourly_matrix, negligible_variables
from ons_energy.utils.logging import get_logger
from ons_energy.utils.seed import set_seed

log = get_logger(__name__)

HOURLY_EVENT_KINDS = ["blackout", "interruption"]
DAILY_EVENT_KINDS = ["hydro_crisis", "covid", "strike", "blackout", "methodology_break"]


class DetectorSpec(StrictModel):
    kind: Literal["pca", "dpca", "rpca", "mscvae"]
    params: dict[str, Any] = Field(default_factory=dict)


class MSPCConfig(StrictModel):
    name: str
    scale: Literal["hourly", "daily"]
    preprocess: Literal["raw", "calendar", "weekly_baseline"]
    preprocess_params: dict[str, Any] = Field(default_factory=dict)
    reference: list[tuple[date, date]]
    monitor: tuple[date, date | None]
    detector: DetectorSpec
    persistence: tuple[int, int] = (3, 4)  # alarm if >= m of the last n points exceed the limit
    event_kinds: list[str] | None = None
    min_interruption_mw: float = 1000.0
    exclude_events_from_reference: bool = True
    min_share: float = 0.01  # drop sources below this share of the subsystem load (reference)
    normal_margin: str = "7D"  # distance from any event for a point to count as "normal"
    event_min_len: str | None = None
    seed: int = 42


# Shared settings per scale. Hourly: two normal windows around the 2023-04 MMGD break (events are
# excluded). Daily: pre-COVID, pre-methodology-break normal operation.
HOURLY_BASE: dict[str, Any] = {
    "scale": "hourly",
    "reference": [(date(2022, 6, 1), date(2023, 4, 28)), (date(2023, 9, 1), date(2024, 8, 31))],
    "monitor": (date(2010, 1, 1), None),
    "persistence": (3, 4),
    "normal_margin": "2D",
}
DAILY_BASE: dict[str, Any] = {
    "scale": "daily",
    "reference": [(date(2016, 1, 1), date(2019, 12, 31))],
    "monitor": (date(2010, 1, 1), None),
    "persistence": (3, 5),
    "normal_margin": "30D",
}


def mspc_config(
    scale: Literal["hourly", "daily"], name: str, preprocess: str, kind: str, label: str | None = None, **params: Any
) -> MSPCConfig:
    """``MSPCConfig`` from the scale preset. ``label`` is the detector's display name; ``params`` go
    to the detector, except keys that are ``MSPCConfig`` fields (e.g. ``persistence``,
    ``reference``, ``preprocess_params``), which override the preset."""
    base = dict(HOURLY_BASE if scale == "hourly" else DAILY_BASE)
    if label:
        params["name"] = label
    for k in list(params):
        if k in MSPCConfig.model_fields and k not in ("name", "detector"):
            base[k] = params.pop(k)
    return MSPCConfig(name=name, preprocess=preprocess, detector=DetectorSpec(kind=kind, params=params), **base)


def make_detector(spec: DetectorSpec) -> Detector:
    p = dict(spec.params)
    if spec.kind == "pca":
        from ons_energy.anomaly.pca import PCAMonitor

        return PCAMonitor(**p)
    if spec.kind == "dpca":
        from ons_energy.anomaly.dpca import DPCAMonitor

        return DPCAMonitor(**p)
    if spec.kind == "rpca":
        from ons_energy.anomaly.rpca import MovingWindowPCA

        return MovingWindowPCA(**p)
    from ons_energy.anomaly.mscvae_adapter import MSCVAEDetector

    if "window_sizes" in p:
        p["window_sizes"] = tuple(p["window_sizes"])
    return MSCVAEDetector(**p)


def _segments(df: pd.DataFrame, drop: np.ndarray, min_len: int) -> list[pd.DataFrame]:
    """Contiguous runs of rows not in ``drop`` with at least ``min_len`` rows."""
    keep = ~drop
    run = np.cumsum(np.r_[True, keep[1:] != keep[:-1]])
    return [g for _, g in df[keep].groupby(run[keep]) if len(g) >= min_len]


def run_mspc(cfg: MSPCConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    hourly = cfg.scale == "hourly"
    freq = "h" if hourly else "D"
    m_start, m_end = cfg.monitor
    first = min([cfg.reference[0][0], m_start])
    warm = pd.Timedelta(days=35)  # history for weekly baselines / lags
    start = (pd.Timestamp(first) - warm).strftime("%Y-%m-%d")
    end = (pd.Timestamp(m_end) + pd.Timedelta(hours=23)).strftime("%Y-%m-%d %H:%M:%S") if m_end else None
    X = hourly_matrix(start, end) if hourly else daily_matrix(start, end)

    events = all_events(cfg.min_interruption_mw).to_pandas()
    kinds = cfg.event_kinds or (HOURLY_EVENT_KINDS if hourly else DAILY_EVENT_KINDS)
    min_len = cfg.event_min_len or ("6h" if hourly else "1D")
    windows_all = event_windows(events, freq, kinds + ["blackout", "interruption", "covid", "strike"], X.index[0], X.index[-1], min_len)

    ref_masks = [(X.index >= pd.Timestamp(a)) & (X.index <= pd.Timestamp(b) + pd.Timedelta(hours=23)) for a, b in cfg.reference]
    dropped = negligible_variables(pd.concat([X[m] for m in ref_masks]), cfg.min_share)
    X = X.drop(columns=dropped)
    log.info("%s: dropped negligible variables %s", cfg.name, dropped)
    ref_raw = [X[m] for m in ref_masks]
    resid = make_residualizer(cfg.preprocess, hourly, **cfg.preprocess_params).fit(ref_raw).transform(X)

    event_rows = np.zeros(len(X), bool)
    if cfg.exclude_events_from_reference:
        for w in windows_all:
            event_rows |= (X.index >= w.start - pd.Timedelta("1D")) & (X.index <= w.end + pd.Timedelta("1D"))
    reference = []
    for m in ref_masks:
        seg_drop = ~m | event_rows | resid.isna().any(axis=1).to_numpy()
        reference += _segments(resid, seg_drop, min_len=48 if hourly else 30)
    log.info("%s: %d reference segments, %d rows", cfg.name, len(reference), sum(len(r) for r in reference))

    det = make_detector(cfg.detector).fit(reference)
    mon = resid[resid.index >= pd.Timestamp(m_start)]
    scores = det.score(mon)
    alarm = persistent_alarm(scores["alarm_raw"], *cfg.persistence)

    windows = [w for w in event_windows(events, freq, kinds, m_start, mon.index[-1], min_len)]
    ev = evaluate_events(scores, alarm, windows, det, mon)
    normal = normal_mask(mon.index, windows_all, cfg.reference, cfg.normal_margin)
    summ = {"name": cfg.name, "detector": det.name, "scale": cfg.scale, "preprocess": cfg.preprocess,
            **summary(scores, alarm, normal, ev), "n_vars": X.shape[1], "dropped": ";".join(dropped)}
    if hasattr(det, "ref_alarm_rate_"):
        summ["ref_alarm_rate"] = det.ref_alarm_rate_
    if hasattr(det, "k_"):
        summ["k"] = det.k_
    if hasattr(det, "threshold_in_sample_"):
        # FAR the author's in-sample SPOT threshold would have produced
        raw_is = (scores["phi"] > det.threshold_in_sample_).fillna(False)
        summ["far_in_sample_threshold"] = float(persistent_alarm(raw_is, *cfg.persistence)[normal & scores["ratio"].notna().to_numpy()].mean())

    scores = scores.assign(alarm=alarm)
    contrib = _event_contributions(det, mon, scores, alarm, windows)
    return {"summary": summ, "events": ev, "scores": scores, "alarm": alarm, "contributions": contrib, "detector": det, "data": mon}


def _event_contributions(det, data, scores, alarm, windows) -> pd.DataFrame:
    """Variable contributions per event window (alarmed points if any, else the whole window)."""
    rows = []
    for w in windows:
        sel = (scores.index >= w.start) & (scores.index <= w.end) & scores["ratio"].notna().to_numpy()
        if not sel.any():
            continue
        mask = sel & alarm.to_numpy() if (sel & alarm.to_numpy()).any() else sel
        try:
            c = det.contributions(data, mask)
        except Exception as exc:
            log.warning("contributions failed for %s: %s", w.event_id, exc)
            continue
        rows.append(c.assign(event_id=w.event_id))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["variable", "subsystem", "score", "pct", "event_id"])


def matched_operating_point(cfg: MSPCConfig, scores: pd.DataFrame, target_far: float = 0.05) -> dict[str, float]:
    """Detection rate when each detector's threshold on ``ratio`` is moved so that its persistent
    false-alarm rate on normal periods equals ``target_far`` (operating-point-free comparison).
    Uses the scores returned by :func:`run_mspc`; nothing is refitted."""
    hourly = cfg.scale == "hourly"
    freq = "h" if hourly else "D"
    events = all_events(cfg.min_interruption_mw).to_pandas()
    kinds = cfg.event_kinds or (HOURLY_EVENT_KINDS if hourly else DAILY_EVENT_KINDS)
    min_len = cfg.event_min_len or ("6h" if hourly else "1D")
    idx = scores.index
    windows_all = event_windows(events, freq, kinds + ["blackout", "interruption", "covid", "strike"], idx[0], idx[-1], min_len)
    windows = event_windows(events, freq, kinds, cfg.monitor[0], idx[-1], min_len)
    normal = normal_mask(idx, windows_all, cfg.reference, cfg.normal_margin) & scores["ratio"].notna().to_numpy()
    ratio = scores["ratio"]

    def far_at(tau: float) -> tuple[float, pd.Series]:
        al = persistent_alarm((ratio > tau).fillna(False), *cfg.persistence)
        return float(al[normal].mean()), al

    lo, hi = float(np.nanquantile(ratio[normal], 0.5)), float(np.nanmax(ratio[normal]))
    for _ in range(50):  # bisection on the threshold (FAR decreases with tau)
        mid = (lo + hi) / 2
        far, _ = far_at(mid)
        lo, hi = (mid, hi) if far > target_far else (lo, mid)
    far, al = far_at(hi)
    hits = [bool(al[(idx >= w.start) & (idx <= w.end)].any()) for w in windows]
    curated = [h for h, w in zip(hits, windows) if w.kind != "interruption"]
    return {"tau": hi, "far": far, "detection_rate": float(np.mean(hits)) if hits else np.nan, "n_events": len(hits),
            "curated_detection_rate": float(np.mean(curated)) if curated else np.nan, "n_curated": len(curated)}
