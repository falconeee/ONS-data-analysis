"""One MSPC experiment: a detector (PCA or MSCVAE) on the hourly matrix, evaluated on known events.

Pipeline, identical for both detectors: hourly matrix -> drop negligible sources -> weekly-baseline
residuals -> reference segments without known events -> fit -> score from ``monitor_start`` ->
persistence rule (m of the last n hours) -> per-event detection, delay and contributions, and the
false-alarm rate (FAR) on normal hours (outside the reference and away from every known event).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ons_energy.anomaly.base import Detector, persistent_alarm
from ons_energy.calendar.events import all_events
from ons_energy.features.mspc import hourly_matrix, negligible_variables, weekly_residuals
from ons_energy.utils import StrictModel, get_logger, set_seed

log = get_logger(__name__)

EVENT_KINDS = ["blackout", "interruption"]  # fast events the detectors should catch
NOT_NORMAL_KINDS = EVENT_KINDS + ["covid", "strike"]  # also kept out of the reference and of the FAR
MIN_EVENT_LEN = pd.Timedelta("6h")


class MSPCConfig(StrictModel):
    name: str
    detector: Literal["pca", "mscvae"]
    params: dict[str, Any] = Field(default_factory=dict)  # passed to PCAMonitor / MSCVAEDetector
    # two normal windows around the 2023-04 MMGD methodology break (known events are removed)
    reference: list[tuple[date, date]] = [(date(2022, 6, 1), date(2023, 4, 28)), (date(2023, 9, 1), date(2024, 8, 31))]
    monitor_start: date = date(2010, 1, 1)
    persistence: tuple[int, int] = (3, 4)  # alarm if >= m of the last n hours exceed the limit
    min_interruption_mw: float = 1000.0  # load interruptions at least this large are events
    min_share: float = 0.01  # drop sources below this share of the subsystem load (reference)
    normal_margin: str = "2D"  # distance from any event for an hour to count as "normal"
    seed: int = 42


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    subsystems: tuple[str, ...]


def make_detector(cfg: MSPCConfig) -> Detector:
    if cfg.detector == "pca":
        from ons_energy.anomaly.pca import PCAMonitor

        return PCAMonitor(**cfg.params)
    from ons_energy.anomaly.mscvae_adapter import MSCVAEDetector

    return MSCVAEDetector(**cfg.params)


def event_windows(min_mw: float, kinds: list[str], start, end) -> list[EventWindow]:
    """Events of ``kinds`` overlapping [start, end], on the hourly grid, lasting at least 6 h."""
    out = []
    for r in all_events(min_mw).to_pandas().itertuples():
        s, e = pd.Timestamp(r.start), pd.Timestamp(r.end)
        if r.kind in kinds and e >= pd.Timestamp(start) and s <= pd.Timestamp(end):
            s = s.floor("h")
            out.append(EventWindow(r.event_id, r.kind, s, max(e.ceil("h"), s + MIN_EVENT_LEN), tuple(r.subsystems)))
    return out


def _in_windows(index: pd.DatetimeIndex, windows: list[EventWindow], margin: pd.Timedelta) -> np.ndarray:
    hit = np.zeros(len(index), bool)
    for w in windows:
        hit |= (index >= w.start - margin) & (index <= w.end + margin)
    return hit


def _in_reference(index: pd.DatetimeIndex, reference: list[tuple[date, date]]) -> list[np.ndarray]:
    return [(index >= pd.Timestamp(a)) & (index < pd.Timestamp(b) + pd.Timedelta("1D")) for a, b in reference]


def _segments(df: pd.DataFrame, keep: np.ndarray, min_len: int = 48) -> list[pd.DataFrame]:
    """Contiguous runs of kept rows with at least ``min_len`` rows."""
    run = np.cumsum(np.r_[True, keep[1:] != keep[:-1]])
    return [g for _, g in df[keep].groupby(run[keep]) if len(g) >= min_len]


def evaluate_events(
    scores: pd.DataFrame, windows: list[EventWindow], detector: Detector, data: pd.DataFrame, top: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per event: detected?, delay of the first persistent alarm, max statistic/limit, and the
    variable contributions over the alarmed hours (all event hours if none alarmed)."""
    idx, alarm = scores.index, scores["alarm"].to_numpy()
    rows, contribs = [], []
    for w in windows:
        sel = (idx >= w.start) & (idx <= w.end)
        hits = sel & alarm
        valid = sel & scores["ratio"].notna().to_numpy()
        mask = valid & alarm if (valid & alarm).any() else valid
        contrib = None
        if mask.any():
            try:
                contrib = detector.contributions(data, mask)
                contribs.append(contrib.assign(event_id=w.event_id))
            except Exception as exc:  # contributions are diagnostic only
                log.warning("contributions failed for %s: %s", w.event_id, exc)
        rows.append({
            "event_id": w.event_id, "kind": w.kind, "start": w.start, "end": w.end,
            "subsystems": ";".join(w.subsystems), "detected": bool(hits.any()),
            "delay_h": (idx[hits][0] - w.start) / pd.Timedelta(hours=1) if hits.any() else np.nan,
            "max_ratio": float(scores.loc[sel, "ratio"].max()) if sel.any() else np.nan,
            "alarm_share": float(alarm[sel].mean()) if sel.any() else np.nan,
            "top_contributors": "; ".join(f"{r.variable} ({r.pct:.0f}%)" for r in contrib.head(top).itertuples()) if contrib is not None else "",
            "top_subsystem_match": bool(contrib.iloc[0]["subsystem"] in w.subsystems) if contrib is not None and len(contrib) else np.nan,
        })
    columns = ["variable", "subsystem", "score", "pct", "event_id"]
    return pd.DataFrame(rows), (pd.concat(contribs, ignore_index=True) if contribs else pd.DataFrame(columns=columns))


def run_mspc(cfg: MSPCConfig) -> dict[str, Any]:
    """Fit on the reference, score from ``monitor_start`` and evaluate. Returns summary, events,
    scores (with persistent ``alarm``), contributions, detector, data and the normal-hours mask."""
    set_seed(cfg.seed)
    first = min(cfg.reference[0][0], cfg.monitor_start)
    X = hourly_matrix((pd.Timestamp(first) - pd.Timedelta(days=35)).strftime("%Y-%m-%d"))  # 4-week warm-up

    ref_masks = _in_reference(X.index, cfg.reference)
    dropped = negligible_variables(pd.concat([X[m] for m in ref_masks]), cfg.min_share)
    log.info("%s: dropped negligible variables %s", cfg.name, dropped)
    resid = weekly_residuals(X.drop(columns=dropped))

    not_normal = event_windows(cfg.min_interruption_mw, NOT_NORMAL_KINDS, resid.index[0], resid.index[-1])
    near_event = _in_windows(resid.index, not_normal, pd.Timedelta("1D"))
    complete = resid.notna().all(axis=1).to_numpy()
    reference = [seg for m in ref_masks for seg in _segments(resid, m & ~near_event & complete)]
    log.info("%s: %d reference segments, %d hours", cfg.name, len(reference), sum(len(r) for r in reference))

    det = make_detector(cfg).fit(reference)
    data = resid[resid.index >= pd.Timestamp(cfg.monitor_start)]
    scores = det.score(data)
    scores["alarm"] = persistent_alarm(scores["alarm_raw"], *cfg.persistence)

    windows = [w for w in not_normal if w.kind in EVENT_KINDS and w.end >= data.index[0]]
    events, contrib = evaluate_events(scores, windows, det, data)
    normal = (
        ~_in_windows(data.index, not_normal, pd.Timedelta(cfg.normal_margin))
        & ~np.any(_in_reference(data.index, cfg.reference), axis=0)
        & scores["ratio"].notna().to_numpy()
    )
    summ = {
        "name": cfg.name, "detector": det.name, "n_vars": len(det.variables_),
        "far": float(scores["alarm"][normal].mean()),
        "detection_rate": float(events["detected"].mean()) if len(events) else np.nan,
        "median_delay_h": float(events["delay_h"].median()) if events["detected"].any() else np.nan,
        "n_events": len(events),
        "top_subsystem_match": float(events["top_subsystem_match"].mean()) if len(events) else np.nan,
        "n_normal_hours": int(normal.sum()),
    }
    if hasattr(det, "ref_alarm_rate_"):  # PCA
        summ |= {"k": det.k_, "ref_alarm_rate": det.ref_alarm_rate_}
    if hasattr(det, "threshold_in_sample_"):  # MSCVAE: FAR the author's in-sample SPOT threshold would give
        raw = (scores["phi"] > det.threshold_in_sample_).fillna(False)
        summ["far_in_sample_threshold"] = float(persistent_alarm(raw, *cfg.persistence)[normal].mean())
    return {"config": cfg, "summary": summ, "events": events, "scores": scores, "contributions": contrib,
            "detector": det, "data": data, "normal": normal, "windows": windows}


def matched_operating_point(result: dict[str, Any], target_far: float = 0.05) -> dict[str, float]:
    """Detection rate when the threshold on ``ratio`` is moved so that the persistent FAR on normal
    hours equals ``target_far``: compares detectors at the same operating point (nothing refitted)."""
    scores, normal, windows = result["scores"], result["normal"], result["windows"]
    persistence, ratio, idx = result["config"].persistence, result["scores"]["ratio"], result["scores"].index

    def alarms(tau: float) -> pd.Series:
        return persistent_alarm((ratio > tau).fillna(False), *persistence)

    lo, hi = float(np.nanquantile(ratio[normal], 0.5)), float(np.nanmax(ratio[normal]))
    for _ in range(50):  # bisection on the threshold (FAR decreases with tau)
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if alarms(mid)[normal].mean() > target_far else (lo, mid)
    al = alarms(hi)
    hits = [bool(al[(idx >= w.start) & (idx <= w.end)].any()) for w in windows]
    curated = [h for h, w in zip(hits, windows) if w.kind != "interruption"]
    return {"detector": result["summary"]["detector"], "tau": hi, "far": float(al[normal].mean()),
            "detection_rate": float(np.mean(hits)) if hits else np.nan, "n_events": len(hits),
            "blackout_detection_rate": float(np.mean(curated)) if curated else np.nan, "n_blackouts": len(curated)}
