"""Event-based evaluation of anomaly detectors (identical for every detector).

* false-alarm rate (FAR): share of "normal" monitored points (outside the reference period and away
  from every known event) in persistent alarm;
* per event: detected?, delay of the first persistent alarm, top contributing variables.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ons_energy.anomaly.base import Detector


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    subsystems: tuple[str, ...]
    description: str


def event_windows(events: pd.DataFrame, freq: str, kinds: list[str], start, end, min_len: str) -> list[EventWindow]:
    """Events of ``kinds`` inside the monitored span; each window lasts at least ``min_len``."""
    out = []
    for r in events.itertuples():
        if r.kind not in kinds:
            continue
        s, e = pd.Timestamp(r.start), pd.Timestamp(r.end)
        if e < pd.Timestamp(start) or s > pd.Timestamp(end):
            continue
        s = s.floor(freq)
        e = max(e.ceil(freq), s + pd.Timedelta(min_len))
        out.append(EventWindow(r.event_id, r.kind, s, e, tuple(r.subsystems), r.description))
    return out


def normal_mask(index: pd.DatetimeIndex, windows: list[EventWindow], reference: list[tuple], margin: str) -> np.ndarray:
    mask = np.ones(len(index), bool)
    m = pd.Timedelta(margin)
    for w in windows:
        mask &= ~((index >= w.start - m) & (index <= w.end + m))
    for a, b in reference:
        mask &= ~((index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b) + pd.Timedelta("1D")))
    return mask


def evaluate_events(
    scores: pd.DataFrame,
    alarm: pd.Series,
    windows: list[EventWindow],
    detector: Detector,
    data: pd.DataFrame,
    lead: str = "0h",
    top: int = 5,
) -> pd.DataFrame:
    rows = []
    for w in windows:
        sel = (scores.index >= w.start - pd.Timedelta(lead)) & (scores.index <= w.end)
        hits = alarm[sel]
        detected = bool(hits.any())
        delay = (hits.index[hits.to_numpy()][0] - w.start) if detected else pd.NaT
        in_event = sel & alarm.to_numpy() if detected else sel
        contrib = None
        try:
            if in_event.any() and scores.loc[in_event, "ratio"].notna().any():
                valid = in_event & scores["ratio"].notna().to_numpy()
                contrib = detector.contributions(data, valid)
        except Exception as exc:  # contributions are diagnostic only
            contrib = None
            print(f"contribution failed for {w.event_id}: {exc}")
        top_vars = "; ".join(f"{r.variable} ({r.pct:.0f}%)" for r in contrib.head(top).itertuples()) if contrib is not None else ""
        rows.append({
            "event_id": w.event_id, "kind": w.kind, "start": w.start, "end": w.end,
            "subsystems": ";".join(w.subsystems), "detected": detected,
            "delay_h": delay / pd.Timedelta(hours=1) if detected else np.nan,
            "max_ratio": float(scores.loc[sel, "ratio"].max()) if sel.any() else np.nan,
            "alarm_share": float(hits.mean()) if len(hits) else np.nan,
            "top_contributors": top_vars,
            "top_subsystem_match": _subsystem_match(contrib, w.subsystems) if contrib is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _subsystem_match(contrib: pd.DataFrame, subsystems: tuple[str, ...]) -> bool:
    return bool(contrib.iloc[0]["subsystem"] in subsystems) if len(contrib) else False


def summary(scores: pd.DataFrame, alarm: pd.Series, normal: np.ndarray, events: pd.DataFrame) -> dict[str, float]:
    valid = scores["ratio"].notna().to_numpy()
    n = normal & valid
    return {
        "far": float(alarm[n].mean()) if n.any() else np.nan,
        "far_raw": float(scores["alarm_raw"][n].mean()) if n.any() else np.nan,
        "n_normal": int(n.sum()),
        "detection_rate": float(events["detected"].mean()) if len(events) else np.nan,
        "median_delay_h": float(events["delay_h"].median()) if events["detected"].any() else np.nan,
        "n_events": int(len(events)),
        "top_subsystem_match": float(events["top_subsystem_match"].mean()) if len(events) else np.nan,
    }
