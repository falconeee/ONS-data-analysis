"""Thin MLflow helpers (local sqlite backend at the project root)."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ons_energy.utils.io import PROCESSED_DIR, PROJECT_ROOT

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"


def data_snapshot() -> str:
    path = PROCESSED_DIR / "_built_at.txt"
    return path.read_text().strip() if path.exists() else "unknown"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, list):
            out[key] = json.dumps(v, default=str)[:500]
        else:
            out[key] = v
    return out


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tables: dict[str, pd.DataFrame] | None = None,
    artifacts: list[Path] | None = None,
) -> str:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ARTIFACT_ROOT / experiment).as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**_flatten(params), "data_snapshot": data_snapshot()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # drop NaN
        for name, df in (tables or {}).items():
            mlflow.log_text(df.to_csv(index=False), f"{name}.csv")
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def log_backtest(cfg, tables: dict[str, pd.DataFrame], cv: pd.DataFrame | None = None) -> str:
    """Log a forecast backtest (``BacktestConfig`` + ``evaluate_backtest`` tables)."""
    overall = tables["overall"]
    metrics = {f"{r.model}.{k}": getattr(r, k) for r in overall.itertuples() for k in ("MAPE", "RMSE", "MASE")}
    if cv is not None:
        metrics |= {f"time_s.{k}": v for k, v in cv.attrs.get("timings", {}).items()}
    return log_run("forecast", cfg.name, cfg.model_dump(mode="json"), metrics, tables)


def log_mspc(cfg, result: dict[str, Any]) -> str:
    """Log an MSPC experiment (``MSPCConfig`` + ``run_mspc`` result)."""
    metrics = {k: v for k, v in result["summary"].items() if isinstance(v, (int, float))}
    return log_run("anomaly", cfg.name, cfg.model_dump(mode="json"), metrics, {"events": result["events"]})
