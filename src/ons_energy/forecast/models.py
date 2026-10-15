"""Model factory: ``ModelSpec`` -> StatsForecast / MLForecast / NeuralForecast objects."""

from typing import Any

from ons_energy.forecast.config import BacktestConfig, ModelSpec

LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbosity": -1,
}
LGBM_FEATURES: dict[str, Any] = {"lags": [1, 2, 3, 24, 48, 168, 336], "date_features": ["hour", "dayofweek"]}
NF_DEFAULTS: dict[str, Any] = {
    "input_size": 336, "max_steps": 1500, "learning_rate": 1e-3, "batch_size": 32,
    "windows_batch_size": 256, "scaler_type": "robust", "val_check_steps": 1_000_000,
}
# MQLoss(level=[80, 90]) produces these columns; mapped to quantile levels.
NF_LEVEL_COLUMNS = {"lo-90": 0.05, "lo-80": 0.1, "median": 0.5, "hi-80": 0.9, "hi-90": 0.95}


def seasonal_naive(specs: list[ModelSpec]):
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    models = [SeasonalNaive(season_length=s.params.get("season_length", 168), alias=s.name) for s in specs]
    return StatsForecast(models=models, freq="h", n_jobs=1)


def lgbm(specs: list[ModelSpec], seed: int):
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    models = {}
    for s in specs:
        params = {**LGBM_DEFAULTS, **{k: v for k, v in s.params.items() if k not in LGBM_FEATURES}}
        models[s.name] = lgb.LGBMRegressor(random_state=seed, **params)
    feats = {**LGBM_FEATURES, **{k: v for s in specs for k, v in s.params.items() if k in LGBM_FEATURES}}
    return MLForecast(
        models=models,
        freq="h",
        lags=feats["lags"],
        lag_transforms={1: [RollingMean(window_size=24)], 24: [RollingMean(window_size=168)], 168: [RollingMean(window_size=672)]},
        date_features=feats["date_features"],
    )


def neural(specs: list[ModelSpec], cfg: BacktestConfig, futr_exog: list[str]):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE, MQLoss
    from neuralforecast.models import NBEATS, NHITS, TFT, NBEATSx

    trainer = {
        "accelerator": cfg.accelerator, "devices": 1, "enable_progress_bar": False, "logger": False,
        "enable_checkpointing": False, "enable_model_summary": False,
    }
    models = []
    for s in specs:
        p = {**NF_DEFAULTS, **s.params}
        common = {"h": cfg.horizon, "random_seed": cfg.seed, "alias": s.name, **trainer, **p}
        if s.kind == "nhits":
            models.append(NHITS(loss=MAE(), futr_exog_list=futr_exog, **common))
        elif s.kind == "nbeatsx":
            models.append(NBEATSx(loss=MAE(), futr_exog_list=futr_exog, stack_types=["identity", "identity"], **common))
        elif s.kind == "nbeats_i":
            # Interpretable N-BEATS: trend + seasonality stacks, univariate (no covariates).
            models.append(NBEATS(loss=MAE(), stack_types=["trend", "seasonality"], **common))
        elif s.kind == "tft":
            models.append(TFT(loss=MQLoss(level=[80, 90]), futr_exog_list=futr_exog, **common))
        else:
            raise ValueError(f"{s.kind} is not a NeuralForecast model")
    return NeuralForecast(models=models, freq="h")
