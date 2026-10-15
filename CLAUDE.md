# ONS Energy — project conventions

Portfolio project on ONS (Brazilian grid operator) open data:
**Module 1** hourly load forecasting per subsystem (TFT, N-BEATS, N-HiTS vs. SeasonalNaive / LightGBM, plus the ONS programmed load as an external benchmark);
**Module 2** multivariate anomaly detection & diagnosis (PCA T²/SPE, DPCA, RPCA, MSCVAE).
Full design, experimental setup and roadmap: `docs/PLAN.md` (pt-BR; §10 = current notebook-based layout). MVP subsystem is SE/CO.

## Stack & environment
- Python 3.12, managed with `uv` (`uv sync` installs core + `dev` + `dl` groups; run everything via `uv run`).
- Package: `src/ons_energy` (src layout). CLI only for data collection: `uv run ons ingest`, `uv run ons build` (typer, `src/ons_energy/cli.py`).
- Experiments run in the notebooks (`notebooks/02_*`, `03_*`, `04_*`), which import the algorithms from `ons_energy`.
- Hardware: Apple M4 / 16 GB / MPS only. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`; accelerator comes from config (`mps|cpu`). Keep DL training history configurable (default 2015+). Benchmarked: N-HiTS / N-BEATS train faster on CPU; TFT only on MPS and in the small config (hidden 32, input 168) — see README "Runtime".
- Forecasting: Nixtla (neuralforecast, mlforecast, statsforecast, utilsforecast) — all models go through the same `cross_validation` backtest. ETL: polars; pandas only at library boundaries.
- Tracking: MLflow (local sqlite, `mlruns/`, gitignored).
- No linters, pre-commit, pandera or CI for now — don't add them unless asked.
- **Never mix LightGBM and torch in one process** (macOS: two libomp builds -> segfault, exit 139, in
  either order). LightGBM lives in `02_forecast_baselines.ipynb`, torch in `03_forecast_deep.ipynb` /
  `04_anomaly_detection.ipynb` (separate kernels); the backtest refuses mixed configs, `set_seed`
  never imports torch, `tests/conftest.py` preloads LightGBM, and tests that *run* torch go to a
  subprocess (see `tests/test_spot.py`). Notebook 03 compares all models by reading the stored
  backtests (`merge_forecasts`), never by importing LightGBM.

## Layout rules
- Algorithms live in `src/`; notebooks run the experiments and show the results (tables + figures inline, incl. actual vs predicted). Notebooks only define configs, call `ons_energy` functions and plot; no algorithm code in notebooks.
- Order: `uv run ons ingest [-s ons|api|weather]` → `uv run ons build` → `01_data_eda` → `02_forecast_baselines` → `03_forecast_deep` → `04_anomaly_detection`.
- Experiment configs are pydantic `StrictModel` objects built in notebook cells: `BacktestConfig(name=..., models=[ModelSpec(...)])` and `mspc_config(scale, name, preprocess, kind, label=..., **params)` (presets `HOURLY_BASE` / `DAILY_BASE` in `anomaly/experiment.py`). The data catalog is the constant `CATALOG` in `data/catalog.py`. No `configs/` folder; paths come from `ons_energy.utils.io`.
- `viz/` functions take DataFrames and return the figure (no saving to disk). Backtests are cached in `data/interim/forecasts/<name>.parquet` (`run_or_load`, `load_forecasts`, `merge_forecasts`); MSPC results are returned in memory by `run_mspc`. Log runs to MLflow with `tracking.log_backtest` / `tracking.log_mspc`.
- `data/` is never committed. Small curated tables (events, UF→subsystem maps) live in `resources/` (versioned).
- Logging via `ons_energy.utils.logging` (`logger` / `get_logger(__name__)`), not `print`.

## Data conventions
- Never assume ONS column names: resolve resources via CKAN `package_show` (`CATALOG` in `data/catalog.py`) and check the dataset JSON dictionary; validate against the expected-schema dicts in `data/schemas.py`. Schema drift must fail loudly.
- Canonical columns: `subsystem` ∈ {SE, S, NE, N, SIN} — normalize from **stripped** `id_subsistema` (has trailing spaces in some files), never from `nom_subsistema` (`SUDESTE` vs `SUDESTE/CENTRO-OESTE`). `ts_local` (America/Sao_Paulo wall clock, as published), `ts_utc`, values float64 in MWmed.
- Old files store numbers as decimal strings (`"0E-8"`) and missing as `""`; coerce strictly. Structural zeros (source did not exist yet, e.g. wind/solar in 2000) ≠ missing: fill 0 + flag.
- DST (pre-2019): the spring-forward hour is missing from ONS files → impute with `is_imputed=True`; the fall-back hour is merged.
- Methodology breaks in load: 2021-03 (non-dispatched generation) and 2023-04-29 (MMGD). Model them with regime dummies; never silently mix regimes.
- ONS data is revised daily: raw downloads are tracked in `data/raw/_manifest.parquet` (ETag, sha256, timestamp); log the extraction date to MLflow.
- Load API (`apicarga.ons.org.br/prd`) is semi-hourly in UTC (value at the END of the half-hour), ≤ 3 months per call, verified from 2018 and programmed from ~2021-03. It returns invalid JSON for missing values (`"key": ,`) — repaired in `api_carga.py`. API "carga global" is ~0.9% above curva-carga (definition); `ONS_adj` is the causal level-adjusted benchmark.
- Plant types include `BOMBEAMENTO` (pumping, mapped to `pumping`); balance `thermal` includes nuclear (split out using plant data).
- Events: `resources/events.csv` (curated) + interruptions ≥ 1000 MW from `interrupcao_carga`, one event per disturbance; interruptions within 3 h of a curated blackout are dropped as duplicates.

## Modeling rules
- No leakage: features at a forecast origin use only data available at that origin (day-ahead: origin D-1 12:00 local, data ≤ D-1 11:00, predict hours of D). Covered by `tests/test_features_leakage.py`.
- Temporal splits only (train ≤ 2022, val 2023-05..12, test 2024+); rolling-origin backtests. No random CV.
- Observed/reanalysis weather used as a future covariate = "perfect-forecast" scenario; always label it. The "honest" scenario uses archived forecasts.
- Report segment metrics by `day_type` always with the number of days; rare segments get bootstrap CIs.
- MSPC: scaling/PCA/limits fitted on the reference period only. T²/SPE/limits/contributions are our own implementation and must be unit-tested.
- All anomaly detectors implement the `Detector` interface in `anomaly/base.py` and are evaluated by the same `anomaly/evaluate.py` (same events, same preprocessing variants). Compare detectors also at a matched false-alarm rate (`matched_operating_point`), since their own limits sit at different operating points.
- Variables below 1% of the subsystem load in the reference are dropped (`min_share`), and PCA standardization uses a 1-unit std floor: near-constant reference variables otherwise explode z-scores when a new source is commissioned.
- RPCA: excluding every alarmed point from the window freezes the model after a permanent change (new plants, methodology breaks). Use `exclude_ratio` / `accept_after`; the hourly RPCA uses no exclusion.
- `anomaly/mscvae.py` is the author's MSCVAE algorithm: do NOT refactor or "fix" it without asking. Project-specific behavior (warm-up with real history, out-of-sample SPOT calibration, `df_sistema` mapping, NaN handling) lives in `anomaly/mscvae_adapter.py`.
- Seeds fixed via `ons_energy.utils.seed.set_seed` (python, numpy, torch, MPS, lightning); pass the same seed to LightGBM / scikit-learn configs.

## Code style
- English identifiers, docstrings and commit messages; keep ONS domain terms (EAR, ENA, MMGD, SIN, SE/CO) as-is. User-facing chat can be in Portuguese.
- Type hints on public functions; match the surrounding style and comment density.
- Tests: `uv run pytest` (network tests marked `@pytest.mark.network` are deselected by default; run them with `-m network`). Use small fixtures in `tests/fixtures/`, never full ONS downloads.
