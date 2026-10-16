# ONS Energy — project conventions

Portfolio project on ONS (Brazilian grid operator) open data:
**Module 1** hourly load forecasting per subsystem (TFT, N-BEATS, N-HiTS vs. SeasonalNaive / LightGBM, plus the ONS programmed load as an external benchmark);
**Module 2** hourly multivariate anomaly detection & diagnosis (PCA T²/SPE vs. MSCVAE).
Full design, experimental setup and roadmap: `docs/PLAN.md` (pt-BR; §10–11 = current notebook-based, simplified layout). MVP subsystem is SE/CO.

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
  `04_anomaly_detection.ipynb` (separate kernels); the backtest refuses mixed configs and `set_seed`
  never imports torch. Notebook 03 compares all models by reading the stored backtests
  (`merge_forecasts`), never by importing LightGBM.

## Layout rules
- Algorithms live in `src/`; notebooks run the experiments and show the results (tables + figures inline, incl. actual vs predicted). Notebooks only define configs, call `ons_energy` functions and plot; no algorithm code in notebooks.
- Order: `uv run ons ingest [-s ons|api|weather]` → `uv run ons build` → `01_data_eda` → `02_forecast_baselines` → `03_forecast_deep` → `04_anomaly_detection`.
- Experiment configs are pydantic `StrictModel` objects built in notebook cells: `BacktestConfig(name=..., models=[ModelSpec(...)])` and `MSPCConfig(name=..., detector="pca"|"mscvae", params={...})`. The data catalog is the constant `CATALOG` in `data/catalog.py`. No `configs/` folder.
- Shared helpers live in the single module `ons_energy.utils`: paths (`PROCESSED_DIR`, ...), `logger` / `get_logger`, `set_seed`, `StrictModel`.
- Keep it lean: prefer extending an existing function over adding modules/options; no variant knobs that no notebook uses.
- `viz/` functions take DataFrames and return the figure (no saving to disk). Backtests are cached in `data/interim/forecasts/<name>.parquet` (`run_or_load`, `load_forecasts`, `merge_forecasts`); MSPC results are returned in memory by `run_mspc`. Log runs to MLflow with `tracking.log_backtest` / `tracking.log_mspc`.
- `data/` is never committed. Small curated tables (events, UF→subsystem maps) live in `resources/` (versioned).
- Logging via `ons_energy.utils` (`logger` / `get_logger(__name__)`), not `print`.

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
- No leakage: features at a forecast origin use only data available at that origin (day-ahead: origin D-1 12:00 local, data ≤ D-1 11:00, predict hours of D); lags are built by the Nixtla engines at each cutoff, conformal intervals use residuals of target days ≤ D-2.
- Temporal splits only (train from 2015, test from `eval_start` = 2024-01-01); rolling-origin backtests. No random CV.
- Observed/reanalysis weather used as a future covariate = "perfect-forecast" scenario; always label it. The "honest" scenario uses archived forecasts.
- Report segment metrics by `day_type` always with the number of days; rare segments get bootstrap CIs.
- MSPC: only two detectors — PCA (`anomaly/pca.py`: T²/SPE, KDE limits, RBC contributions) and MSCVAE — on the hourly matrix as weekly-baseline residuals. Scaling/PCA/limits are fitted on the reference period only. No DPCA/RPCA/daily-scale variants.
- Both implement the `Detector` interface (`anomaly/base.py`) and go through the same `run_mspc` pipeline and event evaluation (`anomaly/experiment.py`). Compare them also at a matched false-alarm rate (`matched_operating_point`), since their own limits sit at different operating points.
- Variables below 1% of the subsystem load in the reference are dropped (`min_share`), and PCA standardization uses a 1-unit std floor: near-constant reference variables otherwise explode z-scores when a new source is commissioned.
- `anomaly/mscvae.py` is the author's MSCVAE algorithm: do NOT refactor or "fix" it without asking. Project-specific behavior (warm-up with real history, out-of-sample SPOT calibration, `df_sistema` mapping, NaN handling) lives in `anomaly/mscvae_adapter.py`.
- Seeds fixed via `ons_energy.utils.set_seed` (python, numpy, torch, MPS, lightning); pass the same seed to LightGBM configs.

## Code style
- English identifiers, docstrings and commit messages; keep ONS domain terms (EAR, ENA, MMGD, SIN, SE/CO) as-is. User-facing chat can be in Portuguese.
- Type hints on public functions; match the surrounding style and comment density.
- No test suite (removed on purpose to keep the repo small). Check changes by importing the modules, running notebook 01 and dry-running the config cells of notebooks 02–04 (don't train models unless asked).
