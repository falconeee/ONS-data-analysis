# ONS Energy — load forecasting & multivariate anomaly detection on Brazil's grid

*Português: [README.pt-BR.md](README.pt-BR.md)*

End-to-end portfolio project on the open data of **ONS** (Operador Nacional do Sistema Elétrico),
the Brazilian grid operator:

1. **Load forecasting.** Day-ahead hourly load per subsystem (SE/CO, S, NE, N). Models: Temporal Fusion
   Transformer, N-BEATS (generic and interpretable) and N-HiTS. Baselines: seasonal naive, LightGBM and
   **the ONS's own programmed load**. Errors are broken down for *special days*: holidays, bridges,
   Carnival, Brazil's World Cup matches and blackouts.
2. **Multivariate anomaly detection.** Hourly load, generation by source and interchange of the four
   subsystems. Two detectors: **PCA** with Hotelling T² and SPE/Q (empirical KDE limits,
   reconstruction-based contributions) and **MSCVAE**, a multi-scale convolutional VAE on signature
   matrices with an extreme-value (SPOT) threshold and root-cause analysis. Both are evaluated against
   known events: the blackouts of 15/08/2023 and 21/03/2018 and every load interruption ≥ 1 GW.

Design document (pt-BR): [`docs/PLAN.md`](docs/PLAN.md). Conventions: [`CLAUDE.md`](CLAUDE.md).

---

## Quick start

```bash
uv sync                              # Python 3.12; core + dev + dl groups
uv run ons ingest                    # ONS files via CKAN (ETag/sha256 manifest), load API, Open-Meteo
uv run ons build                     # canonical Parquet tables in data/processed/ (strict schema checks)
uv run jupyter lab notebooks/        # then run the notebooks in order
```

`ons ingest -s ons|api|weather` downloads one source only. Models and detectors run **in the notebooks**,
which import every algorithm from `src/ons_energy`. Tables and figures, including the actual vs
predicted load charts, stay in the notebook outputs:

| Notebook | Content | Runtime (M4) |
|---|---|---|
| `01_data_eda.ipynb` | load history, generation mix, EAR/ENA, holiday effects, data-quality checks | < 1 min |
| `02_forecast_baselines.ipynb` | SeasonalNaive, LightGBM, ONS programmed load: backtest, actual vs predicted, day-type errors, DM tests, SHAP; optional weather ablation | ~15 min |
| `03_forecast_deep.ipynb` | N-HiTS, N-BEATSx, N-BEATS-I (CPU), TFT (MPS): backtests, intervals, interpretability, **final comparison of all models** | ~1.5 h |
| `04_anomaly_detection.ipynb` | PCA vs MSCVAE (hourly): summary, matched-FAR comparison, control charts around the 2023 and 2018 blackouts, contributions, event tables, alarms per year | ~1 h with MSCVAE |

LightGBM and torch cannot share a process on macOS, so notebook 02 (LightGBM) and notebooks 03/04
(torch) use separate kernels. Backtests are cached in `data/interim/forecasts/`, which is how notebook 03
compares every model without retraining. Experiments are logged to MLflow (`uv run mlflow ui
--backend-store-uri sqlite:///mlflow.db`).

Hardware assumed: Apple Silicon (M4, 16 GB). See [runtime notes](#runtime-on-apple-silicon).

## Data (verified against the portal, Oct/2026)

| Dataset (CKAN id) | Granularity | Since | Used for |
|---|---|---|---|
| `curva-carga` | hourly, subsystem | 2000 | forecasting target |
| `balanco-energia-subsistema` | hourly, subsystem + SIN | 2000 | load / hydro / thermal / wind / solar / interchange |
| `geracao-usina-2` | hourly, plant (yearly → monthly files) | 2000 | nuclear split, reconciliation |
| `ear-diario-por-subsistema`, `ena-diario-por-subsistema` | daily | 2000 | stored energy (% max), inflow (% long-term mean) |
| `interrupcao_carga` | event | 2007 | labelled disturbances (≥ 1 GW) |
| Load API `apicarga.ons.org.br` | 30 min, UTC | 2018 / 2021 | verified load (MMGD split), **ONS programmed load** |
| Open-Meteo archive / previous-runs | hourly | 2015 / 2022 | weather: reanalysis (*perfect*) and day-before forecast (*honest*) |

Cleaning issues handled:
- **Decimal strings.** Old files store numbers as strings (`"0E-8"`) and missing values as `""`.
  Coercion is strict.
- **Subsystem names.** Names change (`SUDESTE` vs `SUDESTE/CENTRO-OESTE`) and ids carry trailing
  spaces. Subsystems are normalized from the stripped id.
- **Daylight saving time.** Published local timestamps skip the spring-forward hour (it is imputed and
  flagged) and merge the fall-back hour.
- **Whole missing days.** Three days are missing in every subsystem (2013-12-01, 2014-02-01,
  2015-04-09). They are filled with the same hour of the previous week and flagged.
- **Nuclear inside thermal.** The balance's *thermal* includes nuclear (checked against plant-level
  data, MAE ≈ 0 MW), so nuclear is split out.
- **Methodology breaks.** The load definition changed in **Mar/2021** (adds the forecast of
  non-dispatched generation) and on **29/04/2023** (adds distributed solar, MMGD). They are modelled
  with regime dummies.
- **Load API.** It returns invalid JSON for missing values, which is repaired. Its definition runs
  ~0.9% above `curva-carga`, which is why a causal level-adjusted benchmark `ONS_adj` exists.

## Module 1 — day-ahead load forecasting

- **Task.** Forecasts are issued at **D-1 12:00** using data up to 11:00, for the 24 hours of D
  (forecast steps 13-36). This is the same information set as the ONS programmed load.
- **Backtest.** Every model runs through the Nixtla `cross_validation` engines with the same daily
  cutoffs. Train from 2015, **test from 2024-01-01 to the last day** (this
  includes the 2026 World Cup). LightGBM refits monthly; deep models refit every 90 or 180 days.
- **Covariates.** Calendar terms: Fourier terms, national holidays, Carnival and Corpus Christi,
  **state holidays weighted by each state's share of the load** and bridges. Event flags for Brazil's
  matches. Methodology-regime dummies and a DST flag. Weather in three scenarios: `none`, `perfect`
  (reanalysis, an upper bound) and `honest` (the forecast issued the day before).
- **Metrics.**
  - Point: MAPE, sMAPE, RMSE and MASE (scaled by in-sample seasonal naive, m=168).
  - Probabilistic: pinball loss and 80/90% coverage. The TFT gives native quantiles; the other models
    get **causal rolling conformal intervals** built only from residuals known at issue time.
  - Diebold-Mariano tests on daily losses.
- **Segments.** Results are broken down by day type (normal, national or state holiday, bridge,
  Carnival, year end, World Cup match, atypical event), with day-bootstrap confidence intervals and the
  number of days in each segment.
- **Interpretability.** TFT variable selection and attention, compared between normal and special
  days. The decomposition of the interpretable N-BEATS. SHAP values for LightGBM.

Experiments: `notebooks/02_forecast_baselines.ipynb` and `notebooks/03_forecast_deep.ipynb`, each one a
`BacktestConfig` defined in a cell.

## Module 2 — multivariate statistical process monitoring

- **Data.** Hourly load, hydro, thermal (ex-nuclear), wind, solar and net interchange per subsystem,
  plus SE nuclear. Sources below 1% of the subsystem load in the reference are dropped.
- **Preprocessing.** Causal weekly-baseline residuals: x_t minus the mean of the same hour over the
  previous 4 weeks. This removes the daily/weekly cycles and follows growth and level shifts.
- **Reference.** Two normal windows around the 2023 methodology break (Jun/2022–Apr/2023 and
  Sep/2023–Aug/2024), with known events removed. Monitoring runs from 2010 to the last day; an alarm
  needs 3 of the last 4 hours above the limit.
- **Detectors behind one `Detector` interface.**
  - **PCA.** Hotelling T² and SPE/Q. Limits are the 99% quantile of a KDE fit on the reference.
    Diagnosis uses **reconstruction-based contributions (RBC)**.
  - **MSCVAE.** The author's algorithm, used unchanged through an adapter. The adapter adds warm-up
    with real history, an out-of-sample SPOT threshold, a bounded SPOT fit and explicit NaN handling.
    Diagnosis uses its root-cause analysis.
- **Evaluation.**
  - False-alarm rate on normal hours (away from events and outside the reference period).
  - Detection rate and delay per event; whether the top contributor is in the affected subsystem.
  - **Detection at a matched 5% false-alarm rate**, to compare both detectors at the same operating
    point.

Slow anomalies such as the 2021 water crisis are shown in the EDA (EAR/ENA, notebook 01).

Experiments: `notebooks/04_anomaly_detection.ipynb` (one `MSPCConfig` per detector).

## Results

The results live in the notebook outputs. Run notebooks 02–04 after `ons build`; notebook 01 is
committed with its outputs.

- **Forecasting:**
  - Metric tables (MAPE, sMAPE, RMSE, MASE, pinball, coverage).
  - Daily actual vs predicted over the whole test period, plus hourly zooms on a normal week,
    Carnival 2025, national holidays, Christmas and Brazil's 2026 World Cup matches.
  - MAPE by day type (with CIs) and by hour, Diebold-Mariano tests.
  - TFT variable selection and attention, N-BEATS decomposition, LightGBM SHAP.
- **Anomaly detection:**
  - PCA vs MSCVAE summary (FAR, detection rate, delay) and comparison at a matched 5% FAR.
  - Control charts around the 15/08/2023 and 21/03/2018 blackouts, contributions (PCA RBC vs MSCVAE
    RCA), event tables, alarms per year.

## Engineering notes (what the data taught us)

- **macOS libomp clash.** torch and LightGBM ship different OpenMP runtimes, and using both in one
  process segfaults (exit 139) whichever runs first. LightGBM and torch therefore live in different
  notebooks (kernels).
- **The energy transition breaks static references.** Wind and solar capacity grows and new plants are
  commissioned (e.g. N wind in 2017, SE wind in 2025). A variable that is almost constant in the
  reference period turns any later movement into a huge z-score. Negligible sources (< 1% of load) are
  dropped and standardization has a floor.
- **Parametric limits under-cover.** On autocorrelated, heavy-tailed residuals the classic F (T²) and
  Box (SPE) limits raise ~10% alarms in the reference at α = 1%. That is why the PCA uses empirical
  KDE limits, which bring it back to ~1–2%.
- **MSCVAE threshold.** Calibrating SPOT on in-sample training scores gives a different (optimistic)
  threshold than calibrating on held-out normal data. Both are reported.

## Runtime on Apple Silicon

Benchmarked training speed on an M4 (SE/CO, 2015–2023 history, steps/s):

| Model | MPS | CPU |
|---|---|---|
| N-HiTS | 14.5 | **20.9** |
| N-BEATSx | 16.0 | **24.2** |
| N-BEATS (interpretable) | 51.7 | **98.5** |
| TFT (hidden 64, input 336) | **1.0** | 0.4 |
| TFT (hidden 32, input 168) | **3.5** | 3.6 |

Hence the N-BEATS family and N-HiTS run on CPU, while the TFT uses the smaller configuration on MPS
with 180-day refits.

## Repository layout

```
notebooks/    01 data EDA/quality · 02 baselines · 03 deep models + comparison · 04 anomaly detection
resources/    curated events (World Cups, blackouts, crises) and UF→subsystem / weather cities
src/ons_energy/
  cli.py      `ons ingest` / `ons build` (data collection only)
  data/       dataset catalog, CKAN + S3 download with manifest, load API, cleaning, schemas, quality checks
  calendar/   holidays (state holidays weighted by load), events, day-type segments
  weather/    Open-Meteo (reanalysis and day-before forecasts)
  features/   forecasting panel (future-known covariates), hourly MSPC matrix
  forecast/   models, backtest, conformal intervals, metrics, DM test, interpretability
  anomaly/    PCA (T²/SPE, KDE limits, RBC), MSCVAE + adapter, experiment pipeline and evaluation
  viz/        figure functions used by the notebooks
  tracking.py MLflow helpers
  utils.py    paths, logger, seeding, config base class
docs/PLAN.md  design document (pt-BR)
```

Data © ONS, Portal de Dados Abertos; weather © Open-Meteo (CC BY 4.0).
