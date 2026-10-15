import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ons_energy.cli import app
from ons_energy.config import StrictModel
from ons_energy.utils.seed import set_seed


def test_cli_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "build"):
        assert cmd in result.output
    assert "backtest" not in result.output  # models run in the notebooks


def test_set_seed_is_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


class _Cfg(StrictModel):
    name: str
    alpha: float = 0.01


def test_strict_config_rejects_unknown_keys():
    assert _Cfg(name="pca", alpha=0.05).alpha == 0.05
    with pytest.raises(ValidationError):
        _Cfg(name="pca", alpah=0.05)


def test_catalog_is_valid():
    from ons_energy.data.catalog import CATALOG, load_catalog

    assert load_catalog() is CATALOG
    assert CATALOG.datasets["load_hourly"].ckan_id == "curva-carga"
    assert {k for k, d in CATALOG.datasets.items() if not d.default} == {"interchange_hourly", "load_daily"}


def test_experiment_configs_build():
    from ons_energy.anomaly.experiment import MSPCConfig, mspc_config
    from ons_energy.forecast.config import BacktestConfig, ModelSpec

    cfg = BacktestConfig(name="seco_tft", refit_every_days=180, models=[ModelSpec(name="TFT", kind="tft", params={"hidden_size": 32})])
    assert cfg.eval_steps == (13, 36) and cfg.origin_hour == 12
    with pytest.raises(ValidationError):
        BacktestConfig(name="x", models=[ModelSpec(name="X", kind="arima")])

    m = mspc_config("hourly", "hourly_pca_weekly", "weekly_baseline", "pca", var_explained=0.9, persistence=(2, 3))
    assert isinstance(m, MSPCConfig) and m.persistence == (2, 3) and m.normal_margin == "2D"
    assert m.detector.params == {"var_explained": 0.9}
    d = mspc_config("daily", "daily_rpca", "calendar", "rpca", window=730)
    assert d.reference[0][0].year == 2016 and d.persistence == (3, 5)
