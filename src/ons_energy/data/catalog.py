"""ONS open-data catalog used by ``ons ingest``.

Resources are resolved at runtime via CKAN ``package_show`` (https://dados.ons.org.br/api/3/action/);
``file_regex`` selects the data files among a package's resources. Verified 2026-10-14.
"""

from typing import Literal

from ons_energy.utils import StrictModel


class DatasetSpec(StrictModel):
    ckan_id: str
    file_regex: str
    granularity: Literal["hourly", "daily", "event"]
    time_col: str | None = None
    period: Literal["yearly", "yearly_then_monthly", "single"]
    default: bool = True  # ingested by `ons ingest` when no dataset is named


class LoadApiSpec(StrictModel):
    base_url: str
    endpoints: dict[str, str]
    max_window_days: int = 90
    areas: list[str]
    start: str = "2018-01-01"
    uf_share_year: int = 2024  # year used to estimate UF load shares


class OpenMeteoSpec(StrictModel):
    archive_url: str
    previous_runs_url: str
    variables: list[str]
    archive_start: str
    forecast_start: str


class Catalog(StrictModel):
    ckan_api: str
    preferred_formats: list[str]
    datasets: dict[str, DatasetSpec]
    load_api: LoadApiSpec
    open_meteo: OpenMeteoSpec


CATALOG = Catalog(
    ckan_api="https://dados.ons.org.br/api/3/action",
    preferred_formats=["parquet", "csv"],  # first available wins, per period
    datasets={
        "load_hourly": DatasetSpec(
            ckan_id="curva-carga", file_regex=r"CURVA_CARGA_(?P<year>\d{4})",
            granularity="hourly", time_col="din_instante", period="yearly",  # 2000 -> current year
        ),
        "balance_hourly": DatasetSpec(
            ckan_id="balanco-energia-subsistema", file_regex=r"BALANCO_ENERGIA_SUBSISTEMA_(?P<year>\d{4})",
            granularity="hourly", time_col="din_instante", period="yearly",
        ),
        "generation_plant_hourly": DatasetSpec(  # yearly files up to 2021, monthly from 2022
            ckan_id="geracao-usina-2", file_regex=r"GERACAO_USINA-2_(?P<year>\d{4})(?:_(?P<month>\d{2}))?",
            granularity="hourly", time_col="din_instante", period="yearly_then_monthly",
        ),
        "ear_daily": DatasetSpec(
            ckan_id="ear-diario-por-subsistema", file_regex=r"EAR_DIARIO_SUBSISTEMA_(?P<year>\d{4})",
            granularity="daily", time_col="ear_data", period="yearly",
        ),
        "ena_daily": DatasetSpec(  # parquet only for recent years; csv otherwise
            ckan_id="ena-diario-por-subsistema", file_regex=r"ENA_DIARIO_SUBSISTEMA_(?P<year>\d{4})",
            granularity="daily", period="yearly",
        ),
        "interchange_hourly": DatasetSpec(  # net interchange is already in balance_hourly
            ckan_id="intercambio-nacional", file_regex=r"INTERCAMBIO_NACIONAL_(?P<year>\d{4})",
            granularity="hourly", period="yearly", default=False,
        ),
        "load_interruptions": DatasetSpec(
            ckan_id="interrupcao_carga", file_regex="INTERRUPCAO_CARGA",
            granularity="event", time_col="din_interrupcaocarga", period="single",
        ),
        "load_daily": DatasetSpec(  # consistency checks only
            ckan_id="carga-energia", file_regex=r"CARGA_ENERGIA_(?P<year>\d{4})",
            granularity="daily", period="yearly", default=False,
        ),
    },
    # Load API (semi-hourly, UTC, by load area: SECO, S, NE, N, SIN and states). Max 3 months per call.
    load_api=LoadApiSpec(
        base_url="https://apicarga.ons.org.br/prd",
        endpoints={"verified": "cargaverificada", "programmed": "cargaprogramada"},  # ~2018 / ~2021-03 ->
        max_window_days=90,
        start="2018-01-01",
        areas=["SECO", "S", "NE", "N", "SIN"],
        uf_share_year=2024,
    ),
    # Weather (no API key). Archive = ERA5-based reanalysis ("perfect-forecast" scenario);
    # previous-runs `*_previous_day1` = forecast issued the day before ("honest" scenario, ~2022+).
    open_meteo=OpenMeteoSpec(
        archive_url="https://archive-api.open-meteo.com/v1/archive",
        previous_runs_url="https://previous-runs-api.open-meteo.com/v1/forecast",
        variables=["temperature_2m", "relative_humidity_2m", "shortwave_radiation"],
        archive_start="2015-01-01",
        forecast_start="2022-01-01",
    ),
)


def load_catalog() -> Catalog:
    return CATALOG
