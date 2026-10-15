import json

import pytest

from ons_energy.data.catalog import DatasetSpec, load_catalog
from ons_energy.data.ckan import select_resources

S3 = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"


def _package(urls: list[str]) -> dict:
    return {"resources": [{"name": u.rsplit("/", 1)[-1], "url": u, "last_modified": None} for u in urls]}


def test_select_resources_prefers_parquet_and_handles_monthly_files():
    spec = load_catalog().datasets["generation_plant_hourly"]
    pkg = _package([
        f"{S3}/geracao_usina_2_ho/DicionarioDados_GeracaoPorUsina.pdf",
        f"{S3}/geracao_usina_2_ho/GERACAO_USINA-2_2021.csv",
        f"{S3}/geracao_usina_2_ho/GERACAO_USINA-2_2021.parquet",
        f"{S3}/geracao_usina_2_ho/GERACAO_USINA-2_2021.xlsx",
        f"{S3}/geracao_usina_2_ho/GERACAO_USINA-2_2022_01.csv",
        f"{S3}/geracao_usina_2_ho/GERACAO_USINA-2_2022_01.parquet",
        f"{S3}/geracao_usina_2_ho/GERACAO_USINA-2_2022_02.csv",
    ])
    res = select_resources("generation_plant_hourly", spec, pkg, ["parquet", "csv"])
    assert [(r.year, r.month, r.fmt) for r in res] == [(2021, None, "parquet"), (2022, 1, "parquet"), (2022, 2, "csv")]


def test_select_resources_ignores_dictionary_and_unmatched_files():
    spec = DatasetSpec(ckan_id="x", file_regex=r"EAR_DIARIO_SUBSISTEMA_(?P<year>\d{4})", granularity="daily", period="yearly")
    pkg = _package([
        f"{S3}/ear_subsistema_di/DicionarioDados_EarPorSubsistema.json",
        f"{S3}/ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_2000.parquet",
        f"{S3}/ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_2000_backup.parquet",
    ])
    res = select_resources("ear_daily", spec, pkg, ["parquet", "csv"])
    assert len(res) == 1 and res[0].year == 2000


def test_select_resources_raises_when_nothing_matches():
    spec = load_catalog().datasets["load_hourly"]
    with pytest.raises(RuntimeError):
        select_resources("load_hourly", spec, _package([f"{S3}/x/OTHER_2000.csv"]), ["parquet", "csv"])


def test_api_empty_value_repair():
    from ons_energy.data.api_carga import _EMPTY_VALUE

    text = '[{"val_cargaglobal": 1.5, "val_cargammgd": ,\n "val_consistencia": }]'
    assert json.loads(_EMPTY_VALUE.sub(r"\1 null\2", text)) == [
        {"val_cargaglobal": 1.5, "val_cargammgd": None, "val_consistencia": None}
    ]


@pytest.mark.network
def test_ckan_live_resolution():
    from ons_energy.data.ckan import http_session, package_show

    cat = load_catalog()
    pkg = package_show(cat.ckan_api, cat.datasets["load_hourly"].ckan_id, http_session())
    res = select_resources("load_hourly", cat.datasets["load_hourly"], pkg, cat.preferred_formats)
    assert res[0].year == 2000 and all(r.fmt == "parquet" for r in res)
