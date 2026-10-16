"""Expected ONS raw schemas (from the CKAN JSON data dictionaries).

Raw files drift in dtype (decimal strings in old years, floats in recent ones), so the checks only
assert that the columns exist with a compatible *kind*; coercion happens in ``clean.py``. A renamed
or missing column raises :class:`SchemaError` instead of silently producing wrong tables.
"""

import polars as pl

NUM, STR, TIME = "num", "str", "time"

RAW: dict[str, dict[str, str]] = {
    "load_hourly": {
        "id_subsistema": STR,
        "din_instante": TIME,
        "val_cargaenergiahomwmed": NUM,
    },
    "balance_hourly": {
        "id_subsistema": STR,
        "din_instante": TIME,
        "val_gerhidraulica": NUM,
        "val_gertermica": NUM,
        "val_gereolica": NUM,
        "val_gersolar": NUM,
        "val_carga": NUM,
        "val_intercambio": NUM,
    },
    "generation_plant_hourly": {
        "din_instante": TIME,
        "id_subsistema": STR,
        "nom_tipousina": STR,
        "val_geracao": NUM,
    },
    "ear_daily": {
        "id_subsistema": STR,
        "ear_data": TIME,
        "ear_max_subsistema": NUM,
        "ear_verif_subsistema_mwmes": NUM,
        "ear_verif_subsistema_percentual": NUM,
    },
    "ena_daily": {
        "id_subsistema": STR,
        "ena_data": TIME,
        "ena_bruta_regiao_mwmed": NUM,
        "ena_bruta_regiao_percentualmlt": NUM,
        "ena_armazenavel_regiao_mwmed": NUM,
        "ena_armazenavel_regiao_percentualmlt": NUM,
    },
    "load_interruptions": {
        "cod_perturbacao": STR,
        "din_interrupcaocarga": TIME,
        "id_subsistema": STR,
        "id_estado": STR,
        "val_cargainterrompida_mw": NUM,
        "val_tempomedio_minutos": NUM,
        "val_energianaosuprida_mwh": NUM,
    },
}


class SchemaError(ValueError):
    pass


def _kind_ok(dtype: pl.DataType, kind: str) -> bool:
    if dtype == pl.String or dtype == pl.Null:
        return True  # coerced strictly later
    if kind == NUM:
        return dtype.is_numeric() or isinstance(dtype, pl.Decimal)
    if kind == TIME:
        return dtype.is_temporal()
    return kind == STR and dtype == pl.Categorical


def validate_raw(df: pl.DataFrame, dataset: str, source: str = "") -> None:
    expected = RAW[dataset]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"{dataset} {source}: missing columns {missing}; got {df.columns}")
    bad = {c: str(df.schema[c]) for c, k in expected.items() if not _kind_ok(df.schema[c], k)}
    if bad:
        raise SchemaError(f"{dataset} {source}: incompatible dtypes {bad}")

