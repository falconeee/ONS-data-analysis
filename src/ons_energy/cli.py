"""Command-line entry point for data collection: ``uv run ons ingest`` then ``uv run ons build``.

Models and anomaly detectors run in the notebooks (``notebooks/``), importing from ``ons_energy``.
"""

import typer

app = typer.Typer(help="ONS open-data collection: download raw data and build the processed tables.", no_args_is_help=True)


@app.command()
def ingest(
    source: str = typer.Option("all", "--source", "-s", help="ons | api | weather | all"),
    datasets: list[str] = typer.Option(None, "--datasets", "-d", help="Dataset keys from ons_energy.data.catalog.CATALOG."),
    years: list[int] = typer.Option(None, "--years", "-y", help="Restrict ONS files to these years."),
) -> None:
    """Download raw data into data/raw: ONS files (CKAN/S3, with manifest), load API, Open-Meteo."""
    from ons_energy.data.catalog import load_catalog

    catalog = load_catalog()
    if source in ("ons", "all"):
        from ons_energy.data.download import ingest_datasets

        new = ingest_datasets(catalog, datasets=datasets, years=years)
        typer.echo(new.group_by("dataset", "status").len().sort("dataset"))
    if source in ("api", "all"):
        from ons_energy.data.api_carga import ingest_load_api
        from ons_energy.data.build import build_uf_load_share

        ingest_load_api(catalog.load_api)
        build_uf_load_share(catalog.load_api.uf_share_year)  # weather weights need it
    if source in ("weather", "all"):
        from ons_energy.weather.openmeteo import ingest_weather

        ingest_weather(catalog.open_meteo)


@app.command()
def build(
    no_api: bool = typer.Option(False, "--no-api", help="Skip the load-API tables."),
    no_weather: bool = typer.Option(False, "--no-weather", help="Skip the weather table."),
) -> None:
    """Clean raw files into canonical Parquet tables (data/processed)."""
    from ons_energy.data.build import build_all

    counts = build_all(include_api=not no_api, include_weather=not no_weather)
    for name, n in counts.items():
        typer.echo(f"{name:<22} {n:>12,d} rows")


if __name__ == "__main__":
    app()
