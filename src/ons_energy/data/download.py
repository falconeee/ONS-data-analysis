"""Download ONS files into ``data/raw/<dataset>/`` and keep a manifest of every fetch.

ONS revises published data, so every file is re-requested with ``If-None-Match`` (S3 ETag):
unchanged files cost one 304 response, revised files are replaced and the manifest records it.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from ons_energy.data.catalog import Catalog
from ons_energy.data.ckan import Resource, http_session, package_show, select_resources
from ons_energy.utils import RAW_DIR, get_logger

log = get_logger(__name__)

MANIFEST_PATH = RAW_DIR / "_manifest.parquet"
MANIFEST_SCHEMA = {
    "dataset": pl.String,
    "filename": pl.String,
    "url": pl.String,
    "year": pl.Int32,
    "month": pl.Int32,
    "etag": pl.String,
    "last_modified": pl.String,
    "sha256": pl.String,
    "bytes": pl.Int64,
    "fetched_at": pl.Datetime("us", "UTC"),
    "status": pl.String,  # downloaded | not_modified
}


def read_manifest(path: Path = MANIFEST_PATH) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=MANIFEST_SCHEMA)


def latest_entries(manifest: pl.DataFrame) -> pl.DataFrame:
    """Most recent manifest row per (dataset, filename)."""
    if manifest.is_empty():
        return manifest
    return manifest.sort("fetched_at").group_by("dataset", "filename").last()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_resource(
    res: Resource, dest_dir: Path, session: requests.Session, etag: str | None = None
) -> dict:
    dest = dest_dir / res.filename
    headers = {"If-None-Match": etag} if etag and dest.exists() else {}
    with session.get(res.url, headers=headers, stream=True, timeout=300) as resp:
        if resp.status_code == 304:
            status = "not_modified"
        else:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            status = "downloaded"
            etag = resp.headers.get("ETag")
        last_modified = resp.headers.get("Last-Modified")

    return {
        "dataset": res.dataset,
        "filename": res.filename,
        "url": res.url,
        "year": res.year,
        "month": res.month,
        "etag": etag,
        "last_modified": last_modified,
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
        "fetched_at": datetime.now(UTC),
        "status": status,
    }


def ingest_datasets(
    catalog: Catalog,
    datasets: list[str] | None = None,
    years: list[int] | None = None,
    raw_dir: Path = RAW_DIR,
) -> pl.DataFrame:
    """Download (or revalidate) the selected datasets; returns the new manifest rows."""
    names = datasets or [k for k, v in catalog.datasets.items() if v.default]
    unknown = set(names) - set(catalog.datasets)
    if unknown:
        raise KeyError(f"Unknown datasets {sorted(unknown)}; see ons_energy.data.catalog.CATALOG")

    session = http_session()
    known = latest_entries(read_manifest(raw_dir / "_manifest.parquet"))
    etags = {(r["dataset"], r["filename"]): r["etag"] for r in known.iter_rows(named=True)}
    rows: list[dict] = []

    for name in names:
        spec = catalog.datasets[name]
        package = package_show(catalog.ckan_api, spec.ckan_id, session)
        resources = select_resources(name, spec, package, catalog.preferred_formats)
        if years:
            resources = [r for r in resources if r.year is None or r.year in years]
        dest_dir = raw_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for res in resources:
            row = fetch_resource(res, dest_dir, session, etags.get((name, res.filename)))
            rows.append(row)
        n_new = sum(r["status"] == "downloaded" for r in rows if r["dataset"] == name)
        log.info("%s: %d downloaded, %d unchanged", name, n_new, len(resources) - n_new)

    new = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
    manifest = pl.concat([read_manifest(raw_dir / "_manifest.parquet"), new], how="vertical")
    manifest.write_parquet(raw_dir / "_manifest.parquet")
    return new


def raw_files(dataset: str, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Raw files of a dataset currently on disk (latest manifest entry per file)."""
    manifest = latest_entries(read_manifest(raw_dir / "_manifest.parquet"))
    names = manifest.filter(pl.col("dataset") == dataset).sort("year", "month")["filename"].to_list()
    paths = [raw_dir / dataset / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest files missing on disk, e.g. {missing[0]}")
    if not paths:
        raise FileNotFoundError(f"No raw files for {dataset!r}; run `ons ingest -d {dataset}` first")
    return paths
