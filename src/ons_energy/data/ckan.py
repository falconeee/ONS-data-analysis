"""Resolve ONS dataset files through the CKAN API instead of hard-coded URLs.

File naming changes over time (yearly files up to 2021, monthly afterwards for plant generation;
Parquet only for recent ENA years), so each dataset declares a regex over the file stem and we
pick one resource per period, preferring the formats listed in the catalog.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

import requests

from ons_energy.data.catalog import DatasetSpec
from ons_energy.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Resource:
    dataset: str
    name: str
    url: str
    fmt: str  # lower-case extension: parquet | csv | xlsx
    year: int | None
    month: int | None
    last_modified: str | None

    @property
    def filename(self) -> str:
        return PurePosixPath(urlparse(self.url).path).name

    @property
    def period_key(self) -> tuple[int, int]:
        return (self.year or 0, self.month or 0)


def http_session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "ons-energy/0.1 (portfolio research)"
    return session


def package_show(api: str, ckan_id: str, session: requests.Session) -> dict:
    resp = session.get(f"{api}/package_show", params={"id": ckan_id}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show failed for {ckan_id!r}: {payload}")
    return payload["result"]


def select_resources(
    dataset: str, spec: DatasetSpec, package: dict, preferred_formats: list[str]
) -> list[Resource]:
    """One resource per (year, month), in the most preferred available format."""
    pattern = re.compile(spec.file_regex + r"$", re.IGNORECASE)
    rank = {f: i for i, f in enumerate(preferred_formats)}
    best: dict[tuple[int, int], Resource] = {}

    for res in package["resources"]:
        url = res.get("url") or ""
        path = PurePosixPath(urlparse(url).path)
        fmt = path.suffix.lstrip(".").lower()
        if fmt not in rank:
            continue
        match = pattern.match(path.stem)
        if not match:
            continue
        groups = match.groupdict()
        cand = Resource(
            dataset=dataset,
            name=res.get("name", path.stem),
            url=url.replace("http://", "https://", 1),
            fmt=fmt,
            year=int(groups["year"]) if groups.get("year") else None,
            month=int(groups["month"]) if groups.get("month") else None,
            last_modified=res.get("last_modified") or res.get("metadata_modified"),
        )
        current = best.get(cand.period_key)
        if current is None or rank[cand.fmt] < rank[current.fmt]:
            best[cand.period_key] = cand

    if not best:
        raise RuntimeError(f"No resource of {spec.ckan_id!r} matched {spec.file_regex!r}")
    resources = [best[k] for k in sorted(best)]
    log.info("%s: %d files resolved (%s)", dataset, len(resources), sorted({r.fmt for r in resources}))
    return resources
