"""National, optional (Carnival, Corpus Christi) and state holidays, weighted per subsystem.

State holidays are aggregated into ``state_holiday_w`` = sum of the load shares of the subsystem's
UFs that are on holiday that day (shares estimated from the ONS load API by UF).
"""

from datetime import date, timedelta
from functools import lru_cache

import holidays
import polars as pl
import yaml
from holidays.constants import OPTIONAL, PUBLIC

from ons_energy.utils.io import PROCESSED_DIR, RESOURCES_DIR

SUBSYSTEMS = ("SE", "S", "NE", "N")


@lru_cache(maxsize=1)
def _states_cfg() -> dict:
    with (RESOURCES_DIR / "subsystem_states.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


SUBSYSTEM_UFS: dict[str, list[str]] = _states_cfg()["subsystems"]
UF_TO_SUBSYSTEM: dict[str, str] = {uf: s for s, ufs in SUBSYSTEM_UFS.items() for uf in ufs}


def holiday_table(years: list[int]) -> pl.DataFrame:
    """Columns: date, uf (null = national), name, category ∈ {national, optional, state}."""
    rows = []
    national = holidays.Brazil(years=years, categories=(PUBLIC,))
    for d, name in national.items():
        rows.append((d, None, name, "national"))
    for d, name in holidays.Brazil(years=years, categories=(OPTIONAL,)).items():
        rows.append((d, None, name, "optional"))
    for uf in UF_TO_SUBSYSTEM:
        for d, name in holidays.Brazil(years=years, subdiv=uf, categories=(PUBLIC,)).items():
            if d not in national:
                rows.append((d, uf, name, "state"))
    return pl.DataFrame(
        rows, schema={"date": pl.Date, "uf": pl.String, "name": pl.String, "category": pl.String}, orient="row"
    ).sort("date")


def uf_load_shares() -> dict[str, dict[str, float]]:
    """Share of each UF in its subsystem's load. Falls back to equal shares if not built yet."""
    path = PROCESSED_DIR / "uf_load_share.parquet"
    if path.exists():
        df = pl.read_parquet(path)
        out: dict[str, dict[str, float]] = {s: {} for s in SUBSYSTEMS}
        for row in df.iter_rows(named=True):
            out[row["subsystem"]][row["uf"]] = row["share"]
        return out
    return {s: {uf: 1 / len(ufs) for uf in ufs} for s, ufs in SUBSYSTEM_UFS.items()}


def daily_calendar(
    start: date, end: date, subsystem: str, shares: dict[str, float] | None = None
) -> pl.DataFrame:
    """One row per date with holiday flags for ``subsystem``.

    Columns: date, dow (0=Mon), is_national_holiday, is_carnival, is_ash_wednesday,
    is_corpus_christi, state_holiday_w, is_bridge, is_year_end, holiday_name.
    """
    years = list(range(start.year, end.year + 1))
    table = holiday_table(years)
    shares = shares if shares is not None else uf_load_shares()[subsystem]

    nat = table.filter(pl.col("category") == "national")
    opt = table.filter(pl.col("category") == "optional")
    state = table.filter((pl.col("category") == "state") & pl.col("uf").is_in(list(shares)))
    state_w = (
        state.unique(["date", "uf"])
        .with_columns(w=pl.col("uf").replace_strict(shares, default=0.0, return_dtype=pl.Float64))
        .group_by("date")
        .agg(state_holiday_w=pl.col("w").sum().clip(0, 1))
    )

    nat_dates = set(nat["date"].to_list())
    carnival = set(opt.filter(pl.col("name") == "Carnaval")["date"].to_list())
    ash = set(opt.filter(pl.col("name") == "Início da Quaresma")["date"].to_list())
    corpus = set(opt.filter(pl.col("name") == "Corpus Christi")["date"].to_list())
    names = dict(
        nat.group_by("date").agg(pl.col("name").first()).iter_rows()
    ) | {d: "Carnaval" for d in carnival} | {d: "Corpus Christi" for d in corpus}

    # Bridge ("ponte"): Monday before a Tuesday holiday / Friday after a Thursday holiday.
    long_holidays = nat_dates | corpus
    bridge = {d - timedelta(days=1) for d in long_holidays if d.weekday() == 1} | {
        d + timedelta(days=1) for d in long_holidays if d.weekday() == 3
    }

    dates = pl.date_range(start, end, interval="1d", eager=True).alias("date")
    df = pl.DataFrame(dates).with_columns(
        dow=pl.col("date").dt.weekday() - 1,
        is_national_holiday=pl.col("date").is_in(list(nat_dates)),
        is_carnival=pl.col("date").is_in(list(carnival)),
        is_ash_wednesday=pl.col("date").is_in(list(ash)),
        is_corpus_christi=pl.col("date").is_in(list(corpus)),
        is_bridge=pl.col("date").is_in(list(bridge)) & ~pl.col("date").is_in(list(long_holidays)),
        is_year_end=((pl.col("date").dt.month() == 12) & (pl.col("date").dt.day() >= 24))
        | ((pl.col("date").dt.month() == 1) & (pl.col("date").dt.day() == 1)),
        holiday_name=pl.col("date").replace_strict(names, default=None, return_dtype=pl.String),
    )
    return df.join(state_w, on="date", how="left").with_columns(pl.col("state_holiday_w").fill_null(0.0))
