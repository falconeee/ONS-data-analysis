"""Remove calendar structure before PCA (PCA assumes stationary, serially independent data).

* ``CalendarResidualizer`` - per-variable OLS on the reference period: intercept, optional linear
  trend, annual Fourier terms, day-of-week (and hour-of-day for hourly data) dummies and a national
  holiday flag. Keeps slow anomalies (a drought shows up as months of negative EAR residuals).
* ``WeeklyBaselineResidualizer`` - causal: x_t minus the mean of the same hour/day in the previous
  ``n_weeks`` weeks. Adapts to growth and methodology level shifts within weeks; suited to fast
  events (blackouts), blind to slow ones.
"""

import numpy as np
import pandas as pd

from ons_energy.calendar.holidays import holiday_table


def _holiday_flags(index: pd.DatetimeIndex) -> np.ndarray:
    years = sorted(set(index.year))
    t = holiday_table(years)
    days = set(t.filter(t["category"].is_in(["national", "optional"]))["date"].to_list())
    return np.array([x in days for x in index.date], float)


class CalendarResidualizer:
    def __init__(self, hourly: bool, fourier_k: int = 3, trend: bool = True):
        self.hourly = hourly
        self.fourier_k = fourier_k
        self.trend = trend

    def _design(self, index: pd.DatetimeIndex) -> np.ndarray:
        cols = [np.ones(len(index))]
        if self.trend:
            cols.append((index - pd.Timestamp("2000-01-01")).days.to_numpy() / 3652.5)
        doy = index.dayofyear.to_numpy()
        for k in range(1, self.fourier_k + 1):
            cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
        dow = index.dayofweek.to_numpy()
        cols += [(dow == d).astype(float) for d in range(1, 7)]
        if self.hourly:
            hour = index.hour.to_numpy()
            cols += [(hour == h).astype(float) for h in range(1, 24)]
        cols.append(_holiday_flags(index))
        return np.column_stack(cols)

    def fit(self, reference: list[pd.DataFrame]) -> "CalendarResidualizer":
        X = pd.concat(reference).dropna()
        A = self._design(X.index)
        self.columns_ = list(X.columns)
        self.coef_, *_ = np.linalg.lstsq(A, X.to_numpy(), rcond=None)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        fitted = self._design(df.index) @ self.coef_
        return df[self.columns_] - fitted


class WeeklyBaselineResidualizer:
    def __init__(self, hourly: bool, n_weeks: int = 4):
        self.period = 168 if hourly else 7
        self.n_weeks = n_weeks

    def fit(self, reference: list[pd.DataFrame]) -> "WeeklyBaselineResidualizer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = sum(df.shift(self.period * k) for k in range(1, self.n_weeks + 1)) / self.n_weeks
        return df - base


class NoResidualizer:
    def fit(self, reference: list[pd.DataFrame]) -> "NoResidualizer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def make_residualizer(kind: str, hourly: bool, **kw):
    if kind == "raw":
        return NoResidualizer()
    if kind == "calendar":
        return CalendarResidualizer(hourly=hourly, **kw)
    if kind == "weekly_baseline":
        return WeeklyBaselineResidualizer(hourly=hourly, **kw)
    raise ValueError(kind)

