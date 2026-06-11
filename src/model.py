"""Chronological train/test evaluation of feature -> future return regressions.

Leakage guards, in one place:
  * split is chronological -- train is strictly earlier than test
  * feature standardization uses train mean/std only, applied frozen to test
  * rows with undefined feature (warm-up) or target (end of day) are dropped
  * OOS R^2 is computed on the held-out test block against the test mean
  * t-stats use Newey-West (HAC) errors because h-event forward returns
    overlap and are mechanically autocorrelated
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class FitResult:
    feature: str
    target: str
    horizon: int
    n_train: int
    n_test: int
    beta: float          # dollars of future move per 1 train-std of feature
    beta_tstat: float    # HAC (Newey-West) t-stat, maxlags = horizon
    r2_train: float
    r2_oos: float        # 1 - SSE/SST on the chronological test block
    corr_oos: float      # signed test-set correlation (sanity check on sign)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def chronological_split(n: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Index masks for a strictly chronological split. Never shuffles."""
    cut = int(n * train_frac)
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def fit_eval(
    x: pd.Series,
    y: pd.Series,
    horizon: int,
    train_frac: float = 0.7,
) -> FitResult:
    """OLS of future return y on standardized feature x, chronological OOS."""
    df = pd.concat([x, y], axis=1).dropna()
    xv = df.iloc[:, 0].to_numpy(dtype=float)
    yv = df.iloc[:, 1].to_numpy(dtype=float)

    tr, te = chronological_split(len(df), train_frac)

    # standardize with train statistics only
    mu, sd = xv[tr].mean(), xv[tr].std()
    xz = (xv - mu) / sd

    X_tr = sm.add_constant(xz[tr])
    fit = sm.OLS(yv[tr], X_tr).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})

    X_te = sm.add_constant(xz[te])
    pred = X_te @ fit.params
    resid = yv[te] - pred
    sst = ((yv[te] - yv[te].mean()) ** 2).sum()
    r2_oos = 1.0 - (resid ** 2).sum() / sst

    return FitResult(
        feature=str(x.name),
        target=str(y.name),
        horizon=horizon,
        n_train=len(tr),
        n_test=len(te),
        beta=float(fit.params[1]),
        beta_tstat=float(fit.tvalues[1]),
        r2_train=float(fit.rsquared),
        r2_oos=float(r2_oos),
        corr_oos=float(np.corrcoef(pred, yv[te])[0, 1]),
    )


def run_grid(
    features: pd.DataFrame,
    book: pd.DataFrame,
    feature_names: list[str],
    horizons: list[int],
    price_cols: list[str],
    train_frac: float = 0.7,
) -> pd.DataFrame:
    """Evaluate every (feature, horizon, price) cell; returns a tidy table."""
    from src.target import future_return

    rows = []
    for price_col in price_cols:
        for h in horizons:
            y = future_return(book[price_col], h)
            for f in feature_names:
                rows.append(fit_eval(features[f], y, horizon=h,
                                     train_frac=train_frac).as_dict())
    return pd.DataFrame(rows)
