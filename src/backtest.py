"""Cost-honest toy strategy on the held-out test block.

Rule: at event t, if the model's predicted h-event move exceeds +tau, go long
one share; below -tau, go short. Hold exactly h events, then exit. While in a
position, new signals are ignored (no overlapping trades, no pyramiding).

Accounting is per-trade, one share, using the touch prices that actually
prevailed:
  gross = mid-to-mid move in the trade direction (the frictionless fantasy)
  net   = executed at market: buy at the ask, sell at the bid, both legs
The difference between the two is exactly the half-spread paid per leg --
charged at the entry and exit spreads of that specific trade, not at an
average.

The model (OLS coefficients, feature mean/std, tau) is fit on the train block
only; the test block is touched once, chronologically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.model import chronological_split


@dataclass
class BacktestResult:
    tau_quantile: float
    tau_dollars: float
    n_trades: int
    hit_rate: float            # share of trades with positive gross pnl
    gross_per_trade: float     # dollars, one share
    cost_per_trade: float
    net_per_trade: float
    gross_total: float
    net_total: float
    sharpe_like: float         # mean(net)/std(net) * sqrt(n_trades)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def predict_oos(x: pd.Series, y: pd.Series, train_frac: float = 0.7):
    """Fit y ~ x on the chronological train block; return test predictions.

    Returns (test_index, predictions, train_predictions) with all
    standardization stats taken from train only.
    """
    df = pd.concat([x, y], axis=1).dropna()
    xv = df.iloc[:, 0].to_numpy(dtype=float)
    yv = df.iloc[:, 1].to_numpy(dtype=float)
    tr, te = chronological_split(len(df), train_frac)

    mu, sd = xv[tr].mean(), xv[tr].std()
    xz = (xv - mu) / sd
    fit = sm.OLS(yv[tr], sm.add_constant(xz[tr])).fit()

    pred_tr = sm.add_constant(xz[tr]) @ fit.params
    pred_te = sm.add_constant(xz[te]) @ fit.params
    return df.index[te], pred_te, pred_tr


def run_backtest(
    book: pd.DataFrame,
    test_index: pd.Index,
    pred_test: np.ndarray,
    pred_train: np.ndarray,
    horizon: int,
    tau_quantile: float,
) -> BacktestResult:
    """Threshold strategy, hold `horizon` events, full touch-price costs."""
    tau = float(np.quantile(np.abs(pred_train), tau_quantile))

    mid = book["mid"].to_numpy()
    bid = book["bid_price_1"].to_numpy()
    ask = book["ask_price_1"].to_numpy()
    rows = test_index.to_numpy()

    gross, net = [], []
    i = 0
    # stop early enough that every trade can complete its full hold
    while i < len(rows):
        p = pred_test[i]
        if abs(p) > tau and i + horizon < len(rows):
            side = 1 if p > 0 else -1
            t0, t1 = rows[i], rows[i + horizon]
            g = side * (mid[t1] - mid[t0])
            if side == 1:                       # buy at ask, sell at bid
                n_ = bid[t1] - ask[t0]
            else:                               # sell at bid, buy back at ask
                n_ = bid[t0] - ask[t1]
            gross.append(g)
            net.append(n_)
            i += horizon                        # busy until the exit event
        else:
            i += 1

    gross = np.array(gross)
    net = np.array(net)
    n = len(gross)
    if n == 0:
        return BacktestResult(tau_quantile, tau, 0, np.nan, np.nan, np.nan,
                              np.nan, 0.0, 0.0, np.nan)
    return BacktestResult(
        tau_quantile=tau_quantile,
        tau_dollars=tau,
        n_trades=n,
        hit_rate=float((gross > 0).mean()),
        gross_per_trade=float(gross.mean()),
        cost_per_trade=float((gross - net).mean()),
        net_per_trade=float(net.mean()),
        gross_total=float(gross.sum()),
        net_total=float(net.sum()),
        sharpe_like=float(net.mean() / net.std() * np.sqrt(n)) if n > 1 else np.nan,
    )


def threshold_sweep(
    book: pd.DataFrame,
    feature: pd.Series,
    target: pd.Series,
    horizon: int,
    quantiles: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99),
    train_frac: float = 0.7,
) -> pd.DataFrame:
    """Run the backtest across entry thresholds; tidy results table."""
    te_idx, pred_te, pred_tr = predict_oos(feature, target, train_frac)
    rows = [
        run_backtest(book, te_idx, pred_te, pred_tr, horizon, q).as_dict()
        for q in quantiles
    ]
    return pd.DataFrame(rows)
