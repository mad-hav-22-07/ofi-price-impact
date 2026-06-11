"""Reproduce every number and figure in results/ from the raw data.

Usage:  python -m src.run_all
Deterministic end to end (OLS has no randomness; no seeds needed).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.backtest import predict_oos, threshold_sweep
from src.features import build_features
from src.loader import load_lobster, sanity_report
from src.model import chronological_split, fit_eval, run_grid
from src.plots import fig_cost_honesty, fig_ofi_vs_return, fig_r2_decay, fig_sanity
from src.target import future_return

MESSAGE = "data/AAPL_2012-06-21_34200000_57600000_message_5.csv"
ORDERBOOK = "data/AAPL_2012-06-21_34200000_57600000_orderbook_5.csv"

HEADLINE_H = 50
DECAY_HORIZONS = [5, 10, 20, 50, 100, 200, 500]
FEATURES = ["ofi_l1", "ofi_5lvl", "qimb", "trade_imb"]


def main() -> None:
    book = load_lobster(MESSAGE, ORDERBOOK)
    stats = sanity_report(book)
    print("data:", json.dumps(stats, default=float))
    fig_sanity(book, "results/fig0_sanity.png")

    # ---- main grid: feature x horizon x target price -----------------------
    frames = []
    for h in (10, 50, 100):
        feats = build_features(book, window=h)
        frames.append(run_grid(feats, book, FEATURES, [h], ["mid", "microprice"]))
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv("results/grid_results.csv", index=False)

    # ---- horizon decay curve (mid target) ----------------------------------
    rows = []
    for h in DECAY_HORIZONS:
        feats = build_features(book, window=h)
        y = future_return(book["mid"], h)
        for f in FEATURES:
            rows.append(fit_eval(feats[f], y, horizon=h).as_dict())
    decay = pd.DataFrame(rows)
    decay.to_csv("results/r2_decay.csv", index=False)
    fig_r2_decay(decay, "results/fig2_r2_decay.png")

    # ---- headline figure: OFI vs future return, OOS ------------------------
    feats = build_features(book, window=HEADLINE_H)
    y = future_return(book["mid"], HEADLINE_H)
    x = feats["ofi_5lvl"]
    res = fit_eval(x, y, horizon=HEADLINE_H)
    print("headline:", json.dumps(res.as_dict(), default=float))

    df = pd.concat([x, y], axis=1).dropna()
    xv = df.iloc[:, 0].to_numpy(float)
    yv = df.iloc[:, 1].to_numpy(float)
    tr, te = chronological_split(len(df))
    xz = (xv - xv[tr].mean()) / xv[tr].std()
    fit = sm.OLS(yv[tr], sm.add_constant(xz[tr])).fit()
    fig_ofi_vs_return(xz[te], yv[te], beta=fit.params[1], alpha=fit.params[0],
                      r2_oos=res.r2_oos, horizon=HEADLINE_H,
                      path="results/fig1_ofi_vs_return.png")

    # ---- cost-honest backtest ----------------------------------------------
    sweep = threshold_sweep(book, feats["ofi_5lvl"], y, HEADLINE_H)
    sweep.to_csv("results/backtest_sweep.csv", index=False)
    fig_cost_honesty(sweep, "results/fig3_cost_honesty.png")
    print(sweep.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
