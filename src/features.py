"""Order-flow features: multi-level OFI, queue imbalance, trade-sign imbalance.

All features at row t use only book states with index <= t. Windowed
aggregations are trailing sums ending at t (inclusive). Nothing here looks
forward; forward alignment is target.py's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ofi_per_event(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """Per-event order flow imbalance at one book level (Cont-Kukanov-Stoikov).

    Compares the book after event n with the book after event n-1, using the
    compact indicator form (equivalent to the price-up/down/unchanged cases):

      bid flow  b_n = q^b_n * 1{P^b_n >= P^b_{n-1}} - q^b_{n-1} * 1{P^b_n <= P^b_{n-1}}
      ask flow  a_n = q^a_n * 1{P^a_n <= P^a_{n-1}} - q^a_{n-1} * 1{P^a_n >= P^a_{n-1}}
      OFI_n = b_n - a_n

    Spelled out for the bid side: price up -> +new size (fresh demand),
    price down -> -old size (demand withdrawn/consumed), price unchanged ->
    +(size change). The ask side mirrors this with signs flipped, so
    positive OFI = net buying pressure. Units: shares.
    """
    bp, bq = df[f"bid_price_{level}"], df[f"bid_size_{level}"]
    ap, aq = df[f"ask_price_{level}"], df[f"ask_size_{level}"]
    bp_prev, bq_prev = bp.shift(1), bq.shift(1)
    ap_prev, aq_prev = ap.shift(1), aq.shift(1)

    bid_flow = bq * (bp >= bp_prev) - bq_prev * (bp <= bp_prev)
    ask_flow = aq * (ap <= ap_prev) - aq_prev * (ap >= ap_prev)

    return (bid_flow - ask_flow).fillna(0.0).rename(f"ofi_event_l{level}")


def ofi_multilevel(df: pd.DataFrame, n_levels: int = 5,
                   decay: float = 1.0) -> pd.Series:
    """Sum of per-event OFI over levels 1..n_levels with weights decay**(lvl-1).

    decay=1.0 weights all levels equally; decay<1 down-weights deeper levels,
    reflecting that flow far from the touch is a weaker (and cheaper-to-fake)
    signal.
    """
    total = sum(
        decay ** (lvl - 1) * ofi_per_event(df, lvl) for lvl in range(1, n_levels + 1)
    )
    return total.rename(f"ofi_event_{n_levels}lvl")


def trailing_sum(per_event: pd.Series, window: int) -> pd.Series:
    """Trailing sum over the last `window` events, inclusive of the current one.

    Row t aggregates events (t-window+1 .. t): strictly past-and-present data.
    """
    return per_event.rolling(window, min_periods=window).sum()


def queue_imbalance(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """(bid size - ask size) / (bid size + ask size) at one level, in [-1, 1].

    The static cousin of OFI: a level, not a flow. Serves as a baseline.
    """
    bq, aq = df[f"bid_size_{level}"], df[f"ask_size_{level}"]
    return ((bq - aq) / (bq + aq)).rename(f"qimb_l{level}")


def trade_sign_flow(df: pd.DataFrame) -> pd.Series:
    """Per-event signed trade volume (positive = buyer-initiated), in shares.

    LOBSTER's `direction` is the side of the *resting* limit order, so the
    aggressor side is its negative: an execution against a sell limit
    (direction=-1) is a buyer-initiated trade. Non-execution events are 0.
    """
    is_trade = df["event_type"].isin([4, 5])
    signed = np.where(is_trade, -df["direction"] * df["size"], 0.0)
    return pd.Series(signed, index=df.index, name="trade_flow_event")


def build_features(df: pd.DataFrame, window: int, n_levels: int = 5) -> pd.DataFrame:
    """Assemble the feature matrix: OFI (L1 and multi-level), baselines.

    `window` is in events; all features at row t use data from rows <= t.
    """
    out = pd.DataFrame(index=df.index)
    out["ofi_l1"] = trailing_sum(ofi_per_event(df, 1), window)
    out["ofi_5lvl"] = trailing_sum(ofi_multilevel(df, n_levels), window)
    out["qimb"] = queue_imbalance(df, 1)
    out["trade_imb"] = trailing_sum(trade_sign_flow(df), window)
    return out
