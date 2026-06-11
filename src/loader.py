"""Load LOBSTER message + orderbook files into one event-indexed DataFrame.

LOBSTER format (https://lobsterdata.com):
  message file:   [time, event_type, order_id, size, price, direction]
  orderbook file: [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...]
                  repeated for each level. Row i of the orderbook file is the
                  book state *after* event i of the message file.

Prices in both files are integers scaled by 10,000 (i.e. 5853300 = $585.33).
That scaling is undone here, once, and nowhere else in the codebase.

Event types: 1=new limit order, 2=partial cancel, 3=full delete,
4=execution of visible order, 5=execution of hidden order,
6=auction cross trade, 7=trading halt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SCALE = 10_000

MESSAGE_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def _orderbook_cols(n_levels: int) -> list[str]:
    cols = []
    for lvl in range(1, n_levels + 1):
        cols += [f"ask_price_{lvl}", f"ask_size_{lvl}",
                 f"bid_price_{lvl}", f"bid_size_{lvl}"]
    return cols


def load_lobster(
    message_path: str,
    orderbook_path: str,
    n_levels: int = 5,
    trim_minutes: float = 15.0,
) -> pd.DataFrame:
    """Return an event-indexed DataFrame of book states with derived prices.

    Each row is one order-book event (the book state immediately after it).
    The first/last `trim_minutes` of the session are dropped: the open and
    close are dominated by auction dynamics that the continuous-trading OFI
    model is not meant to describe.

    Columns: time, event_type, size, direction, per-level bid/ask price+size
    (in dollars / shares), mid, microprice, spread.
    """
    msg = pd.read_csv(message_path, header=None, names=MESSAGE_COLS)
    book = pd.read_csv(orderbook_path, header=None, names=_orderbook_cols(n_levels))
    if len(msg) != len(book):
        raise ValueError(f"message rows ({len(msg)}) != orderbook rows ({len(book)})")

    df = pd.concat([msg[["time", "event_type", "size", "direction"]], book], axis=1)

    # undo LOBSTER's integer price scaling -- here and only here
    price_cols = [c for c in df.columns if "price" in c]
    df[price_cols] = df[price_cols] / PRICE_SCALE

    # drop auction crosses and halts; they are not continuous-trading events
    df = df[~df["event_type"].isin([6, 7])]

    # trim the open/close auction-adjacent windows
    start = df["time"].min() + trim_minutes * 60
    end = df["time"].max() - trim_minutes * 60
    df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)

    bid_p, bid_s = df["bid_price_1"], df["bid_size_1"]
    ask_p, ask_s = df["ask_price_1"], df["ask_size_1"]

    df["mid"] = (bid_p + ask_p) / 2
    # microprice cross-weights sizes: a heavy bid queue pulls fair value
    # toward the ask, because the ask side is the one likely to break first
    df["microprice"] = (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s)
    df["spread"] = ask_p - bid_p

    return df


def sanity_report(df: pd.DataFrame) -> dict:
    """Quick invariant checks; raises on hard violations, returns summary stats."""
    if (df["spread"] <= 0).any():
        raise AssertionError("crossed or locked book found (spread <= 0)")
    if not df["time"].is_monotonic_increasing:
        raise AssertionError("timestamps are not sorted")
    return {
        "n_events": len(df),
        "session_minutes": (df["time"].max() - df["time"].min()) / 60,
        "events_per_second": len(df) / (df["time"].max() - df["time"].min()),
        "mid_first": df["mid"].iloc[0],
        "mid_last": df["mid"].iloc[-1],
        "median_spread": df["spread"].median(),
        "mean_spread": df["spread"].mean(),
        "median_l1_depth_shares": float(
            np.median(df["bid_size_1"] + df["ask_size_1"])
        ),
    }
