"""Hand-checked OFI cases. Run: python -m pytest tests/ or python tests/test_ofi.py"""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import ofi_per_event, trade_sign_flow


def _book(rows):
    return pd.DataFrame(rows, columns=["bid_price_1", "bid_size_1",
                                       "ask_price_1", "ask_size_1"])


def test_ofi_four_event_sequence():
    """Start: bid 150.01x1200, ask 150.02x400.
    1) market buy eats 300 from ask          -> OFI +300
    2) +500 shares join the bid              -> OFI +500
    3) ask cancelled, best ask -> 150.03x900 -> OFI +100
    4) market sell eats 1000 from bid        -> OFI -1000
    """
    book = _book([
        (150.01, 1200, 150.02, 400),
        (150.01, 1200, 150.02, 100),
        (150.01, 1700, 150.02, 100),
        (150.01, 1700, 150.03, 900),
        (150.01,  700, 150.03, 900),
    ])
    ofi = ofi_per_event(book, level=1)
    assert ofi.iloc[0] == 0          # no previous book -> defined as 0
    assert ofi.iloc[1] == 300
    assert ofi.iloc[2] == 500
    assert ofi.iloc[3] == 100
    assert ofi.iloc[4] == -1000


def test_ofi_bid_price_improvement():
    """New best bid at a higher price counts its full size as fresh demand."""
    book = _book([
        (150.01, 1200, 150.03, 400),
        (150.02,  600, 150.03, 400),
    ])
    assert ofi_per_event(book, level=1).iloc[1] == 600


def test_ofi_ask_price_improvement_is_negative():
    """New, more aggressive ask = selling pressure = negative OFI."""
    book = _book([
        (150.01, 1200, 150.03, 400),
        (150.01, 1200, 150.02, 250),
    ])
    assert ofi_per_event(book, level=1).iloc[1] == -250


def test_trade_sign_aggressor_convention():
    """Execution against a sell limit (direction=-1) is buyer-initiated."""
    df = pd.DataFrame({
        "event_type": [4, 4, 1],
        "direction": [-1, 1, 1],
        "size": [100, 50, 999],
    })
    flow = trade_sign_flow(df)
    assert flow.iloc[0] == 100    # buyer-initiated
    assert flow.iloc[1] == -50    # seller-initiated
    assert flow.iloc[2] == 0      # limit order, not a trade


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
