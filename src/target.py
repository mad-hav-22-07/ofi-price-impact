"""Forward-return targets, aligned so features can never see them.

Convention: a feature at row t is built from rows <= t (see features.py).
The target at row t is the price change over rows (t, t+h]. The two windows
share the single boundary point t and overlap nowhere else.
"""

from __future__ import annotations

import pandas as pd


def future_return(price: pd.Series, horizon_events: int) -> pd.Series:
    """price(t + h) - price(t), h measured in book events. Dollars.

    The last h rows have no defined target and come back NaN; they are
    dropped at modeling time, never filled.
    """
    return (price.shift(-horizon_events) - price).rename(
        f"fret_{price.name}_{horizon_events}"
    )
