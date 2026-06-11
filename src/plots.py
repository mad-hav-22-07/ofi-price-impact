"""Deliverable figures. Every figure carries its takeaway in the title/caption."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CENTS = 100  # dollars -> cents


def fig_sanity(book: pd.DataFrame, path: str) -> None:
    """Mid-price path and spread distribution: the data passes the eye test."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    hours = book["time"] / 3600
    axes[0].plot(hours, book["mid"], lw=0.5, color="navy")
    axes[0].set(xlabel="hour of day", ylabel="mid ($)",
                title="AAPL mid price, 2012-06-21")
    spread_c = book["spread"] * CENTS
    axes[1].hist(spread_c[spread_c < spread_c.quantile(0.99)], bins=40,
                 color="darkorange", edgecolor="black", lw=0.3)
    axes[1].axvline(spread_c.median(), color="black", ls="--",
                    label=f"median {spread_c.median():.0f}c")
    axes[1].set(xlabel="bid-ask spread (cents)", ylabel="events",
                title="Spread distribution (1st-99th pct)")
    axes[1].legend()
    fig.suptitle("Sanity: clean mid path; the spread the strategy must pay",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ofi_vs_return(
    x_test: np.ndarray,
    y_test: np.ndarray,
    beta: float,
    alpha: float,
    r2_oos: float,
    horizon: int,
    path: str,
    n_bins: int = 40,
) -> None:
    """Binned scatter of standardized OFI vs realized future return (test set),
    with the line fitted on TRAIN overlaid. The linearity is the CKS result."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x_test, qs)
    idx = np.clip(np.searchsorted(edges, x_test, side="right") - 1, 0, n_bins - 1)
    bx = np.array([x_test[idx == b].mean() for b in range(n_bins)])
    by = np.array([y_test[idx == b].mean() * CENTS for b in range(n_bins)])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(bx, by, s=28, color="navy", zorder=3,
               label=f"test data, {n_bins} equal-count bins")
    # span the binned range, not the raw extremes -- a handful of >10-sd
    # outlier events would otherwise squash the visible data into a corner
    xx = np.linspace(bx.min(), bx.max(), 100)
    ax.plot(xx, (alpha + beta * xx) * CENTS, color="crimson", lw=1.8,
            label=f"train fit: beta={beta * CENTS:.2f} cents/sd")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set(xlabel="multi-level OFI over past 50 events (train-standardized)",
           ylabel=f"mid change over next {horizon} events (cents)",
           title=f"OFI vs future return -- out-of-sample R$^2$ = {r2_oos:.3f}")
    ax.legend()
    fig.text(0.5, -0.04,
             "Impact is approximately linear in OFI (Cont-Kukanov-Stoikov). "
             "Line fitted on train, points are held-out test data.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_r2_decay(decay: pd.DataFrame, path: str) -> None:
    """OOS R^2 vs horizon per feature. The information is real but perishable."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {"ofi_l1": ("navy", "o"), "ofi_5lvl": ("crimson", "s"),
              "trade_imb": ("darkgreen", "^"), "qimb": ("gray", "d")}
    for feat, grp in decay.groupby("feature"):
        c, m = styles.get(feat, ("black", "x"))
        grp = grp.sort_values("horizon")
        ax.plot(grp["horizon"], grp["r2_oos"], marker=m, color=c, label=feat)
    ax.set(xscale="log", xlabel="prediction horizon (events, log scale)",
           ylabel="out-of-sample R$^2$",
           title="Predictability vs horizon (mid-price target)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.legend(title="feature")
    fig.text(0.5, -0.04,
             "R$^2$ peaks at intermediate horizons (the tick grid pins very "
             "short moves) and decays beyond ~100 events: the signal is perishable.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cost_honesty(sweep: pd.DataFrame, path: str) -> None:
    """Gross vs net per trade across entry thresholds: where the edge dies."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(sweep))
    w = 0.38
    ax.bar(x - w / 2, sweep["gross_per_trade"] * CENTS, w,
           color="seagreen", label="gross (mid-to-mid)")
    ax.bar(x + w / 2, sweep["net_per_trade"] * CENTS, w,
           color="firebrick", label="net (executed at the touch)")
    for i, row in sweep.reset_index().iterrows():
        ax.annotate(f"{int(row['n_trades'])} trades\nhit {row['hit_rate']:.0%}",
                    (i - w / 2, row["gross_per_trade"] * CENTS),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(top=float(sweep["gross_per_trade"].max()) * CENTS + 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.0%}" for q in sweep["tau_quantile"]])
    ax.set(xlabel="entry threshold (train quantile of |predicted move|)",
           ylabel="P&L per trade (cents/share)",
           title="The signal is real; the spread eats it")
    ax.legend()
    fig.text(0.5, -0.04,
             "Gross P&L per trade is positive at every threshold; charging the "
             "actual touch prices flips all of them negative.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
