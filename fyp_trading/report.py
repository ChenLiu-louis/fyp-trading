from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_ml_backtest(bt_df: pd.DataFrame, title: str, proba_threshold: float) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax0, ax1, ax2 = axes
    ax0.plot(bt_df["date"], bt_df["strategy_equity"], label="Strategy (after cost)", color="tab:blue", linewidth=2)
    ax0.plot(bt_df["date"], bt_df["buyhold_equity"], label="Buy & Hold", color="tab:orange", linestyle="--")
    ax0.set_ylabel("Equity")
    ax0.set_title(title)
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    ax1.step(bt_df["date"], bt_df["position"], where="post", color="tab:green", linewidth=1.5)
    ax1.set_ylabel("Position")
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(["Short", "Flat", "Long"])
    ax1.grid(True, alpha=0.3)

    if "proba_up" in bt_df.columns and "proba_down" in bt_df.columns:
        ax2.plot(bt_df["date"], bt_df["proba_up"], label="P(Up)", color="tab:blue", alpha=0.8)
        ax2.plot(bt_df["date"], bt_df["proba_down"], label="P(Down)", color="tab:red", alpha=0.8)
        ax2.axhline(proba_threshold, color="gray", linestyle="--", alpha=0.6, label="Confidence thr")
        ax2.legend(loc="upper right")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_classic_backtest(bt_df: pd.DataFrame, title: str) -> plt.Figure:
    """
    Plot equity curve + position for classic (rule-based) strategies.

    Expected columns:
    - date (datetime-like) OR use the DataFrame index as date axis
    - strategy_equity, buyhold_equity
    - position (optional, but recommended)
    """
    if "date" in bt_df.columns:
        x = pd.to_datetime(bt_df["date"])
    else:
        x = pd.to_datetime(bt_df.index)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax0, ax1 = axes
    ax0.plot(x, bt_df["strategy_equity"], label="Strategy (after cost)", color="tab:blue", linewidth=2)
    ax0.plot(x, bt_df["buyhold_equity"], label="Buy & Hold", color="tab:orange", linestyle="--")
    ax0.set_ylabel("Equity")
    ax0.set_title(title)
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    if "position" in bt_df.columns:
        ax1.step(x, bt_df["position"], where="post", color="tab:green", linewidth=1.5)
        ax1.set_ylabel("Position")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.axis("off")

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


