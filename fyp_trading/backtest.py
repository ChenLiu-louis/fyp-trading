from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .labeling import CLASS_ID_DOWN, CLASS_ID_NEUTRAL, CLASS_ID_UP


def generate_positions_from_proba(
    proba: np.ndarray,
    proba_threshold: float,
    min_holding_period: int,
    allow_short: bool = True,
) -> np.ndarray:
    """Convert class probabilities into discrete positions (-1/0/+1)."""
    proba_down = proba[:, CLASS_ID_DOWN]
    proba_up = proba[:, CLASS_ID_UP]
    raw_signal = np.zeros(len(proba), dtype=int)

    up_mask = (proba_up >= proba_threshold) & (proba_up >= proba_down)
    raw_signal[up_mask] = 1

    if allow_short:
        down_mask = (proba_down >= proba_threshold) & (proba_down > proba_up)
        raw_signal[down_mask] = -1

    positions = np.zeros(len(proba), dtype=int)
    holding = 0
    current_pos = 0

    for i in range(len(proba)):
        if holding > 0:
            positions[i] = current_pos
            holding -= 1
            continue
        new_signal = raw_signal[i]
        if new_signal != 0:
            current_pos = new_signal
            positions[i] = current_pos
            holding = max(min_holding_period - 1, 0)
        else:
            current_pos = 0
            positions[i] = 0
            holding = 0
    return positions


def prepare_backtest_df(cv_preds: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    df = cv_preds.sort_values("date").copy()
    if "date" not in df.columns:
        raise ValueError("cv_preds must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    ret_df = feat_df[["next_return"]].reset_index()
    first_col = ret_df.columns[0]
    if first_col != "date":
        ret_df = ret_df.rename(columns={first_col: "date"})
    ret_df["date"] = pd.to_datetime(ret_df["date"]).dt.tz_localize(None)

    bt = df.merge(ret_df, how="left", on="date")
    bt["simple_return"] = np.exp(bt["next_return"].fillna(0.0)) - 1.0
    return bt


def equity_and_stats_from_positions(
    simple_return: pd.Series,
    position: pd.Series,
    transaction_cost_bp: float = 2.0,
    benchmark_return: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Vectorized backtest with cost charged on abs(position change)."""
    df = pd.DataFrame({"simple_return": simple_return, "position": position}).copy()
    df["strategy_simple_return"] = df["position"] * df["simple_return"]

    pos_change = df["position"].diff().abs()
    pos_change.iloc[0] = abs(df["position"].iloc[0])
    cost_rate = transaction_cost_bp / 10000.0
    df["transaction_cost"] = pos_change * cost_rate
    df["strategy_after_cost"] = df["strategy_simple_return"] - df["transaction_cost"]

    df["strategy_equity"] = (1 + df["strategy_after_cost"]).cumprod()
    if benchmark_return is None:
        df["buyhold_equity"] = (1 + df["simple_return"]).cumprod()
    else:
        df["buyhold_equity"] = (1 + benchmark_return).cumprod()

    total_days = len(df)
    years = max(total_days / 252.0, 1.0 / 252.0)
    strategy_final = float(df["strategy_equity"].iloc[-1])
    buyhold_final = float(df["buyhold_equity"].iloc[-1])

    daily_mean = float(df["strategy_after_cost"].mean())
    daily_std = float(df["strategy_after_cost"].std())

    stats = {
        "days": float(total_days),
        "total_return": strategy_final - 1.0,
        "buyhold_total_return": buyhold_final - 1.0,
        "annualized_return": strategy_final ** (1.0 / years) - 1.0,
        "buyhold_annualized_return": buyhold_final ** (1.0 / years) - 1.0,
        "annualized_volatility": daily_std * np.sqrt(252.0),
        "sharpe_ratio": (daily_mean / daily_std) * np.sqrt(252.0) if daily_std > 1e-8 else float("nan"),
        "max_drawdown": float((df["strategy_equity"] / df["strategy_equity"].cummax() - 1.0).min()),
        "avg_trade_day_return": float(df.loc[df["position"] != 0, "strategy_after_cost"].mean())
        if (df["position"] != 0).any()
        else float("nan"),
        "coverage": float((df["position"] != 0).mean()),
        "transaction_cost_bp": float(transaction_cost_bp),
    }
    return df, stats


def backtest_from_cv_preds(
    cv_preds: pd.DataFrame,
    feat_df: pd.DataFrame,
    proba_threshold: float,
    min_holding_period: int,
    transaction_cost_bp: float,
    backtest_days: int = 252,
    allow_short: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    bt = prepare_backtest_df(cv_preds, feat_df)
    if len(bt) > backtest_days:
        bt = bt.iloc[-backtest_days:].copy()

    proba_array = bt[["proba_down", "proba_neutral", "proba_up"]].values
    bt["position"] = generate_positions_from_proba(
        proba_array,
        proba_threshold=proba_threshold,
        min_holding_period=min_holding_period,
        allow_short=allow_short,
    )

    eq_df, stats = equity_and_stats_from_positions(
        simple_return=bt["simple_return"],
        position=bt["position"],
        transaction_cost_bp=transaction_cost_bp,
    )
    # Keep key columns + computed columns
    out = bt[["date", "fold", "actual_class", "pred_class", "proba_down", "proba_neutral", "proba_up", "next_return", "simple_return", "position"]].copy()
    out = pd.concat([out.reset_index(drop=True), eq_df.reset_index(drop=True)[["strategy_simple_return","transaction_cost","strategy_after_cost","strategy_equity","buyhold_equity"]]], axis=1)
    stats.update(
        {
            "proba_threshold": float(proba_threshold),
            "min_holding_period": float(min_holding_period),
            "allow_short": float(1.0 if allow_short else 0.0),
        }
    )
    return out, stats


