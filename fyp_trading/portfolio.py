from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .backtest import equity_and_stats_from_positions, generate_positions_from_proba


def portfolio_backtest_equal_weight_from_preds(
    preds_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    proba_threshold: float,
    min_holding_period: int,
    transaction_cost_bp: float,
    backtest_days: int = 252,
    allow_short: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Equal-weight portfolio backtest across tickers using per-(date,ticker) class probabilities.

    Required columns:
      preds_df: date,ticker,proba_down,proba_neutral,proba_up
      returns_df: date,ticker,simple_return

    Output:
      portfolio daily time series with equity + stats dict (same keys as single-asset backtest).
    """
    p = preds_df.copy()
    r = returns_df.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None)
    r["date"] = pd.to_datetime(r["date"]).dt.tz_localize(None)

    merged = p.merge(r[["date", "ticker", "simple_return"]], on=["date", "ticker"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping (date,ticker) between preds_df and returns_df.")

    # Restrict to last N days by date (portfolio horizon)
    merged = merged.sort_values(["date", "ticker"]).copy()
    unique_dates = pd.Index(merged["date"].unique()).sort_values()
    if len(unique_dates) > backtest_days:
        keep_dates = set(unique_dates[-backtest_days:])
        merged = merged[merged["date"].isin(keep_dates)].copy()

    # Compute discrete positions per ticker (groupby)
    def _pos_one(g: pd.DataFrame) -> pd.Series:
        proba = g[["proba_down", "proba_neutral", "proba_up"]].values
        pos = generate_positions_from_proba(
            proba=proba,
            proba_threshold=proba_threshold,
            min_holding_period=min_holding_period,
            allow_short=allow_short,
        )
        return pd.Series(pos, index=g.index)

    merged["position"] = merged.groupby("ticker", group_keys=False).apply(_pos_one)

    # Equal-weight daily portfolio return: mean over tickers of (pos * ret)
    merged["strategy_simple_return"] = merged["position"] * merged["simple_return"]

    # Transaction cost: sum over tickers of abs(delta position) * cost_rate, then average by ticker count
    cost_rate = transaction_cost_bp / 10000.0
    merged["pos_change"] = merged.groupby("ticker")["position"].diff().abs()
    # first day per ticker
    first_mask = merged.groupby("ticker")["date"].transform("min") == merged["date"]
    merged.loc[first_mask, "pos_change"] = merged.loc[first_mask, "position"].abs()
    merged["transaction_cost"] = merged["pos_change"] * cost_rate

    # Aggregate per date
    daily = (
        merged.groupby("date", as_index=False)
        .agg(
            {
                "strategy_simple_return": "mean",
                "simple_return": "mean",  # equal-weight buy&hold benchmark
                "transaction_cost": "mean",
                "position": "mean",  # average exposure
            }
        )
        .sort_values("date")
        .copy()
    )
    daily["strategy_after_cost"] = daily["strategy_simple_return"] - daily["transaction_cost"]
    daily["strategy_equity"] = (1 + daily["strategy_after_cost"]).cumprod()
    daily["buyhold_equity"] = (1 + daily["simple_return"]).cumprod()

    # Reuse the existing stats helper by treating "position" as the portfolio exposure
    # (for coverage/turnover/trades diagnostics, this is approximate; detailed per-ticker
    # diagnostics can be added later if needed).
    eq_df, stats = equity_and_stats_from_positions(
        simple_return=daily.set_index("date")["simple_return"],
        position=daily.set_index("date")["position"].round().astype(int) if allow_short else (daily.set_index("date")["position"] > 0).astype(int),
        transaction_cost_bp=transaction_cost_bp,
        benchmark_return=None,
    )
    # Overwrite equity with portfolio equity
    eq_df = eq_df.reset_index().rename(columns={"index": "date"})
    eq_df["date"] = pd.to_datetime(eq_df["date"]).dt.tz_localize(None)
    out = daily.merge(eq_df[["date", "strategy_equity", "buyhold_equity"]], on="date", how="left", suffixes=("", "_tmp"))
    out["strategy_equity"] = out["strategy_equity_tmp"].fillna(out["strategy_equity"])
    out["buyhold_equity"] = out["buyhold_equity_tmp"].fillna(out["buyhold_equity"])
    out = out.drop(columns=[c for c in out.columns if c.endswith("_tmp")])

    stats.update(
        {
            "proba_threshold": float(proba_threshold),
            "min_holding_period": float(min_holding_period),
            "allow_short": float(1.0 if allow_short else 0.0),
        }
    )
    return out, stats


