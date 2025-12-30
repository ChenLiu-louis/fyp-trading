from __future__ import annotations

"""
Sweep trading parameters WITHOUT retraining models.

Rationale:
- Training deep models is slow/expensive.
- A large part of realized performance comes from the trading rule:
  threshold, holding period, (optionally) allow_short.
This script reuses an existing backtest timeseries CSV (which already contains
date + simple_return + class probabilities) and recomputes positions/equity/stats
for a grid of parameters, then saves a ranked table to outputs/reports/.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fyp_trading.backtest import equity_and_stats_from_positions, generate_positions_from_proba
from fyp_trading.utils import resolve_outputs_dir, save_json


def main() -> None:
    out = resolve_outputs_dir()
    reports_dir: Path = out["reports"]

    # --- Choose which existing run to sweep ---
    # Default: latest informer_opt backtest timeseries
    candidates = sorted(reports_dir.glob("informer_opt_backtest_timeseries_*.csv"))
    if not candidates:
        raise FileNotFoundError("No informer_opt_backtest_timeseries_*.csv found under outputs/reports/.")
    src = candidates[-1]
    print("Using:", src.name)

    df = pd.read_csv(src)
    # Ensure required columns exist
    need = ["date", "simple_return", "proba_down", "proba_neutral", "proba_up"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {src}")

    simple_ret = pd.Series(df["simple_return"].values, index=pd.to_datetime(df["date"]))
    proba = df[["proba_down", "proba_neutral", "proba_up"]].values.astype(float)

    # --- Parameter grid (edit freely) ---
    thresholds = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
    holds = [1, 2, 3, 5, 10]
    allow_shorts = [False, True]

    results = []
    for thr in thresholds:
        for hold in holds:
            for allow_short in allow_shorts:
                pos = generate_positions_from_proba(
                    proba=proba,
                    proba_threshold=float(thr),
                    min_holding_period=int(hold),
                    allow_short=bool(allow_short),
                )
                pos_s = pd.Series(pos, index=simple_ret.index)
                _, stats = equity_and_stats_from_positions(
                    simple_return=simple_ret,
                    position=pos_s,
                    transaction_cost_bp=2.0,
                )
                stats.update({"proba_threshold": float(thr), "min_holding_period": float(hold), "allow_short": float(1.0 if allow_short else 0.0)})
                results.append(stats)

    res_df = pd.DataFrame(results)
    # Rank primarily by excess_total_return (vs Buy&Hold), then Sharpe
    sort_cols = [c for c in ["excess_total_return", "sharpe_ratio"] if c in res_df.columns]
    res_df = res_df.sort_values(sort_cols, ascending=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = reports_dir / f"trading_param_sweep_{ts}.csv"
    res_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv.name)

    # Save best config snapshot for convenience
    best = res_df.iloc[0].to_dict() if len(res_df) else {}
    out_json = reports_dir / f"trading_param_sweep_best_{ts}.json"
    save_json(out_json, best)
    print("Saved:", out_json.name)


if __name__ == "__main__":
    main()


