from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fyp_trading.backtest import equity_and_stats_from_positions
from fyp_trading.data import fetch_prices
from fyp_trading.report import plot_classic_backtest, save_figure
from fyp_trading.strategies import (
    BollingerConfig,
    BollingerSqueezeConfig,
    DonchianConfig,
    DualMaConfig,
    MacdConfig,
    RsiConfig,
    VolTargetMomentumConfig,
    bollinger_mean_reversion_signal,
    bollinger_squeeze_breakout_signal,
    donchian_breakout_signal,
    dual_ma_trend_signal,
    macd_crossover_signal,
    rsi_mean_reversion_signal,
    vol_target_momentum_position,
)
from fyp_trading.utils import resolve_outputs_dir, save_json, to_jsonable


def _slug(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def _jsonable_cfg(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return to_jsonable(x)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = resolve_outputs_dir()
    reports_dir: Path = out["reports"]
    plots_dir: Path = out["plots"]

    # --- Common backtest settings (align with notebooks) ---
    TICKER = "2800.HK"
    PERIOD = "3y"
    INTERVAL = "1d"
    COST_BP = 2.0
    ALLOW_SHORT = False

    df = fetch_prices(TICKER, PERIOD, INTERVAL)
    close = df["Close"].copy()
    high = df["High"].copy()
    low = df["Low"].copy()

    next_ret = np.log(close.shift(-1) / close)
    simple_ret = (np.exp(next_ret) - 1.0).dropna()

    # Strategy definitions
    strategies: List[Dict[str, Any]] = [
        {
            "name": "MACD",
            "position": lambda: macd_crossover_signal(close, MacdConfig(allow_short=ALLOW_SHORT)),
            "cfg": MacdConfig(allow_short=ALLOW_SHORT),
        },
        {
            "name": "RSI",
            "position": lambda: rsi_mean_reversion_signal(close, RsiConfig(allow_short=ALLOW_SHORT)),
            "cfg": RsiConfig(allow_short=ALLOW_SHORT),
        },
        {
            "name": "Bollinger(MR)",
            "position": lambda: bollinger_mean_reversion_signal(close, BollingerConfig(allow_short=ALLOW_SHORT)),
            "cfg": BollingerConfig(allow_short=ALLOW_SHORT),
        },
        {
            "name": "DualMA(10/50)",
            "position": lambda: dual_ma_trend_signal(close, DualMaConfig(fast=10, slow=50, allow_short=ALLOW_SHORT)),
            "cfg": DualMaConfig(fast=10, slow=50, allow_short=ALLOW_SHORT),
        },
        {
            "name": "Donchian(20/10)",
            "position": lambda: donchian_breakout_signal(high, low, DonchianConfig(entry=20, exit=10, allow_short=ALLOW_SHORT)),
            "cfg": DonchianConfig(entry=20, exit=10, allow_short=ALLOW_SHORT),
        },
        {
            "name": "VolTargetMom(60,20)",
            "position": lambda: vol_target_momentum_position(
                close,
                VolTargetMomentumConfig(
                    mom_lookback=60,
                    vol_lookback=20,
                    target_vol_annual=0.10,
                    max_leverage=1.0,
                    allow_short=True,
                ),
            ),
            "cfg": VolTargetMomentumConfig(
                mom_lookback=60,
                vol_lookback=20,
                target_vol_annual=0.10,
                max_leverage=1.0,
                allow_short=True,
            ),
        },
        {
            "name": "BollSqueeze(20,q0.2)",
            "position": lambda: bollinger_squeeze_breakout_signal(
                close,
                BollingerSqueezeConfig(
                    period=20,
                    num_std=2.0,
                    squeeze_lookback=120,
                    squeeze_quantile=0.2,
                    allow_short=ALLOW_SHORT,
                ),
            ),
            "cfg": BollingerSqueezeConfig(
                period=20,
                num_std=2.0,
                squeeze_lookback=120,
                squeeze_quantile=0.2,
                allow_short=ALLOW_SHORT,
            ),
        },
    ]

    all_stats: List[Dict[str, Any]] = []

    for s in strategies:
        name = str(s["name"])
        slug = _slug(name)
        cfg = s["cfg"]

        pos = s["position"]()
        pos = pos.loc[simple_ret.index]

        bt_df, stats = equity_and_stats_from_positions(
            simple_return=simple_ret,
            position=pos,
            transaction_cost_bp=COST_BP,
        )

        # Ensure a 'date' column for CSV/plots
        bt_out = bt_df.copy()
        bt_out.insert(0, "date", pd.to_datetime(bt_out.index))

        # Save artifacts
        save_json(
            reports_dir / f"classic_{slug}_run_config_{run_id}.json",
            {
                "run_id": run_id,
                "ticker": TICKER,
                "period": PERIOD,
                "interval": INTERVAL,
                "transaction_cost_bp": COST_BP,
                "strategy": name,
                "strategy_cfg": _jsonable_cfg(cfg),
            },
        )
        save_json(reports_dir / f"classic_{slug}_backtest_stats_{run_id}.json", stats)
        bt_out.to_csv(reports_dir / f"classic_{slug}_backtest_timeseries_{run_id}.csv", index=False)

        fig = plot_classic_backtest(bt_out, title=f"{name} | {TICKER} | cost={COST_BP:.0f}bps")
        save_figure(fig, plots_dir / f"classic_{slug}_backtest_{run_id}.png")

        all_stats.append({"strategy": name, **stats})

    # Summary table (CSV + LaTeX)
    summary = pd.DataFrame(all_stats)
    summary["excess_total_return"] = summary["total_return"] - summary["buyhold_total_return"]
    summary["excess_annualized_return"] = summary["annualized_return"] - summary["buyhold_annualized_return"]

    cols = [
        "strategy",
        "total_return",
        "buyhold_total_return",
        "excess_total_return",
        "annualized_return",
        "buyhold_annualized_return",
        "excess_annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "coverage",
        "transaction_cost_bp",
    ]
    summary = summary[[c for c in cols if c in summary.columns]].sort_values("sharpe_ratio", ascending=False)
    summary.to_csv(reports_dir / f"classic_strategy_compare_{run_id}.csv", index=False)

    latex = summary.to_latex(index=False, float_format="%.6f")
    (reports_dir / f"classic_strategy_compare_{run_id}.tex").write_text(latex, encoding="utf-8")

    print(f"[OK] Saved classic strategy artifacts to:")
    print(f"  - {reports_dir}")
    print(f"  - {plots_dir}")
    print(f"[RUN_ID] {run_id}")


if __name__ == "__main__":
    main()


