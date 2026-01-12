"""
Informer + FinBERT Sentiment Pipeline.

This script combines:
1. Technical features (extended set from build_extended_features)
2. FinBERT sentiment features (P_neg, P_neu, P_pos from news headlines)

The FinBERT model is used as a FIXED feature extractor (no fine-tuning).
For days without news, sentiment features are filled with zeros.

Usage:
    python run_informer_sentiment_pipeline.py

Results saved to outputs/reports/, outputs/plots/, outputs/models/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fyp_trading.backtest import backtest_from_cv_preds
from fyp_trading.config import LabelingConfig, PipelineConfig, TrainConfig
from fyp_trading.data import fetch_prices
from fyp_trading.features import build_extended_features
from fyp_trading.informer_cv import fixed_window_cv_informer
from fyp_trading.informer_models import InformerConfig
from fyp_trading.labeling import apply_3class_labeling
from fyp_trading.report import plot_ml_backtest, save_figure
from fyp_trading.sequences import make_sequences
from fyp_trading.sentiment_features import (
    build_sentiment_features_for_ticker,
    merge_sentiment_with_features,
)
from fyp_trading.utils import get_torch_device, resolve_outputs_dir, save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Informer + FinBERT Sentiment Pipeline")
    parser.add_argument("--ticker", type=str, default="2800.HK", help="Stock ticker")
    parser.add_argument("--period", type=str, default="10y", help="Data period (e.g., 3y, 5y, 10y)")
    parser.add_argument("--news-csv", type=str, default="data/news/news.csv", help="Path to news CSV")
    parser.add_argument("--fill-method", type=str, default="zero", choices=["zero", "ffill", "neutral"],
                        help="How to fill missing sentiment (days without news)")
    parser.add_argument("--no-sentiment", action="store_true", help="Run without sentiment (baseline)")
    parser.add_argument("--use-all-news", action="store_true", default=True,
                        help="Use ALL news regardless of ticker (default: True, good for ETFs)")
    parser.add_argument("--ticker-news-only", action="store_true",
                        help="Only use news for the specific ticker (overrides --use-all-news)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = args.seed
    set_global_seed(seed)
    device = get_torch_device()
    # Determine whether to use all news or ticker-specific
    use_all_news = args.use_all_news and not args.ticker_news_only

    print(f"Device: {device}")
    print(f"Ticker: {args.ticker}, Period: {args.period}")
    print(f"News CSV: {args.news_csv}")
    print(f"Sentiment fill method: {args.fill_method}")
    print(f"Use sentiment features: {not args.no_sentiment}")
    print(f"Use all news (market-wide): {use_all_news}")
    print("-" * 60)

    # ========== Configurations ==========
    label_cfg = LabelingConfig(k_dynamic=0.5)

    train_cfg = TrainConfig(
        epochs=120,
        batch_size=64,
        patience=20,
        lr=8e-4,
        weight_decay=2e-4,
        dropout=0.2,
        loss_mode="full_ce",
        label_smoothing=0.05,
        verbose=False,
    )

    pipe_cfg = PipelineConfig(
        ticker=args.ticker,
        period=args.period,
        lookback=90,
        train_window=420,
        val_size=21,
        test_size=21,
        step_size=21,
        horizon=1,
        proba_threshold=0.34,
        min_holding_period=5,
        transaction_cost_bp=2.0,
        backtest_days=252,
    )

    model_cfg = InformerConfig(
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.15,
        use_cls_token=True,
        factor=5,
        distil=True,
    )

    out_dirs = resolve_outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "informer_sent" if not args.no_sentiment else "informer_nosent"

    # ========== Load Price Data & Technical Features ==========
    df_raw = fetch_prices(pipe_cfg.ticker, pipe_cfg.period, pipe_cfg.interval)
    print(f"Price data: {df_raw.index.min().date()} ~ {df_raw.index.max().date()}, N={len(df_raw)}")

    feat_df, tech_feature_cols = build_extended_features(df_raw, horizon=pipe_cfg.horizon, use_log_return=True)
    print(f"Technical features: {len(tech_feature_cols)} columns")

    # ========== Load Sentiment Features (if enabled) ==========
    sentiment_cols = []
    if not args.no_sentiment:
        news_path = Path(args.news_csv)
        if news_path.exists():
            print("\n--- Extracting FinBERT Sentiment Features ---")
            daily_sent = build_sentiment_features_for_ticker(
                ticker=args.ticker,
                news_csv_path=news_path,
                model_name="ProsusAI/finbert",
                batch_size=32,
                device=str(device),
                agg_method="mean",
                verbose=True,
                use_all_news=use_all_news,
            )

            if len(daily_sent) > 0:
                feat_df, sentiment_cols = merge_sentiment_with_features(
                    feat_df, daily_sent, fill_method=args.fill_method
                )
                print(f"Added {len(sentiment_cols)} sentiment features: {sentiment_cols}")

                # Report coverage
                sent_coverage = (feat_df["sent_neg"] != 0).sum() / len(feat_df) * 100
                print(f"Sentiment coverage: {sent_coverage:.1f}% of trading days have news")
            else:
                print("WARNING: No sentiment data extracted. Proceeding without sentiment features.")
        else:
            print(f"WARNING: News file not found at {news_path}. Proceeding without sentiment features.")

    # Combine all feature columns
    all_feature_cols = tech_feature_cols + sentiment_cols
    print(f"\nTotal features: {len(all_feature_cols)} ({len(tech_feature_cols)} tech + {len(sentiment_cols)} sentiment)")

    # ========== Apply Labels ==========
    labels = apply_3class_labeling(feat_df, label_cfg)
    feat_df = feat_df.copy()
    feat_df["target_class"] = labels
    feat_df = feat_df.dropna(subset=["target_class"]).copy()
    feat_df["target_class"] = feat_df["target_class"].astype(int)

    # Check all features exist
    missing_cols = [c for c in all_feature_cols if c not in feat_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # ========== Create Sequences ==========
    X_seq, label_dict, seq_index = make_sequences(
        feat_df,
        feature_cols=all_feature_cols,
        lookback=pipe_cfg.lookback,
        label_cols=["target_class", "next_return"],
    )
    y_all = label_dict["target_class"].astype("int64")

    print(f"\nSequences: X_seq.shape={X_seq.shape}, labels={len(y_all)}")
    print(f"Date range: {seq_index.min().date()} ~ {seq_index.max().date()}")

    # ========== Walk-Forward CV with Informer ==========
    print("\n--- Running Walk-Forward CV ---")
    metrics_df, preds_df, last_artifacts = fixed_window_cv_informer(
        X_seq_all=X_seq,
        y_all=y_all,
        seq_index=seq_index,
        cfg_pipe=pipe_cfg,
        cfg_train=train_cfg,
        cfg_model=model_cfg,
        device=torch.device(device) if isinstance(device, str) else device,
        save_last_fold_artifacts=True,
    )

    # ========== Save CV Results ==========
    reports_dir = out_dirs["reports"]
    models_dir = out_dirs["models"]
    plots_dir = out_dirs["plots"]

    metrics_path = reports_dir / f"{run_name}_cv_metrics_{ts}.csv"
    preds_path = reports_dir / f"{run_name}_cv_preds_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    print(f"\nSaved: {metrics_path.name}, {preds_path.name}")

    if not preds_df.empty:
        d0 = pd.to_datetime(preds_df["date"]).min().date()
        d1 = pd.to_datetime(preds_df["date"]).max().date()
        print(f"OOS preds date range: {d0} ~ {d1}, N={len(preds_df)}")
    else:
        print("WARNING: preds_df is empty. Check CV windows.")

    # ========== Backtest ==========
    print("\n--- Running Backtest ---")
    bt_df, bt_stats = backtest_from_cv_preds(
        cv_preds=preds_df,
        feat_df=feat_df,
        proba_threshold=pipe_cfg.proba_threshold,
        min_holding_period=pipe_cfg.min_holding_period,
        transaction_cost_bp=pipe_cfg.transaction_cost_bp,
        backtest_days=pipe_cfg.backtest_days,
        allow_short=False,
    )

    bt_ts_path = reports_dir / f"{run_name}_backtest_timeseries_{ts}.csv"
    bt_stats_path = reports_dir / f"{run_name}_backtest_stats_{ts}.json"
    bt_df.to_csv(bt_ts_path, index=False)
    save_json(bt_stats_path, bt_stats)
    print(f"Saved: {bt_ts_path.name}, {bt_stats_path.name}")

    # Print key stats
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:       {bt_stats.get('total_return', 0):.4f} ({bt_stats.get('total_return', 0)*100:.2f}%)")
    print(f"  Buy&Hold Return:    {bt_stats.get('buyhold_total_return', 0):.4f} ({bt_stats.get('buyhold_total_return', 0)*100:.2f}%)")
    print(f"  Excess Return:      {bt_stats.get('excess_total_return', 0):.4f}")
    print(f"  Sharpe Ratio:       {bt_stats.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown:       {bt_stats.get('max_drawdown', 0):.4f}")
    print(f"  Coverage:           {bt_stats.get('coverage', 0):.4f}")
    print(f"  Num Trades:         {bt_stats.get('num_trades', 0)}")
    print(f"  Backtest Days:      {bt_stats.get('days', 0)}")
    print("=" * 60)

    if float(bt_stats.get("days", 0)) < float(pipe_cfg.backtest_days):
        print(
            f"WARNING: backtest uses only {bt_stats.get('days')} days "
            f"(< backtest_days={pipe_cfg.backtest_days})."
        )

    # ========== Plot ==========
    sent_str = "+Sentiment" if sentiment_cols else "NoSent"
    title = (
        f"Informer {sent_str} ({pipe_cfg.ticker}) | "
        f"thr={pipe_cfg.proba_threshold:.2f}, hold≥{pipe_cfg.min_holding_period}"
    )
    fig = plot_ml_backtest(bt_df, title=title, proba_threshold=pipe_cfg.proba_threshold)
    fig_path = plots_dir / f"{run_name}_backtest_{ts}.png"
    save_figure(fig, fig_path)
    print(f"Saved: {fig_path.name}")

    # ========== Save Model Artifacts ==========
    if last_artifacts is not None:
        model_path = models_dir / f"{run_name}_last_fold_{ts}.pt"
        torch.save(last_artifacts, model_path)
        print(f"Saved: {model_path.name}")

    # ========== Save Config ==========
    cfg_path = reports_dir / f"{run_name}_run_config_{ts}.json"
    save_json(
        cfg_path,
        {
            "seed": seed,
            "ticker": args.ticker,
            "period": args.period,
            "news_csv": args.news_csv,
            "fill_method": args.fill_method,
            "use_sentiment": not args.no_sentiment,
            "label_cfg": label_cfg,
            "train_cfg": train_cfg,
            "pipe_cfg": pipe_cfg,
            "model_cfg": model_cfg,
            "tech_feature_cols": tech_feature_cols,
            "sentiment_cols": sentiment_cols,
            "all_feature_cols": all_feature_cols,
        },
    )
    print(f"Saved: {cfg_path.name}")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()

