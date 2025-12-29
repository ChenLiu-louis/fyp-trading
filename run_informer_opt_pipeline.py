from __future__ import annotations

"""
Optimized Informer-style pipeline.

This script intentionally changes multiple knobs (features, lookback, training window,
loss mode, and trading constraints) to pursue better risk-adjusted performance vs Buy&Hold,
while keeping the evaluation time-respecting (walk-forward CV) and outputs standardized
under outputs/.
"""

from datetime import datetime

import torch
import pandas as pd

from fyp_trading.backtest import backtest_from_cv_preds
from fyp_trading.config import LabelingConfig, PipelineConfig, TrainConfig
from fyp_trading.data import fetch_prices
from fyp_trading.features import build_extended_features
from fyp_trading.informer_cv import fixed_window_cv_informer
from fyp_trading.informer_models import InformerConfig
from fyp_trading.labeling import apply_3class_labeling
from fyp_trading.report import plot_ml_backtest, save_figure
from fyp_trading.sequences import make_sequences
from fyp_trading.utils import get_torch_device, resolve_outputs_dir, save_json, set_global_seed


def main() -> None:
    seed = 42
    set_global_seed(seed)
    device = get_torch_device()
    print("Device:", device)

    # --- Key optimizations informed by prior findings ---
    # 1) Labels: easier directional threshold -> more Up/Down samples (less "all-neutral")
    label_cfg = LabelingConfig(k_dynamic=0.5)

    # 2) Training: use full 3-class CE (better calibrated probabilities than masked UD-only)
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

    # 3) Pipeline windows: longer context + larger train window for time-series models
    pipe_cfg = PipelineConfig(
        # IMPORTANT: with long lookback + large train window + longer indicators (e.g. SMA_100),
        # period=3y may leave too few out-of-sample test days (e.g. only ~63 days).
        # Use a longer history so the backtest can cover ~252 trading days consistently.
        period="10y",
        lookback=90,
        train_window=420,
        val_size=21,
        test_size=21,
        step_size=21,
        horizon=1,
        # With 3-class CE, probabilities are often less "peaky"; too high a threshold
        # can lead to zero trades (flat equity). Start lower, then tune based on coverage/turnover.
        proba_threshold=0.34,
        min_holding_period=5,
        transaction_cost_bp=2.0,
        backtest_days=252,
    )

    # 4) Informer-style model: slightly larger capacity; ProbSparse + distilling
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

    df_raw = fetch_prices(pipe_cfg.ticker, pipe_cfg.period, pipe_cfg.interval)
    print(f"Data: {df_raw.index.min().date()} ~ {df_raw.index.max().date()}, N={len(df_raw)}")

    feat_df, feature_cols = build_extended_features(df_raw, horizon=pipe_cfg.horizon, use_log_return=True)
    labels = apply_3class_labeling(feat_df, label_cfg)
    feat_df = feat_df.copy()
    feat_df["target_class"] = labels
    feat_df = feat_df.dropna(subset=["target_class"]).copy()
    feat_df["target_class"] = feat_df["target_class"].astype(int)

    X_seq, label_dict, seq_index = make_sequences(
        feat_df,
        feature_cols=feature_cols,
        lookback=pipe_cfg.lookback,
        label_cols=["target_class", "next_return"],
    )
    y_all = label_dict["target_class"].astype("int64")

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

    reports_dir = out_dirs["reports"]
    models_dir = out_dirs["models"]
    plots_dir = out_dirs["plots"]

    metrics_path = reports_dir / f"informer_opt_cv_metrics_{ts}.csv"
    preds_path = reports_dir / f"informer_opt_cv_preds_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    print("Saved:", metrics_path.name, preds_path.name)
    if not preds_df.empty:
        d0 = pd.to_datetime(preds_df["date"]).min().date()
        d1 = pd.to_datetime(preds_df["date"]).max().date()
        print(f"OOS preds date range: {d0} ~ {d1}, N={len(preds_df)}")
    else:
        print("WARNING: preds_df is empty (no out-of-sample predictions). Check CV windows/filters.")

    # Long-only trading is often more realistic for ETFs and avoids systematic short drag in bull regimes.
    bt_df, bt_stats = backtest_from_cv_preds(
        cv_preds=preds_df,
        feat_df=feat_df,
        proba_threshold=pipe_cfg.proba_threshold,
        min_holding_period=pipe_cfg.min_holding_period,
        transaction_cost_bp=pipe_cfg.transaction_cost_bp,
        backtest_days=pipe_cfg.backtest_days,
        allow_short=False,
    )
    bt_ts_path = reports_dir / f"informer_opt_backtest_timeseries_{ts}.csv"
    bt_stats_path = reports_dir / f"informer_opt_backtest_stats_{ts}.json"
    bt_df.to_csv(bt_ts_path, index=False)
    save_json(bt_stats_path, bt_stats)
    print("Saved:", bt_ts_path.name, bt_stats_path.name)
    if float(bt_stats.get("days", 0.0)) < float(pipe_cfg.backtest_days):
        print(
            f"WARNING: backtest uses only {bt_stats.get('days')} days (< backtest_days={pipe_cfg.backtest_days}). "
            "This usually means too few OOS predictions (period too short or windows too large)."
        )

    title = (
        f"Informer-OPT Backtest ({pipe_cfg.ticker}) | thr={pipe_cfg.proba_threshold:.2f}, "
        f"hold≥{pipe_cfg.min_holding_period}, cost={pipe_cfg.transaction_cost_bp:.0f}bps, long-only"
    )
    fig = plot_ml_backtest(bt_df, title=title, proba_threshold=pipe_cfg.proba_threshold)
    fig_path = plots_dir / f"informer_opt_backtest_{ts}.png"
    save_figure(fig, fig_path)
    print("Saved:", fig_path.name)

    if last_artifacts is not None:
        model_path = models_dir / f"informer_opt_last_fold_{ts}.pt"
        torch.save(last_artifacts, model_path)
        print("Saved:", model_path.name)

    cfg_path = reports_dir / f"informer_opt_run_config_{ts}.json"
    save_json(
        cfg_path,
        {
            "seed": seed,
            "label_cfg": label_cfg,
            "train_cfg": train_cfg,
            "pipe_cfg": pipe_cfg,
            "model_cfg": model_cfg,
            "feature_cols": feature_cols,
        },
    )
    print("Saved:", cfg_path.name)


if __name__ == "__main__":
    main()


