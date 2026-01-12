from __future__ import annotations

"""
PatchTST experiment runner.

This mirrors the existing LSTM/Transformer/Informer pipelines:
data -> features -> labeling -> sequences -> walk-forward CV -> backtest -> plots -> outputs/*

Why this script exists:
- PatchTST is a strong recent baseline for time-series Transformers.
- It patchifies the time axis and uses channel-independence, which often improves stability.
"""

from datetime import datetime

import argparse
import torch

from fyp_trading.backtest import backtest_from_cv_preds
from fyp_trading.config import LabelingConfig, PipelineConfig, TrainConfig
from fyp_trading.data import fetch_prices
from fyp_trading.features import build_extended_features, build_simple_features
from fyp_trading.labeling import apply_3class_labeling
from fyp_trading.patchtst_cv import fixed_window_cv_patchtst
from fyp_trading.patchtst_models import PatchTSTConfig
from fyp_trading.report import plot_ml_backtest, save_figure
from fyp_trading.sequences import make_sequences
from fyp_trading.utils import get_torch_device, resolve_outputs_dir, save_json, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["simple", "extended"], default="extended")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--train-window", type=int, default=420)
    parser.add_argument("--loss-mode", choices=["masked_ud", "full_ce"], default="full_ce")
    parser.add_argument("--label-k", type=float, default=0.5)
    parser.add_argument("--thr", type=float, default=0.36)
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--allow-short", action="store_true", default=False)
    args = parser.parse_args()

    seed = 42
    set_global_seed(seed)
    device = get_torch_device()
    print("Device:", device)

    label_cfg = LabelingConfig(k_dynamic=float(args.label_k))
    train_cfg = TrainConfig(
        epochs=120,
        batch_size=64,
        patience=20,
        lr=8e-4,
        weight_decay=2e-4,
        dropout=0.2,
        loss_mode=str(args.loss_mode),
        label_smoothing=0.05 if args.loss_mode == "full_ce" else 0.0,
        verbose=False,
    )
    pipe_cfg = PipelineConfig(
        period=str(args.period),
        lookback=int(args.lookback),
        train_window=int(args.train_window),
        val_size=21,
        test_size=21,
        step_size=21,
        horizon=1,
        proba_threshold=float(args.thr),
        min_holding_period=int(args.hold),
        transaction_cost_bp=2.0,
        backtest_days=252,
    )

    # PatchTST config (good default for 96-step lookback: 16 patches of len 6)
    model_cfg = PatchTSTConfig(
        patch_len=6,
        stride=6,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.15,
        token_pool="mean",
        channel_agg="concat",
        head_dropout=0.2,
    )

    out_dirs = resolve_outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_raw = fetch_prices(pipe_cfg.ticker, pipe_cfg.period, pipe_cfg.interval)
    print(f"Data: {df_raw.index.min().date()} ~ {df_raw.index.max().date()}, N={len(df_raw)}")

    if args.features == "extended":
        feat_df, feature_cols = build_extended_features(df_raw, horizon=pipe_cfg.horizon, use_log_return=True)
    else:
        feat_df, feature_cols = build_simple_features(df_raw, horizon=pipe_cfg.horizon, use_log_return=True)

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

    metrics_df, preds_df, last_artifacts = fixed_window_cv_patchtst(
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

    metrics_path = reports_dir / f"patchtst_cv_metrics_{ts}.csv"
    preds_path = reports_dir / f"patchtst_cv_preds_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    print("Saved:", metrics_path.name, preds_path.name)

    bt_df, bt_stats = backtest_from_cv_preds(
        cv_preds=preds_df,
        feat_df=feat_df,
        proba_threshold=pipe_cfg.proba_threshold,
        min_holding_period=pipe_cfg.min_holding_period,
        transaction_cost_bp=pipe_cfg.transaction_cost_bp,
        backtest_days=pipe_cfg.backtest_days,
        allow_short=bool(args.allow_short),
    )
    bt_ts_path = reports_dir / f"patchtst_backtest_timeseries_{ts}.csv"
    bt_stats_path = reports_dir / f"patchtst_backtest_stats_{ts}.json"
    bt_df.to_csv(bt_ts_path, index=False)
    save_json(bt_stats_path, bt_stats)
    print("Saved:", bt_ts_path.name, bt_stats_path.name)

    title = (
        f"PatchTST Backtest ({pipe_cfg.ticker}) | feat={args.features}, thr={pipe_cfg.proba_threshold:.2f}, "
        f"hold≥{pipe_cfg.min_holding_period}, cost={pipe_cfg.transaction_cost_bp:.0f}bps, "
        f"{'long-short' if args.allow_short else 'long-only'}"
    )
    fig = plot_ml_backtest(bt_df, title=title, proba_threshold=pipe_cfg.proba_threshold)
    fig_path = plots_dir / f"patchtst_backtest_{ts}.png"
    save_figure(fig, fig_path)
    print("Saved:", fig_path.name)

    if last_artifacts is not None:
        model_path = models_dir / f"patchtst_last_fold_{ts}.pt"
        torch.save(last_artifacts, model_path)
        print("Saved:", model_path.name)

    cfg_path = reports_dir / f"patchtst_run_config_{ts}.json"
    save_json(
        cfg_path,
        {
            "seed": seed,
            "label_cfg": label_cfg,
            "train_cfg": train_cfg,
            "pipe_cfg": pipe_cfg,
            "model_cfg": model_cfg,
            "feature_cols": feature_cols,
            "cli": vars(args),
        },
    )
    print("Saved:", cfg_path.name)


if __name__ == "__main__":
    main()


