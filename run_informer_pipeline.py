from __future__ import annotations

from datetime import datetime

import torch

from fyp_trading.backtest import backtest_from_cv_preds
from fyp_trading.config import LabelingConfig, PipelineConfig, TrainConfig
from fyp_trading.data import fetch_prices
from fyp_trading.features import build_simple_features
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

    label_cfg = LabelingConfig()
    train_cfg = TrainConfig()
    pipe_cfg = PipelineConfig()

    # Informer-style encoder classifier (classification adaptation):
    # - ProbSparse attention controlled by factor
    # - distil=True downsamples between encoder layers
    model_cfg = InformerConfig(
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        use_cls_token=True,
        factor=5,
        distil=True,
    )

    out_dirs = resolve_outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_raw = fetch_prices(pipe_cfg.ticker, pipe_cfg.period, pipe_cfg.interval)
    print(f"Data: {df_raw.index.min().date()} ~ {df_raw.index.max().date()}, N={len(df_raw)}")

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

    metrics_path = reports_dir / f"informer_cv_metrics_{ts}.csv"
    preds_path = reports_dir / f"informer_cv_preds_{ts}.csv"
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
        allow_short=True,
    )
    bt_ts_path = reports_dir / f"informer_backtest_timeseries_{ts}.csv"
    bt_stats_path = reports_dir / f"informer_backtest_stats_{ts}.json"
    bt_df.to_csv(bt_ts_path, index=False)
    save_json(bt_stats_path, bt_stats)
    print("Saved:", bt_ts_path.name, bt_stats_path.name)

    title = (
        f"Informer-style Backtest ({pipe_cfg.ticker}) | thr={pipe_cfg.proba_threshold:.2f}, "
        f"hold≥{pipe_cfg.min_holding_period}, cost={pipe_cfg.transaction_cost_bp:.0f}bps"
    )
    fig = plot_ml_backtest(bt_df, title=title, proba_threshold=pipe_cfg.proba_threshold)
    fig_path = plots_dir / f"informer_backtest_{ts}.png"
    save_figure(fig, fig_path)
    print("Saved:", fig_path.name)

    if last_artifacts is not None:
        model_path = models_dir / f"informer_last_fold_{ts}.pt"
        torch.save(last_artifacts, model_path)
        print("Saved:", model_path.name)

    cfg_path = reports_dir / f"informer_run_config_{ts}.json"
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


