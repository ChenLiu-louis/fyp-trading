from __future__ import annotations

"""
Next-stage pipeline: fine-tune a pretrained financial text model (FinBERT) on HK tickers,
then produce standardized artifacts under outputs/ and run a cost-aware backtest.

Inputs:
- Multi-ticker OHLCV from yfinance (price labels)
- News CSV: data/news/news.csv with schema: date,ticker,text

Outputs:
- outputs/reports/finbert_* (train config, metrics, preds, backtest stats/timeseries)
- outputs/plots/finbert_backtest_*.png

Notes:
- This script requires additional deps:
  pip install transformers datasets accelerate sentencepiece
"""

from datetime import datetime

import numpy as np
import pandas as pd

from fyp_trading.config import LabelingConfig, PipelineConfig, TrainConfig
from fyp_trading.news import NewsConfig
from fyp_trading.news_labels import build_news_training_table, build_price_label_table_for_universe
from fyp_trading.portfolio import portfolio_backtest_equal_weight_from_preds
from fyp_trading.report import plot_ml_backtest, save_figure
from fyp_trading.text_models import FinBertConfig, finbert_model, finbert_tokenizer, predict_proba_finbert
from fyp_trading.universe import UniverseConfig, default_hk_universe_small
from fyp_trading.utils import resolve_outputs_dir, save_json, set_global_seed


def main() -> None:
    seed = 42
    set_global_seed(seed)

    # Universe (HK)
    uni = UniverseConfig(tickers=default_hk_universe_small(), period="10y", interval="1d")

    # Labeling and pipeline settings (for trade mapping)
    label_cfg = LabelingConfig(k_dynamic=0.6)
    pipe_cfg = PipelineConfig(
        ticker="MULTI_HK",
        period=uni.period,
        interval=uni.interval,
        # used only by trading layer
        proba_threshold=0.55,
        min_holding_period=2,
        transaction_cost_bp=2.0,
        backtest_days=252,
    )

    # FinBERT fine-tune config (text model)
    fm_cfg = FinBertConfig(model_name="ProsusAI/finbert", epochs=3, batch_size=16, lr=2e-5)
    news_cfg = NewsConfig(csv_path="data/news/news.csv")

    out = resolve_outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = out["reports"]
    plots_dir = out["plots"]
    models_dir = out["models"]

    # 1) Build supervised table (weak supervision): (date,ticker,text)->target_class
    df = build_news_training_table(
        tickers=uni.tickers,
        period=uni.period,
        interval=uni.interval,
        news_cfg=news_cfg,
        label_cfg=label_cfg,
    )
    if df.empty:
        raise RuntimeError(
            "No news training data found. Please create data/news/news.csv with columns date,ticker,text.\n"
            "Example row:\n"
            "  2025-01-15,2800.HK,Some news headline...\n"
        )

    # 2) Time-respecting split by date (train on first 80% dates, test on last 20% dates)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = pd.Index(df["date"].unique()).sort_values()
    cut = int(len(dates) * 0.8)
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    # 3) Fine-tune (simple, HuggingFace Trainer)
    tok = finbert_tokenizer(fm_cfg)
    model = finbert_model(fm_cfg, num_labels=3)

    # Minimal Trainer loop (optional; requires transformers + datasets + accelerate)
    try:
        import torch
        from datasets import Dataset  # type: ignore
        from transformers import TrainingArguments, Trainer  # type: ignore

        def encode(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=fm_cfg.max_length)

        train_ds = Dataset.from_pandas(train_df[["text", "target_class"]].rename(columns={"target_class": "labels"}))
        test_ds = Dataset.from_pandas(test_df[["text", "target_class"]].rename(columns={"target_class": "labels"}))
        train_ds = train_ds.map(encode, batched=True)
        test_ds = test_ds.map(encode, batched=True)
        cols = ["input_ids", "attention_mask", "labels"]
        train_ds.set_format(type="torch", columns=cols)
        test_ds.set_format(type="torch", columns=cols)

        args = TrainingArguments(
            output_dir=str(models_dir / f"finbert_ckpt_{ts}"),
            per_device_train_batch_size=fm_cfg.batch_size,
            per_device_eval_batch_size=fm_cfg.batch_size,
            learning_rate=fm_cfg.lr,
            num_train_epochs=fm_cfg.epochs,
            weight_decay=fm_cfg.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            report_to=[],
            seed=fm_cfg.seed,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            pred = logits.argmax(axis=-1)
            acc = float((pred == labels).mean())
            return {"accuracy": acc}

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
        trainer.train()
        eval_metrics = trainer.evaluate()
    except Exception as e:
        # Fallback: skip fine-tuning and just use pretrained head
        eval_metrics = {"warning": f"trainer_not_run: {type(e).__name__}: {e}"}

    # 4) Predict probabilities on test set
    proba = predict_proba_finbert(model, tok, test_df["text"].tolist(), batch_size=fm_cfg.batch_size)
    preds = proba.argmax(axis=1)
    maxp = proba.max(axis=1)
    pred_class = preds.copy()
    pred_class[maxp < pipe_cfg.proba_threshold] = 1  # Neutral

    preds_df = pd.DataFrame(
        {
            "fold": 0,
            "date": pd.to_datetime(test_df["date"]).dt.tz_localize(None),
            "actual_class": test_df["target_class"].astype(int).values,
            "pred_class": pred_class.astype(int),
            "proba_down": proba[:, 0],
            "proba_neutral": proba[:, 1],
            "proba_up": proba[:, 2],
            "ticker": test_df["ticker"].astype(str).values,
        }
    )

    # 5) Portfolio backtest (equal-weight across tickers)
    price_tbl = build_price_label_table_for_universe(uni.tickers, uni.period, uni.interval, label_cfg)
    returns_df = price_tbl[["date", "ticker", "simple_return"]].copy()
    bt_df, bt_stats = portfolio_backtest_equal_weight_from_preds(
        preds_df=preds_df,
        returns_df=returns_df,
        proba_threshold=pipe_cfg.proba_threshold,
        min_holding_period=pipe_cfg.min_holding_period,
        transaction_cost_bp=pipe_cfg.transaction_cost_bp,
        backtest_days=pipe_cfg.backtest_days,
        allow_short=False,
    )

    # 6) Save artifacts
    metrics_path = reports_dir / f"finbert_eval_metrics_{ts}.json"
    save_json(metrics_path, eval_metrics)
    preds_path = reports_dir / f"finbert_news_preds_{ts}.csv"
    preds_df.to_csv(preds_path, index=False)

    bt_ts_path = reports_dir / f"finbert_portfolio_backtest_timeseries_{ts}.csv"
    bt_stats_path = reports_dir / f"finbert_portfolio_backtest_stats_{ts}.json"
    bt_df.to_csv(bt_ts_path, index=False)
    save_json(bt_stats_path, bt_stats)

    # Save model weights (best effort)
    try:
        import torch

        torch.save(model.state_dict(), models_dir / f"finbert_state_{ts}.pt")
    except Exception:
        pass

    title = (
        f"FinBERT-News Portfolio Backtest (HK) | thr={pipe_cfg.proba_threshold:.2f}, "
        f"hold≥{pipe_cfg.min_holding_period}, cost={pipe_cfg.transaction_cost_bp:.0f}bps, equal-weight"
    )
    fig = plot_ml_backtest(bt_df, title=title, proba_threshold=pipe_cfg.proba_threshold)
    fig_path = plots_dir / f"finbert_portfolio_backtest_{ts}.png"
    save_figure(fig, fig_path)

    cfg_path = reports_dir / f"finbert_run_config_{ts}.json"
    save_json(
        cfg_path,
        {
            "seed": seed,
            "universe": uni,
            "label_cfg": label_cfg,
            "pipe_cfg": pipe_cfg,
            "finbert_cfg": fm_cfg,
            "news_cfg": news_cfg,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        },
    )

    print("Saved:", metrics_path.name, preds_path.name, bt_ts_path.name, bt_stats_path.name, fig_path.name, cfg_path.name)

    # --- Intended outputs (after return join is implemented) ---
    # metrics_path = reports_dir / f"finbert_cv_metrics_{ts}.csv"
    # preds_path = reports_dir / f"finbert_cv_preds_{ts}.csv"
    # save_json(reports_dir / f"finbert_run_config_{ts}.json", {...})
    # save model: torch.save(model.state_dict(), models_dir / f"finbert_{ts}.pt")
    # bt_df, bt_stats = backtest_from_cv_preds(...)
    # plot + save_figure(...)


if __name__ == "__main__":
    main()


