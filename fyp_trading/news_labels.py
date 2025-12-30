from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import LabelingConfig
from .features import build_simple_features
from .labeling import apply_3class_labeling
from .multiticker import align_on_common_dates, fetch_prices_multi
from .news import NewsConfig, aggregate_news_per_day, load_news


@dataclass(frozen=True)
class NewsLabelJoinConfig:
    """
    Join config for (news text) -> (next-day direction label).
    """

    # If True, align all tickers to common dates (helps pooled fine-tune stability).
    align_common_dates: bool = True


def build_price_labels_for_universe(
    tickers: List[str],
    period: str,
    interval: str,
    label_cfg: LabelingConfig,
) -> Dict[str, pd.DataFrame]:
    """
    For each ticker:
    - fetch prices
    - build features (simple)
    - apply 3-class labels
    Output: {ticker: df with index=date and columns next_return,target_class}
    """
    dfs = fetch_prices_multi(tickers, period, interval)
    if label_cfg is None:
        label_cfg = LabelingConfig()

    if dfs and all(isinstance(v, pd.DataFrame) for v in dfs.values()):
        dfs = align_on_common_dates(dfs)  # optional; safe default

    out: Dict[str, pd.DataFrame] = {}
    for t, df_raw in dfs.items():
        feat_df, _ = build_simple_features(df_raw, horizon=1, use_log_return=True)
        labels = apply_3class_labeling(feat_df, label_cfg)
        dd = feat_df.copy()
        dd["target_class"] = labels
        dd = dd.dropna(subset=["target_class"]).copy()
        dd["target_class"] = dd["target_class"].astype(int)
        out[t] = dd[["next_return", "target_class"]].copy()
    return out


def build_price_label_table_for_universe(
    tickers: List[str],
    period: str,
    interval: str,
    label_cfg: LabelingConfig,
) -> pd.DataFrame:
    """
    Build a long-form table:
      date, ticker, next_return, simple_return, target_class
    """
    per = build_price_labels_for_universe(tickers, period, interval, label_cfg)
    rows = []
    for t, df in per.items():
        dd = df.copy()
        dd = dd.reset_index()
        date_col = dd.columns[0]
        if date_col != "date":
            dd = dd.rename(columns={date_col: "date"})
        dd["date"] = pd.to_datetime(dd["date"]).dt.tz_localize(None)
        dd["ticker"] = t
        dd["simple_return"] = np.exp(dd["next_return"].astype(float)) - 1.0
        rows.append(dd[["date", "ticker", "next_return", "simple_return", "target_class"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["date", "ticker", "next_return", "simple_return", "target_class"]
    )


def build_news_training_table(
    tickers: List[str],
    period: str,
    interval: str,
    news_cfg: NewsConfig,
    label_cfg: LabelingConfig,
    join_cfg: NewsLabelJoinConfig = NewsLabelJoinConfig(),
) -> pd.DataFrame:
    """
    Build a supervised table for fine-tuning a text model:
      (date, ticker, text) -> target_class

    Notes:
    - News is aggregated per (date,ticker). It is assumed to be available by end of day t.
    - Label is next-day direction based on next_return at t+1 (from price series), which is
      a standard setup but still noisy and should be treated as weak supervision.
    """
    news = load_news(news_cfg)
    news = aggregate_news_per_day(news)
    if news.empty:
        return pd.DataFrame(columns=["date", "ticker", "text", "target_class"])

    price_labels = build_price_labels_for_universe(tickers, period, interval, label_cfg)

    rows = []
    for t in tickers:
        if t not in price_labels:
            continue
        pl = price_labels[t].reset_index().rename(columns={price_labels[t].index.name or "index": "date"})
        pl["date"] = pd.to_datetime(pl["date"]).dt.tz_localize(None)

        nn = news[news["ticker"] == t].copy()
        nn["date"] = pd.to_datetime(nn["date"]).dt.tz_localize(None)
        merged = nn.merge(pl[["date", "target_class"]], on="date", how="inner")
        if merged.empty:
            continue
        rows.append(merged[["date", "ticker", "text", "target_class"]])

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "ticker", "text", "target_class"])
    # Optional: ensure common dates across tickers
    if join_cfg.align_common_dates and not out_df.empty:
        common_dates = out_df.groupby("ticker")["date"].apply(lambda s: set(s.unique()))
        inter = None
        for _, s in common_dates.items():
            inter = s if inter is None else inter.intersection(s)
        if inter:
            out_df = out_df[out_df["date"].isin(sorted(inter))].copy()
    return out_df


