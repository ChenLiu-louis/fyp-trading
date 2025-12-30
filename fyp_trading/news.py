from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import project_root


@dataclass(frozen=True)
class NewsConfig:
    """
    News ingestion config.

    This repo intentionally keeps the news source pluggable.
    In practice, you will provide a CSV exported from a provider (e.g., GDELT, RavenPack,
    or a manual export), then cache embeddings/sentiment under outputs/.
    """

    # Expected schema: date,ticker,text (date can be YYYY-MM-DD)
    csv_path: Optional[str] = "data/news/news.csv"


def load_news(cfg: NewsConfig) -> pd.DataFrame:
    """
    Load news CSV (local). Returns DataFrame with columns: date,ticker,text.
    """
    if cfg.csv_path is None:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    path = Path(cfg.csv_path)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        # Return empty but with correct schema
        return pd.DataFrame(columns=["date", "ticker", "text"])
    df = pd.read_csv(path)
    for col in ["date", "ticker", "text"]:
        if col not in df.columns:
            raise ValueError(f"news csv missing column: {col}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def aggregate_news_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple news items per (date,ticker) into a single text blob.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    g = df.groupby(["date", "ticker"], as_index=False)["text"].apply(lambda s: "\n".join(s.tolist()))
    return g


