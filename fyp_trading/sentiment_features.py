"""
FinBERT Sentiment Feature Extractor.

Uses pre-trained FinBERT as a fixed feature extractor to produce daily sentiment features
from news headlines. These 3-dimensional features (P_neg, P_neu, P_pos) can be concatenated
with technical indicators for downstream models like Informer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def extract_finbert_sentiment(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    device: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract sentiment probabilities from texts using pre-trained FinBERT.

    Returns:
        np.ndarray of shape (N, 3): [P_negative, P_neutral, P_positive] for each text.
        Note: ProsusAI/finbert outputs labels in order: negative(0), neutral(1), positive(2).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"Loading FinBERT model: {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    all_probs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate very long texts
            batch = [t[:512] if len(t) > 512 else t for t in batch]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

            if verbose and (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(texts)} texts...")

    if verbose:
        print(f"  Done. Processed {len(texts)} texts.")

    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))


def load_news_and_extract_sentiment(
    news_csv_path: str | Path,
    ticker: str = "2800.HK",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    device: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
    verbose: bool = True,
    use_all_news: bool = False,
) -> pd.DataFrame:
    """
    Load news from CSV, filter by ticker (optional), and extract FinBERT sentiment.

    If cache_path is provided and exists, loads from cache instead of re-extracting.

    Args:
        use_all_news: If True, use ALL news regardless of ticker (useful for ETFs like 2800.HK
                      which track broad market indices).

    Returns:
        DataFrame with columns: date, ticker, text, sent_neg, sent_neu, sent_pos
    """
    news_csv_path = Path(news_csv_path)
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"Loading cached sentiment from {cache_path}")
            return pd.read_csv(cache_path, parse_dates=["date"])

    if not news_csv_path.exists():
        raise FileNotFoundError(f"News file not found: {news_csv_path}")

    news_df = pd.read_csv(news_csv_path)
    if verbose:
        print(f"Loaded {len(news_df)} news entries from {news_csv_path}")

    # Normalize column names
    news_df.columns = news_df.columns.str.lower().str.strip()

    # Filter by ticker if specified and ticker column exists
    if not use_all_news and ticker and "ticker" in news_df.columns:
        # Also include general HK market news (no specific ticker or empty ticker)
        news_df["ticker"] = news_df["ticker"].fillna("").str.strip()
        mask = (news_df["ticker"] == ticker) | (news_df["ticker"] == "")
        news_df = news_df[mask].copy()
        if verbose:
            print(f"After filtering for {ticker} + general news: {len(news_df)} entries")
    elif use_all_news:
        if verbose:
            print(f"Using ALL news (use_all_news=True): {len(news_df)} entries")

    if len(news_df) == 0:
        if verbose:
            print("No news data after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "ticker", "text", "sent_neg", "sent_neu", "sent_pos"])

    # Parse date
    if "date" in news_df.columns:
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    else:
        raise ValueError("News CSV must have a 'date' column")

    # Get texts
    if "text" not in news_df.columns:
        raise ValueError("News CSV must have a 'text' column")

    texts = news_df["text"].fillna("").astype(str).tolist()

    # Extract sentiment
    probs = extract_finbert_sentiment(
        texts, model_name=model_name, batch_size=batch_size, device=device, verbose=verbose
    )

    news_df["sent_neg"] = probs[:, 0]
    news_df["sent_neu"] = probs[:, 1]
    news_df["sent_pos"] = probs[:, 2]

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        news_df.to_csv(cache_path, index=False)
        if verbose:
            print(f"Cached sentiment to {cache_path}")

    return news_df


def aggregate_daily_sentiment(
    news_with_sentiment: pd.DataFrame,
    agg_method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate news sentiment by date.

    Args:
        news_with_sentiment: DataFrame with date, sent_neg, sent_neu, sent_pos columns.
        agg_method: 'mean' (simple average) or 'weighted' (weight by text length).

    Returns:
        DataFrame indexed by date with columns: sent_neg, sent_neu, sent_pos, news_count
    """
    df = news_with_sentiment.copy()

    if len(df) == 0:
        return pd.DataFrame(columns=["sent_neg", "sent_neu", "sent_pos", "news_count"])

    df["date"] = pd.to_datetime(df["date"])

    if agg_method == "weighted" and "text" in df.columns:
        # Weight by text length (longer = more informative)
        df["weight"] = df["text"].fillna("").str.len().clip(lower=1)
        df["w_neg"] = df["sent_neg"] * df["weight"]
        df["w_neu"] = df["sent_neu"] * df["weight"]
        df["w_pos"] = df["sent_pos"] * df["weight"]

        agg = df.groupby("date").agg(
            w_neg=("w_neg", "sum"),
            w_neu=("w_neu", "sum"),
            w_pos=("w_pos", "sum"),
            total_weight=("weight", "sum"),
            news_count=("text", "count"),
        )
        agg["sent_neg"] = agg["w_neg"] / agg["total_weight"]
        agg["sent_neu"] = agg["w_neu"] / agg["total_weight"]
        agg["sent_pos"] = agg["w_pos"] / agg["total_weight"]
        agg = agg[["sent_neg", "sent_neu", "sent_pos", "news_count"]]
    else:
        # Simple mean aggregation
        agg = df.groupby("date").agg(
            sent_neg=("sent_neg", "mean"),
            sent_neu=("sent_neu", "mean"),
            sent_pos=("sent_pos", "mean"),
            news_count=("sent_neg", "count"),
        )

    return agg


def merge_sentiment_with_features(
    feat_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    fill_method: str = "zero",
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Merge daily sentiment features into the technical feature DataFrame.

    Args:
        feat_df: DataFrame with DatetimeIndex containing technical features.
        daily_sentiment: DataFrame with date index and sent_neg/neu/pos columns.
        fill_method: How to handle days without news:
            - 'zero': fill with zeros (neutral assumption)
            - 'ffill': forward-fill from last known sentiment
            - 'neutral': fill with [0.0, 1.0, 0.0] (explicit neutral)

    Returns:
        (merged_df, sentiment_feature_cols)
    """
    result = feat_df.copy()

    # Ensure both have compatible date index
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("feat_df must have a DatetimeIndex")

    # Convert daily_sentiment index to datetime if needed
    if len(daily_sentiment) > 0:
        daily_sentiment = daily_sentiment.copy()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

    sentiment_cols = ["sent_neg", "sent_neu", "sent_pos"]

    # Join on date
    result = result.join(daily_sentiment[sentiment_cols], how="left")

    # Handle missing values
    if fill_method == "zero":
        for col in sentiment_cols:
            result[col] = result[col].fillna(0.0)
    elif fill_method == "ffill":
        for col in sentiment_cols:
            result[col] = result[col].fillna(method="ffill").fillna(0.0)
    elif fill_method == "neutral":
        result["sent_neg"] = result["sent_neg"].fillna(0.0)
        result["sent_neu"] = result["sent_neu"].fillna(1.0)
        result["sent_pos"] = result["sent_pos"].fillna(0.0)
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    return result, sentiment_cols


def build_sentiment_features_for_ticker(
    ticker: str = "2800.HK",
    news_csv_path: str | Path = "data/news/news.csv",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    device: Optional[str] = None,
    agg_method: str = "mean",
    verbose: bool = True,
    use_all_news: bool = False,
) -> pd.DataFrame:
    """
    End-to-end: load news, extract sentiment, aggregate by date.

    Args:
        use_all_news: If True, use ALL news regardless of ticker. Recommended for
                      index ETFs like 2800.HK.

    Returns:
        DataFrame indexed by date with columns: sent_neg, sent_neu, sent_pos, news_count
    """
    # Check for cached sentiment
    cache_suffix = "all" if use_all_news else ticker.replace(".", "_")
    cache_path = Path(news_csv_path).parent / f"sentiment_cache_{cache_suffix}.csv"

    news_with_sent = load_news_and_extract_sentiment(
        news_csv_path=news_csv_path,
        ticker=ticker,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        cache_path=cache_path,
        verbose=verbose,
        use_all_news=use_all_news,
    )

    if len(news_with_sent) == 0:
        if verbose:
            print("No news data available. Returning empty sentiment features.")
        return pd.DataFrame(columns=["sent_neg", "sent_neu", "sent_pos", "news_count"])

    daily_sent = aggregate_daily_sentiment(news_with_sent, agg_method=agg_method)

    if verbose:
        print(f"Daily sentiment features: {len(daily_sent)} days with news")
        if len(daily_sent) > 0:
            print(f"  Date range: {daily_sent.index.min().date()} ~ {daily_sent.index.max().date()}")
            print(f"  Avg sentiment: neg={daily_sent['sent_neg'].mean():.3f}, "
                  f"neu={daily_sent['sent_neu'].mean():.3f}, pos={daily_sent['sent_pos'].mean():.3f}")

    return daily_sent

