from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .data import fetch_prices


def fetch_prices_multi(
    tickers: List[str],
    period: str,
    interval: str,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple tickers (separate DataFrames).

    Notes:
    - Keeping separate DataFrames avoids hard-to-debug alignment issues.
    - Later steps can align by date intersection if needed.
    """
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        out[t] = fetch_prices(t, period, interval)
    return out


def align_on_common_dates(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align all ticker DataFrames on the intersection of dates.
    """
    if not dfs:
        return {}
    common = None
    for _, df in dfs.items():
        idx = pd.DatetimeIndex(df.index)
        common = idx if common is None else common.intersection(idx)
    assert common is not None
    return {k: v.loc[common].copy() for k, v in dfs.items()}


def summarize_date_range(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    if not dfs:
        raise ValueError("empty dfs")
    mins = [df.index.min() for df in dfs.values()]
    maxs = [df.index.max() for df in dfs.values()]
    lens = [len(df) for df in dfs.values()]
    return min(mins), max(maxs), int(min(lens))


