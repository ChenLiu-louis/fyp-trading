from __future__ import annotations

import pandas as pd

import yfinance as yf


def fetch_prices(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV (auto-adjusted) from yfinance.

    Index is timezone-naive DateTimeIndex.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=5 * 370)
        df = tk.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"Cannot fetch data for {ticker}.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


