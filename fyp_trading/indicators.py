from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    bandwidth = (upper - lower) / mid
    percent_b = (close - lower) / (upper - lower)
    return mid, upper, lower, bandwidth, percent_b


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of change (simple return over period)."""
    return series.pct_change(period)


def zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR). Requires High/Low/Close."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV). Requires Close and Volume."""
    close = df["Close"]
    vol = df["Volume"].fillna(0.0)
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * vol).cumsum()


