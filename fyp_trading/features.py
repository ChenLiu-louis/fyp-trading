from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .indicators import atr, bollinger_bands, macd, obv, roc, rsi, zscore


def build_simple_features(
    df: pd.DataFrame,
    horizon: int = 1,
    use_log_return: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Ported from `LSTM_2.ipynb` (Simple features: SMA, Var, RSI, MACD, Bollinger).

    Returns:
      - data: feature dataframe + `next_return` + `logret_1d`
      - feature_cols: explicit list of model feature columns (excludes `logret_1d`)
    """
    c = df["Close"].copy()

    feat = pd.DataFrame(index=df.index)
    feat["logret_1d"] = np.log(c / c.shift(1))

    feat["sma_5"] = c.rolling(5).mean()
    feat["sma_10"] = c.rolling(10).mean()
    feat["sma_20"] = c.rolling(20).mean()
    feat["sma_50"] = c.rolling(50).mean()

    feat["var_5d"] = feat["logret_1d"].rolling(5).var()
    feat["var_10d"] = feat["logret_1d"].rolling(10).var()
    feat["var_20d"] = feat["logret_1d"].rolling(20).var()

    feat["rsi_14"] = rsi(c, 14)

    macd_line, signal_line, macd_hist = macd(c)
    feat["macd"] = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"] = macd_hist

    bb_mid, bb_up, bb_lo, bb_bw, bb_pb = bollinger_bands(c, 20, 2.0)
    feat["bb_mid"] = bb_mid
    feat["bb_upper"] = bb_up
    feat["bb_lower"] = bb_lo
    feat["bb_bw"] = bb_bw
    feat["bb_percent_b"] = bb_pb

    if use_log_return:
        next_ret = np.log(c.shift(-horizon) / c)
    else:
        next_ret = c.pct_change(horizon).shift(-horizon)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    data = feat.copy()
    data["next_return"] = next_ret

    feature_cols = [
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "var_5d",
        "var_10d",
        "var_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_bw",
        "bb_percent_b",
    ]

    data = data.dropna().copy()
    return data, feature_cols


def build_extended_features(
    df: pd.DataFrame,
    horizon: int = 1,
    use_log_return: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extended feature set for stronger baselines (still purely technical / price-volume based).

    Adds:
    - multi-horizon returns and volatilities
    - ATR-based volatility proxy
    - volume features (log vol, vol z-score, OBV)
    - price z-scores relative to moving windows
    - price-to-SMA ratios
    """
    c = df["Close"].copy()
    v = df.get("Volume", pd.Series(index=df.index, dtype=float)).copy()

    feat = pd.DataFrame(index=df.index)
    feat["logret_1d"] = np.log(c / c.shift(1))

    # SMAs
    for w in [5, 10, 20, 50, 100]:
        feat[f"sma_{w}"] = c.rolling(w).mean()
        feat[f"px_sma_ratio_{w}"] = c / feat[f"sma_{w}"] - 1.0

    # Multi-horizon returns (simple + log)
    for w in [5, 10, 20, 60]:
        feat[f"roc_{w}"] = roc(c, w)
        feat[f"logret_{w}"] = np.log(c / c.shift(w))

    # Volatility proxies
    for w in [5, 10, 20, 60]:
        feat[f"std_{w}"] = feat["logret_1d"].rolling(w).std()
        feat[f"var_{w}"] = feat["logret_1d"].rolling(w).var()

    feat["atr_14"] = atr(df, 14) / c  # scale by price

    # RSI / MACD / Bollinger
    feat["rsi_14"] = rsi(c, 14)
    macd_line, signal_line, macd_hist = macd(c)
    feat["macd"] = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"] = macd_hist
    bb_mid, bb_up, bb_lo, bb_bw, bb_pb = bollinger_bands(c, 20, 2.0)
    feat["bb_mid"] = bb_mid
    feat["bb_upper"] = bb_up
    feat["bb_lower"] = bb_lo
    feat["bb_bw"] = bb_bw
    feat["bb_percent_b"] = bb_pb

    # Volume features
    feat["log_vol"] = np.log(v.replace(0, np.nan))
    feat["vol_z_20"] = zscore(v, 20)
    feat["obv"] = obv(df)
    feat["obv_z_20"] = zscore(feat["obv"], 20)

    # Price z-scores
    for w in [20, 60]:
        feat[f"px_z_{w}"] = zscore(c, w)

    # Next return
    if use_log_return:
        next_ret = np.log(c.shift(-horizon) / c)
    else:
        next_ret = c.pct_change(horizon).shift(-horizon)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    data = feat.copy()
    data["next_return"] = next_ret

    feature_cols = [col for col in feat.columns if col != "logret_1d"]
    data = data.dropna().copy()
    return data, feature_cols


