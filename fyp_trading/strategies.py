from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import bollinger_bands, macd, rsi


def _signal_to_position(signal: pd.Series, allow_short: bool) -> pd.Series:
    if allow_short:
        return signal.clip(-1, 1)
    return signal.clip(lower=0, upper=1)


@dataclass(frozen=True)
class MacdConfig:
    fast: int = 12
    slow: int = 26
    signal: int = 9
    allow_short: bool = False


def macd_crossover_signal(close: pd.Series, cfg: MacdConfig = MacdConfig()) -> pd.Series:
    macd_line, signal_line, _ = macd(close, fast=cfg.fast, slow=cfg.slow, signal=cfg.signal)
    # +1 when MACD above signal, -1 otherwise
    sig = (macd_line > signal_line).astype(int) * 2 - 1
    return _signal_to_position(sig, allow_short=cfg.allow_short)


@dataclass(frozen=True)
class RsiConfig:
    period: int = 14
    low: float = 30.0
    high: float = 70.0
    allow_short: bool = False


def rsi_mean_reversion_signal(close: pd.Series, cfg: RsiConfig = RsiConfig()) -> pd.Series:
    r = rsi(close, period=cfg.period)
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[r < cfg.low] = 1
    if cfg.allow_short:
        sig[r > cfg.high] = -1
    return _signal_to_position(sig, allow_short=cfg.allow_short)


@dataclass(frozen=True)
class BollingerConfig:
    period: int = 20
    num_std: float = 2.0
    allow_short: bool = False


def bollinger_mean_reversion_signal(close: pd.Series, cfg: BollingerConfig = BollingerConfig()) -> pd.Series:
    mid, upper, lower, _, _ = bollinger_bands(close, period=cfg.period, num_std=cfg.num_std)
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[close < lower] = 1
    if cfg.allow_short:
        sig[close > upper] = -1
    return _signal_to_position(sig, allow_short=cfg.allow_short)


@dataclass(frozen=True)
class DualMaConfig:
    fast: int = 10
    slow: int = 50
    allow_short: bool = False


def dual_ma_trend_signal(close: pd.Series, cfg: DualMaConfig = DualMaConfig()) -> pd.Series:
    ma_fast = close.rolling(cfg.fast).mean()
    ma_slow = close.rolling(cfg.slow).mean()
    sig = (ma_fast > ma_slow).astype(int) * 2 - 1
    return _signal_to_position(sig, allow_short=cfg.allow_short)


