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


@dataclass(frozen=True)
class DonchianConfig:
    """Turtle / Donchian channel breakout (trend-following)."""

    entry: int = 20
    exit: int = 10
    allow_short: bool = False


def donchian_breakout_signal(high: pd.Series, low: pd.Series, cfg: DonchianConfig = DonchianConfig()) -> pd.Series:
    """Classic Donchian breakout:
    - Long when price breaks above N-day high
    - Exit when price breaks below M-day low
    (Optional symmetric short side)
    """
    hh = high.rolling(cfg.entry).max()
    ll = low.rolling(cfg.entry).min()
    exit_ll = low.rolling(cfg.exit).min()
    exit_hh = high.rolling(cfg.exit).max()

    pos = pd.Series(0, index=high.index, dtype=int)
    current = 0
    for i in range(len(pos)):
        if i == 0:
            pos.iat[i] = 0
            continue
        if current == 0:
            # Entry
            if high.iat[i] >= hh.iat[i]:
                current = 1
            elif cfg.allow_short and low.iat[i] <= ll.iat[i]:
                current = -1
        elif current == 1:
            # Exit long
            if low.iat[i] <= exit_ll.iat[i]:
                current = 0
        elif current == -1:
            # Exit short
            if high.iat[i] >= exit_hh.iat[i]:
                current = 0
        pos.iat[i] = current
    return _signal_to_position(pos, allow_short=cfg.allow_short)


@dataclass(frozen=True)
class VolTargetMomentumConfig:
    """Time-series momentum with volatility targeting (single instrument)."""

    mom_lookback: int = 60
    vol_lookback: int = 20
    target_vol_annual: float = 0.10
    max_leverage: float = 1.0
    allow_short: bool = True


def vol_target_momentum_position(close: pd.Series, cfg: VolTargetMomentumConfig = VolTargetMomentumConfig()) -> pd.Series:
    """Position = sign(momentum) * (target_vol / realized_vol), clipped by max_leverage.

    - Momentum: sign of lookback return
    - Realized vol: rolling std of daily returns
    """
    ret = close.pct_change()
    mom = close / close.shift(cfg.mom_lookback) - 1.0
    mom_sign = np.sign(mom).replace(0, 0)
    if not cfg.allow_short:
        mom_sign = (mom > 0).astype(int)

    vol_daily = ret.rolling(cfg.vol_lookback).std()
    vol_annual = vol_daily * np.sqrt(252.0)
    leverage = (cfg.target_vol_annual / vol_annual).clip(upper=cfg.max_leverage)
    pos = mom_sign * leverage
    pos = pos.fillna(0.0)
    return pos.astype(float)


@dataclass(frozen=True)
class BollingerSqueezeConfig:
    """Bollinger Band squeeze breakout (trend-following after low volatility)."""

    period: int = 20
    num_std: float = 2.0
    squeeze_lookback: int = 120
    squeeze_quantile: float = 0.2
    allow_short: bool = False


def bollinger_squeeze_breakout_signal(close: pd.Series, cfg: BollingerSqueezeConfig = BollingerSqueezeConfig()) -> pd.Series:
    mid, upper, lower, bandwidth, _ = bollinger_bands(close, period=cfg.period, num_std=cfg.num_std)
    # Define squeeze as bandwidth being in the lowest q quantile over a rolling window
    bw_q = bandwidth.rolling(cfg.squeeze_lookback).quantile(cfg.squeeze_quantile)
    in_squeeze = bandwidth <= bw_q

    pos = pd.Series(0, index=close.index, dtype=int)
    current = 0
    for i in range(len(pos)):
        if i == 0:
            continue
        if current == 0:
            # Only enter after a squeeze day
            if in_squeeze.iat[i - 1]:
                if close.iat[i] > upper.iat[i]:
                    current = 1
                elif cfg.allow_short and close.iat[i] < lower.iat[i]:
                    current = -1
        else:
            # Exit when price crosses mid (simple)
            if current == 1 and close.iat[i] < mid.iat[i]:
                current = 0
            elif current == -1 and close.iat[i] > mid.iat[i]:
                current = 0
        pos.iat[i] = current
    return _signal_to_position(pos, allow_short=cfg.allow_short)


