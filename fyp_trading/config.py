from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LabelingConfig:
    """3-class labeling based on next_return vs (dynamic) threshold."""

    use_dynamic_threshold: bool = True
    vol_window: int = 20
    min_vol: float = 1e-4
    k_dynamic: float = 0.8  # threshold = k * rolling_vol


@dataclass(frozen=True)
class TrainConfig:
    """Training settings for LSTM multi-class classifier."""

    epochs: int = 100
    batch_size: int = 64
    patience: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 8
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    verbose: bool = False

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.4

    # 0=Down, 1=Neutral, 2=Up
    neutral_class_id: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline settings: data, windowing, signal threshold, backtest."""

    ticker: str = "2800.HK"
    period: str = "3y"
    interval: str = "1d"
    lookback: int = 30

    train_window: int = 250
    val_size: int = 21
    test_size: int = 21
    step_size: Optional[int] = 21

    horizon: int = 1
    proba_threshold: float = 0.57

    # Backtest
    min_holding_period: int = 2
    transaction_cost_bp: float = 2.0
    backtest_days: int = 252

    def resolved_step_size(self) -> int:
        return int(self.step_size if self.step_size is not None else self.test_size)


