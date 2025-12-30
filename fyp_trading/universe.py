from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class UniverseConfig:
    """
    Multi-instrument universe configuration.

    Note:
    - Keep the universe small at first (5-20 tickers) to iterate quickly.
    - Later expand to a broader HK universe for better fine-tuning signal.
    """

    tickers: List[str]
    period: str = "10y"
    interval: str = "1d"


def default_hk_universe_small() -> List[str]:
    """
    A small HK universe for quick iteration.
    (You can replace this with a Hang Seng constituents list later.)
    """

    return [
        "2800.HK",  # Tracker Fund (HSI proxy)
        "0700.HK",  # Tencent
        "9988.HK",  # Alibaba-SW
        "3690.HK",  # Meituan
        "0005.HK",  # HSBC
    ]


