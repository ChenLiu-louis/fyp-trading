from __future__ import annotations

"""
Fetch news for a HK ticker universe using GDELT 2.1 DOC API and write to:
  - data/news/news_raw_gdelt_<ts>.csv
  - data/news/news.csv                (date,ticker,text)  <-- used by FinBERT pipeline
  - outputs/reports/news_gdelt_stats_<ts>.json

This script requires network access and `requests`:
  pip install requests
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from fyp_trading.news_fetch import GdeltDocConfig, fetch_news_for_universe_gdelt
from fyp_trading.universe import default_hk_universe_small
from fyp_trading.utils import project_root, resolve_outputs_dir, save_json


def default_aliases() -> Dict[str, List[str]]:
    """
    Basic alias mapping. Expand this list as you expand universe.
    """
    return {
        "2800.HK": ["Tracker Fund of Hong Kong", "Hong Kong Tracker Fund", "Hang Seng Index ETF", "2800.HK"],
        "0700.HK": ["Tencent", "Tencent Holdings", "0700.HK"],
        "9988.HK": ["Alibaba", "Alibaba Group", "Alibaba-SW", "9988.HK"],
        "3690.HK": ["Meituan", "Meituan Dianping", "3690.HK"],
        # Avoid short alias "HSBC" (may trigger GDELT "phrase too short")
        "0005.HK": ["HSBC Holdings", "HSBC Holdings plc", "The Hongkong and Shanghai Banking Corporation"],
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = project_root()

    tickers = default_hk_universe_small()
    aliases = default_aliases()

    cfg = GdeltDocConfig(
        # Use a longer window for "lots of data" (adjust as needed)
        startdatetime="20180101000000",
        enddatetime="20251231235959",
        maxrecords=250,
        max_pages_per_query=10,  # increase if you want more per ticker
        language="English",
        sleep_seconds=0.25,
        sort="datedesc",
    )

    raw, agg = fetch_news_for_universe_gdelt(tickers, aliases, cfg)

    data_dir = root / "data" / "news"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / f"news_raw_gdelt_{ts}.csv"
    agg_path = data_dir / "news.csv"

    raw.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)

    out = resolve_outputs_dir()
    stats = {
        "ts": ts,
        "tickers": tickers,
        "cfg": cfg,
        "raw_rows": int(len(raw)),
        "agg_rows": int(len(agg)),
        "date_min": str(pd.to_datetime(agg["date"]).min().date()) if not agg.empty else None,
        "date_max": str(pd.to_datetime(agg["date"]).max().date()) if not agg.empty else None,
        "raw_path": str(raw_path),
        "agg_path": str(agg_path),
    }
    save_json(out["reports"] / f"news_gdelt_stats_{ts}.json", stats)

    print("[OK] Saved:")
    print(" -", raw_path)
    print(" -", agg_path)
    print(" -", out["reports"] / f"news_gdelt_stats_{ts}.json")
    print("[INFO] To inspect: open data/news/news.csv (date,ticker,text)")


if __name__ == "__main__":
    main()


