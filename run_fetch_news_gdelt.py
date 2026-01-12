from __future__ import annotations

"""
Fetch news for a HK ticker universe using GDELT 2.1 DOC API and write to:
  - data/news/news_raw_gdelt_<ts>.csv
  - data/news/news_raw_gdelt_all.csv  (deduped URL-level cache; grows over time)
  - data/news/news.csv                (date,ticker,text)  <-- used by FinBERT pipeline (regenerated from raw_all)
  - outputs/reports/news_gdelt_stats_<ts>.json

This script requires network access and `requests`:
  pip install requests
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

import argparse

from fyp_trading.news_fetch import GdeltDocConfig, fetch_news_for_universe_gdelt, fetch_news_for_universe_gdelt_sliced
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20180101000000", help="YYYYMMDDHHMMSS (UTC)")
    parser.add_argument("--end", default="20251231235959", help="YYYYMMDDHHMMSS (UTC)")
    parser.add_argument("--slice-days", type=int, default=30, help="Time slicing window in days (0 disables slicing)")
    parser.add_argument("--pages-per-slice", type=int, default=6, help="Max pages per slice per ticker (each page<=250)")
    parser.add_argument("--language", default="English", help="GDELT language filter (or empty to disable)")
    parser.add_argument("--sort", default="datedesc", help="GDELT sort for non-sliced mode")
    parser.add_argument("--maxrecords", type=int, default=250)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = project_root()

    tickers = default_hk_universe_small()
    aliases = default_aliases()

    cfg = GdeltDocConfig(
        startdatetime=str(args.start),
        enddatetime=str(args.end),
        maxrecords=int(args.maxrecords),
        max_pages_per_query=int(args.pages_per_slice),
        language=(str(args.language) if str(args.language).strip() else None),
        sleep_seconds=float(args.sleep),
        sort=str(args.sort),
    )

    if int(args.slice_days) > 0:
        raw, agg = fetch_news_for_universe_gdelt_sliced(
            tickers,
            aliases,
            cfg,
            slice_days=int(args.slice_days),
            max_pages_per_slice=int(args.pages_per_slice),
        )
    else:
        raw, agg = fetch_news_for_universe_gdelt(tickers, aliases, cfg)

    data_dir = root / "data" / "news"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / f"news_raw_gdelt_{ts}.csv"
    raw_all_path = data_dir / "news_raw_gdelt_all.csv"
    agg_path = data_dir / "news.csv"

    raw.to_csv(raw_path, index=False)

    # Maintain a growing, URL-level cache (dedup by url), then regenerate daily agg used by FinBERT.
    if raw_all_path.exists():
        try:
            raw_old = pd.read_csv(raw_all_path)
        except Exception:
            raw_old = pd.DataFrame()
    else:
        raw_old = pd.DataFrame()

    raw_all = pd.concat([raw_old, raw], ignore_index=True) if not raw_old.empty else raw.copy()
    if not raw_all.empty and "url" in raw_all.columns:
        raw_all = raw_all.drop_duplicates(subset=["url"]).copy()
    raw_all.to_csv(raw_all_path, index=False)

    # Regenerate daily aggregation from raw_all (ensures dedup + stable growth)
    if raw_all.empty:
        agg_all = pd.DataFrame(columns=["date", "ticker", "text"])
    else:
        raw_all = raw_all.copy()
        raw_all["date"] = pd.to_datetime(raw_all.get("seendate"), errors="coerce").dt.tz_localize(None)
        raw_all = raw_all.dropna(subset=["date"]).copy()
        raw_all["date"] = pd.to_datetime(raw_all["date"]).dt.normalize()
        if "title" not in raw_all.columns:
            raw_all["title"] = ""
        if "snippet" not in raw_all.columns:
            raw_all["snippet"] = ""
        if "ticker" not in raw_all.columns:
            raw_all["ticker"] = ""
        raw_all["title"] = raw_all["title"].fillna("").astype(str)
        raw_all["snippet"] = raw_all["snippet"].fillna("").astype(str)
        raw_all["ticker"] = raw_all["ticker"].fillna("").astype(str)
        raw_all["text"] = (raw_all["title"].str.strip() + "\n" + raw_all["snippet"].str.strip()).str.strip()
        raw_all = raw_all[raw_all["text"].astype(str).str.len() > 0].copy()
        agg_all = (
            raw_all.groupby(["date", "ticker"], as_index=False)["text"]
            .apply(lambda s: "\n\n".join(s.tolist()))
            .reset_index(drop=True)
        )

    agg_all.to_csv(agg_path, index=False)

    out = resolve_outputs_dir()
    stats = {
        "ts": ts,
        "tickers": tickers,
        "cfg": cfg,
        "raw_rows_new": int(len(raw)),
        "agg_rows_new": int(len(agg)),
        "raw_rows_all": int(len(raw_all)),
        "agg_rows_all": int(len(agg_all)),
        "date_min": str(pd.to_datetime(agg_all["date"]).min().date()) if not agg_all.empty else None,
        "date_max": str(pd.to_datetime(agg_all["date"]).max().date()) if not agg_all.empty else None,
        "raw_path": str(raw_path),
        "raw_all_path": str(raw_all_path),
        "agg_path": str(agg_path),
        "args": vars(args),
    }
    save_json(out["reports"] / f"news_gdelt_stats_{ts}.json", stats)

    print("[OK] Saved:")
    print(" -", raw_path)
    print(" -", raw_all_path)
    print(" -", agg_path)
    print(" -", out["reports"] / f"news_gdelt_stats_{ts}.json")
    print("[INFO] To inspect: open data/news/news.csv (date,ticker,text)")


if __name__ == "__main__":
    main()


