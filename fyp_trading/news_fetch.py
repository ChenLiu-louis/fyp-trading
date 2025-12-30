from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from json import JSONDecodeError


@dataclass(frozen=True)
class GdeltDocConfig:
    """
    Fetch news from GDELT 2.1 DOC API (public, no key required).

    Docs (reference):
      https://blog.gdeltproject.org/gdelt-2-1-api-debuts/
      https://api.gdeltproject.org/api/v2/doc/doc
    """

    # API endpoint
    endpoint: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Time range (UTC), format: YYYYMMDDHHMMSS
    startdatetime: str = "20180101000000"
    enddatetime: str = "20251231235959"

    # Pagination
    maxrecords: int = 250  # per page (GDELT supports up to 250)
    max_pages_per_query: int = 20  # safety cap

    # Query options
    mode: str = "artlist"  # returns articles list
    format: str = "json"
    sort: str = "datedesc"  # 'datedesc' or 'dateasc'

    # Language filter (optional). Common: "English"
    # See GDELT docs for exact language names.
    language: Optional[str] = "English"

    # Politeness / rate limiting
    sleep_seconds: float = 0.2

    # Retry behavior (GDELT may occasionally return HTML/503/empty responses)
    max_retries: int = 4
    backoff_base_seconds: float = 0.8


def _require_requests():
    try:
        import requests  # type: ignore

        return requests
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This feature requires `requests`. Please `pip install requests`.") from e


def _gdelt_query_for_ticker(ticker: str, aliases: List[str]) -> str:
    """
    Build a reasonably broad GDELT query for one ticker.
    We OR together aliases and add a Hong Kong hint to reduce noise.
    """
    # GDELT query syntax supports OR and quoted phrases.
    # IMPORTANT: GDELT rejects very short phrases (e.g. "HK").
    # Filter aliases that are too short to avoid HTML errors like:
    #   "The specified phrase is too short."
    cleaned = []
    for a in aliases:
        a = (a or "").strip()
        if not a:
            continue
        # keep tickers like "0005.HK", but drop very short tokens like "HK"
        if len(a) < 3 and "." not in a:
            continue
        # GDELT DOC API may reject short phrases; in practice, 4-letter tickers like "HSBC"
        # can still trigger: "The specified phrase is too short."
        # Keep longer aliases for such cases.
        if a.isalpha() and len(a) < 5:
            continue
        cleaned.append(a)

    terms = [f'"{a}"' for a in cleaned]
    if not terms:
        terms = [f'"{ticker}"']
    q = " OR ".join(terms)
    # Add region hint WITHOUT any 2-letter tokens.
    # NOTE: GDELT DOC API may reject 2-letter tokens even when used as operators (e.g. sourceCountry:HK),
    # returning an HTML error: "The specified phrase is too short."
    # Use longer textual constraints instead.
    return f"({q}) AND (\"Hong Kong\" OR Hongkong OR HKSAR)"


def fetch_gdelt_docs(
    cfg: GdeltDocConfig,
    query: str,
) -> pd.DataFrame:
    """
    Fetch articles for a single query. Returns DataFrame with best-effort columns:
    - date (datetime)
    - title, seendate, url, domain, sourceCountry, language, snippet
    """
    requests = _require_requests()

    rows: List[Dict[str, object]] = []
    startrecord = 1
    for _page in range(cfg.max_pages_per_query):
        params = {
            "query": query,
            "mode": cfg.mode,
            "format": cfg.format,
            "startdatetime": cfg.startdatetime,
            "enddatetime": cfg.enddatetime,
            "maxrecords": int(cfg.maxrecords),
            "startrecord": int(startrecord),
            "sort": cfg.sort,
        }
        if cfg.language:
            params["language"] = cfg.language

        headers = {
            "User-Agent": "fyp-trading-news-fetch/1.0 (+https://example.com)",
            "Accept": "application/json",
        }

        last_err: Optional[Exception] = None
        resp = None
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = requests.get(cfg.endpoint, params=params, headers=headers, timeout=30)
                # Some transient errors return 200 with HTML content; handle below
                if resp.status_code >= 400:
                    resp.raise_for_status()
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                sleep(cfg.backoff_base_seconds * (2**attempt))
        if resp is None:
            raise RuntimeError(f"GDELT request failed (no response). Last error: {last_err}")

        # GDELT may return HTML/text on overload or query validation errors; ensure JSON
        ctype = (resp.headers.get("Content-Type") or "").lower()
        try:
            data = resp.json()
        except (JSONDecodeError, ValueError) as e:
            text_preview = (resp.text or "")[:500].replace("\n", " ")
            # Auto-fallback: if query contains a 2-letter token constraint and GDELT complains,
            # retry once with a simplified query (remove region clause).
            if "phrase is too short" in text_preview.lower() and " and (" in query.lower():
                simple_q = query.split(") AND (", 1)[0] + ")"
                params["query"] = simple_q
                resp2 = requests.get(cfg.endpoint, params=params, headers=headers, timeout=30)
                ctype2 = (resp2.headers.get("Content-Type") or "").lower()
                try:
                    data = resp2.json()
                except Exception:
                    text_preview2 = (resp2.text or "")[:500].replace("\n", " ")
                    raise RuntimeError(
                        "GDELT response is not valid JSON (even after fallback query).\n"
                        f"- status_code: {resp2.status_code}\n"
                        f"- content_type: {ctype2}\n"
                        f"- url: {resp2.url}\n"
                        f"- body_preview: {text_preview2!r}\n"
                    ) from e
                else:
                    arts = data.get("articles", []) or []
                    if not arts:
                        break
                    # continue normal flow by setting data/ctype and falling through
                    ctype = ctype2
            raise RuntimeError(
                "GDELT response is not valid JSON.\n"
                f"- status_code: {resp.status_code}\n"
                f"- content_type: {ctype}\n"
                f"- url: {resp.url}\n"
                f"- body_preview: {text_preview!r}\n"
                "This is usually due to rate limiting or transient server errors.\n"
                "Try again later, reduce max_pages_per_query, or increase sleep_seconds/backoff."
            ) from e

        arts = data.get("articles", []) or []
        if not arts:
            break

        for a in arts:
            rows.append(
                {
                    "seendate": a.get("seendate"),
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "domain": a.get("domain"),
                    "sourceCountry": a.get("sourceCountry"),
                    "language": a.get("language"),
                    "snippet": a.get("snippet"),
                }
            )

        # GDELT pagination: advance by maxrecords
        startrecord += int(cfg.maxrecords)
        sleep(cfg.sleep_seconds)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Normalize date
    df["date"] = pd.to_datetime(df["seendate"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).copy()
    return df


def fetch_news_for_universe_gdelt(
    tickers: List[str],
    ticker_aliases: Dict[str, List[str]],
    cfg: GdeltDocConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch GDELT news for each ticker (by query aliases), returning:
    - raw articles table (date,ticker,title,snippet,url,domain,...)
    - aggregated per-day table (date,ticker,text)
    """
    all_rows = []
    for t in tickers:
        aliases = ticker_aliases.get(t, [t])
        q = _gdelt_query_for_ticker(t, aliases)
        df = fetch_gdelt_docs(cfg, q)
        if df.empty:
            continue
        df = df.copy()
        df["ticker"] = t
        all_rows.append(df)

    raw = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if raw.empty:
        agg = pd.DataFrame(columns=["date", "ticker", "text"])
        return raw, agg

    # Dedup by URL
    if "url" in raw.columns:
        raw = raw.drop_duplicates(subset=["url"]).copy()

    # Build text field: title + snippet
    raw["title"] = raw["title"].fillna("").astype(str)
    raw["snippet"] = raw["snippet"].fillna("").astype(str)
    raw["text"] = (raw["title"].str.strip() + "\n" + raw["snippet"].str.strip()).str.strip()
    raw = raw[raw["text"].astype(str).str.len() > 0].copy()

    # Aggregate per (date,ticker)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    agg = (
        raw.groupby(["date", "ticker"], as_index=False)["text"]
        .apply(lambda s: "\n\n".join(s.tolist()))
        .reset_index(drop=True)
    )
    return raw, agg


