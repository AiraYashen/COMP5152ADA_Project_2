"""
GDELT news downloader (raw JSON) with patch templates for failed windows.

Workflow:
1) Run this script to download weekly JSON files into data/external/gdelt_json/
2) If some windows fail due to rate limiting / non-JSON / network issues:
   - a CSV list is generated: gdelt_failed_windows.csv
   - a patch template JSON is generated for each failed window:
       patch_YYYYMMDD_YYYYMMDD.json
     You can manually copy JSON from the GDELT web/API into the "articles" list.
3) Later, collector.py will read ALL *.json files in that folder (including patch_*.json)
   to build daily sentiment CSV offline.

Notes:
- Respect GDELT policy: one request every 5 seconds.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd

from src.config import END_DATE, EXTERNAL_DATA_DIR, START_DATE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class WindowResult:
    start: pd.Timestamp
    end: pd.Timestamp
    ok: bool
    path: Optional[Path] = None
    reason: str = ""
    url: str = ""


def _window_json_name(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"gdelt_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"


def _patch_json_name(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"patch_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"


def _build_gdelt_url(
    *,
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    maxrecords: int,
    sort: str,
) -> str:
    # Build a browser-friendly URL for manual fetch
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    start_str = start.strftime("%Y%m%d000000")
    end_str = end.strftime("%Y%m%d235959")

    # URL encode query
    q = quote(query, safe="")
    return (
        f"{base}?query={q}&mode=ArtList&format=json"
        f"&startdatetime={start_str}&enddatetime={end_str}"
        f"&maxrecords={int(maxrecords)}&sort={sort}"
    )


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_patch_template(
    *,
    out_dir: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    query: str,
    sort: str,
    maxrecords: int,
    reason: str,
) -> Path:
    """Create patch template JSON for manual filling (if it doesn't already exist)."""
    patch_path = out_dir / _patch_json_name(window_start, window_end)
    if patch_path.exists():
        # Do not overwrite manual work
        return patch_path

    manual_url = _build_gdelt_url(
        query=query,
        start=window_start,
        end=window_end,
        maxrecords=maxrecords,
        sort=sort,
    )

    template: dict[str, Any] = {
        "meta": {
            "type": "gdelt_patch_template",
            "window_start": window_start.date().isoformat(),
            "window_end": window_end.date().isoformat(),
            "query": query,
            "sort": sort,
            "maxrecords": int(maxrecords),
            "reason": reason,
            "manual_fetch_url": manual_url,
            "how_to_fill": (
                "Open manual_fetch_url in browser, copy the JSON response's "
                '"articles" array items and paste them into this file\'s articles list. '
                "Order does not matter."
            ),
        },
        "articles": [],
    }
    _write_json(patch_path, template)
    return patch_path


def download_gdelt_news_json(
    *,
    query: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    out_dir: Optional[Path] = None,
    window_days: int = 7,
    maxrecords_per_window: int = 50,
    sort: str = "datedesc",
    sleep_s: float = 5.2,
    timeout_s: int = 30,
    max_retries: int = 2,
    max_backoff_s: float = 30.0,
    force_refresh: bool = False,
) -> list[WindowResult]:
    """Download GDELT docs API results as JSON files per weekly window.

    On failure, creates a patch template JSON for that window.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required. Install with: pip install requests") from exc

    out_dir = Path(out_dir or (Path(EXTERNAL_DATA_DIR) / "gdelt_json"))
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; COMP5152ADA/1.0; +https://github.com/)",
        "Accept": "application/json",
    }

    results: list[WindowResult] = []

    for window_start in pd.date_range(start=start, end=end, freq=f"{window_days}D"):
        window_end = min(window_start + pd.Timedelta(days=window_days - 1), end)

        file_name = _window_json_name(window_start, window_end)
        out_path = out_dir / file_name

        manual_url = _build_gdelt_url(
            query=query,
            start=window_start,
            end=window_end,
            maxrecords=maxrecords_per_window,
            sort=sort,
        )

        if out_path.exists() and not force_refresh:
            logger.info("Exists, skip: %s", out_path)
            results.append(WindowResult(window_start, window_end, True, out_path, "cached", manual_url))
            continue

        start_str = window_start.strftime("%Y%m%d000000")
        end_str = window_end.strftime("%Y%m%d235959")

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
            "maxrecords": int(maxrecords_per_window),
            "sort": sort,
        }

        payload: Optional[dict[str, Any]] = None
        ok = False
        reason = ""

        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout_s)

                # rate limit
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = float(retry_after)
                    else:
                        wait = min(max_backoff_s, (2 ** (attempt - 1)) * 5.0)
                    wait = max(5.0, wait)
                    reason = f"429 rate limited; wait {wait:.1f}s"
                    logger.warning(
                        "429 for %s..%s. Sleep %.1fs then retry (%d/%d).",
                        window_start.date(),
                        window_end.date(),
                        wait,
                        attempt,
                        max_retries,
                    )
                    time.sleep(wait)
                    continue

                r.raise_for_status()

                ctype = (r.headers.get("Content-Type") or "").lower()
                if "json" not in ctype:
                    snippet = (r.text or "")[:200].replace("\n", " ")
                    wait = min(max_backoff_s, (2 ** (attempt - 1)) * 5.0)
                    wait = max(5.0, wait)
                    reason = f"non-json content-type={ctype}; body~ {snippet}"
                    logger.warning(
                        "Non-JSON for %s..%s (ctype=%s). Sleep %.1fs then retry (%d/%d).",
                        window_start.date(),
                        window_end.date(),
                        ctype,
                        wait,
                        attempt,
                        max_retries,
                    )
                    time.sleep(wait)
                    continue

                payload_any = r.json()
                if isinstance(payload_any, dict):
                    payload = payload_any
                    ok = True
                    reason = "ok"
                    break

                reason = "unexpected json type"
                time.sleep(max(5.0, sleep_s))

            except Exception as exc:  # noqa: BLE001
                wait = min(max_backoff_s, (2 ** (attempt - 1)) * 5.0)
                wait = max(5.0, wait)
                reason = f"exception: {exc}"
                logger.warning(
                    "Request failed for %s..%s: %s. Sleep %.1fs then retry (%d/%d).",
                    window_start.date(),
                    window_end.date(),
                    exc,
                    wait,
                    attempt,
                    max_retries,
                )
                time.sleep(wait)

        if ok and payload is not None:
            _write_json(out_path, payload)
            logger.info("Saved %s..%s -> %s", window_start.date(), window_end.date(), out_path)
            results.append(WindowResult(window_start, window_end, True, out_path, "downloaded", manual_url))
        else:
            logger.error("FAILED %s..%s (%s)", window_start.date(), window_end.date(), reason)
            patch_path = _ensure_patch_template(
                out_dir=out_dir,
                window_start=window_start,
                window_end=window_end,
                query=query,
                sort=sort,
                maxrecords=maxrecords_per_window,
                reason=reason,
            )
            logger.error("Created patch template: %s", patch_path)
            results.append(WindowResult(window_start, window_end, False, None, reason, manual_url))

        # Respect GDELT limit: one request every 5 seconds
        time.sleep(max(5.0, sleep_s))

    failed = [r for r in results if not r.ok]
    if failed:
        fail_csv = out_dir / "gdelt_failed_windows.csv"
        pd.DataFrame(
            [
                {
                    "window_start": r.start.date().isoformat(),
                    "window_end": r.end.date().isoformat(),
                    "reason": r.reason,
                    "manual_fetch_url": r.url,
                    "patch_file": _patch_json_name(r.start, r.end),
                }
                for r in failed
            ]
        ).to_csv(fail_csv, index=False)
        logger.warning("Some windows failed. Saved list to: %s", fail_csv)
    else:
        logger.info("All windows downloaded successfully.")

    return results


if __name__ == "__main__":
    default_query = '("Nasdaq 100" OR "NASDAQ-100" OR NDX)'
    download_gdelt_news_json(
        query=default_query,
        window_days=7,
        maxrecords_per_window=50,
        sleep_s=10,  # >= 5 seconds to respect GDELT rule
        max_retries=3,  # keep small to avoid long waits; patch workflow handles failures
        max_backoff_s=60.0,
    )