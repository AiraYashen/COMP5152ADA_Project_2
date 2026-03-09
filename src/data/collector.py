"""
Data collection module.

Downloads:
- NASDAQ-100 (^NDX) OHLCV data via yfinance
- Macroeconomic indicators from FRED via pandas_datareader
- News sentiment using GDELT Doc API + VADER
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from src.config import (
    END_DATE,
    EXTERNAL_DATA_DIR,
    FRED_SERIES,
    RAW_DATA_DIR,
    START_DATE,
    TICKER,
)

logger = logging.getLogger(__name__)


class DataCollector:
    """Handles all data acquisition for the forecasting pipeline."""

    def __init__(
        self,
        ticker: str = TICKER,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        raw_dir: Optional[Path] = None,
        external_dir: Optional[Path] = None,
    ) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.raw_dir = Path(raw_dir or RAW_DATA_DIR)
        self.external_dir = Path(external_dir or EXTERNAL_DATA_DIR)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stock price data
    # ------------------------------------------------------------------

    def download_stock_data(self, save: bool = True) -> pd.DataFrame:
        """Download OHLCV data for *self.ticker* from yfinance.

        Parameters
        ----------
        save : bool
            If ``True``, persist the raw CSV to ``raw_dir``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns Open, High, Low, Close, Adj Close, Volume.
        """
        logger.info(
            "Downloading %s from %s to %s", self.ticker, self.start_date, self.end_date
        )
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise ValueError(
                f"No data returned for {self.ticker} between "
                f"{self.start_date} and {self.end_date}."
            )
        # Flatten MultiIndex columns that yfinance may return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        logger.info("Downloaded %d rows of stock data.", len(df))
        if save:
            out = self.raw_dir / f"{self.ticker.replace('^', '')}_ohlcv.csv"
            df.to_csv(out)
            logger.info("Saved raw stock data to %s", out)
        return df

    # ------------------------------------------------------------------
    # Macroeconomic data from FRED
    # ------------------------------------------------------------------

    def download_macro_data(self, save: bool = True) -> pd.DataFrame:
        """Fetch macroeconomic series from FRED via pandas_datareader.

        Falls back gracefully if a single series is unavailable.

        Returns
        -------
        pd.DataFrame
            Combined macro DataFrame at monthly (or original) frequency.
        """
        try:
            import pandas_datareader.data as web
        except ImportError as exc:
            raise ImportError(
                "pandas_datareader is required for macro data collection. "
                "Install it with: pip install pandas-datareader"
            ) from exc

        frames: Dict[str, pd.Series] = {}
        for name, series_id in FRED_SERIES.items():
            try:
                logger.info("Fetching FRED series %s (%s)", series_id, name)
                s = web.DataReader(series_id, "fred", self.start_date, self.end_date)
                frames[name] = s[series_id]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch %s (%s): %s", name, series_id, exc)

        if not frames:
            logger.warning("No FRED data could be fetched; returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        if save:
            out = self.external_dir / "macro_data.csv"
            df.to_csv(out)
            logger.info("Saved macro data to %s", out)
        return df

    # ------------------------------------------------------------------
    # News sentiment via GDELT
    # ------------------------------------------------------------------

    def fetch_news_sentiment(
            self,
            save: bool = True,
            force_refresh: bool = False,
            gdelt_json_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Build daily sentiment from locally saved GDELT JSON files (offline).

        Enhancements:
        - De-duplicate articles by URL across all JSON files (including patch files).
        - Bucket articles by US/Eastern calendar day (better aligned with US market trading days).
        """
        out = self.external_dir / "news_sentiment.csv"
        if save and (not force_refresh) and out.exists():
            logger.info("Loading cached news sentiment from %s", out)
            df_cached = pd.read_csv(out, parse_dates=["Date"], index_col="Date")
            for col in ["sentiment_pos", "sentiment_neg", "sentiment_compound"]:
                if col not in df_cached.columns:
                    df_cached[col] = pd.NA
            return df_cached[["sentiment_pos", "sentiment_neg", "sentiment_compound"]]

        # Sentiment analyzer
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
            import nltk  # type: ignore

            try:
                sia = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                sia = SentimentIntensityAnalyzer()
        except ImportError as exc:
            raise ImportError(
                "nltk is required for sentiment analysis. Install it with: pip install nltk"
            ) from exc

        import json

        # Note: start/end are normalized to midnight *local naive*. We'll compare using naive dates after bucketing.
        start = pd.to_datetime(self.start_date).normalize()
        end = pd.to_datetime(self.end_date).normalize()

        gdelt_json_dir = Path(gdelt_json_dir or (self.external_dir / "gdelt_json"))
        if not gdelt_json_dir.exists():
            logger.warning("GDELT JSON dir not found: %s", gdelt_json_dir)
            df_empty = pd.DataFrame(
                columns=["sentiment_pos", "sentiment_neg", "sentiment_compound"]
            )
            if save:
                df_empty.to_csv(out, index=False)
            return df_empty

        json_files = sorted(gdelt_json_dir.glob("*.json"))
        if not json_files:
            logger.warning("No JSON files found in: %s", gdelt_json_dir)
            df_empty = pd.DataFrame(
                columns=["sentiment_pos", "sentiment_neg", "sentiment_compound"]
            )
            if save:
                df_empty.to_csv(out, index=False)
            return df_empty

        records: list[dict] = []
        seen_urls: set[str] = set()

        for fp in json_files:
            try:
                with fp.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read JSON %s: %s", fp, exc)
                continue

            articles = payload.get("articles") if isinstance(payload, dict) else None
            if not articles:
                continue

            for a in articles:
                title = (a.get("title") or "").strip()
                if not title:
                    continue

                url = (a.get("url") or "").strip()
                if url:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                dt_raw = (
                        a.get("seendate")
                        or a.get("seenDate")
                        or a.get("sourceCollectionDate")
                        or a.get("sourceCollectionDateTime")
                        or a.get("datetime")
                        or a.get("date")
                )

                if dt_raw:
                    dt = pd.to_datetime(dt_raw, errors="coerce", utc=True)
                else:
                    dt = pd.NaT

                if pd.isna(dt):
                    continue

                # Bucket by US/Eastern calendar day, then drop tz for joining with other daily series
                day = dt.tz_convert("US/Eastern").normalize().tz_localize(None)

                if day < start or day > end:
                    continue

                scores = sia.polarity_scores(title)
                records.append(
                    {
                        "Date": day,
                        "sentiment_pos": scores["pos"],
                        "sentiment_neg": scores["neg"],
                        "sentiment_compound": scores["compound"],
                    }
                )

        if not records:
            logger.warning("No sentiment records built from local JSON; returning empty sentiment DF.")
            df_empty = pd.DataFrame(
                columns=["sentiment_pos", "sentiment_neg", "sentiment_compound"]
            )
            if save:
                df_empty.to_csv(out, index=False)
            return df_empty

        df = pd.DataFrame(records).groupby("Date", as_index=True).mean()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        if save:
            df.to_csv(out)
            logger.info("Saved news sentiment to %s", out)

        return df[["sentiment_pos", "sentiment_neg", "sentiment_compound"]]

    # ------------------------------------------------------------------
    # Convenience: run all collection steps
    # ------------------------------------------------------------------

    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """Download stock data + macro data, and build news sentiment from local GDELT JSON."""
        result: Dict[str, pd.DataFrame] = {}

        result["stock"] = self.download_stock_data()
        time.sleep(0.5)

        try:
            result["macro"] = self.download_macro_data()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Macro data collection failed: %s", exc)
            result["macro"] = pd.DataFrame()

        try:
            # Offline: read local JSON files from external_dir/gdelt_json and build daily sentiment CSV
            result["sentiment"] = self.fetch_news_sentiment(
                gdelt_json_dir=self.external_dir / "gdelt_json"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment processing failed: %s", exc)
            result["sentiment"] = pd.DataFrame()

        return result

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    collector = DataCollector()
    data = collector.collect_all()
    for k, v in data.items():
        print(f"{k}: {v.shape}")