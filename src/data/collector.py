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
            query: Optional[str] = None,
            max_records_per_day: int = 250,
            sleep_s: float = 0.2,
    ) -> pd.DataFrame:
        """Fetch historical news from GDELT and compute daily VADER sentiment.

        This replaces yfinance-based news collection, which usually only returns
        very recent headlines.

        Parameters
        ----------
        save : bool
            Save the aggregated daily sentiment to ``external_dir/news_sentiment.csv``.
        force_refresh : bool
            If False and CSV exists, load from disk (offline-friendly).
        query : Optional[str]
            Custom GDELT query. If None, uses a default query suitable for NDX/Nasdaq-100.
        max_records_per_day : int
            Upper bound of articles to pull per day (controls runtime and API volume).
        sleep_s : float
            Small sleep between API calls to be polite and reduce rate-limit issues.

        Returns
        -------
        pd.DataFrame
            Index: Date (daily)
            Columns: sentiment_pos, sentiment_neg, sentiment_compound
        """
        out = self.external_dir / "news_sentiment.csv"
        if save and (not force_refresh) and out.exists():
            logger.info("Loading cached news sentiment from %s", out)
            df_cached = pd.read_csv(out, parse_dates=["Date"], index_col="Date")
            # Ensure expected columns exist even if cached file differs
            for col in ["sentiment_pos", "sentiment_neg", "sentiment_compound"]:
                if col not in df_cached.columns:
                    df_cached[col] = pd.NA
            return df_cached[["sentiment_pos", "sentiment_neg", "sentiment_compound"]]

        # Sentiment analyzer (local, no external API for sentiment)
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
            import nltk  # type: ignore

            try:
                sia = SentimentIntensityAnalyzer()
            except LookupError:
                # NOTE: This downloads data if missing (needs internet).
                # For strict offline runs, pre-package the lexicon or skip sentiment.
                nltk.download("vader_lexicon", quiet=True)
                sia = SentimentIntensityAnalyzer()
        except ImportError as exc:
            raise ImportError(
                "nltk is required for sentiment analysis. Install it with: pip install nltk"
            ) from exc

        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests is required for GDELT collection. Install it with: pip install requests"
            ) from exc

        # Default query: broad enough to return results across years
        # (GDELT works best with keywords rather than tickers)
        q = query or '"Nasdaq 100" OR "NASDAQ-100" OR NDX OR "NASDAQ 100"'

        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)

        # GDELT date format: YYYYMMDDHHMMSS (we query per-day windows)
        records: list[dict] = []

        # Iterate day by day to control volume and make aggregation easy
        for day in pd.date_range(start=start, end=end, freq="D"):
            day_start = day.strftime("%Y%m%d000000")
            day_end = day.strftime("%Y%m%d235959")

            params = {
                "query": q,
                "mode": "ArtList",
                "format": "json",
                "startdatetime": day_start,
                "enddatetime": day_end,
                "maxrecords": int(max_records_per_day),
                "sort": "HybridRel",
            }

            try:
                r = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params=params,
                    timeout=30,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GDELT request failed for %s: %s", day.date(), exc)
                time.sleep(sleep_s)
                continue

            articles = payload.get("articles", []) or []
            if not articles:
                time.sleep(sleep_s)
                continue

            for a in articles:
                title = (a.get("title") or "").strip()
                # Sometimes GDELT titles can be missing; skip empty
                if not title:
                    continue

                scores = sia.polarity_scores(title)
                records.append(
                    {
                        "Date": day.normalize(),
                        "sentiment_pos": scores["pos"],
                        "sentiment_neg": scores["neg"],
                        "sentiment_compound": scores["compound"],
                    }
                )

            time.sleep(sleep_s)

        if not records:
            logger.warning("No GDELT news records collected; returning empty sentiment DF.")
            df_empty = pd.DataFrame(
                columns=["sentiment_pos", "sentiment_neg", "sentiment_compound"]
            )
            if save:
                # Save empty file so downstream code doesn't crash on missing file
                df_empty.to_csv(out, index=False)
            return df_empty

        df = pd.DataFrame(records)
        df = df.groupby("Date", as_index=True).mean()
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
        """Download stock data, macro data, and news sentiment in one call."""
        result: Dict[str, pd.DataFrame] = {}
        result["stock"] = self.download_stock_data()
        time.sleep(0.5)

        try:
            result["macro"] = self.download_macro_data()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Macro data collection failed: %s", exc)
            result["macro"] = pd.DataFrame()

        try:
            # uses GDELT now
            result["sentiment"] = self.fetch_news_sentiment()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment collection failed: %s", exc)
            result["sentiment"] = pd.DataFrame()

        return result

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    collector = DataCollector()
    data = collector.collect_all()
    for k, v in data.items():
        print(f"{k}: {v.shape}")