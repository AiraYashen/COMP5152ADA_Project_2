"""
Data collection module.

Downloads:
- NASDAQ-100 (^NDX) OHLCV data via yfinance
- Macroeconomic indicators from FRED via pandas_datareader
- Basic news sentiment using yfinance news items + VADER
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
    # News sentiment
    # ------------------------------------------------------------------

    def fetch_news_sentiment(self, save: bool = True) -> pd.DataFrame:
        """Retrieve recent news headlines and compute VADER sentiment scores.

        Uses *yfinance* to obtain news items and NLTK's VADER lexicon for
        sentiment scoring.  Returns a daily aggregate sentiment score.

        Returns
        -------
        pd.DataFrame
            Columns: ``sentiment_pos``, ``sentiment_neg``, ``sentiment_compound``.
        """
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
                "nltk is required for sentiment analysis. "
                "Install it with: pip install nltk"
            ) from exc

        ticker_obj = yf.Ticker(self.ticker)
        news = ticker_obj.news
        if not news:
            logger.warning("No news returned by yfinance for %s.", self.ticker)
            return pd.DataFrame(
                columns=["sentiment_pos", "sentiment_neg", "sentiment_compound"]
            )

        records = []
        for item in news:
            title = item.get("title", "")
            ts = item.get("providerPublishTime")
            if not ts:
                continue
            scores = sia.polarity_scores(title)
            records.append(
                {
                    "Date": pd.Timestamp(ts, unit="s").normalize(),
                    "sentiment_pos": scores["pos"],
                    "sentiment_neg": scores["neg"],
                    "sentiment_compound": scores["compound"],
                }
            )

        if not records:
            return pd.DataFrame(
                columns=["sentiment_pos", "sentiment_neg", "sentiment_compound"]
            )

        df = pd.DataFrame(records)
        df = df.groupby("Date").mean()
        df.index = pd.to_datetime(df.index)

        if save:
            out = self.external_dir / "news_sentiment.csv"
            df.to_csv(out)
            logger.info("Saved news sentiment to %s", out)
        return df

    # ------------------------------------------------------------------
    # Convenience: run all collection steps
    # ------------------------------------------------------------------

    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """Download stock data, macro data, and news sentiment in one call.

        Returns
        -------
        dict
            Keys: ``stock``, ``macro``, ``sentiment``.
        """
        result: Dict[str, pd.DataFrame] = {}
        result["stock"] = self.download_stock_data()
        time.sleep(0.5)
        try:
            result["macro"] = self.download_macro_data()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Macro data collection failed: %s", exc)
            result["macro"] = pd.DataFrame()
        try:
            result["sentiment"] = self.fetch_news_sentiment()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment collection failed: %s", exc)
            result["sentiment"] = pd.DataFrame()
        return result
