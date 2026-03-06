"""
Data preprocessing module.

Handles:
- Merging stock, macro, and sentiment data
- Missing value imputation
- Train / validation / test split
- Feature scaling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from src.config import (
    PROCESSED_DATA_DIR,
    TRAIN_END,
    VAL_END,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Merge, clean, and split time-series features for model training."""

    def __init__(self, processed_dir: Optional[Path] = None) -> None:
        self.processed_dir = Path(processed_dir or PROCESSED_DATA_DIR)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.scaler: Optional[MinMaxScaler] = None

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge_datasets(
        self,
        stock_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge stock, macro, and sentiment data on a common daily index.

        Macro data (which may be monthly/quarterly) is forward-filled to
        daily frequency.  Sentiment data is outer-joined and filled with 0.

        Parameters
        ----------
        stock_df : pd.DataFrame
            Daily OHLCV data from ``DataCollector.download_stock_data``.
        macro_df : pd.DataFrame, optional
            Macro data from ``DataCollector.download_macro_data``.
        sentiment_df : pd.DataFrame, optional
            Daily sentiment scores from ``DataCollector.fetch_news_sentiment``.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame aligned to stock trading days.
        """
        df = stock_df.copy()
        df.index = pd.to_datetime(df.index)

        if macro_df is not None and not macro_df.empty:
            macro_df = macro_df.copy()
            macro_df.index = pd.to_datetime(macro_df.index)
            # Reindex to daily stock dates, then forward-fill monthly/quarterly series
            macro_daily = macro_df.reindex(df.index, method="ffill")
            df = df.join(macro_daily, how="left")

        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df = sentiment_df.copy()
            sentiment_df.index = pd.to_datetime(sentiment_df.index)
            sentiment_daily = sentiment_df.reindex(df.index)
            # Fill missing sentiment with 0 (neutral)
            sentiment_daily = sentiment_daily.fillna(0)
            df = df.join(sentiment_daily, how="left")

        logger.info("Merged dataset shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values.

        Strategy:
        - Forward-fill for price/macro columns.
        - Backward-fill for any remaining NaNs at the start of the series.
        - Fill any remaining NaNs with 0 (e.g. sentiment on the first day).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with no NaN values.
        """
        n_before = df.isna().sum().sum()
        df = df.ffill().bfill().fillna(0)
        n_after = df.isna().sum().sum()
        logger.info("Filled %d NaN values (%d remaining).", n_before - n_after, n_after)
        return df

    # ------------------------------------------------------------------
    # Train / validation / test split
    # ------------------------------------------------------------------

    def split_data(
        self,
        df: pd.DataFrame,
        train_end: str = TRAIN_END,
        val_end: str = VAL_END,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation, and test sets.

        - Train  : start of series through *train_end* (inclusive)
        - Val    : day after *train_end* through *val_end* (inclusive)
        - Test   : day after *val_end* through end of series

        Parameters
        ----------
        df : pd.DataFrame
            Full merged dataset with DatetimeIndex.
        train_end : str
            ISO-format date string for end of training set.
        val_end : str
            ISO-format date string for end of validation set.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train, val, test) DataFrames.
        """
        df.index = pd.to_datetime(df.index)
        train = df.loc[:train_end]
        val = df.loc[
            pd.Timestamp(train_end) + pd.Timedelta(days=1): pd.Timestamp(val_end)
        ]
        test = df.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1):]
        logger.info(
            "Split sizes – train: %d, val: %d, test: %d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        feature_range: Tuple[float, float] = (0, 1),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Min-max scale features, fitting only on the training set.

        The fitted scaler is stored in ``self.scaler`` for inverse transforms.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Scaled (train, val, test) DataFrames.
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        train_scaled = pd.DataFrame(
            self.scaler.fit_transform(train),
            index=train.index,
            columns=train.columns,
        )
        val_scaled = pd.DataFrame(
            self.scaler.transform(val),
            index=val.index,
            columns=val.columns,
        )
        test_scaled = pd.DataFrame(
            self.scaler.transform(test),
            index=test.index,
            columns=test.columns,
        )
        return train_scaled, val_scaled, test_scaled

    def inverse_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform a scaled DataFrame back to original space."""
        if self.scaler is None:
            raise RuntimeError("Call scale_features before inverse_scale.")
        return pd.DataFrame(
            self.scaler.inverse_transform(df),
            index=df.index,
            columns=df.columns,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run(
        self,
        stock_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Full preprocessing pipeline.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train, val, test) splits of the cleaned, merged dataset.
        """
        df = self.merge_datasets(stock_df, macro_df, sentiment_df)
        df = self.handle_missing_values(df)
        if save:
            path = self.processed_dir / "full_dataset.csv"
            df.to_csv(path)
            logger.info("Saved full dataset to %s", path)
        return self.split_data(df)
