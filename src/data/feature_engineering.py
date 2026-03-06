"""
Feature engineering module.

Computes technical indicators:
  - Simple Moving Averages (MA5, MA10, MA20)
  - RSI (Relative Strength Index)
  - MACD and Signal line
  - Bollinger Bands (upper, lower, %B)

All calculations are performed on the 'Close' column by default.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    MA_WINDOWS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PROCESSED_DATA_DIR,
    RSI_PERIOD,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute and attach technical indicators to a price DataFrame."""

    def __init__(self, price_col: str = "Close") -> None:
        self.price_col = price_col

    # ------------------------------------------------------------------
    # Individual indicator methods
    # ------------------------------------------------------------------

    def add_moving_averages(
        self, df: pd.DataFrame, windows: Optional[list] = None
    ) -> pd.DataFrame:
        """Add simple moving average columns (MA{w}) for each window in *windows*."""
        windows = windows or MA_WINDOWS
        for w in windows:
            df[f"MA{w}"] = df[self.price_col].rolling(window=w).mean()
        logger.debug("Added MA indicators: %s", [f"MA{w}" for w in windows])
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
        """Add RSI (Relative Strength Index) column."""
        delta = df[self.price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        logger.debug("Added RSI(%d).", period)
        return df

    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL,
    ) -> pd.DataFrame:
        """Add MACD, MACD Signal, and MACD Histogram columns."""
        ema_fast = df[self.price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[self.price_col].ewm(span=slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        logger.debug("Added MACD(%d,%d,%d).", fast, slow, signal)
        return df

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = BOLLINGER_WINDOW,
        num_std: float = BOLLINGER_STD,
    ) -> pd.DataFrame:
        """Add Bollinger Band upper/lower and %B columns."""
        rolling = df[self.price_col].rolling(window=window)
        mid = rolling.mean()
        std = rolling.std()
        df["BB_upper"] = mid + num_std * std
        df["BB_lower"] = mid - num_std * std
        df["BB_mid"] = mid
        band_width = df["BB_upper"] - df["BB_lower"]
        df["BB_pct"] = (df[self.price_col] - df["BB_lower"]) / band_width.replace(
            0, np.nan
        )
        logger.debug("Added Bollinger Bands(%d, %.1f).", window, num_std)
        return df

    def add_lag_features(self, df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
        """Add lagged close-price return columns (lag_1 … lag_*lags*)."""
        returns = df[self.price_col].pct_change()
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = returns.shift(lag)
        logger.debug("Added %d lag features.", lags)
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume moving average and on-balance volume (OBV)."""
        if "Volume" not in df.columns:
            logger.warning("'Volume' column not found; skipping volume features.")
            return df
        df["Volume_MA5"] = df["Volume"].rolling(5).mean()
        df["Volume_MA20"] = df["Volume"].rolling(20).mean()
        # OBV
        direction = np.sign(df[self.price_col].diff().fillna(0))
        df["OBV"] = (direction * df["Volume"]).cumsum()
        logger.debug("Added volume features.")
        return df

    # ------------------------------------------------------------------
    # Master method
    # ------------------------------------------------------------------

    def engineer_all(
        self,
        df: pd.DataFrame,
        save: bool = False,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply all feature engineering steps to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least a ``Close`` column.
        save : bool
            Persist the result to disk if ``True``.
        save_path : str, optional
            Override the default output path.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame.
        """
        df = df.copy()
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_lag_features(df)
        df = self.add_volume_features(df)
        logger.info(
            "Feature engineering complete. Shape: %s. Columns: %s",
            df.shape,
            list(df.columns),
        )
        if save:
            from src.config import PROCESSED_DATA_DIR  # local import to avoid circular

            out = save_path or str(PROCESSED_DATA_DIR / "features.csv")
            df.to_csv(out)
            logger.info("Saved engineered features to %s", out)
        return df
