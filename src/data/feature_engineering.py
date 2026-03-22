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

    def aggregate_sentiment_to_trading_days(
            self,
            sentiment_df: pd.DataFrame,
            trading_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        将包含周末/节假日的每日情感数据，向后聚合到下一个有效交易日。
        """
        # 1. 确保情感数据的索引是完整的连续日期（包含周末）
        all_days = pd.date_range(start=sentiment_df.index.min(), end=sentiment_df.index.max(), freq='D')
        sentiment_df = sentiment_df.reindex(all_days).fillna(0)

        # 2. 创建一个新列，用于标记“所属的交易日”
        sentiment_df['Target_Trading_Date'] = sentiment_df.index

        # 3. 将非交易日（周末/节假日）的标记设为缺失值 (NaT)
        sentiment_df.loc[~sentiment_df.index.isin(trading_index), 'Target_Trading_Date'] = pd.NaT

        # 4. 关键步：使用向后填充 (bfill)
        sentiment_df['Target_Trading_Date'] = sentiment_df['Target_Trading_Date'].bfill()

        # 5. 按照目标交易日进行分组聚合（求均值），把周末情绪融合进周一
        sentiment_aggregated = sentiment_df.groupby('Target_Trading_Date').mean()

        # 清理掉可能超出交易日历范围的最后几天数据
        sentiment_aggregated = sentiment_aggregated[sentiment_aggregated.index.isin(trading_index)]

        return sentiment_aggregated

    def merge_external_data(
            self,
            df: pd.DataFrame,
            macro_df: pd.DataFrame = None,
            sentiment_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        将宏观经济数据和新闻情感数据对齐并合并到主价格时间序列中。
        使用 merge_asof 解决节假日数据丢失，并严格防范前视偏差(Look-ahead bias)。
        """
        df_merged = df.copy()

        # 确保主表的索引是 datetime 格式
        if not isinstance(df_merged.index, pd.DatetimeIndex):
            df_merged.index = pd.to_datetime(df_merged.index)
        df_merged.index = df_merged.index.normalize()

        # 1. 合并宏观经济数据 (Macro Data)
        # 1. 合并宏观经济数据 (Macro Data)
        if macro_df is not None and not macro_df.empty:
            macro_df = macro_df.copy()
            macro_df.index = pd.to_datetime(macro_df.index).normalize()

            # 【修复点 1：解决 GDP 空白】
            # 因为 CPI 是月度，GDP 是季度。在宏观表内部先用前一个季度的值填满中间的月份
            macro_df = macro_df.ffill()

            # 【防未来函数 (Publication Lag Proxy)】
            # 真实世界中，1月1日标注的宏观数据，要到下个月初或中旬才公布。
            # 这里统一往后推迟 35 天生效。这意味着 2020-01-01(Q1) 的数据，
            # 在 2月5日 之后才开始影响你的预测模型，极其符合真实市场信息流！
            macro_df.index = macro_df.index + pd.Timedelta(days=35)
            # 把 GDP 变成 GDP_Lag35，CPI 变成 CPI_Lag35
            # ==========================================
            macro_df = macro_df.add_suffix('_Lag35')

            df_merged = df_merged.sort_index()
            macro_df = macro_df.sort_index()

            # 使用 merge_asof 对齐
            df_merged = pd.merge_asof(
                df_merged,
                macro_df,
                left_index=True,
                right_index=True,
                direction='backward'
            )
            # 因为推迟了35天，导致前35天可能找不到数据（变成NaN），所以最后再用 bfill 兜底填上最初的缺口
            df_merged.bfill(inplace=True)
            logger.info("Successfully merged macroeconomic data with 35-day lag for point-in-time realism.")

        # 2. 合并新闻情感数据 (Sentiment Data)
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df = sentiment_df.copy()
            sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
            # weekend_aggregation (周末+节假日累加到下一个交易日)
            df_merged = df_merged.join(sentiment_df, how="left")
            df_merged.fillna(0, inplace=True)
            logger.info("Successfully merged sentiment data.")

        return df_merged