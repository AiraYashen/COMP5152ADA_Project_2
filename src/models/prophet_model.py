"""
Prophet model for stock index forecasting.

Wraps Facebook/Meta Prophet with support for additional regressors
(macro indicators, technical indicators).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import (
    MODELS_SAVED_DIR,
    PROPHET_CHANGEPOINT_PRIOR,
    PROPHET_DAILY_SEASONALITY,
    PROPHET_SEASONALITY_PRIOR,
    PROPHET_WEEKLY_SEASONALITY,
    PROPHET_YEARLY_SEASONALITY,
)

logger = logging.getLogger(__name__)


class ProphetModel:
    """Wrapper around ``prophet.Prophet`` with convenience helpers."""

    def __init__(
        self,
        changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR,
        seasonality_prior_scale: float = PROPHET_SEASONALITY_PRIOR,
        yearly_seasonality: bool = PROPHET_YEARLY_SEASONALITY,
        weekly_seasonality: bool = PROPHET_WEEKLY_SEASONALITY,
        daily_seasonality: bool = PROPHET_DAILY_SEASONALITY,
        extra_regressors: Optional[List[str]] = None,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.extra_regressors: List[str] = extra_regressors or []
        self.models_dir = Path(models_dir or MODELS_SAVED_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_prophet_df(df: pd.DataFrame, target_col: str = "Close") -> pd.DataFrame:
        """Reshape DataFrame to Prophet's ``ds``/``y`` format."""
        pdf = df[[target_col]].copy().reset_index()
        pdf.columns = ["ds", "y"]
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        return pdf

    def _build_model(self):
        try:
            from prophet import Prophet  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "prophet is required. Install it with: pip install prophet"
            ) from exc

        model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )
        for reg in self.extra_regressors:
            model.add_regressor(reg)
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "Close",
    ) -> "ProphetModel":
        """Fit Prophet on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with DatetimeIndex.
        target_col : str
            Column to treat as the target.

        Returns
        -------
        ProphetModel
            Self.
        """
        pdf = self._to_prophet_df(df, target_col)
        # Attach extra regressors
        for reg in self.extra_regressors:
            if reg in df.columns:
                pdf[reg] = df[reg].values
            else:
                logger.warning("Regressor '%s' not found in dataframe; skipping.", reg)
                self.extra_regressors.remove(reg)
        self._model = self._build_model()
        logger.info("Fitting Prophet model on %d rows.", len(pdf))
        self._model.fit(pdf)
        logger.info("Prophet fitting complete.")
        return self

    def predict(
        self, periods: int, freq: str = "B", future_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Forecast *periods* business days ahead.

        Parameters
        ----------
        periods : int
            Number of future periods to forecast.
        freq : str
            Pandas frequency string ('B' = business days).
        future_df : pd.DataFrame, optional
            If provided, use these rows for regressor values in the future.

        Returns
        -------
        pd.DataFrame
            Prophet forecast DataFrame.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        future = self._model.make_future_dataframe(periods=periods, freq=freq)
        if future_df is not None:
            for reg in self.extra_regressors:
                if reg in future_df.columns:
                    future = future.merge(
                        future_df[["Date", reg]].rename(columns={"Date": "ds"}),
                        on="ds",
                        how="left",
                    )
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def save(self, filename: str = "prophet_model.pkl") -> Path:
        """Save the model with pickle."""
        if self._model is None:
            raise RuntimeError("No fitted model to save.")
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("Saved Prophet model to %s", path)
        return path

    def load(self, filename: str = "prophet_model.pkl") -> "ProphetModel":
        """Load a pickled Prophet model."""
        path = self.models_dir / filename
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("Loaded Prophet model from %s", path)
        return self

    def train_and_refit(self, train_df, val_df, test_df):
        import pandas as pd
        import numpy as np

        # === Phase 1 (2022-2023 -> 2024) ===
        train_p1 = train_df.loc['2022':'2023']
        self.fit(train_p1)

        # 显式指定 periods 为天数(整数交易日)，并将数据传给 future_df 
        val_preds_raw = self.predict(periods=len(val_df), freq='B', future_df=val_df)

        # Prophet 默认返回一个包含多列的 DataFrame，提取 'yhat' (预测值)
        # 并取最后 len(val_df) 天的结果，使用 np.array 强制剥离索引防 NaN
        if isinstance(val_preds_raw, pd.DataFrame) and 'yhat' in val_preds_raw.columns:
            val_preds_arr = val_preds_raw['yhat'].values[-len(val_df):]
        else:
            val_preds_arr = np.array(val_preds_raw)[-len(val_df):]

        val_preds = pd.Series(val_preds_arr, index=val_df.index, name='Prophet')

        # === Phase 2 (2023-2024 -> 2025) ===
        train_full = pd.concat([train_df, val_df])
        refit_data = train_full.loc['2023':'2024']

        # 重新初始化底层 Prophet 对象，彻底清空旧的参数状态
        self.__init__()
        self.fit(refit_data)

        # 显式传参
        test_preds_raw = self.predict(periods=len(test_df), freq='B', future_df=test_df)

        if isinstance(test_preds_raw, pd.DataFrame) and 'yhat' in test_preds_raw.columns:
            test_preds_arr = test_preds_raw['yhat'].values[-len(test_df):]
        else:
            test_preds_arr = np.array(test_preds_raw)[-len(test_df):]

        test_preds = pd.Series(test_preds_arr, index=test_df.index, name='Prophet')

        return val_preds, test_preds