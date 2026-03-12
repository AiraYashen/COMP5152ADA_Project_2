"""
ARIMA model for stock index forecasting.

Uses statsmodels ARIMA to fit a univariate autoregressive model on the
closing price series.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ARIMA_ORDER, MODELS_SAVED_DIR

logger = logging.getLogger(__name__)


class ARIMAModel:
    """Wrapper around ``statsmodels.tsa.arima.model.ARIMA``."""

    def __init__(
        self,
        order: Tuple[int, int, int] = ARIMA_ORDER,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.order = order
        self.models_dir = Path(models_dir or MODELS_SAVED_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._fitted_model = None

    def fit(self, series: pd.Series) -> "ARIMAModel":
        """Fit the ARIMA model on a univariate price series.

        Parameters
        ----------
        series : pd.Series
            Training price series with DatetimeIndex.

        Returns
        -------
        ARIMAModel
            Fitted model (self).
        """
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore

        logger.info("Fitting ARIMA%s on %d observations.", self.order, len(series))
        model = ARIMA(series, order=self.order)
        self._fitted_model = model.fit()
        logger.info("ARIMA fitting complete. AIC=%.4f", self._fitted_model.aic)
        return self

    def predict(self, steps: int) -> pd.Series:
        """Forecast *steps* steps ahead from the end of the training series.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.

        Returns
        -------
        pd.Series
            Forecasted values.
        """
        if self._fitted_model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        forecast = self._fitted_model.forecast(steps=steps)
        return forecast

    def predict_in_sample(self) -> pd.Series:
        """Return in-sample fitted values."""
        if self._fitted_model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._fitted_model.fittedvalues

    def save(self, filename: str = "arima_model.pkl") -> Path:
        """Pickle the fitted model to disk."""
        if self._fitted_model is None:
            raise RuntimeError("No fitted model to save.")
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(self._fitted_model, f)
        logger.info("Saved ARIMA model to %s", path)
        return path

    def load(self, filename: str = "arima_model.pkl") -> "ARIMAModel":
        """Load a pickled model from disk."""
        path = self.models_dir / filename
        with open(path, "rb") as f:
            self._fitted_model = pickle.load(f)
        logger.info("Loaded ARIMA model from %s", path)
        return self

    def train_and_refit(self, train_df, val_df, test_df, target_col: str):
        """
        高度封装的两阶段训练与预测：
        Phase 1: 仅用 2023 年数据训练，预测 2024 给集成模型
        Phase 2: 仅用 2024 年数据重塑记忆，预测 2025 最终结果
        """
        import pandas as pd
        import numpy as np

        # === Phase 1: 为 Ensemble 提供 2024 模拟考成绩 ===
        train_p1 = train_df.loc['2023']
        self.fit(train_p1[target_col])
        val_preds_arr = self.predict(steps=len(val_df))

        # 【修复点】：使用 np.array() 剥离默认的数字索引，强行对齐日期
        val_preds = pd.Series(np.array(val_preds_arr), index=val_df.index, name='ARIMA')

        # === Phase 2: 为 2025 实战重铸记忆 ===
        self.fit(val_df[target_col])
        test_preds_arr = self.predict(steps=len(test_df))

        # 【修复点】：同理剥离索引
        test_preds = pd.Series(np.array(test_preds_arr), index=test_df.index, name='ARIMA')

        return val_preds, test_preds