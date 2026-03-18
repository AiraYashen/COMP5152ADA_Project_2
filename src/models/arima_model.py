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

    def _walk_forward_rolling_predict(
        self,
        history: pd.Series,
        future: pd.Series,
        window: int,
    ) -> pd.Series:
        """One-step walk-forward forecasting with rolling training window.

        At each step, the model is re-fitted on the most recent *window* points,
        then forecasts one step ahead. The newly revealed true value is appended
        to history before the next step.
        """
        if history.empty:
            raise ValueError("History series is empty; cannot run walk-forward forecast.")
        if future.empty:
            return pd.Series(dtype=float, index=future.index, name="ARIMA")
        if window < 30:
            raise ValueError("window must be >= 30 for stable ARIMA fitting.")

        history_values = history.copy()
        preds = []

        for ts, y_true in future.items():
            train_slice = history_values.iloc[-window:]
            self.fit(train_slice)
            yhat = float(np.asarray(self.predict(steps=1)).ravel()[0])
            preds.append(yhat)
            history_values.loc[ts] = y_true

        return pd.Series(preds, index=future.index, name="ARIMA")

    def train_and_refit(
        self,
        train_df,
        val_df,
        test_df,
        target_col: str,
        window: int = 126,
    ):
        """
        高度封装的两阶段训练与预测：
        Phase 1: 仅用 2023 年数据训练，预测 2024 给集成模型
        Phase 2: 仅用 2024 年数据重塑记忆，预测 2025 最终结果
        """
        # === Phase 1: Walk-forward on validation with rolling window ===
        train_p1 = train_df.loc["2023"]
        history_p1 = train_p1[target_col]
        val_target = val_df[target_col]
        val_preds = self._walk_forward_rolling_predict(
            history=history_p1,
            future=val_target,
            window=window,
        )

        # === Phase 2: Walk-forward on test with rolling window ===
        history_p2 = val_target.copy()
        test_target = test_df[target_col]
        test_preds = self._walk_forward_rolling_predict(
            history=history_p2,
            future=test_target,
            window=window,
        )

        return val_preds, test_preds