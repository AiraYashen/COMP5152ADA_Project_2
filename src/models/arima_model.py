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
