"""
XGBoost model for stock index forecasting.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import MODELS_SAVED_DIR, XGBOOST_PARAMS

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost regressor with feature importance reporting."""

    def __init__(
        self,
        params: Optional[Dict] = None,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.params = params or XGBOOST_PARAMS
        self.models_dir = Path(models_dir or MODELS_SAVED_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostModel":
        """Fit the XGBoost model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix for training.
        y_train : pd.Series
            Target series for training.
        X_val : pd.DataFrame, optional
            Validation features for early stopping.
        y_val : pd.Series, optional
            Validation target for early stopping.

        Returns
        -------
        XGBoostModel
            Self.
        """
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "xgboost is required. Install it with: pip install xgboost"
            ) from exc

        self.feature_names = list(X_train.columns)
        params = dict(self.params)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        logger.info("Training XGBoost on %d rows, %d features.", *X_train.shape)
        self._model = xgb.XGBRegressor(**params)
        fit_kwargs: Dict = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        self._model.fit(X_train, y_train, **fit_kwargs)
        logger.info("XGBoost training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for feature matrix *X*."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending.

        Returns
        -------
        pd.Series
            Importance scores indexed by feature name.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        scores = self._model.feature_importances_
        series = pd.Series(scores, index=self.feature_names)
        return series.sort_values(ascending=False)

    def save(self, filename: str = "xgboost_model.pkl") -> Path:
        """Pickle the fitted model."""
        if self._model is None:
            raise RuntimeError("No fitted model to save.")
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("Saved XGBoost model to %s", path)
        return path

    def load(self, filename: str = "xgboost_model.pkl") -> "XGBoostModel":
        """Load a pickled XGBoost model."""
        path = self.models_dir / filename
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("Loaded XGBoost model from %s", path)
        return self
