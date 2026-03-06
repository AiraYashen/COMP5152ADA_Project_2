"""
Ensemble model that combines predictions from base models using
a linear regression meta-learner trained on the validation set.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # type: ignore

from src.config import ENSEMBLE_BASE_MODELS, MODELS_SAVED_DIR

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Stacking ensemble using a Ridge regression meta-learner.

    Base model predictions are used as features for the meta-learner.
    """

    def __init__(
        self,
        base_model_names: Optional[List[str]] = None,
        alpha: float = 1.0,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.base_model_names = base_model_names or list(ENSEMBLE_BASE_MODELS)
        self.alpha = alpha
        self.models_dir = Path(models_dir or MODELS_SAVED_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._meta_learner = Ridge(alpha=self.alpha)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta_features(predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack base model predictions into a 2-D feature matrix."""
        arrays = [np.asarray(v).ravel() for v in predictions.values()]
        min_len = min(len(a) for a in arrays)
        return np.column_stack([a[:min_len] for a in arrays])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        val_predictions: Dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "EnsembleModel":
        """Train the meta-learner on validation-set predictions.

        Parameters
        ----------
        val_predictions : dict
            Keys are model names, values are 1-D arrays of predictions on
            the validation set.
        y_val : np.ndarray
            Ground-truth validation target values.

        Returns
        -------
        EnsembleModel
            Self.
        """
        X_meta = self._build_meta_features(val_predictions)
        y_meta = np.asarray(y_val).ravel()[: len(X_meta)]
        logger.info(
            "Fitting ensemble meta-learner on %d samples, %d base models.",
            *X_meta.shape,
        )
        self._meta_learner.fit(X_meta, y_meta)
        self._is_fitted = True
        logger.info(
            "Ensemble coefficients: %s",
            dict(zip(val_predictions.keys(), self._meta_learner.coef_)),
        )
        return self

    def predict(self, test_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine test-set predictions using the trained meta-learner.

        Parameters
        ----------
        test_predictions : dict
            Keys are model names, values are 1-D arrays of predictions on
            the test set.

        Returns
        -------
        np.ndarray
            1-D ensemble predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble has not been fitted. Call fit() first.")
        X_meta = self._build_meta_features(test_predictions)
        return self._meta_learner.predict(X_meta)

    def get_weights(self) -> pd.Series:
        """Return the meta-learner coefficients as a named Series."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble has not been fitted.")
        return pd.Series(
            self._meta_learner.coef_,
            index=list(self.base_model_names)[: len(self._meta_learner.coef_)],
        )

    def save(self, filename: str = "ensemble_model.pkl") -> Path:
        """Pickle the ensemble meta-learner."""
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(self._meta_learner, f)
        logger.info("Saved ensemble model to %s", path)
        return path

    def load(self, filename: str = "ensemble_model.pkl") -> "EnsembleModel":
        """Load a pickled ensemble meta-learner."""
        path = self.models_dir / filename
        with open(path, "rb") as f:
            self._meta_learner = pickle.load(f)
        self._is_fitted = True
        logger.info("Loaded ensemble model from %s", path)
        return self
