"""
Ensemble model that combines predictions from base models using
a linear regression meta-learner trained on the validation set.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    @staticmethod
    def _to_meta_frame(predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert prediction dict to aligned feature frame."""
        frame = pd.DataFrame(predictions).copy()
        for col in frame.columns:
            frame[col] = np.asarray(frame[col]).ravel()
        if frame.empty:
            raise ValueError("No base-model predictions provided.")
        return frame

    def train_and_refit_walk_forward(
        self,
        val_predictions: Dict[str, np.ndarray],
        y_val: np.ndarray,
        test_predictions: Dict[str, np.ndarray],
        y_test: Optional[np.ndarray] = None,
        window: int = 126,
        min_train_size: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Walk-forward stacking with rolling-window meta-learner updates.

        Phase 1 (validation): one-step-ahead predictions are generated for each
        validation sample using only past validation meta-history. The true
        validation target is then appended to the history.

        Phase 2 (test): one-step-ahead test predictions are generated using the
        rolling meta-history from Phase 1; if *y_test* is provided, true test
        values are appended after each step for strict backtesting updates.
        """
        if window < 20:
            raise ValueError("window must be >= 20 for stable meta-learner fitting.")

        X_val = self._to_meta_frame(val_predictions)
        y_val_arr = np.asarray(y_val).ravel()[: len(X_val)]
        if len(y_val_arr) != len(X_val):
            raise ValueError("y_val length does not match validation predictions.")

        X_test = self._to_meta_frame(test_predictions)
        n_features = X_val.shape[1]
        min_train_size = min_train_size or max(20, n_features * 3)

        history_X = X_val.iloc[:0].copy()
        history_y = pd.Series(dtype=float)
        val_out = []

        # Phase 1: sequentially learn on validation period.
        for i in range(len(X_val)):
            x_t = X_val.iloc[[i]]
            y_t = float(y_val_arr[i])

            if len(history_y) >= min_train_size:
                X_train = history_X.iloc[-window:]
                y_train = history_y.iloc[-window:]
                self._meta_learner = Ridge(alpha=self.alpha)
                self._meta_learner.fit(X_train, y_train)
                yhat_t = float(self._meta_learner.predict(x_t)[0])
                self._is_fitted = True
            else:
                # Warm-up before enough samples are available.
                yhat_t = float(x_t.mean(axis=1).iloc[0])

            val_out.append(yhat_t)
            history_X = pd.concat([history_X, x_t], axis=0)
            history_y.loc[len(history_y)] = y_t

        # Phase 2: sequentially predict test period with rolling updates.
        y_test_arr = None
        if y_test is not None:
            y_test_arr = np.asarray(y_test).ravel()[: len(X_test)]
            if len(y_test_arr) != len(X_test):
                raise ValueError("y_test length does not match test predictions.")

        test_out = []
        for i in range(len(X_test)):
            x_t = X_test.iloc[[i]]

            if len(history_y) >= min_train_size:
                X_train = history_X.iloc[-window:]
                y_train = history_y.iloc[-window:]
                self._meta_learner = Ridge(alpha=self.alpha)
                self._meta_learner.fit(X_train, y_train)
                yhat_t = float(self._meta_learner.predict(x_t)[0])
                self._is_fitted = True
            else:
                yhat_t = float(x_t.mean(axis=1).iloc[0])

            test_out.append(yhat_t)

            y_update = float(y_test_arr[i]) if y_test_arr is not None else yhat_t
            history_X = pd.concat([history_X, x_t], axis=0)
            history_y.loc[len(history_y)] = y_update

        val_series = pd.Series(val_out, index=X_val.index, name="Ensemble")
        test_series = pd.Series(test_out, index=X_test.index, name="Ensemble")
        logger.info(
            "Walk-forward ensemble complete. val=%d, test=%d, window=%d",
            len(val_series),
            len(test_series),
            window,
        )
        return val_series, test_series
