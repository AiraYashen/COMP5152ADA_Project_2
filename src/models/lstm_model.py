"""
LSTM model for stock index forecasting using TensorFlow/Keras.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,
    LSTM_SEQUENCE_LENGTH,
    LSTM_UNITS,
    MODELS_SAVED_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class LSTMModel:
    """Multi-layer LSTM for multivariate time-series forecasting."""

    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        lstm_units: List[int] = LSTM_UNITS,
        dropout: float = LSTM_DROPOUT,
        learning_rate: float = LSTM_LEARNING_RATE,
        batch_size: int = LSTM_BATCH_SIZE,
        epochs: int = LSTM_EPOCHS,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.models_dir = Path(models_dir or MODELS_SAVED_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self.history = None

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

    def create_sequences(
        self, data: np.ndarray, target_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a 2-D array into overlapping (X, y) sequences.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features).
        target_idx : int
            Column index of the target variable.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X of shape (n, seq_len, n_features), y of shape (n,).
        """
        xs, ys = [], []
        for i in range(self.sequence_length, len(data)):
            xs.append(data[i - self.sequence_length: i])
            ys.append(data[i, target_idx])
        return np.array(xs), np.array(ys)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, n_features: int):
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(RANDOM_SEED)
        model = tf.keras.Sequential()
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            if i == 0:
                model.add(
                    tf.keras.layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        input_shape=(self.sequence_length, n_features),
                    )
                )
            else:
                model.add(
                    tf.keras.layers.LSTM(units, return_sequences=return_sequences)
                )
            model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        target_idx: int = 0,
    ) -> "LSTMModel":
        """Train the LSTM model.

        Parameters
        ----------
        train_data : np.ndarray
            Scaled training array of shape (n_train, n_features).
        val_data : np.ndarray, optional
            Scaled validation array for early stopping.
        target_idx : int
            Column index of the target variable in *train_data*.

        Returns
        -------
        LSTMModel
            Self.
        """
        import tensorflow as tf  # type: ignore

        X_train, y_train = self.create_sequences(train_data, target_idx)
        n_features = X_train.shape[2]
        self._model = self._build_model(n_features)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_data is not None else "loss",
                patience=10,
                restore_best_weights=True,
            )
        ]
        validation_data = None
        if val_data is not None:
            X_val, y_val = self.create_sequences(val_data, target_idx)
            validation_data = (X_val, y_val)

        logger.info(
            "Training LSTM: X_train=%s, epochs=%d, batch=%d",
            X_train.shape,
            self.epochs,
            self.batch_size,
        )
        self.history = self._model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        logger.info("LSTM training complete.")
        return self

    def predict(self, data: np.ndarray, target_idx: int = 0) -> np.ndarray:
        """Generate predictions for *data*.

        Parameters
        ----------
        data : np.ndarray
            Scaled input array of shape (n, n_features).
        target_idx : int
            Column index for y (used only to build sequences).

        Returns
        -------
        np.ndarray
            1-D predicted values array.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X, _ = self.create_sequences(data, target_idx)
        return self._model.predict(X, verbose=0).ravel()

    def save(self, filename: str = "lstm_model") -> Path:
        """Save the Keras model to disk."""
        if self._model is None:
            raise RuntimeError("No fitted model to save.")
        path = self.models_dir / filename
        self._model.save(str(path))
        logger.info("Saved LSTM model to %s", path)
        return path

    def load(self, filename: str = "lstm_model") -> "LSTMModel":
        """Load a saved Keras model from disk."""
        import tensorflow as tf  # type: ignore

        path = self.models_dir / filename
        self._model = tf.keras.models.load_model(str(path))
        logger.info("Loaded LSTM model from %s", path)
        return self
