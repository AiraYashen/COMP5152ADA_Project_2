"""
Evaluation metrics for time-series forecasting.

Provides MSE, MAE, MAPE, RMSE, and directional accuracy helpers.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.Series]


def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert pandas Series or list-like to a 1-D numpy array."""
    return np.asarray(x, dtype=float).ravel()


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return Mean Squared Error."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return Mean Absolute Error."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return Mean Absolute Percentage Error (in %).

    Zero-valued actuals are excluded to avoid division by zero.
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        logger.warning("All true values are zero; MAPE is undefined.")
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of correctly predicted price movement directions.

    Both *y_true* and *y_pred* should be price-level series (not returns).
    The direction is computed as sign of first-difference.

    Returns
    -------
    float
        Value in [0, 1].
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    if len(y_true) < 2:
        logger.warning("Need at least two observations to compute directional accuracy.")
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


def calculate_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Compute a full suite of forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth price values.
    y_pred : array-like
        Predicted price values.

    Returns
    -------
    dict
        Keys: ``mse``, ``rmse``, ``mae``, ``mape``, ``directional_accuracy``.
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    logger.info("Metrics: %s", metrics)
    return metrics
