"""
Tests for model implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    calculate_metrics,
    directional_accuracy,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def _arrays(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(10_000, 20_000, 100)
        y_pred = y_true + rng.normal(0, 100, 100)
        return y_true, y_pred

    def test_mse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_rmse_positive(self):
        y_true, y_pred = self._arrays()
        assert root_mean_squared_error(y_true, y_pred) >= 0

    def test_mae_positive(self):
        y_true, y_pred = self._arrays()
        assert mean_absolute_error(y_true, y_pred) >= 0

    def test_mape_zero_values(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.5, 2.5])
        mape = mean_absolute_percentage_error(y_true, y_pred)
        # Only non-zero values are used; should not raise
        assert not np.isnan(mape) or mape >= 0

    def test_directional_accuracy_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1, 2.1, 3.2, 4.3, 5.4], dtype=float)
        assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_directional_accuracy_inverse(self):
        y_true = np.array([5, 4, 3, 2, 1], dtype=float)
        y_pred = np.array([1, 2, 3, 4, 5], dtype=float)
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_directional_accuracy_single_obs(self):
        da = directional_accuracy(np.array([1.0]), np.array([1.0]))
        assert np.isnan(da)

    def test_calculate_metrics_keys(self):
        y_true, y_pred = self._arrays()
        metrics = calculate_metrics(y_true, y_pred)
        for key in ("mse", "rmse", "mae", "mape", "directional_accuracy"):
            assert key in metrics

    def test_calculate_metrics_pandas_series(self):
        y_true = pd.Series([100, 200, 300, 400, 500], dtype=float)
        y_pred = pd.Series([110, 190, 310, 390, 510], dtype=float)
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["mse"] > 0


# ---------------------------------------------------------------------------
# ARIMA model (statsmodels required)
# ---------------------------------------------------------------------------

class TestARIMAModel:
    @pytest.fixture
    def series(self):
        rng = np.random.default_rng(0)
        n = 200
        dates = pd.bdate_range("2020-01-02", periods=n)
        values = 10_000 + np.cumsum(rng.normal(0, 10, n))
        return pd.Series(values, index=dates)

    def test_fit_predict(self, series):
        try:
            from src.models.arima_model import ARIMAModel
        except ImportError:
            pytest.skip("statsmodels not available")

        model = ARIMAModel(order=(2, 1, 0))
        model.fit(series)
        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_in_sample_prediction_length(self, series):
        try:
            from src.models.arima_model import ARIMAModel
        except ImportError:
            pytest.skip("statsmodels not available")

        model = ARIMAModel(order=(1, 1, 0))
        model.fit(series)
        fitted = model.predict_in_sample()
        assert len(fitted) == len(series)


# ---------------------------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------------------------

class TestXGBoostModel:
    @pytest.fixture
    def dataset(self):
        rng = np.random.default_rng(1)
        n = 300
        dates = pd.bdate_range("2020-01-02", periods=n)
        X = pd.DataFrame(
            rng.standard_normal((n, 5)),
            columns=["f1", "f2", "f3", "f4", "f5"],
            index=dates,
        )
        y = pd.Series(rng.standard_normal(n), index=dates)
        return X, y

    def test_fit_predict(self, dataset):
        try:
            from src.models.xgboost_model import XGBoostModel
        except ImportError:
            pytest.skip("xgboost not available")

        X, y = dataset
        model = XGBoostModel(params={"n_estimators": 10, "random_state": 0, "objective": "reg:squarederror"})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_feature_importance_shape(self, dataset):
        try:
            from src.models.xgboost_model import XGBoostModel
        except ImportError:
            pytest.skip("xgboost not available")

        X, y = dataset
        model = XGBoostModel(params={"n_estimators": 10, "random_state": 0, "objective": "reg:squarederror"})
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert (importance >= 0).all()


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    def test_fit_predict(self):
        from src.models.ensemble import EnsembleModel

        rng = np.random.default_rng(2)
        n = 50
        val_preds = {
            "model_a": rng.standard_normal(n),
            "model_b": rng.standard_normal(n),
        }
        y_val = rng.standard_normal(n)
        ensemble = EnsembleModel()
        ensemble.fit(val_preds, y_val)

        test_preds = {
            "model_a": rng.standard_normal(n),
            "model_b": rng.standard_normal(n),
        }
        out = ensemble.predict(test_preds)
        assert len(out) == n

    def test_get_weights_shape(self):
        from src.models.ensemble import EnsembleModel

        rng = np.random.default_rng(3)
        n = 40
        preds = {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
        y = rng.standard_normal(n)
        ensemble = EnsembleModel(base_model_names=["a", "b"])
        ensemble.fit(preds, y)
        weights = ensemble.get_weights()
        assert len(weights) == 2
