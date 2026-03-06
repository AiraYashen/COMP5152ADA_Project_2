"""
Tests for data-related modules: DataCollector, DataPreprocessor, FeatureEngineer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessor import DataPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 500, start: str = "2020-01-02") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start=start, periods=n)
    close = 10_000 + np.cumsum(rng.normal(0, 50, n))
    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.uniform(-0.005, 0.005, n)),
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Low": close * (1 - rng.uniform(0, 0.01, n)),
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# FeatureEngineer tests
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    def setup_method(self):
        self.df = _make_ohlcv()
        self.fe = FeatureEngineer(price_col="Close")

    def test_moving_averages_columns_created(self):
        df = self.fe.add_moving_averages(self.df.copy(), windows=[5, 10, 20])
        assert "MA5" in df.columns
        assert "MA10" in df.columns
        assert "MA20" in df.columns

    def test_moving_averages_values(self):
        df = self.fe.add_moving_averages(self.df.copy(), windows=[5])
        expected = self.df["Close"].rolling(5).mean()
        pd.testing.assert_series_equal(df["MA5"], expected, check_names=False)

    def test_rsi_range(self):
        df = self.fe.add_rsi(self.df.copy())
        valid = df["RSI"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_columns_created(self):
        df = self.fe.add_macd(self.df.copy())
        for col in ["MACD", "MACD_signal", "MACD_hist"]:
            assert col in df.columns

    def test_bollinger_bands_columns_created(self):
        df = self.fe.add_bollinger_bands(self.df.copy())
        for col in ["BB_upper", "BB_lower", "BB_mid", "BB_pct"]:
            assert col in df.columns

    def test_bollinger_bands_upper_above_lower(self):
        df = self.fe.add_bollinger_bands(self.df.copy())
        valid = df[["BB_upper", "BB_lower"]].dropna()
        assert (valid["BB_upper"] >= valid["BB_lower"]).all()

    def test_lag_features(self):
        df = self.fe.add_lag_features(self.df.copy(), lags=3)
        for lag in range(1, 4):
            assert f"lag_{lag}" in df.columns

    def test_volume_features(self):
        df = self.fe.add_volume_features(self.df.copy())
        assert "Volume_MA5" in df.columns
        assert "OBV" in df.columns

    def test_engineer_all_no_error(self):
        df = self.fe.engineer_all(self.df.copy())
        assert df.shape[0] == len(self.df)
        assert df.shape[1] > self.df.shape[1]


# ---------------------------------------------------------------------------
# DataPreprocessor tests
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    def setup_method(self):
        self.df = _make_ohlcv(n=1000)
        self.preprocessor = DataPreprocessor()

    def test_merge_without_macro_sentiment(self):
        merged = self.preprocessor.merge_datasets(self.df)
        assert set(merged.columns) >= set(self.df.columns)
        assert len(merged) == len(self.df)

    def test_merge_with_macro(self):
        # Monthly macro data reindexed to stock dates
        macro_dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        macro = pd.DataFrame(
            {"GDP": np.random.rand(60) * 100, "CPI": np.random.rand(60) * 300},
            index=macro_dates,
        )
        merged = self.preprocessor.merge_datasets(self.df, macro_df=macro)
        assert "GDP" in merged.columns

    def test_handle_missing_values(self):
        df_with_nan = self.df.copy()
        df_with_nan.iloc[5:10, 0] = np.nan
        df_clean = DataPreprocessor.handle_missing_values(df_with_nan)
        assert df_clean.isna().sum().sum() == 0

    def test_split_data_sizes(self):
        train, val, test = self.preprocessor.split_data(
            self.df, train_end="2021-12-31", val_end="2022-12-31"
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_split_data_no_overlap(self):
        train, val, test = self.preprocessor.split_data(
            self.df, train_end="2021-12-31", val_end="2022-12-31"
        )
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_scale_features_range(self):
        train, val, test = self.preprocessor.split_data(
            self.df, train_end="2021-12-31", val_end="2022-12-31"
        )
        t_sc, v_sc, te_sc = self.preprocessor.scale_features(train, val, test)
        # Training set should be in [0, 1]
        assert t_sc.min().min() >= -0.01
        assert t_sc.max().max() <= 1.01

    def test_inverse_scale_round_trip(self):
        train, val, test = self.preprocessor.split_data(
            self.df, train_end="2021-12-31", val_end="2022-12-31"
        )
        t_sc, _, _ = self.preprocessor.scale_features(train, val, test)
        t_inv = self.preprocessor.inverse_scale(t_sc)
        pd.testing.assert_frame_equal(t_inv.round(4), train.round(4))
