"""
Configuration file for the Stock Index Forecasting project.

All hyperparameters, file paths, and settings are centralised here
so they can be adjusted in one place.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolved relative to this file)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
MODELS_SAVED_DIR = ROOT_DIR / "models_saved"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
TICKER = "^NDX"          # NASDAQ-100 index
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
# Everything after VAL_END is test data (2025)

# FRED series identifiers
FRED_SERIES = {
    "GDP": "GDP",            # Real Gross Domestic Product (quarterly)
    "CPI": "CPIAUCSL",       # Consumer Price Index
    "FED_RATE": "FEDFUNDS",  # Federal Funds Rate
    "UNRATE": "UNRATE",      # Unemployment Rate
}

# ---------------------------------------------------------------------------
# Technical indicator parameters
# ---------------------------------------------------------------------------
MA_WINDOWS = [5, 10, 20]    # Simple Moving Average windows
RSI_PERIOD = 14              # RSI look-back period
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# ---------------------------------------------------------------------------
# Model hyperparameters – ARIMA
# ---------------------------------------------------------------------------
ARIMA_ORDER = (5, 1, 0)      # (p, d, q)

# ---------------------------------------------------------------------------
# Model hyperparameters – Prophet
# ---------------------------------------------------------------------------
PROPHET_CHANGEPOINT_PRIOR = 0.05
PROPHET_SEASONALITY_PRIOR = 10
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = False

# ---------------------------------------------------------------------------
# Model hyperparameters – LSTM
# ---------------------------------------------------------------------------
LSTM_SEQUENCE_LENGTH = 60    # number of past days used as input
LSTM_UNITS = [64, 32]        # hidden units per LSTM layer
LSTM_DROPOUT = 0.2
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_LEARNING_RATE = 5e-4
LSTM_SHUFFLE = False

# ---------------------------------------------------------------------------
# Model hyperparameters – XGBoost
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 1,
    "colsample_bytree": 1,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "objective": "reg:squarederror",
    "early_stopping_rounds": 120,
}

# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
ENSEMBLE_BASE_MODELS = ["arima", "prophet", "lstm", "xgboost"]

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
