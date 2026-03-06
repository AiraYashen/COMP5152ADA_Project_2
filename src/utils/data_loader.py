"""
Helpers for loading and saving processed datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def load_processed_data(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a CSV file from the processed data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file (with or without ``.csv`` extension).
    data_dir : Path, optional
        Override directory. Defaults to ``config.PROCESSED_DATA_DIR``.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a ``DatetimeIndex``.
    """
    data_dir = data_dir or PROCESSED_DATA_DIR
    if not filename.endswith(".csv"):
        filename += ".csv"
    filepath = Path(data_dir) / filename
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def save_processed_data(
    df: pd.DataFrame,
    filename: str,
    data_dir: Optional[Path] = None,
) -> Path:
    """Save a DataFrame to the processed data directory as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Output filename (with or without ``.csv`` extension).
    data_dir : Path, optional
        Override directory. Defaults to ``config.PROCESSED_DATA_DIR``.

    Returns
    -------
    Path
        Full path of the saved file.
    """
    data_dir = Path(data_dir or PROCESSED_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".csv"):
        filename += ".csv"
    filepath = data_dir / filename
    df.to_csv(filepath)
    logger.info("Saved data to %s", filepath)
    return filepath
