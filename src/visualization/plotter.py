"""
Visualization utilities for model comparison and EDA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Use a non-interactive backend so that figures can be saved without a display
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402


class Plotter:
    """Collection of plotting helpers for the forecasting project."""

    def __init__(self, figures_dir: Optional[Path] = None) -> None:
        self.figures_dir = Path(figures_dir or FIGURES_DIR)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_or_show(self, fig: plt.Figure, filename: Optional[str]) -> None:
        if filename:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved figure to %s", path)
        plt.close(fig)

    # ------------------------------------------------------------------
    # EDA plots
    # ------------------------------------------------------------------

    def plot_price_history(
            self,
            df: pd.DataFrame,
            price_col: str = "Close",
            title: str = "NASDAQ-100 Price History",
            filename: Optional[str] = "price_history.png",
            xlim: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> None:
        """Plot the closing price time series.

        Tick strategy (option 3):
        - Major ticks every 6 months at Jun/Dec (so 2025-12 appears naturally).
        - If xlim is provided, force the x-axis range to avoid extra padding.
        """
        df_plot = df.copy()
        df_plot.index = pd.to_datetime(df_plot.index)
        df_plot = df_plot.sort_index()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_plot.index, df_plot[price_col], linewidth=1.2, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")

        if xlim is not None:
            ax.set_xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))

        # Major ticks at June/December (every 6 months)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12], bymonthday=-1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.autofmt_xdate()
        ax.grid(alpha=0.3)
        self._save_or_show(fig, filename)

    def plot_technical_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        filename: Optional[str] = "technical_indicators.png",
    ) -> None:
        """Four-panel chart: price+MAs, RSI, MACD, Bollinger Bands."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # Price + Moving Averages
        ax = axes[0]
        ax.plot(df.index, df[price_col], label=price_col, linewidth=1)
        for col in ["MA5", "MA10", "MA20"]:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col, linewidth=0.8, linestyle="--")
        ax.set_title("Price & Moving Averages")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # RSI
        ax = axes[1]
        if "RSI" in df.columns:
            ax.plot(df.index, df["RSI"], color="purple", linewidth=0.8)
            ax.axhline(70, color="red", linestyle="--", linewidth=0.7, label="Overbought")
            ax.axhline(30, color="green", linestyle="--", linewidth=0.7, label="Oversold")
        ax.set_title("RSI")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # MACD
        ax = axes[2]
        if "MACD" in df.columns:
            ax.plot(df.index, df["MACD"], label="MACD", linewidth=0.8)
        if "MACD_signal" in df.columns:
            ax.plot(
                df.index, df["MACD_signal"], label="Signal", linewidth=0.8, linestyle="--"
            )
        if "MACD_hist" in df.columns:
            ax.bar(
                df.index,
                df["MACD_hist"],
                width=1,
                alpha=0.3,
                label="Histogram",
                color="grey",
            )
        ax.set_title("MACD")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Bollinger Bands
        ax = axes[3]
        ax.plot(df.index, df[price_col], linewidth=1, label=price_col)
        if "BB_upper" in df.columns:
            ax.fill_between(
                df.index,
                df["BB_lower"],
                df["BB_upper"],
                alpha=0.15,
                color="orange",
                label="Bollinger Bands",
            )
        ax.set_title("Bollinger Bands")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12], bymonthday=-1))
        fig.autofmt_xdate()
        fig.tight_layout()
        self._save_or_show(fig, filename)

    # ------------------------------------------------------------------
    # Model comparison plots
    # ------------------------------------------------------------------

    def plot_predictions_comparison(
        self,
        y_true: Union[pd.Series, np.ndarray],
        predictions: Dict[str, Union[pd.Series, np.ndarray]],
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Model Predictions vs Actual",
        filename: Optional[str] = "predictions_comparison.png",
    ) -> None:
        """Overlay actual prices with predictions from multiple models."""
        fig, ax = plt.subplots(figsize=(14, 6))
        x = dates if dates is not None else np.arange(len(y_true))
        ax.plot(x, y_true, label="Actual", linewidth=1.5, color="black")
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (name, pred) in enumerate(predictions.items()):
            ax.plot(x[: len(pred)], pred, label=name, linewidth=1, linestyle="--",
                    color=colors[i % len(colors)])
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(alpha=0.3)
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
        self._save_or_show(fig, filename)

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        filename: Optional[str] = "metrics_comparison.png",
    ) -> None:
        """Bar chart comparing evaluation metrics across models."""
        df = pd.DataFrame(metrics_dict).T  # models × metrics
        metric_cols = [c for c in ["rmse", "mae", "mape"] if c in df.columns]
        if not metric_cols:
            logger.warning("No standard metric columns found.")
            return

        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metric_cols):
            df[metric].sort_values().plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(metric.upper())
            ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save_or_show(fig, filename)

    def plot_feature_importance(
        self,
        importance: pd.Series,
        top_n: int = 20,
        title: str = "XGBoost Feature Importance",
        filename: Optional[str] = "feature_importance.png",
    ) -> None:
        """Horizontal bar chart for feature importances."""
        top = importance.head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n // 2)))
        top[::-1].plot(kind="barh", ax=ax, color="teal")
        ax.set_title(title)
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save_or_show(fig, filename)

    def plot_loss_history(
        self,
        history,
        filename: Optional[str] = "lstm_loss.png",
    ) -> None:
        """Plot LSTM training and validation loss curves."""
        if history is None:
            logger.warning("No history object provided.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history["loss"], label="Train Loss")
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="Val Loss")
        ax.set_title("LSTM Training History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_or_show(fig, filename)

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = "correlation_heatmap.png",
    ) -> None:
        """Plot a correlation heatmap for the feature DataFrame."""
        try:
            import seaborn as sns  # type: ignore
        except ImportError:
            logger.warning("seaborn not installed; skipping correlation heatmap.")
            return

        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            center=0,
            annot=False,
            fmt=".2f",
            linewidths=0.5,
        )
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        self._save_or_show(fig, filename)
