\section{Introduction}
\label{sec:introduction}

\subsection{Project Overview}
\subsection{Workflow}

\section{Domain Selection and Dataset Collection}
\subsection{Data Collecting}
\subsubsection{Market Data Acquisition (Primary Time Series)}
\subsubsection{Alternative Data Acquisition (Macro + Sentiment)}

\section{Discovery of Data Characteristics and Preprocessing}
\subsection{Exploratory Analysis of Raw Market Data}
\subsection{Data Pre-processing}
\subsubsection{Calendar Alignment and Timestamp}
\subsubsection{Missing Value Handling}
\subsubsection{Data Splitting}

\section{Feature Engneering \& Data Processing }
\subsection{Technical Index Calculation}
\subsubsection{List of Technical Indicators}
\subsubsection{Preventing Future Leakage in Indicator Calculation}
\subsection{Sentiment Data Aggregation}
\subsection{Feature Scaling and Inverse Transformation}

\section{Model Architecture \& Backtesting Framework}

\subsection{Baseline Models and Ensemble Model}
\subsubsection{ARIMA}

As a classical statistical baseline, we adopted an ARIMA model to capture the linear autocorrelation structure in the closing-price series. ARIMA is defined by three orders, $(p,d,q)$, where $p$ is the autoregressive lag order, $d$ is the differencing order used to improve stationarity, and $q$ is the moving-average lag order. Before fitting, the training split was inspected for trend and non-stationarity, and first-order differencing was applied when needed to stabilize the mean process.

To determine a robust configuration under walk-forward backtesting, we conducted a targeted parameter comparison on the 2024 validation split across both ARIMA order and rolling window length. The tested settings were: (i) $(5,1,0)$ with window $=126$, (ii) $(2,1,0)$ with window $=126$, and (iii) $(2,1,0)$ with window $=252$. The corresponding validation metrics are summarized in Table~\ref{tab:arima_param_search}.

\begin{table}[htbp]
\centering
\caption{ARIMA parameter comparison on 2024 validation set (walk-forward + rolling window)}
\label{tab:arima_param_search}
\begin{tabular}{lcccccc}
\hline
Configuration $(p,d,q)$ & Window & MSE & RMSE & MAE & MAPE (\%) & Directional Accuracy \\
\hline
$(5,1,0)$ & 126 & 51232.3929 & 226.3457 & 165.6568 & 0.8687 & 0.5458 \\
$(2,1,0)$ & 126 & 49939.2451 & 223.4709 & 166.4884 & 0.8732 & 0.5259 \\
$(2,1,0)$ & 252 & 48958.5025 & 221.2657 & 164.8913 & 0.8648 & 0.5179 \\
\hline
\end{tabular}
\end{table}

Based on this comparison, we selected the final ARIMA configuration as $(p,d,q)=(2,1,0)$ with rolling window $=252$. Although $(5,1,0)$ yielded slightly higher directional accuracy, $(2,1,0)$ with the longer window achieved the best overall error profile across MSE, RMSE, MAE, and MAPE, which better matches our project objective of stable price-level forecasting and downstream ensemble integration.

Although ARIMA is interpretable and robust for short-horizon forecasting, its linear structure limits its ability to represent nonlinear interactions from technical, macro, and sentiment signals. Therefore, in this project ARIMA serves as a reference benchmark rather than the final production model, providing an interpretable lower bound against which machine-learning and deep-learning models can be assessed.


\subsubsection{Prophet}

Prophet assumes an additive decomposition of the target series, written as
\[
y(t)=g(t)+s(t)+h(t)+\varepsilon_t,
\]
where $g(t)$ denotes the long-term trend, $s(t)$ captures seasonal patterns, $h(t)$ models holiday/event effects, and $\varepsilon_t$ is the residual noise term.

As a second baseline, we used Prophet to model additive components of the closing-price time series, including trend and seasonality. In our two-phase protocol, Phase 1 fits on pre-validation history and predicts the 2024 validation period, while Phase 2 refits on the updated history and predicts the 2025 test period. During diagnostics, we observed that using calendar-day frequency ($\texttt{freq='D'}$) could introduce visually periodic oscillations when forecasting trading-day financial data. After switching to business-day frequency ($\texttt{freq='B'}$), this sawtooth artifact was substantially reduced, improving temporal alignment with market calendars.

The final Prophet performance under the two-phase setting is summarized in Table~\ref{tab:prophet_two_phase_metrics}.

\begin{table}[htbp]
\centering
\caption{Prophet performance in two-phase backtesting (business-day frequency)}
\label{tab:prophet_two_phase_metrics}
\begin{tabular}{lccccc}
\hline
Phase & MSE & RMSE & MAE & MAPE (\%) & Directional Accuracy \\
\hline
2024 Validation (Phase 1) & 6093697.2898 & 2468.5415 & 2121.6002 & 10.9696 & 0.5339 \\
2025 Test (Phase 2) & 5636727.8084 & 2374.1794 & 2062.8247 & 9.6167 & 0.5282 \\
\hline
\end{tabular}
\end{table}

Compared with Phase 1, Phase 2 shows lower MSE, RMSE, MAE, and MAPE, indicating better overall level prediction on the test year after refitting. Directional accuracy remains stable around 0.53, suggesting that Prophet captures broad movement structure but still has limited sensitivity to short-term direction changes in this market setting.

\subsubsection{XGBoost}

\subsubsection{LSTM}

\subsubsection{Stacking Ensemble}

\subsection{Two-Phase Refitting Mechanism}

\subsubsection{Phase 1 (Validation/Meta-Training)}

\subsubsection{Phase 2 (Testing/Refitting)}

\section{Model Comparison and Evaluation}
\subsection{Metrics Evaluation}
\subsection{Results Discussion}


\section{Conclusion \& Future Directions}# ---- Phase 1-only model for feature importance (train) ----
xgb_phase1 = XGBoostModel()
xgb_phase1.fit(
    train_xgb_df[xgb_feature_cols],
    train_xgb_df[target_col],
    X_val=val_xgb_df[xgb_feature_cols],
    y_val=val_xgb_df[target_col],
)

xgb_importance_train = pd.Series(
    xgb_phase1._model.feature_importances_,
    index=xgb_feature_cols
)

plotter.plot_feature_importance(
    xgb_importance_train,
    top_n=20,
    title="XGBoost Feature Importance (train)",
    filename="xgboost_feature_importance_train.png",
    annotate=True,
    decimals=4,
)

# ---- Phase 2 refitted importance ----
xgb_importance_refit = pd.Series(
    xgb._model.feature_importances_,
    index=xgb_feature_cols
)

plotter.plot_feature_importance(
    xgb_importance_refit,
    top_n=20,
    title="XGBoost Feature Importance (Refitted)",
    filename="xgboost_feature_importance.png",
    annotate=True,
    decimals=4,
)