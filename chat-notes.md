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

\subsubsection{Prophet}
\subsubsection{XGBoost}

\subsubsection{LSTM}

\subsubsection{Stacking Ensemble}

\subsection{Two-Phase Refitting Mechanism}

\subsubsection{Phase 1 (Validation/Meta-Training)}

\subsubsection{Phase 2 (Testing/Refitting)}

\section{Model Comparison and Evaluation}
\subsection{Metrics Evaluation}
\subsection{Results Discussion}


\section{Conclusion \& Future Directions}

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

帮我用英文写这章的内容，只讨论2024年的结果，使用latex格式插入表格，保留位小数--- Training XGBoost (2024) ---

2024 Validation Metrics:
{'mse': 8254268.134358606, 'rmse': 2873.0242140223263, 'mae': 2556.0283474392363, 'mape': 12.968319521493482, 'directional_accuracy': 0.4940239043824701}

\subsubsection{LSTM}

\subsubsection{Stacking Ensemble}

\subsection{Two-Phase Refitting Mechanism}

\subsubsection{Phase 1 (Validation/Meta-Training)}

\subsubsection{Phase 2 (Testing/Refitting)}

\section{Model Comparison and Evaluation}
\subsection{Metrics Evaluation}
\subsection{Results Discussion}


\section{Conclusion \& Future Directions}

Prophet 可以把它理解成一个“可解释的时间序列分解模型”，核心思想是：

$$
y(t)=g(t)+s(t)+h(t)+\varepsilon_t
$$

1. `g(t)`：长期趋势（trend）  
例如整体上涨/下跌速度，Prophet 会自动在一些时间点允许“斜率变化”（changepoints）。

2. `s(t)`：季节性（seasonality）  
例如周周期、年周期。它用傅里叶级数去拟合这种重复模式。

3. `h(t)`：节假日/特殊事件效应（holidays/events）  
如果你提供节假日或事件日期，模型会学习这些日期的额外影响。

4. `\varepsilon_t`：噪声  
不能解释的随机波动。

它是怎么“预测”的：
1. 先用历史数据拟合上面这些组件参数。
2. 再把未来日期喂给模型。
3. 模型会把“未来趋势 + 未来季节性 + 未来事件效应”加总，得到未来 `yhat`，同时给不确定性区间（`yhat_lower`, `yhat_upper`）。

为什么初学者常用 Prophet：
1. 对缺失值/异常值相对鲁棒。
2. 参数少，开箱即用。
3. 可解释性强，能看到 trend、weekly、yearly 各部分贡献。

在你项目这份代码里，Prophet 的实现大致是：
1. 把数据转成 `ds`（日期）和 `y`（目标值）格式；
2. 拟合 Prophet；
3. 生成未来日期并预测；
4. 输出 `yhat` 作为预测值。  
你这里还做了 two-phase 训练（先 2024 验证，再 2025 测试），属于时序回测上的工程封装。

如果你愿意，我下一步可以用你当前 prophet_model.py 逐行给你画一个“从 fit 到 predict”的简化流程图。


为什么在xgboost中使用的数据用了lag1，但arima和prephet 没有
这是因为三种模型“吃数据的方式”不一样：

1. XGBoost 是监督学习回归器  
它需要明确的输入特征 \(X\) 和目标 \(y\)。  
如果你要预测“明天的 Close”，就必须把“今天的信息”显式做成特征，所以你在 notebook 里做了 `shift(1)`（lag1）来避免同日泄露。  
否则它会偷偷用到当日信息，产生数据泄露。

2. ARIMA 本身就内置“滞后项”  
ARIMA 的 `AR` 部分就是用过去的 \(y_{t-1}, y_{t-2}, ...\) 来预测 \(y_t\)，不需要你再手工做 lag1 特征列。  
所以它看起来“没做 lag1”，其实模型内部已经在做。

3. Prophet 是时间分解模型，不是特征表回归  
Prophet 默认用时间戳学习趋势+季节性来预测 \(y\)，不会像 XGBoost 那样要求你手动构造 lag 特征矩阵。  
它可以加额外回归器，但不是必须；当前你的实现主要还是时间分解主导。

一句话总结：  
XGBoost 需要你“喂”滞后特征；ARIMA 把滞后写在模型结构里；Prophet 主要靠时间结构（trend/seasonality）而不是显式 lag 特征工程。




你这组结果非常像“欠拟合”而不是“过拟合”：

1. 2024 预测波动很小，说明模型输出被压成了平滑均值线。  
2. directional_accuracy 接近 0.5，说明方向信息几乎没学到。  
3. 你现在参数正则很强（深度浅、min_child_weight 大、gamma>0、采样率低），这会直接导致这种“平滑但没走势”的表现。

建议按这个顺序继续调，不要一次乱改：

一. 先减弱正则，让模型能动起来  
当前方向：max_depth 偏小、min_child_weight 偏大、gamma 偏高、subsample 和 colsample 偏低。  
下一轮可先试一组“放松版”：
- max_depth: 4 -> 6
- min_child_weight: 6 -> 2
- gamma: 0.2 -> 0
- subsample: 0.7 -> 0.9
- colsample_bytree: 0.7 -> 0.9
- learning_rate 保持 0.03
- n_estimators 保持较大并用 early stopping

二. 再做一轮小网格，不要大网格  
优先扫这三个最敏感参数：
- max_depth in 4, 5, 6
- min_child_weight in 1, 2, 4
- gamma in 0, 0.1, 0.2  
其他参数先固定，快速看 2024 directional_accuracy 和 MAPE。

三. 检查你当前特征是否“被二次滞后”  
你在 notebook 对所有特征统一 shift(1)，而原数据里已经有 lag_1 到 lag_5 列。  
这会把这些本来就滞后的特征再滞后一层，信号会变弱，预测更平。  
强烈建议做一个对照：
- 方案A：对所有特征 shift(1)（你现在）
- 方案B：只对非 lag 特征 shift(1)，lag_1 到 lag_5 不再二次 shift  
这个改动往往比单纯调参影响更大。

四. 不要用 bfill 填补 shift 后首行  
bfill 会把未来行信息“倒灌”到第一行。虽然影响只有首行，但建议更干净：直接 drop 首行。  
这也让验证更可信。

五. 评估目标建议  
如果你最终关心走势，单看 MSE 不够。  
在 2024 选参时建议主看：
- directional_accuracy
- MAPE
- 走势振幅是否合理（不是一条平滑线）

你现在最该优先做的是第三点（避免二次滞后）+第一点（放松正则）。  
如果你愿意，我可以下一步直接给你一版“只改最少代码”的特征构造替换方案，用于做 A/B 对照实验。






这次不是你调参没生效，而是遇到了 XGBoost 在这个任务里的结构性上限。

你现在的参数已经明显“放开”了，但 2024 仍跟不上，通常说明核心问题不是树深度或正则，而是这两点：

1. 分布漂移 + 外推能力限制  
你在 Phase 1 是用 2020-2023 训练，去预测整个 2024。  
如果 2024 出现新高区间，树模型很难外推出没见过的价格水平，会系统性低估上行。

2. 静态一次性预测 2024  
当前流程在 xgboost_model.py 是一次 fit 后直接给出整段 2024 预测，不会在 2024 内滚动吸收新信息，所以涨势容易跟丢。

先做 3 个“确认性检查”（很关键）：

1. 检查是否外推失败（天花板效应）  
在 notebook 的 XGB 单元（03_model_baseline.ipynb）后面打印：
- train 阶段目标最大值（2020-2023）
- 2024 真实最大值
- 2024 预测最大值  
如果 2024 真实最大值明显高于 train 最大值，且预测最大值卡住，基本就是外推限制。

2. 检查“振幅压缩”  
计算 std(pred_2024)/std(true_2024)。  
如果远小于 1（例如 0.5-0.7），说明模型在压幅，不是偶然误差。

3. 检查残差趋势  
看 residual = pred - true 是否随时间越来越负。  
如果后半段更负，说明跟涨失败是系统性的。

你下一步真正有效的方向（不改成 return）：

1. 把验证期改成滚动预测，而不是一次性整段预测  
也就是在 2024 内按周或按月递增窗口重训再预测下一段。  
这通常比继续调超参数更能解决“跟不上涨势”。

2. 给模型增加显式趋势特征（仍预测价格）  
在特征中加入：
- 时间序号（交易日编号）
- 20/60 日价格斜率
- 距离近 60 日高点
- 动量类特征（例如 5/20 日涨幅）  
这些比单纯加深树更有帮助。

3. 对训练样本做“近端加权”  
在 fit 时给 2023 末期更高权重，让模型更重视临近 2024 的状态。  
你现在在 xgboost_model.py 的 fit 可以传 sample_weight，这对跟涨通常有效。

额外说明你看到的“误差大但 MAPE 还行”：
- 价格水平本身高时，绝对误差 2000-3000 点对应百分比未必很夸张；
- RMSE/MAE看起来大，MAPE仍可能在 10%-13% 区间，这不矛盾。

结论：
你当前已经把“参数不足”这个问题基本排除了。下一阶段应从“训练方式和特征表达”入手，优先做滚动验证和趋势特征，而不是继续单纯调深度/正则。  
如果你愿意，我可以给你一版最小改动的滚动验证伪代码，直接套进 xgboost_model.py 的 train_and_refit 流程。



是的，这个思路通常更合适，尤其你现在遇到的是“上涨阶段跟不上”的问题。

核心原因：
1. `Close` 非平稳、量纲大，树模型更容易学到“水平区间”，外推到新高区间会有天花板效应。  
2. `return`（尤其 log return）更接近平稳，XGBoost 更容易学到“变化规律”而不是“绝对价位”。

建议你这样做：

1. 目标改为对数收益率  
$$
r_t=\log\left(\frac{C_t}{C_{t-1}}\right)
$$

2. 用 XGBoost 预测 $\hat r_t$。

3. 递推还原到价格（用于最终 ensemble）  
给定起点价格 $C_{t_0}$：
$$
\hat C_t=\hat C_{t-1}\cdot e^{\hat r_t}
$$
如果你用普通收益率 $R_t=\frac{C_t-C_{t-1}}{C_{t-1}}$，则
$$
\hat C_t=\hat C_{t-1}(1+\hat R_t)
$$

4. 和其他模型做 ensemble 时，统一在 `Close` 空间融合（你这个方向是对的）。

你会得到的好处：
1. 上涨阶段通常更容易“跟节奏”。  
2. 对 regime shift（比如 2024 这种新高段）鲁棒性更好。  
3. 训练目标尺度稳定，调参更有效。

要注意的代价：
1. 价格是递推还原，误差会累积。  
2. 若收益率预测有小偏差，长区间价格会漂移。  
3. 需要严格防泄漏（收益率和所有特征都只能用 $t-1$ 及更早信息）。

结论：  
在你当前场景下，`XGBoost 预测 return -> 还原 Close -> 再 ensemble` 是一个更合理、也更常见的工程路径。  
如果你愿意，我可以下一步给你一版最小改动的实现框架（训练目标替换、还原函数、以及和现有 ensemble 对齐的接口）。


ensemble 数据不对


目前，xgboost的预测是直接预测close price，但是这对于不断新高的的涨势而言，会出现“外推失败“。为了避免这种情况，帮我修改agboost_model.py代码，将预测当日close price改为当日return，完成当日return预测后再反推当日close price。结构上保留两阶段模型，即phase1 使用2020-2023的数据train，用2024数据validation。phase2 使用2021-2024数据train，2025数据test。

在03_model_baseline.ipynb cell9 中，生成了gboost_feature_importance_train.png,gboost_feature_importance_refit.png,[Phase 1] 2024 Validation Metrics,[Phase 2] 2025 Test Metrics,现在修改预测对象为return，这些图片或metrics的对象是否也要更改为return？


xgboost_val_forecast_2024.png,xgboost_test_forecast_2025.png 可以各自同时生成与实际return、close price的对比吗


的作图对象03_model_baseline.ipynb 



目前的训练中，使用了31个特征，这31个特征包含return lag1-5，是否包含当日return？ 如果采用当日return作为prediction，目前已经shift（1）的特征工程是否会导致数据泄露？




[Return Metrics] 2024 Validation:
{'mse': 0.00013005511669111864, 'rmse': 0.011404171021653378, 'mae': 0.008439852063790903, 'mape': 117.0598056530046, 'directional_accuracy': 0.549800796812749}

[Return Metrics] 2025 Test:
{'mse': 0.00022191839147266368, 'rmse': 0.014896925571159429, 'mae': 0.009438129342500752, 'mape': 128.6803086365768, 'directional_accuracy': 0.4435483870967742}





arima 

 [Phase 1] 2024 Validation Metrics (pdq=(2, 1, 0) | window=252):
{'mse': 48958.50246036772, 'rmse': 221.26568297042297, 'mae': 164.8913022803177, 'mape': 0.8648196995287013, 'directional_accuracy': 0.5179282868525896}

 [Phase 2] 2025 Test Metrics (pdq=(2, 1, 0) | window=252):
{'mse': 91474.42424544797, 'rmse': 302.4473908722771, 'mae': 205.92952024209595, 'mape': 0.9490222936062171, 'directional_accuracy': 0.4798387096774194}

 [Phase 1] 2024 Validation Metrics (pdq=(2, 1, 0) | window=126):
{'mse': 49939.24514256128, 'rmse': 223.47090446534932, 'mae': 166.4883539667886, 'mape': 0.873165564551175, 'directional_accuracy': 0.5258964143426295}

 [Phase 2] 2025 Test Metrics (pdq=(2, 1, 0) | window=126):
{'mse': 93149.88570780962, 'rmse': 305.2046620020894, 'mae': 206.94499881618145, 'mape': 0.9532863321272554, 'directional_accuracy': 0.5}


[Phase 1] 2024 Validation Metrics (pdq=(2, 1, 0) | window=63):
{'mse': 50480.60030869135, 'rmse': 224.67888264964142, 'mae': 168.21369450740502, 'mape': 0.8824652386186435, 'directional_accuracy': 0.545816733067729}

 [Phase 2] 2025 Test Metrics (pdq=(2, 1, 0) | window=63):
{'mse': 93609.88869864299, 'rmse': 305.9573315000688, 'mae': 211.19166778713327, 'mape': 0.97160322916027, 'directional_accuracy': 0.46774193548387094}

 [Phase 1] 2024 Validation Metrics (pdq=(5, 1, 0) | window=63):
{'mse': 53847.74150120466, 'rmse': 232.05116138732134, 'mae': 172.23911575041112, 'mape': 0.903761146194099, 'directional_accuracy': 0.5258964143426295}

 [Phase 2] 2025 Test Metrics (pdq=(5, 1, 0) | window=63):
{'mse': 98718.37769888641, 'rmse': 314.1948085167647, 'mae': 219.42468819326274, 'mape': 1.0124280251453277, 'directional_accuracy': 0.45161290322580644}


[Phase 1] 2024 Validation Metrics (pdq=(5, 1, 0) | window=126):
{'mse': 51232.39291645581, 'rmse': 226.34573757076984, 'mae': 165.65682981448043, 'mape': 0.8687142028760877, 'directional_accuracy': 0.545816733067729}

 [Phase 2] 2025 Test Metrics (pdq=(5, 1, 0) | window=126):
{'mse': 97395.31124426315, 'rmse': 312.08221872491094, 'mae': 216.2611427762305, 'mape': 0.9988200302470549, 'directional_accuracy': 0.47580645161290325}

 [Phase 1] 2024 Validation Metrics (pdq=(5, 1, 0) | window=252):
{'mse': 49879.182669860005, 'rmse': 223.3364785919667, 'mae': 164.82685792395583, 'mape': 0.863945941998008, 'directional_accuracy': 0.5258964143426295}

 [Phase 2] 2025 Test Metrics (pdq=(5, 1, 0) | window=252):
{'mse': 94423.38710259195, 'rmse': 307.28388682550855, 'mae': 213.01239693796123, 'mape': 0.9841185911406043, 'directional_accuracy': 0.45161290322580644}


[Phase 1] 2024 Validation Metrics (pdq=(1, 1, 1) | window=63):
{'mse': 50080.87509016948, 'rmse': 223.78756688022122, 'mae': 167.28748073063232, 'mape': 0.8773408048833722, 'directional_accuracy': 0.549800796812749}

 [Phase 2] 2025 Test Metrics (pdq=(1, 1, 1) | window=63):
{'mse': 95243.28659194846, 'rmse': 308.6151107641174, 'mae': 211.49127979349322, 'mape': 0.9730398575695974, 'directional_accuracy': 0.4879032258064516}


prophet
默认 
 [Phase 1] 2024 Validation Metrics (cps0.05_cpr0.8_sps10):
{'mse': 6093697.289845788, 'rmse': 2468.541530913707, 'mae': 2121.600205429631, 'mape': 10.969608345511983, 'directional_accuracy': 0.5338645418326693}

 [Phase 2] 2025 Test Metrics (cps0.05_cpr0.8_sps10):
{'mse': 5636727.808419877, 'rmse': 2374.1793968484935, 'mae': 2062.824690898856, 'mape': 9.61669237815255, 'directional_accuracy': 0.5282258064516129}


[Phase 1] 2024 Validation Metrics (cps0.05_cpr0.8_sps5):
{'mse': 5617805.75543343, 'rmse': 2370.1910799413263, 'mae': 2032.4446736252492, 'mape': 10.517458516543645, 'directional_accuracy': 0.5338645418326693}

 [Phase 2] 2025 Test Metrics (cps0.05_cpr0.8_sps5):
{'mse': 4938523.438512208, 'rmse': 2222.278884053981, 'mae': 1886.122973835855, 'mape': 8.84234023028247, 'directional_accuracy': 0.5362903225806451}

 [Phase 1] 2024 Validation Metrics (cps0.05_cpr0.8_sps1):
{'mse': 5803897.673469285, 'rmse': 2409.1279902631336, 'mae': 2066.4617656693827, 'mape': 10.688132713410992, 'directional_accuracy': 0.5298804780876494}

 [Phase 2] 2025 Test Metrics (cps0.05_cpr0.8_sps1):
{'mse': 5091953.951142058, 'rmse': 2256.535829793548, 'mae': 1924.5591676460372, 'mape': 9.011528187042932, 'directional_accuracy': 0.5362903225806451}

[Phase 1] 2024 Validation Metrics (cps0.05_cpr0.95_sps5):
{'mse': 11132235.653108133, 'rmse': 3336.5005099816985, 'mae': 2900.013958733696, 'mape': 14.898787528875127, 'directional_accuracy': 0.545816733067729}

 [Phase 2] 2025 Test Metrics (cps0.05_cpr0.95_sps5):
{'mse': 3290144.770233631, 'rmse': 1813.8756214894204, 'mae': 1472.432555429895, 'mape': 6.90458506155608, 'directional_accuracy': 0.532258064516129}




[Phase 1] 2024 Validation Metrics (cps0.3_cpr0.95_sps10):
{'mse': 128413972.15315513, 'rmse': 11331.988887797019, 'mae': 10170.557241519917, 'mape': 51.95728244313702, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.3_cpr0.95_sps10):
{'mse': 2226249.322028987, 'rmse': 1492.0621039450693, 'mae': 1137.1693506732192, 'mape': 5.409989776552927, 'directional_accuracy': 0.5362903225806451}




 [Phase 1] 2024 Validation Metrics (cps0.3_cpr0.95_sps1.0):
{'mse': 126551258.6048757, 'rmse': 11249.500371344307, 'mae': 10107.308510794972, 'mape': 51.645083820298765, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.3_cpr0.95_sps1.0):
{'mse': 2316746.4499003734, 'rmse': 1522.0862163163995, 'mae': 1170.8544948879178, 'mape': 5.568185957473176, 'directional_accuracy': 0.532258064516129}


[Phase 1] 2024 Validation Metrics (cps0.1_cpr0.95_sps5):
{'mse': 30882899.06259413, 'rmse': 5557.238438522692, 'mae': 5039.816748689302, 'mape': 25.892204622027045, 'directional_accuracy': 0.5737051792828686}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.95_sps5):
{'mse': 3202574.581532442, 'rmse': 1789.5738547297906, 'mae': 1448.908699024442, 'mape': 6.530134479446237, 'directional_accuracy': 0.5201612903225806}
[Phase 1] 2024 Validation Metrics (cps0.1_cpr0.95_sps10):
{'mse': 30877026.398343384, 'rmse': 5556.710033674907, 'mae': 5036.777369279403, 'mape': 25.873171114385862, 'directional_accuracy': 0.5737051792828686}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.95_sps10):
{'mse': 3406332.452648222, 'rmse': 1845.6252199859596, 'mae': 1496.351686181652, 'mape': 6.708757943253024, 'directional_accuracy': 0.5282258064516129}

[Phase 1] 2024 Validation Metrics (cps0.1_cpr0.95_sps1):
{'mse': 44713053.8530474, 'rmse': 6686.782025238104, 'mae': 6062.897674637411, 'mape': 31.114557645075415, 'directional_accuracy': 0.5537848605577689}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.95_sps1):
{'mse': 3418148.740079376, 'rmse': 1848.823609779845, 'mae': 1497.4778739327994, 'mape': 6.718965063533306, 'directional_accuracy': 0.5201612903225806}



[Phase 1] 2024 Validation Metrics (cps0.3_cpr0.95_sps2):
{'mse': 129792102.1218102, 'rmse': 11392.633678031178, 'mae': 10230.377079466421, 'mape': 52.26694055778315, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.3_cpr0.95_sps2):
{'mse': 2370573.2221145695, 'rmse': 1539.6665944660128, 'mae': 1187.3364629672624, 'mape': 5.645511060934179, 'directional_accuracy': 0.5403225806451613}

[Phase 1] 2024 Validation Metrics (cps0.1_cpr0.8_sps10):
{'mse': 24451744.743962165, 'rmse': 4944.870548756779, 'mae': 4467.2610390732225, 'mape': 23.009992354394242, 'directional_accuracy': 0.549800796812749}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.8_sps10):
{'mse': 3037325.0140957725, 'rmse': 1742.792303774541, 'mae': 1406.9555617153108, 'mape': 6.6383756111187, 'directional_accuracy': 0.5282258064516129}

[Phase 1] 2024 Validation Metrics (cps0.1_cpr0.8_sps1):
{'mse': 26134253.61388977, 'rmse': 5112.167213021281, 'mae': 4609.135463669279, 'mape': 23.723710668630197, 'directional_accuracy': 0.549800796812749}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.8_sps1):
{'mse': 2864815.6052401755, 'rmse': 1692.5766172437145, 'mae': 1357.0490776298798, 'mape': 6.409671431435257, 'directional_accuracy': 0.5362903225806451}






[Phase 1] 2024 Validation Metrics (cps0.05_cpr0.95_sps1):
{'mse': 13875533.483811287, 'rmse': 3724.987715927569, 'mae': 3235.4587178590555, 'mape': 16.585862878979498, 'directional_accuracy': 0.5697211155378487}

 [Phase 2] 2025 Test Metrics (cps0.05_cpr0.95_sps1):
{'mse': 3294373.843336068, 'rmse': 1815.04100321069, 'mae': 1463.4125960298961, 'mape': 6.879785805541714, 'directional_accuracy': 0.5362903225806451}








 [Phase 1] 2024 Validation Metrics (cps0.1_cpr0.8_sps5):
{'mse': 28464847.258132398, 'rmse': 5335.245754239668, 'mae': 4819.805481630917, 'mape': 24.807847112092016, 'directional_accuracy': 0.545816733067729}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.8_sps5):
{'mse': 3045123.3160885805, 'rmse': 1745.0281705716332, 'mae': 1422.1183330519682, 'mape': 6.692156588740722, 'directional_accuracy': 0.5362903225806451}




 [Phase 1] 2024 Validation Metrics (cps0.1_cpr0.95_sps2):
{'mse': 26142371.967469364, 'rmse': 5112.961174062381, 'mae': 4672.145707124767, 'mape': 24.041173270930553, 'directional_accuracy': 0.5617529880478087}

 [Phase 2] 2025 Test Metrics (cps0.1_cpr0.95_sps2):
{'mse': 3001000.187362469, 'rmse': 1732.3395127290923, 'mae': 1414.8352017451145, 'mape': 6.431301989597372, 'directional_accuracy': 0.5241935483870968}









 [Phase 1] 2024 Validation Metrics (cps0.5_cpr0.95_sps10):
{'mse': 159173340.45316204, 'rmse': 12616.391736671862, 'mae': 11405.79098012284, 'mape': 58.349832489282804, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.5_cpr0.95_sps10):
{'mse': 3042481.5122563387, 'rmse': 1744.271054697732, 'mae': 1438.9940040851413, 'mape': 6.75625123720929, 'directional_accuracy': 0.5080645161290323}

 [Phase 1] 2024 Validation Metrics (cps0.5_cpr0.95_sps2):
{'mse': 152728876.7447735, 'rmse': 12358.352509326374, 'mae': 11186.686955115567, 'mape': 57.248426128930376, 'directional_accuracy': 0.5737051792828686}

 [Phase 2] 2025 Test Metrics (cps0.5_cpr0.95_sps2):
{'mse': 3243198.024937634, 'rmse': 1800.8881211606772, 'mae': 1509.7158736496463, 'mape': 7.057457613482075, 'directional_accuracy': 0.5}


 [Phase 1] 2024 Validation Metrics (cps0.5_cpr0.95_sps1):
{'mse': 166781176.5843332, 'rmse': 12914.378675891969, 'mae': 11632.290581346659, 'mape': 59.462475028790074, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.5_cpr0.95_sps1):
{'mse': 3455024.82710922, 'rmse': 1858.7697079275904, 'mae': 1576.8480012934383, 'mape': 7.3478659092481315, 'directional_accuracy': 0.5}

[Phase 1] 2024 Validation Metrics (cps0.5_cpr0.95_sps5):
{'mse': 164301117.33368424, 'rmse': 12817.999739962715, 'mae': 11557.456078418707, 'mape': 59.09273670065429, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.5_cpr0.95_sps5):
{'mse': 3769903.3299038, 'rmse': 1941.6238899188997, 'mae': 1686.9631736680185, 'mape': 7.792971694111285, 'directional_accuracy': 0.5201612903225806}

[Phase 1] 2024 Validation Metrics (cps0.3_cpr0.95_sps5):
{'mse': 135152289.45723036, 'rmse': 11625.501686259839, 'mae': 10430.156228145788, 'mape': 53.27686395286754, 'directional_accuracy': 0.5776892430278885}

 [Phase 2] 2025 Test Metrics (cps0.3_cpr0.95_sps5):
{'mse': 2260096.526715457, 'rmse': 1503.3617418025035, 'mae': 1141.6034449980825, 'mape': 5.448002453491998, 'directional_accuracy': 0.532258064516129}


lstm
[Phase 1] 2024 Validation Metrics:
{'mse': 3171384.769482222, 'rmse': 1780.8382210302602, 'mae': 1636.3164967164373, 'mape': 8.191940656742993, 'directional_accuracy': 0.5078534031413613}

[Phase 2] 2025 Test Metrics:
{'mse': 1555966.882535228, 'rmse': 1247.3840156644737, 'mae': 1129.0000579500002, 'mape': 4.804767967228786, 'directional_accuracy': 0.5425531914893617}



alpha = 10

--- Training Ensemble with Walk-forward + Rolling Window (126) ---

 [Phase 1] 2024 Validation Metrics (Walk-forward Ensemble):
{'mse': 100026.63701717321, 'rmse': 316.2698800347153, 'mae': 232.2652756721724, 'mape': 1.2176653266282325, 'directional_accuracy': 0.4973821989528796}

 [Phase 2] 2025 Test Metrics (Walk-forward Ensemble):
{'mse': 103031.25250353487, 'rmse': 320.9848166246106, 'mae': 202.68734398960055, 'mape': 0.9165100590914684, 'directional_accuracy': 0.48936170212765956}

 Latest Rolling Ensemble Weights:
lstm       1.233401
prophet    0.049055
xgboost   -0.090307
arima     -0.265254
dtype: float64

🏆 ULTIMATE 2025 TEST SET METRICS 🏆

[ARIMA]
MSE:  102120.28 | RMSE: 319.56
MAE:  212.37 | MAPE: 0.96%
Dir Acc: 45.21%

[Prophet]
MSE:  5328889.54 | RMSE: 2308.44
MAE:  1983.35 | MAPE: 9.18%
Dir Acc: 53.72%

[XGBoost]
MSE:  92764.35 | RMSE: 304.57
MAE:  196.39 | MAPE: 0.89%
Dir Acc: 47.87%

[LSTM]
MSE:  2300642.37 | RMSE: 1516.79
MAE:  1246.56 | MAPE: 5.19%
Dir Acc: 52.66%

[Ensemble]
MSE:  103031.25 | RMSE: 320.98
MAE:  202.69 | MAPE: 0.92%
Dir Acc: 48.94%
All final plots generated and saved to reports/figures/ !

 Fresh metrics successfully saved to /workspaces/COMP5152ADA_Project_2/reports/results/ultimate_2025_metrics.csv !
                   mse         rmse          mae      mape  \
ARIMA     1.021203e+05   319.562644   212.374219  0.964985   
Prophet   5.328890e+06  2308.438766  1983.348273  9.176204   
XGBoost   9.276435e+04   304.572407   196.391501  0.888126   
LSTM      2.300642e+06  1516.786857  1246.558821  5.187972   
Ensemble  1.030313e+05   320.984817   202.687344  0.916510   

          directional_accuracy  
ARIMA                 0.452128  
Prophet               0.537234  
XGBoost               0.478723  
LSTM                  0.526596  
Ensemble              0.489362  
All 2025 final predictions successfully saved to /workspaces/COMP5152ADA_Project_2/reports/results/ultimate_2025_predictions.csv !

Preview of first 5 rows:
            Actual_Close         ARIMA       Prophet       XGBoost  \
Date                                                                 
2025-04-01  19436.419922  19354.674536  23124.194720  19264.954363   
2025-04-02  19581.779297  19481.240525  23164.654820  19457.574148   
2025-04-03  18521.470703  19720.409740  23202.551916  19603.175766   
2025-04-04  17397.699219  18575.731652  23233.074053  18559.630381   
2025-04-07  17430.679688  17297.791592  23249.067429  17433.854692   

                    LSTM      Ensemble  
Date                                    
2025-04-01  19522.308649  19404.569021  
2025-04-02  19465.566356  19624.363122  
2025-04-03  19397.799144  19685.538242  
2025-04-04  19207.821944  18573.797927  
2025-04-07  18917.682230  17324.347353  



alpha = 10 , positive = true 


--- Training Ensemble with Walk-forward + Rolling Window (126) ---

 [Phase 1] 2024 Validation Metrics (Walk-forward Ensemble):
{'mse': 98895.55124167197, 'rmse': 314.4766306765448, 'mae': 227.56751313354948, 'mape': 1.1937108700989085, 'directional_accuracy': 0.5235602094240838}

 [Phase 2] 2025 Test Metrics (Walk-forward Ensemble):
{'mse': 96149.30764205402, 'rmse': 310.0795182562918, 'mae': 199.02636498206047, 'mape': 0.89794900016961, 'directional_accuracy': 0.4734042553191489}

 Latest Rolling Ensemble Weights:
lstm       0.922774
prophet    0.095004
arima      0.000000
xgboost    0.000000
dtype: float64

🏆 ULTIMATE 2025 TEST SET METRICS 🏆

[ARIMA]
MSE:  102120.28 | RMSE: 319.56
MAE:  212.37 | MAPE: 0.96%
Dir Acc: 45.21%

[Prophet]
MSE:  5328889.54 | RMSE: 2308.44
MAE:  1983.35 | MAPE: 9.18%
Dir Acc: 53.72%

[XGBoost]
MSE:  92764.35 | RMSE: 304.57
MAE:  196.39 | MAPE: 0.89%
Dir Acc: 47.87%

[LSTM]
MSE:  2300642.37 | RMSE: 1516.79
MAE:  1246.56 | MAPE: 5.19%
Dir Acc: 52.66%

[Ensemble]
MSE:  96149.31 | RMSE: 310.08
MAE:  199.03 | MAPE: 0.90%
Dir Acc: 47.34%
All final plots generated and saved to reports/figures/ !

 Fresh metrics successfully saved to /workspaces/COMP5152ADA_Project_2/reports/results/ultimate_2025_metrics.csv !
                   mse         rmse          mae      mape  \
ARIMA     1.021203e+05   319.562644   212.374219  0.964985   
Prophet   5.328890e+06  2308.438766  1983.348273  9.176204   
XGBoost   9.276435e+04   304.572407   196.391501  0.888126   
LSTM      2.300642e+06  1516.786857  1246.558821  5.187972   
Ensemble  9.614931e+04   310.079518   199.026365  0.897949   

          directional_accuracy  
ARIMA                 0.452128  
Prophet               0.537234  
XGBoost               0.478723  
LSTM                  0.526596  
Ensemble              0.473404  
All 2025 final predictions successfully saved to /workspaces/COMP5152ADA_Project_2/reports/results/ultimate_2025_predictions.csv !

Preview of first 5 rows:
            Actual_Close         ARIMA       Prophet       XGBoost  \
Date                                                                 
2025-04-01  19436.419922  19354.674536  23124.194720  19264.954363   
2025-04-02  19581.779297  19481.240525  23164.654820  19457.574148   
2025-04-03  18521.470703  19720.409740  23202.551916  19603.175766   
2025-04-04  17397.699219  18575.731652  23233.074053  18559.630381   
2025-04-07  17430.679688  17297.791592  23249.067429  17433.854692   

                    LSTM      Ensemble  
Date                                    
2025-04-01  19522.308649  19431.701835  
2025-04-02  19465.566356  19540.355849  
2025-04-03  19397.799144  19699.613978  
2025-04-04  19207.821944  18595.657056  
2025-04-07  18917.682230  17390.168960  



[Return Metrics] 2024 Validation:
{'mse': 0.00013005511669111864, 'rmse': 0.011404171021653378, 'mae': 0.008439852063790903, 'mape': 117.0598056530046, 'directional_accuracy': 0.549800796812749}
Val return preds (%):
count    252.000000
mean       0.095119
std        0.036680
min       -0.080355
25%        0.082211
50%        0.094058
75%        0.114740
max        0.190939
Name: XGBoost_Return, dtype: float64

[Return Metrics] 2025 Test:
{'mse': 0.00022191839147266368, 'rmse': 0.014896925571159429, 'mae': 0.009438129342500752, 'mape': 128.6803086365768, 'directional_accuracy': 0.4435483870967742}
Test return preds (%):
count    249.000000
mean       0.100177
std        0.043919
min       -0.103402
25%        0.094058
50%        0.104416
75%        0.128940
max        0.216387

Applying shift(1) to 32 columns: OHLCV + Volume_MA + MA/RSI/MACD/Bollinger + OBV + Sentiment
XGBoost uses 41 features.
Shifted columns are suffixed with _lag1.
--- Training XGBoost (Two-Phase Refitting) ---
{'n_estimators': 1200, 'learning_rate': 0.05, 'max_depth': 4, 'objective': 'reg:squarederror', 'random_state': 42, 'min_child_weight': 1, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 0.7, 'subsample': 0.85, 'colsample_bytree': 0.75, 'colsample_bylevel': 0.6, 'early_stopping_rounds': 50}
{'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 1, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 0.7}

[Phase 1] 2024 Validation Metrics:
{'mse': 48141.89510142491, 'rmse': 219.4126138156713, 'mae': 163.26386563090463, 'mape': 0.8567352890097668, 'directional_accuracy': 0.5099601593625498}

[Phase 2] 2025 Test Metrics:
{'mse': 91623.34813478873, 'rmse': 302.69348875519063, 'mae': 205.71978735320604, 'mape': 0.9473773207230668, 'directional_accuracy': 0.4596774193548387}

[Return Metrics] 2024 Validation:
{'mse': 0.00013087137012191038, 'rmse': 0.011439902539878142, 'mae': 0.008556625262338254, 'mape': 102.49223423972744, 'directional_accuracy': 0.09561752988047809}
Val return preds (%):
count    252.000000
mean       0.042383
std        0.018878
min       -0.096310
25%        0.041808
50%        0.041808
75%        0.041808
max        0.173535
Name: XGBoost_Return, dtype: float64

[Return Metrics] 2025 Test:
{'mse': 0.00022258076685908578, 'rmse': 0.014919140955801906, 'mae': 0.009484880478146527, 'mape': 121.57862990527333, 'directional_accuracy': 0.07258064516129033}
Test return preds (%):
count    249.000000
mean       0.044232
std        0.022617
min       -0.023843
25%        0.041808
50%        0.041808
75%        0.041808
max        0.173535
Name: XGBoost_Return, dtype: float64


[Return Metrics] 2024 Validation: reg:absoluteerror
Val return preds (%):
count    252.000000
mean       0.095119
std        0.036680
min       -0.080355
25%        0.082211
50%        0.094058
75%        0.114740
max        0.190939


[Return Metrics] 2025 Test: reg:absoluteerror
Test return preds (%):
count    249.000000
mean       0.100177
std        0.043919
min       -0.103402
25%        0.094058
50%        0.104416
75%        0.128940
max        0.216387

[Return Metrics] 2024 Validation: reg:squarederror
Val return preds (%):
count    252.000000
mean       0.042383
std        0.018878
min       -0.096310
25%        0.041808
50%        0.041808
75%        0.041808
max        0.173535


[Return Metrics] 2025 Test: reg:squarederror
Test return preds (%):
count    249.000000
mean       0.044232
std        0.022617
min       -0.023843
25%        0.041808
50%        0.041808
75%        0.041808
max        0.173535