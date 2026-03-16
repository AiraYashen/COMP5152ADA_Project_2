# COMP5152ADA Project – Stock Index Forecasting

Time-series forecasting of the NASDAQ-100 index (^NDX) using multiple machine-learning and statistical models:
**ARIMA · Prophet · LSTM · XGBoost · Ensemble**

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Data Collection](#data-collection)
4. [Feature Engineering](#feature-engineering)
5. [Models](#models)
6. [Evaluation](#evaluation)
7. [Notebooks](#notebooks)
8. [Running Tests](#running-tests)
9. [Configuration](#configuration)

---

## Project Structure

```
.
├── data/
│   ├── raw/              ← downloaded OHLCV and Volume data
│   ├── processed/        ← cleaned & feature-enriched datasets(train2020-2023, val2024, test2025)
│   └── external/         ← macro & sentiment data
├── models_saved/         ← serialised trained models
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_model_lstm.ipynb
│   └── 05_ensemble.ipynb
├── reports/
│   ├── figures/
│   ├── results/
│   └── final_report/
├── src/
│   ├── config.py         ← all hyperparameters & paths
│   ├── data/
│   │   ├── collector_sentiment_GDELT.py ← collect sentiment data / .json
│   │   ├── collector.py ← collect price data and macro data & processing sentiment data 
│   │   ├── preprocessor.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── lstm_model.py
│   │   ├── xgboost_model.py
│   │   └── ensemble.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── metrics.py
│   └── visualization/
│       └── plotter.py
└── tests/
    ├── test_data.py
    └── test_models.py
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AiraYashen/COMP5152ADA_Project_2.git
cd COMP5152ADA_Project_2
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (needed for sentiment analysis)- python

```python
import nltk
nltk.download('vader_lexicon')
```

---
After these steps, the environment is fully set up. You can proceed to run the notebooks — no additional data collection is required.
---



## Data Collection

> **All raw data files are already included in this repository.**  
> Running the collection steps again is only needed if you want to refresh the data.

The following files are pre-committed and ready to use:

| File | Location | Description |
|------|----------|-------------|
| `NDX_ohlcv.csv` | `data/raw/` | NASDAQ-100 OHLCV price data |
| `macro_data.csv` | `data/external/` | FRED macroeconomic indicators |
| `news_sentiment.csv` | `data/external/` | Processed daily sentiment scores |
| `gdelt_*.json` | `data/external/gdelt_json/` | Raw GDELT news articles (2020–2025) |

If you wish to re-download stock and macro data:

```python
from src.data.collector import DataCollector

collector = DataCollector()
data = collector.collect_all()   # re-downloads stock & macro; reads cached sentiment
```
Stock data is saved to `NDX_ohlcv.csv` and macro data to `macro_data.csv`.
Sentiment is loaded from the cached `news_sentiment.csv` (already committed).
To regenerate `news_sentiment.csv` from the committed GDELT JSON files:

```python
collector = DataCollector()
sentiment_df = collector.fetch_news_sentiment(force_refresh=True)
```
>Note: To download fresh GDELT news (e.g. for a new date range), run `collector_sentiment_GDELT.py` — this is realy slow due to GDELT's 5-second rate limit.
---

## Feature Engineering

```python
from src.data.feature_engineering import FeatureEngineer

fe = FeatureEngineer(price_col="Close")
df_features = fe.engineer_all(data["stock"], save=True)
```

Technical indicators added: MA5, MA10, MA20, RSI, MACD, Bollinger Bands, OBV, lagged returns.

---

## Models

| Model    | Module                         | Key dependency  |
|----------|--------------------------------|-----------------|
| ARIMA    | `src.models.arima_model`       | statsmodels     |
| Prophet  | `src.models.prophet_model`     | prophet         |
| LSTM     | `src.models.lstm_model`        | tensorflow      |
| XGBoost  | `src.models.xgboost_model`     | xgboost         |
| Ensemble | `src.models.ensemble`          | scikit-learn    |

### Example – ARIMA

```python
from src.models.arima_model import ARIMAModel

model = ARIMAModel(order=(5, 1, 0))
model.fit(train["Close"])
forecast = model.predict(steps=len(test))
```

### Example – LSTM

```python
from src.models.lstm_model import LSTMModel
from src.data.preprocessor import DataPreprocessor

pre = DataPreprocessor()
train, val, test = pre.split_data(df_features)
t_sc, v_sc, te_sc = pre.scale_features(train, val, test)

model = LSTMModel(sequence_length=60, epochs=50)
model.fit(t_sc.values, val_data=v_sc.values)
preds_scaled = model.predict(te_sc.values)
```

### Example – Ensemble

```python
from src.models.ensemble import EnsembleModel

ensemble = EnsembleModel()
ensemble.fit(val_predictions, y_val)     # val_predictions is a dict of arrays
final_preds = ensemble.predict(test_predictions)
```

---

## Evaluation

```python
from src.utils.metrics import calculate_metrics

metrics = calculate_metrics(y_true, y_pred)
# Returns: mse, rmse, mae, mape, directional_accuracy
```

---

## Notebooks

Open Jupyter Lab and run the notebooks in order:

```bash
jupyter lab
```

| Notebook                           | Purpose                               |
|------------------------------------|---------------------------------------|
| `01_eda.ipynb`                     | Exploratory data analysis             |
| `02_feature_engineering.ipynb`     | Technical indicators & feature prep   |
| `03_model_baseline.ipynb`          | ARIMA, Prophet, XGBoost baselines     |
| `04_model_lstm.ipynb`              | LSTM training and evaluation          |
| `05_ensemble.ipynb`                | Ensemble model and final comparison   |

---

## Optional:Running Tests

```bash
pytest tests/ -v
```

Optional:With coverage

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All settings live in `src/config.py`:

| Parameter               | Default          | Description                              |
|-------------------------|------------------|------------------------------------------|
| `TICKER`                | `^NDX`           | Yahoo Finance ticker                     |
| `START_DATE`            | `2020-01-01`     | Data start date                          |
| `END_DATE`              | `2025-12-31`     | Data end date                            |
| `TRAIN_END`             | `2023-12-31`     | End of training period                   |
| `VAL_END`               | `2024-12-31`     | End of validation period                 |
| `LSTM_SEQUENCE_LENGTH`  | `60`             | Look-back window for LSTM                |
| `LSTM_EPOCHS`           | `50`             | Maximum training epochs                  |
| `ARIMA_ORDER`           | `(5, 1, 0)`      | (p, d, q) for ARIMA                      |
| `RANDOM_SEED`           | `42`             | Global random seed                       |



---
## License

MIT – see [LICENSE](LICENSE).

---
## AI Assistance Statement

The initial draft of this project was developed with AI-assisted coding tools .  
All code, experiments, and final decisions were reviewed, validated, and refined by the authors.