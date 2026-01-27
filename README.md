# Realized Volatility Forecasting: Deep Learning vs. Econometric Benchmarks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow.svg)]()

**A comprehensive empirical study comparing GRU-based deep learning architectures against classical GARCH econometric models for one-day-ahead realized volatility forecasting on the S&P 500 index.**

[Key Findings](#key-findings) •
[Installation](#installation) •
[Usage](#usage) •
[Methodology](#methodology) •
[Results](#results) •
[Documentation](#documentation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Methodology](#methodology)
  - [Realized Volatility Construction](#realized-volatility-construction)
  - [GARCH Benchmark](#garch-benchmark)
  - [GRU Neural Network](#gru-neural-network)
- [Evaluation Framework](#evaluation-framework)
- [Results](#results)
- [Future Directions](#future-directions)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

This research project investigates a fundamental question in quantitative finance:

> **Can flexible, data-driven deep learning models capture volatility dynamics that parametric econometric specifications miss, or do the additional degrees of freedom lead to overfitting without predictive gain?**

The project implements and rigorously compares:

| Model | Type | Description |
|-------|------|-------------|
| **GARCH(1,1)** | Econometric | Classical volatility model with Student-t innovations |
| **GRU** | Deep Learning | Gated Recurrent Unit neural network |
| **Naive Persistence** | Baseline | Random walk benchmark ($\hat{\sigma}_{t+1} = \sigma_t$) |

The study uses **high-frequency intraday data** (1-minute SPY prices from 2008-2021) to construct realized volatility from 5-minute returns, ensuring a high-quality target variable for model evaluation.

---

## Key Findings

### Primary Results

| Metric | GARCH(1,1) | GRU | Naive | Winner |
|--------|------------|-----|-------|--------|
| **MAE** | **0.505** | 0.543 | 0.631 | GARCH (+7%) |
| **MSE** | **0.998** | 1.121 | 1.579 | GARCH (+12%) |
| **QLIKE** | **0.317** | 0.395 | 0.508 | GARCH (+20%) |
| **R²** | — | 0.612 | 0.452 | — |

### Key Insights

1. **GARCH dominates on all metrics** — The parsimonious econometric model achieves superior out-of-sample performance during the test period (Jan 2020 – May 2021).

2. **Both models significantly outperform naive persistence** — Confirming exploitable autocorrelation structure in volatility.

3. **QLIKE advantage is economically meaningful** — GARCH's 20% lower QLIKE indicates better risk assessment, particularly important for underestimation penalties.

4. **COVID-19 stress test** — The test period includes extreme market conditions (March 2020 crash), providing a challenging but potentially non-representative evaluation regime.

### Limitations

- GRU uses only squared returns as input (no VIX, volume, sentiment features)
- No formal Diebold-Mariano statistical significance test conducted
- Fixed train/test split without rolling window evaluation
- Test period dominated by exceptional market conditions

---

## Project Structure

```
MR Project/
├── Code/
│   ├── Data/
│   │   ├── dataset/               # Raw and processed data
│   │   │   ├── spy_1min_2008_2021_cleaned.csv    # Primary dataset
│   │   │   ├── spy_5min_2008_2021_realized_vol.csv
│   │   │   ├── eurusd*.csv        # EUR/USD datasets
│   │   │   └── ...
│   │   ├── FAISS/                 # Vector indices for retrieval
│   │   ├── model/                 # Saved Keras/TensorFlow models
│   │   │   ├── lstm_volatility_model.keras
│   │   │   └── sp500_gru_vol_change.keras
│   │   └── scaler/                # Data preprocessing scalers
│   │
│   ├── Tests/                     # Experimental notebooks
│   │   ├── ARCH-benchmark.ipynb   # GARCH(1,1) implementation & evaluation
│   │   ├── GRU-benchmark.ipynb    # GRU model training & evaluation
│   │   ├── GRU-GARCH-Benchmark.ipynb  # Head-to-head comparison
│   │   ├── Desc-stats.ipynb       # Exploratory data analysis
│   │   ├── DataEngineering.ipynb  # Data preprocessing pipeline
│   │   ├── SP500_GK-vol-GRU-comp-GARCH.ipynb  # Garman-Klass volatility
│   │   ├── SP500-Close_vol-GRU-comp-GARCH.ipynb
│   │   ├── S&P500_vol_change_LSTM.ipynb
│   │   ├── S&P500_vol_LSTM_naive.ipynb
│   │   ├── RNN.ipynb              # RNN experiments
│   │   ├── Dataset.ipynb          # Data exploration
│   │   └── utils/
│   │       └── Features.py        # Feature engineering utilities
│   │
│   ├── DataEngineering/           # Production data pipelines
│   ├── pyproject.toml             # Project dependencies
│   └── README.md                  # This file
│
├── Docs/
│   ├── report/
│   │   └── Realized_Volatility_Forecasting_Report.md  # Full technical report
│   ├── Evaluation.md              # Evaluation framework guide
│   ├── Retrieval Augmented Forecasting Advancement.md
│   ├── keywords.md
│   ├── Embeddings/
│   ├── Illustrations/
│   └── papers/
│
└── README_RAF.md                  # Retrieval-Augmented Forecasting overview
```

---

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager (preferably uv)

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd "MR Project/Code"
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -e .
   # Or using uv
   uv sync
   ```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥2.20.0 | Deep learning framework (GRU/LSTM) |
| `arch` | ≥8.0.0 | GARCH model implementation |
| `faiss-cpu` | ≥1.13.1 | Similarity search for retrieval |
| `pandas` | ≥2.3.3 | Data manipulation |
| `numpy` | ≥2.3.5 | Numerical computing |
| `scikit-learn` | ≥1.7.2 | Preprocessing & metrics |
| `statsmodels` | ≥0.14.5 | Statistical tests |
| `yfinance` | ≥0.2.66 | Financial data acquisition |
| `matplotlib` / `seaborn` | — | Visualization |
| `chronos-forecasting` | ≥2.1.0 | Foundation model experiments |

---

## Data

### Primary Dataset

| Attribute | Value |
|-----------|-------|
| **Ticker** | SPY (S&P 500 ETF) |
| **Frequency** | 1-minute intraday |
| **Period** | January 2008 – May 2021 |
| **Source** | `spy_1min_2008_2021_cleaned.csv` |
| **Size** | ~3,300 trading days |

### Train/Test Split

| Set | Period | Percentage |
|-----|--------|------------|
| **Training** | 2008 – Dec 2019 | 90% |
| **Testing** | Jan 2020 – May 2021 | 10% |

> ⚠️ **Note:** The test period includes the COVID-19 market crash (March 2020), providing a stress test scenario but potentially non-representative conditions.

---

## Methodology

### Realized Volatility Construction

The daily realized variance is computed as the sum of **intraday** and **overnight** components:

$$RV_t^{total} = RV_t^{intraday} + RV_t^{overnight}$$

#### Intraday Component (5-minute returns)

$$r_{t,i} = \ln\left(\frac{P_{t,i}}{P_{t,i-1}}\right)$$

$$RV_t^{intraday} = \sum_{i=1}^{N} r_{t,i}^2$$

#### Overnight Component

$$r_t^{overnight} = \ln\left(\frac{P_{t,open}}{P_{t-1,close}}\right)$$

$$RV_t^{overnight} = \left(r_t^{overnight}\right)^2$$

#### Scaling

The variance is scaled to match GARCH percentage-return conventions:

$$\sigma_t = \sqrt{RV_t^{total} \times 10,000}$$

**Design Choices:**

| Decision | Justification |
|----------|---------------|
| 5-minute sampling | Balances noise reduction vs. information loss (Andersen et al., 2001) |
| Overnight inclusion | Captures earnings, macro releases, and global market movements |
| ×10,000 scaling | Standard GARCH convention for percentage returns |

---

### GARCH Benchmark

#### Model Specification

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim t_\nu$$

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Interpretation |
|-----------|----------------|
| $\mu$ | Constant mean return |
| $\omega$ | Long-run variance floor |
| $\alpha$ | Shock impact (ARCH effect) |
| $\beta$ | Persistence (GARCH effect) |
| $\nu$ | Degrees of freedom (Student-t) |

**Implementation:**

```python
from arch import arch_model

model = arch_model(returns, vol="Garch", p=1, q=1, dist="t", mean="constant")
result = model.fit(last_obs=split_date, disp="off")
forecasts = result.forecast(start=split_date, method="analytic")
```

---

### GRU Neural Network

#### Architecture

```
┌─────────────────────────────────────┐
│     Input: (20, 1) - Lookback       │
├─────────────────────────────────────┤
│   GRU(64 units, tanh activation)    │
├─────────────────────────────────────┤
│        Dropout(0.3)                 │
├─────────────────────────────────────┤
│     Dense(32, ReLU activation)      │
├─────────────────────────────────────┤
│   Dense(1, Linear) - Forecast       │
└─────────────────────────────────────┘
```

| Component | Specification |
|-----------|---------------|
| **Lookback** | 20 days (~1 trading month) |
| **Horizon** | 1 day ahead |
| **Features** | Squared daily returns |
| **Optimizer** | Adam |
| **Loss** | Mean Squared Error |
| **Regularization** | Dropout (0.3), Early Stopping (patience=15), LR Reduction |

**Implementation:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input

model = Sequential([
    Input(shape=(20, 1)),
    GRU(units=64, return_sequences=False, activation="tanh"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="linear"),
])
```

---

## Evaluation Framework

### Loss Functions

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MAE** | $\frac{1}{n}\sum\|\hat{\sigma}_t - \sigma_t\|$ | Robust to outliers |
| **MSE** | $\frac{1}{n}\sum(\hat{\sigma}_t - \sigma_t)^2$ | Penalizes large errors |
| **QLIKE** | $\frac{1}{n}\sum\left(\frac{\sigma_t^2}{\hat{\sigma}_t^2} - \ln\frac{\sigma_t^2}{\hat{\sigma}_t^2} - 1\right)$ | Economically motivated |

**QLIKE** is the preferred metric for volatility forecasting because:

- Derived from Gaussian log-likelihood for variance
- Penalizes underestimation more heavily (risk management)
- Robust to noise in realized volatility proxy (Patton, 2011)

### Benchmark Hierarchy

1. **Naive Persistence** — Minimum threshold ($\hat{\sigma}_{t+1} = \sigma_t$)
2. **GARCH(1,1)** — Econometric standard to beat
3. **HAR-RV** — For realized volatility forecasting (uses daily, weekly, monthly lags)

<!-- ### Statistical Significance

For production use, apply the **Diebold-Mariano Test**:

```python
def diebold_mariano_test(actual, pred1, pred2, horizon=1):
    e1 = (actual - pred1)**2
    e2 = (actual - pred2)**2
    d = e1 - e2
    dm_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value
``` -->

---

## Results

### Performance Comparison

| Model | MAE | MSE | QLIKE | Improvement over Naive |
|-------|-----|-----|-------|------------------------|
| Naive Baseline | 0.631 | 1.579 | 0.508 | — |
| **GARCH(1,1)** | **0.505** | **0.998** | **0.317** | 20% MAE, 37% QLIKE |
| GRU | 0.543 | 1.121 | 0.395 | 14% MAE, 22% QLIKE |

### Visual Analysis

Both models:

- Track broad volatility dynamics (spikes and troughs)
- Underestimate extreme spikes (March 2020 COVID crash)
- Capture mean-reversion patterns

### Interpretation

1. **GARCH's efficiency**: With only 4-5 parameters, GARCH captures first-order volatility dynamics parsimoniously
2. **GRU's handicap**: Using only squared returns limits neural network's potential
3. **Regime sensitivity**: COVID-19 period may favor mean-reverting models

---

## Future Directions

### High Priority

| Task | Rationale |
|------|-----------|
| **Diebold-Mariano test** | Establish statistical significance |
| **Rolling-window evaluation** | Assess robustness across regimes |
| **Expand GRU features** | Add VIX, volume, sentiment, cross-asset correlations |

### Medium Priority

- Compare asymmetric variants (GJR-GARCH, EGARCH)
- Implement HAR-RV benchmark
- Hyperparameter optimization for GRU
- Ensemble methods (GARCH + NN)

### Research Extensions

- Retrieval-Augmented Forecasting (RAF) integration
- Longer forecast horizons (5-day, 21-day)
- Multi-asset portfolio volatility
- Value at Risk (VaR) backtesting

---

## Documentation

| Document | Description |
|----------|-------------|
| [Technical Report](../Docs/report/Realized_Volatility_Forecasting_Report.md) | Full research report with methodology details |
| [Evaluation Framework](../Docs/Evaluation.md) | Guide to volatility model evaluation |
| [RAF Overview](../README_RAF.md) | Retrieval-Augmented Forecasting methodology |

---

## Notebook Guide

| Notebook | Purpose |
|----------|---------|
| `DataEngineering.ipynb` | Data preprocessing pipeline |
| `Desc-stats.ipynb` | Exploratory analysis (skewness, kurtosis, Hurst) |
| `ARCH-benchmark.ipynb` | GARCH model implementation |
| `GRU-benchmark.ipynb` | GRU training and evaluation |
| `GRU-GARCH-Benchmark.ipynb` | Head-to-head comparison | You want direct comparison results |
| `SP500_GK-vol-GRU-comp-GARCH.ipynb` | Garman-Klass volatility experiments | You want alternative volatility measures |

---

## Utility Functions

### Feature Engineering (`utils/Features.py`)

```python
from utils.Features import calculate_garman_klass_volatility, create_sequences

# Garman-Klass volatility estimator
gk_vol = calculate_garman_klass_volatility(ohlc_df)

# Create sequences for time series forecasting
X, y = create_sequences(data, lookback=20, horizon=1, feature_cols=["Return"])
```

---

## Acknowledgments

- **Data**: SPY intraday prices sourced via standard financial data providers
- **Libraries**: TensorFlow, arch, scikit-learn, pandas ecosystems
- **Literature**: Andersen et al. (2001), Patton (2011), Hansen & Lunde (2005)

---

<div align="center">

**Built for quantitative finance research** | **S&P 500 Volatility Forecasting** | **Deep Learning vs. Econometrics**

</div>
