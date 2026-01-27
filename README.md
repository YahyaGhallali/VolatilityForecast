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
| **Hybrid** | Ensemble | GARCH Residuals modeled by GRU to capture non-linearities |
| **Naive Persistence** | Baseline | Random walk benchmark ($\hat{\sigma}_{t+1} = \sigma_t$) |

The study uses **high-frequency intraday data** (1-minute SPY prices from 2008-2021) to construct realized volatility from 5-minute returns, ensuring a high-quality target variable for model evaluation.

---

## Key Findings

### Primary Results

| Metric | GARCH(1,1) | GRU | Hybrid | Naive | Winner |
|--------|------------|-----|--------|-------|--------|
| **MAE** | 0.507 | 0.543 | **0.436** | 0.632 | Hybrid (+14%) |
| **MSE** | 1.001 | 1.121 | **0.833** | 1.583 | Hybrid (+17%) |
| **QLIKE** | 0.318 | 0.395 | **0.249** | 0.506 | Hybrid (+21%) |

### Key Insights

1. **Hybrid Model Dominates** — The Hybrid Approach (GARCH + GRU Residuals) outperforms both the pure GARCH(1,1) and pure GRU models across all metrics. It effectively captures the non-linear structure in the residuals that GARCH misses.

2. **Significant Improvement in QLIKE** — The Hybrid model achieves a 21% reduction in QLIKE compared to GARCH, indicating superior performance in tail risk assessment and handling extreme volatility events.

3. **Econometric vs Deep Learning** — While pure GARCH beats pure GRU, combining them yields the best results. The Deep Learning component adds value by correcting the rigid parametric assumptions of GARCH.

4. **COVID-19 Stress Test** — The test period includes the extreme March 2020 volatility. The Hybrid model's ability to adapt to these residuals suggests it is more robust to regime shifts.

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
│   ├── Hybrid/
│   │   └── Main.ipynb         # Hybrid GARCH+GRU model implementation
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

### Hybrid GARCH-GRU Model

#### Methodology

The Hybrid model combines the strengths of both approaches by using GARCH(1,1) to model the linear volatility component and a GRU network to model the residuals (errors).

1. **Stage 1 (Econometric)**: Fit GARCH(1,1) to returns and generate volatility forecasts $\sigma_{GARCH}$.
2. **Stage 2 (Residual Calculation)**: Compute standardized residuals or forecast errors ($e_t = \sigma_{realized} - \sigma_{GARCH}$).
3. **Stage 3 (Deep Learning)**: Train GRU to predict these residuals $\hat{e}_{t+1}$ using past data.
4. **Stage 4 (Ensemble)**: Combine forecasts: $\sigma_{final} = \sigma_{GARCH} + \hat{e}_{t+1}$.

This approach allows the neural network to focus solely on the *structure that GARCH misses*, rather than learning the entire volatility dynamic from scratch.

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
| Naive Baseline | 0.632 | 1.583 | 0.506 | — |
| GARCH(1,1) | 0.507 | 1.001 | 0.318 | 20% MAE |
| GRU (Pure) | 0.543 | 1.121 | 0.395 | 14% MAE |
| **Hybrid (GARCH+GRU)** | **0.436** | **0.833** | **0.249** | **31% MAE, 51% QLIKE** |

### Visual Analysis

Both models:

- Track broad volatility dynamics (spikes and troughs)
- Underestimate extreme spikes (March 2020 COVID crash)
- Capture mean-reversion patterns

### Interpretation

1. **Hybrid Efficiency**: The Hybrid model's success confirms that GARCH explains most of the variance, but systematic errors remain. The GRU component successfully learns to predict these residuals, effectively "boosting" the GARCH forecast.
2. **GARCH's Role**: Acts as a robust baseline, capturing the heavy-tailed distribution (Student-t) better than a pure neural network with MSE loss.
3. **Regime sensitivity**: The large improvement suggests that during high-volatility regimes (COVID-19), the relationship between returns and volatility becomes non-linear in ways simple GARCH cannot capture.

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
| `Hybrid/Main.ipynb` | **Hybrid GARCH+GRU model** (Best Performer) |
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
