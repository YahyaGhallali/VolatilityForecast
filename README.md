# Realized Volatility Forecasting: Hybrid GARCH-GRU vs. Econometric Benchmarks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.X-orange)]()
[![Status](https://img.shields.io/badge/Status-Research%20Complete-green)]()

**A quantitative research framework comparing Deep Learning, Econometric, and Hybrid methodologies for forecasting S&P 500 Realized Volatility.**

</div>

---

## Executive Summary

This project investigates whether flexible, data-driven deep learning models can enhance the predictive power of classical parametric econometric models in financial volatility forecasting.

Using high-frequency **S&P 500 (SPY)** data from 2008 to 2021, we benchmark **GARCH(1,1)** against **GRU (Gated Recurrent Units)** networks. The study culminates in a **Hybrid GARCH-GRU** model that processes GARCH residuals using deep learning, achieving a **~14% reduction in MAE** and a **~21% reduction in QLIKE** loss compared to the standard econometric benchmark.

> **[Read the Principal Research Report](Docs/Report/Main.pdf)** for the complete theoretical framework and detailed analysis.

## Key Findings

Our empirical results demonstrate that while pure Deep Learning models struggle to outperform GARCH due to the complexity of volatility clustering, a Hybrid approach yields superior results by combining the strengths of both:

| Model | MAE | MSE | QLIKE | Improvement (MAE) |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline** | 0.632 | 1.583 | 0.506 | — |
| **GARCH(1,1)** (Student-t) | 0.507 | 1.001 | 0.318 | +19.8% vs Naive |
| **GRU** (Pure) | 0.543 | 1.121 | 0.395 | +14.1% vs Naive |
| **Hybrid GARCH-GRU** | **0.436** | **0.833** | **0.249** | **+14.0% vs GARCH** |

*Results based on Out-of-Sample testing (Jan 2020 – May 2021).*

## Project Structure

The repository is organized to separate production-candidate models from experimental benchmarks.

```plaintext
MR Project/
├── Code/
│   ├── Hybrid/
│   │   └── garch-gru-bench.ipynb      # Hybrid GARCH-GRU implementation (Best Model)
│   │
│   ├── Tests/                         # Experimental Benchmarks
│   │   ├── ARCH-benchmark.ipynb       # Econometric GARCH(1,1) baseline
│   │   ├── GRU-benchmark.ipynb        # Pure Deep Learning GRU baseline
│   │   ├── DataEngineering.ipynb      # High-frequency data cleaning pipeline
│   │   ├── Desc-stats.ipynb           # Statistical analysis (ADF, Hurst, ACF/PACF)
│   │   └── ...
│   │
│   ├── Data/                          # Dataset artifacts
│   └── utils/
│       └── Features.py                # Volatility calculation & sequence generation
│
└── Docs/                              # Research documentation & Illustrations
```

## Methodology

### 1. Data Engineering

We construct a robust target variable, **Daily Realized Volatility**, by aggregating high-frequency returns:

* **Intraday:** Sum of squared 5-minute returns.
* **Overnight:** Squared return between previous close and current open.
* **Validation:** Data is split chronologically (Train: 2008-2019, Test: 2020-2021) to respect time-series causality.

### 2. Modeling Approaches

#### A. Econometric Benchmark (ARCH-benchmark.ipynb)

Standard **GARCH(1,1)** with Student-t innovations. This model captures volatility clustering and heavy tails but is limited by its linear parametric assumptions.

#### B. Deep Learning Benchmark (GRU-benchmark.ipynb)

A **Gated Recurrent Unit (GRU)** network taking raw squared returns as input. While flexible, the pure GRU struggles to learn the "level" of volatility as effectively as GARCH during extreme shifts (e.g., COVID-19 crash).

#### C. Hybrid Approach (garch-gru-bench.ipynb)

The proposed solution uses a residual learning strategy:

1. **Fit GARCH(1,1)** to capture the linear volatility component.
2. **Extract Residuals:** Calculate forecast errors ($e_t = \sigma_{realized} - \sigma_{GARCH}$).
3. **Train GRU:** The network predicts the *nonlinear* residual component using past residuals and returns.
4. **Ensemble:** Final Forecast = $\sigma_{GARCH} + \hat{e}_{GRU}$.

## Installation & Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mr-project.svg
   cd "MR Project/Code"
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

   *Key libraries: `tensorflow`, `arch`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`.*

3. **Running the Models:**
   * To reproduce the **Winning Result**, open garch-gru-bench.ipynb.
   * To view the statistical properties of the data, open Desc-stats.ipynb.

## Future Directions

* **Multivariate Analysis:** Incorporating VIX and volume data into the GRU component.
* **Statistical Significance:** Conducting Diebold-Mariano tests to formally assess the Hybrid model's superiority.
* **Retrieval Augmented Forecasting (RAF):** Early experiments (documented in README_RAF.md) suggest identifying similar historical volatility regimes using Vector Databases (FAISS) can further improve forecasts.

---
