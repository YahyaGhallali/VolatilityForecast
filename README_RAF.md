# Retrieval-Augmented Forecasting (RAF) for Financial Volatility

## Overview

This project explores the application of **Retrieval-Augmented Forecasting (RAF)** to financial time series, specifically focusing on forecasting the volatility of the **S&P 500** index. By integrating traditional parametric models (like LSTM and ARCH) with non-parametric retrieval mechanisms (using FAISS), this project aims to address common challenges in financial forecasting such as non-stationarity and the prediction of rare, high-impact events ("black swans").

## Key Features

* **Retrieval-Augmented Forecasting (RAF):** Implements a retrieval mechanism to fetch historically similar market conditions to inform current predictions.
* **Hybrid Modeling:** Combines Deep Learning (LSTM, Attention mechanisms) with statistical methods (ARCH/GARCH).
* **Vector Database Integration:** Uses **FAISS** for efficient similarity search of high-dimensional time series patches.
* **Comparative Analysis:** Benchmarks RAF approaches against standard parametric baselines.
* **Data Engineering:** Robust pipelines for fetching, cleaning, and transforming financial data (log returns, volatility).

## Project Structure

The project is organized as follows:

```
MR Project/
├── Code/                   # Source code and configuration
│   ├── Data/               # Data artifacts
│   │   ├── dataset/        # Raw and processed CSV datasets
│   │   ├── FAISS/          # FAISS vector indices for retrieval
│   │   ├── model/          # Saved Keras/TensorFlow models
│   │   └── scaler/         # Data scalers (MinMax, Standard, etc.)
│   ├── Tests/              # Jupyter Notebooks for experimentation
│   │   ├── ARCH.ipynb      # Statistical volatility modeling
│   │   ├── RAF.ipynb       # Core Retrieval-Augmented Forecasting implementation
│   │   ├── RNN.ipynb       # RNN/LSTM model training and evaluation
│   │   ├── Attention.ipynb # Attention mechanism experiments
│   │   └── ...             # Other experiments (Data Engineering, Windowing)
│   └── pyproject.toml      # Python dependencies and project config
├── Docs/                   # Documentation and theoretical background
│   ├── Retrieval...md      # In-depth explanation of RAF theory
│   └── ...
└── README.md               # This file
```

## Installation

### Prerequisites

* Python 3.12 or higher
* pip (or a compatible package manager)

### Setup

1. Clone the repository.
2. Navigate to the `Code` directory.
3. Install the dependencies defined in `pyproject.toml`.

```bash
cd Code
pip install -r requirements.txt # If a requirements file exists
# OR install directly from pyproject.toml if using a tool that supports it
pip install .
```

**Key Dependencies:**

* `tensorflow`: Deep learning framework.
* `faiss-cpu`: Efficient similarity search.
* `arch`: ARCH/GARCH models.
* `chronos-forecasting`: Time series forecasting foundation models.
* `yfinance`: Financial data acquisition.
* `pandas`, `numpy`, `scikit-learn`: Data manipulation and analysis.

## Methodology

This project implements the **RAF paradigm**, which decouples "memory" (historical patterns) from "computation" (inference). Instead of relying solely on a model's weights to memorize history, the system queries a Knowledge Base of historical time series patches.

1. **Patching:** Historical data is segmented into sliding windows (Key: Lookback, Value: Forecast).
2. **Retrieval:** For a given input window, the system retrieves the most similar historical windows using metrics like Pearson Correlation (to capture shape/trend) or Euclidean distance.
3. **Augmentation:** These retrieved contexts are fed into the forecasting model to improve accuracy and adaptability to new market regimes.

## Usage

The core logic is contained within the Jupyter Notebooks in the `Code/Tests/` directory.

* Start with `DataEngineering.ipynb` to understand data preparation.
* Explore `RAF.ipynb` to see the retrieval mechanism in action.
* Check `model.ipynb` or `RNN.ipynb` for the baseline deep learning models.

## Data

The project uses historical S&P 500 data, processed to calculate log returns and volatility. The data is stored in `Code/Data/dataset/`.

---
*Created for the MR Project on Financial Volatility Forecasting.*
