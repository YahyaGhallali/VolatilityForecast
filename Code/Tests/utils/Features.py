import pandas as pd
import numpy as np


def calculate_garman_klass_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    Parameters:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        window (int): Rolling window size for volatility calculation.
    Returns:
        pd.Series: Garman-Klass volatility estimates.
    """
    # Term 1: Intraday Range (High vs Low)
    log_hl = (np.log(df["High"] / df["Low"])) ** 2

    # Term 2: Trend / Overnight (Close vs Open)
    log_co = (np.log(df["Close"] / df["Open"])) ** 2

    # GK Formula for Daily Variance
    gk_variance = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk_variance)


def create_sequences(
    data: pd.DataFrame,
    lookback: int,
    horizon: int,
    feature_cols: list[str] = ["Vol_Diff"],
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences of data for time series forecasting.

    Args:
        data (pd.DataFrame): The input data containing features.
        lookback (int): The number of past timesteps to include in each input sequence.
        horizon (int): The number of future timesteps to predict.

    Returns:
        Tuple[np.ndarray]: The input sequences (X) and the corresponding targets (y).
    """
    X, y = [], []

    data_values = data[feature_cols].values

    for i in range(len(data) - lookback - horizon):
        window = data_values[i : i + lookback]
        future_window = data_values[i + lookback : i + lookback + horizon]

        X.append(window)
        y.append(future_window)

    return np.array(X), np.array(y)
