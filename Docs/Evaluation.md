Evaluating a volatility model is trickier than evaluating a price prediction model because **"true" volatility is latent**—you cannot observe it directly in the market. You only see returns, and from those, you *estimate* volatility.

To measure error effectively and prove your model beats State-of-the-Art (SOTA), you must follow a rigorous 4-step framework used in academic quantitative finance (e.g., by researchers like Engle, Patton, or Hansen).

---

### **Step 1: Define the "Ground Truth" (The Proxy)**

Before calculating error, you must decide what "Actual Volatility" is. If your target is noisy, your error metric is meaningless.

* **The Amateur Mistake:** Using squared daily returns () as the target. This is unbiased but extremely noisy.
* **The Standard Approach:** Using **Realized Volatility (RV)** based on high-frequency data (e.g., sum of 5-minute squared returns).
* **Your Constraint (Daily Data):** Since you likely use daily data (OHLC), your "Ground Truth" should be the **Garman-Klass** or **Parkinson** estimator, or a **21-day Rolling Standard Deviation** (though the latter introduces lag).

**Key Rule:** Consistent comparison requires that *all* models (yours and the benchmarks) are evaluated against the *same* high-quality proxy.

---

### **Step 2: Choose the Right Loss Functions**

Standard MSE is often insufficient for volatility because the data is non-negative and has fat tails.

#### **1. MSE (Mean Squared Error) & RMSE**

* **Formula:**
* **Pros:** Standard, easy to interpret.
* **Cons:** Very sensitive to outliers. A single market crash day can dominate your error score.

#### **2. QLIKE (Quasi-Likelihood Loss) – *The Pro Choice***

This is widely preferred in econometrics (e.g., Patton, 2011) because it is **robust to noise in the volatility proxy**. It penalizes the model heavily if it predicts low volatility when reality is high (the most dangerous risk management error).

* **Formula:**

* **Why use it?** If your "Actual Volatility" is an estimate (which it always is), QLIKE is mathematically proven to identify the "true" best model more often than MAE.

---

### **Step 3: Compare Against "SOTA" Benchmarks**

You cannot claim your LSTM/GRU is "good" unless it beats these three specific opponents. If you don't beat #2, your model is useless.

#### **Benchmark 1: The "Naive" Baseline (Random Walk)**

* **Logic:** Forecast that tomorrow's volatility will be exactly the same as today's.
*

* **Test:** If your LSTM cannot beat this, it is just adding noise.

#### **Benchmark 2: The Econometric Standard (GARCH-1,1)**

* **Logic:** The industry standard for decades. It captures mean reversion and clustering.
* **Test:** Fit a standard GARCH(1,1) on the same data. Your Deep Learning model justifies its complexity *only* if it beats GARCH significantly.

#### **Benchmark 3: The "Realized" King (HAR-RV)**

* **Logic:** If you are forecasting Realized Volatility, the **HAR (Heterogeneous Autoregressive)** model is the one to beat, not GARCH. It uses daily, weekly, and monthly lags of volatility.
*

* **Status:** In many academic papers, simple HAR models still outperform complex LSTMs.

---

### **Step 4: Statistical Significance (The Diebold-Mariano Test)**

You cannot just say "My MSE is 0.04 and GARCH is 0.05, so I win." The difference might be luck.

You must use the **Diebold-Mariano (DM) Test**.

* **Hypothesis ():** The forecasts of Model A and Model B have equal accuracy.
* **Result:** A p-value.
* If , you reject the null. Your model is *statistically significantly* better.
* If , the improvement is just noise.

### **Step 5: Economic Evaluation (Value at Risk)**

Business stakeholders (traders/risk managers) don't care about MSE. They care about money. Validate your model using **Value at Risk (VaR)** backtesting.

* **Calculate VaR:**
* **The Kupiec Test:** Count how many times the actual price drop exceeded your predicted VaR.
* For 95% VaR, it should happen exactly 5% of the time.
* If it happens 8% of the time, your model underestimates risk (Bad).
* If it happens 2% of the time, your model is too conservative (Inefficient capital use).

### **Python Implementation for Evaluation**

Here is a snippet to calculate QLIKE and run the DM test.

```python
import numpy as np
import pandas as pd
from scipy import stats

def qlike_loss(y_true, y_pred):
    """
    Calculate QLIKE loss.
    y_true: Realized Variance (sigma^2)
    y_pred: Predicted Variance (sigma^2)
    """
    # Ensure no division by zero or log of zero
    epsilon = 1e-6
    ratio = y_true / (y_pred + epsilon)
    loss = ratio - np.log(ratio) - 1
    return np.mean(loss)

def diebold_mariano_test(actual, pred1, pred2, horizon=1):
    """
    Compare predictive accuracy of two models.
    actual: Ground truth series
    pred1: Forecasts from Model A (e.g., Your LSTM)
    pred2: Forecasts from Model B (e.g., GARCH Benchmark)
    """
    e1 = (actual - pred1)**2  # Squared error for Model A
    e2 = (actual - pred2)**2  # Squared error for Model B
    d = e1 - e2               # Loss differential
    
    mean_d = np.mean(d)
    n = len(d)
    autocov = np.mean((d[:n-horizon] - mean_d) * (d[horizon:] - mean_d))
    var_d = (np.var(d) + 2*autocov) / n # Simplified variance for h=1
    
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat))) # Two-tailed test
    
    return dm_stat, p_value

# Usage Example:
# model_rmse = np.sqrt(mean_squared_error(df['rv'], df['lstm_pred']))
# garch_rmse = np.sqrt(mean_squared_error(df['rv'], df['garch_pred']))
# model_qlike = qlike_loss(df['rv']**2, df['lstm_pred']**2)

# dm_stat, p_val = diebold_mariano_test(df['rv'], df['lstm_pred'], df['garch_pred'])

# print(f"DM Statistic: {dm_stat:.4f}, P-Value: {p_val:.4f}")
# if p_val < 0.05:
#     print("Difference is statistically significant!")

```

### **Summary Checklist**

1. **Metric:** Use **RMSE** for general error and **QLIKE** for robustness.
2. **Benchmark:** Compare against **GARCH(1,1)** and **Random Walk**.
3. **Validation:** Run the **Diebold-Mariano Test** to prove the win isn't luck.
4. **Application:** Run a **VaR Backtest** (Kupiec Test) to prove financial utility.
