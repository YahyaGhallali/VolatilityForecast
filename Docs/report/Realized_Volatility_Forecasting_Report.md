# Realized Volatility Forecasting on the S&P 500: Deep Learning vs. Econometric Benchmarks

**Technical Research Report**

## EXECUTIVE SUMMARY

### Problem Statement

This report investigates whether deep learning architectures (specifically Gated Recurrent Units) can outperform classical econometric models (GARCH) in forecasting one-day-ahead realized volatility of the S&P 500 index. The research addresses a fundamental question in quantitative finance: **can flexible, data-driven models capture volatility dynamics that parametric specifications miss, or do the additional degrees of freedom lead to overfitting without predictive gain?**

### Key Findings

1. **GARCH(1,1) with Student-t innovations achieves superior forecasting performance** on the out-of-sample test period (January 2020 – May 2021), with MAE of 0.505 versus GRU's 0.543, and QLIKE of 0.317 versus 0.395.

2. **Both models significantly outperform the naive persistence baseline** (MAE ~0.63), confirming that volatility exhibits exploitable autocorrelation structure.

3. **The GRU model, despite its flexibility, does not demonstrate statistically meaningful improvement** over GARCH under the current experimental design—using squared daily returns as the sole input feature.

4. **The test period (2020–2021) includes extreme market stress** (COVID-19 crash), providing a challenging but potentially non-representative evaluation regime.

5. **Realized volatility constructed from 5-minute intraday returns plus overnight gaps** provides a consistent, high-frequency target variable aligned with GARCH scaling conventions.

### Limitations

- The GRU model uses only squared returns as input; additional features (e.g., VIX, volume, sentiment) were not incorporated.
- No formal statistical test (e.g., Diebold-Mariano) was conducted to assess significance of forecast differences.
- Train/test split is temporal but fixed; no rolling or expanding window evaluation was performed.
- The test period is dominated by the COVID-19 market regime, which may inflate error metrics for all models.

### Technical Summary for Leadership

The econometric baseline (GARCH) remains the preferred model for realized volatility forecasting under the current experimental configuration. The GRU architecture, while successfully trained and producing reasonable forecasts, does not justify its additional complexity given the marginal (and negative) difference in predictive accuracy. Before production deployment, the deep learning approach requires: (i) expanded feature sets, (ii) hyperparameter optimization, and (iii) rolling-window robustness testing across multiple market regimes.

---

## PART I: FOUNDATIONS

### 1.1 Research Question

**Primary Question:**  
Does a GRU-based neural network provide superior one-step-ahead forecasts of daily realized volatility compared to a GARCH(1,1) benchmark, as measured by MAE, MSE, and QLIKE loss functions?

**Secondary Questions:**

- How sensitive are the results to the choice of train/test split and market regime?
- Does the inclusion of overnight returns in realized volatility construction affect model comparability?

**Falsifiability Criterion:**  
The GRU model is considered superior if it achieves lower QLIKE (the economically motivated loss function for variance forecasting) on out-of-sample data with statistical significance.

### 1.2 Data Provenance and Integrity

#### 1.2.1 Data Source

| Attribute | Value |
|-----------|-------|
| **Ticker** | SPY (S&P 500 ETF) |
| **Frequency** | 1-minute intraday |
| **Period** | 2008-01-01 to 2021-05-31 |
| **Source File** | `spy_1min_2008_2021_cleaned.csv` |
| **Location** | `Code/Data/dataset/` |

**Evidence:** From `GRU-benchmark.ipynb` (Cell 3):

```python
data = pd.read_csv("../Data/dataset/spy_1min_2008_2021_cleaned.csv", 
                   index_col=0, parse_dates=True)
```

#### 1.2.2 Data Cleaning Status

The filename suffix `_cleaned` indicates preprocessing was applied prior to analysis. However, **the specific cleaning procedures (handling of missing values, outlier treatment, corporate action adjustments) are not documented in the provided notebooks.**

> *This information is not available in the provided materials.*

**Recommendation:** A dedicated data provenance notebook documenting all preprocessing steps should be created for audit purposes.

#### 1.2.3 Integrity Verification

The notebooks do not include explicit integrity checks such as:

- Row count validation
- Date continuity checks
- Price reasonableness bounds
- Volume anomaly detection

> *Formal data integrity verification is not documented in the provided notebooks.*

### 1.3 Realized Volatility Construction

#### 1.3.1 Mathematical Definition

The daily realized variance is constructed as the sum of intraday and overnight components:

$$RV_t^{total} = RV_t^{intraday} + RV_t^{overnight}$$

**Intraday Component:**

5-minute log returns are computed as:
$$r_{t,i} = \ln\left(\frac{P_{t,i}}{P_{t,i-1}}\right)$$

The intraday realized variance is the sum of squared returns:
$$RV_t^{intraday} = \sum_{i=1}^{N} r_{t,i}^2$$

where $N$ is the number of 5-minute intervals in trading day $t$.

**Overnight Component:**

The overnight return captures the gap between previous close and current open:
$$r_t^{overnight} = \ln\left(\frac{P_{t,open}}{P_{t-1,close}}\right)$$

$$RV_t^{overnight} = \left(r_t^{overnight}\right)^2$$

**Scaling Convention:**

The variance is scaled by $10,000$ (equivalently $100^2$) to match GARCH percentage-return conventions:
$$\sigma_t = \sqrt{RV_t^{total} \times 10,000}$$

**Evidence:** From `GRU-benchmark.ipynb` (Cell 5) and `ARCH-benchmark.ipynb` (Cell 4):

```python
# Intraday Variance
close_5mn = data.close.resample("5min").last().dropna()
rt_intraday = np.log(close_5mn / close_5mn.shift(1)).dropna()
rv_intraday = (rt_intraday ** 2).groupby(rt_intraday.index.date).sum()

# Overnight Variance
daily_ohlc = data.close.resample("1D").agg(['first', 'last']).dropna()
daily_ohlc["prev_close"] = daily_ohlc['last'].shift(1)
rt_overnight = np.log(daily_ohlc['first'] / daily_ohlc['prev_close']).dropna()
rv_overnight = rt_overnight ** 2

# Total
rv_total['RV_Daily_Var'] = rv_total['intraday'] + rv_total['overnight']
target_variance = rv_total['RV_Daily_Var'] * 10000
```

#### 1.3.2 Methodological Justifications

| Decision | Justification |
|----------|---------------|
| **5-minute sampling** | Balances noise reduction against information loss. Literature (Andersen et al., 2001) suggests 5-minute intervals minimize microstructure noise while preserving volatility signal. Higher frequencies introduce bid-ask bounce; lower frequencies lose intraday dynamics. |
| **Overnight inclusion** | Overnight returns capture earnings announcements, macroeconomic releases, and global market movements. Excluding them would underestimate true daily variance, particularly during periods of overnight news flow. |
| **$\times 10,000$ scaling** | Standard GARCH implementations model percentage returns (multiplied by 100). Squaring yields variance in units of $100^2 = 10,000$. This ensures direct comparability between realized variance and GARCH conditional variance forecasts. |
| **Sum of squared returns** | Non-parametric, model-free estimator of integrated variance under standard assumptions. Does not impose distributional assumptions on returns. |

#### 1.3.3 Known Limitations

1. **Microstructure noise:** Even at 5-minute frequency, bid-ask bounce and non-synchronous trading can introduce bias. Realized kernels or two-scale estimators were not employed.

2. **Overnight gap treatment:** Treating overnight variance as a single squared return assumes log-normality and ignores potential intraday dynamics during extended-hours trading.

3. **No jump adjustment:** The realized variance measure does not separate continuous and jump components. High-volatility days may be dominated by jump contributions.

---

## PART II: EXPLORATORY ANALYSIS

### 2.1 Return Distribution Properties

#### 2.1.1 Stylized Facts Verification

**Evidence:** From `Desc-stats.ipynb`, the following stylized facts of financial returns are examined:

| Stylized Fact | Expected | Observed | Implication |
|---------------|----------|----------|-------------|
| **Negative Skewness** | Skew < 0 | Confirmed (visual inspection) | Left tail heavier; crash risk present |
| **Excess Kurtosis** | Kurt > 0 | Confirmed (leptokurtic) | Fat tails; extreme events more frequent than Gaussian |
| **Volatility Clustering** | Autocorrelation in $r_t^2$ | Confirmed | Justifies GARCH-type and memory-based models |

**From `Desc-stats.ipynb` (Cell 8):**

```python
skew_val = data['Log_Return'].skew()
kurt_val = data['Log_Return'].kurtosis()
# Results: Negative skewness, positive excess kurtosis
```

#### 2.1.2 Distributional Analysis

The histogram overlay with a fitted normal distribution demonstrates significant departure from Gaussianity:

1. **Peak is sharper** (leptokurtosis)
2. **Tails are heavier** (more extreme observations)
3. **Left tail extends further** (negative skewness)

**Implication for Modeling:**

- Gaussian-innovation GARCH models will underestimate tail risk
- Student-t innovations (as used) partially address this
- Neural networks make no distributional assumptions on residuals—potentially advantageous

#### 2.1.3 Time-Varying Higher Moments

Rolling skewness and kurtosis analysis (`Desc-stats.ipynb`, Cell 8) reveals:

- **Skewness becomes deeply negative during market stress** (e.g., March 2020)
- **Kurtosis spikes during regime transitions**

These findings support the use of time-varying volatility models but also highlight that higher moments (skewness, kurtosis) are non-stationary—a limitation for both GARCH and GRU approaches that model only the second moment.

### 2.2 Autocorrelation Structure

#### 2.2.1 Log Returns

ACF/PACF of raw log returns (not shown explicitly in notebooks) typically exhibit:

- Near-zero autocorrelation (market efficiency)
- Returns are approximately unpredictable

#### 2.2.2 Squared Returns (Volatility Proxy)

The presence of significant autocorrelation in squared returns is the fundamental justification for volatility forecasting:

> **These findings validate the use of GARCH-type models** (which explicitly model conditional variance as autoregressive) **and recurrent neural networks** (which learn temporal dependencies from historical sequences).

### 2.3 Hurst Exponent Analysis

**Evidence:** From `Desc-stats.ipynb` (Cell 7):

| Series Type | Hurst Exponent | Interpretation |
|-------------|----------------|----------------|
| Mean-reverting | H < 0.5 | Price pulls back to mean |
| Random walk | H ≈ 0.5 | Unpredictable |
| Trending | H > 0.5 | Persistence in direction |

The Hurst exponent applied to S&P 500 prices provides insight into regime characteristics. Values near 0.5 suggest limited long-term predictability in price levels, consistent with weak-form efficiency.

**Note:** Hurst analysis on prices differs from analysis on volatility. Volatility series typically exhibit H > 0.5, indicating persistence—which is captured by GARCH's $\alpha + \beta$ parameter sum approaching 1.

### 2.4 Summary: Exploratory Findings and Modeling Implications

| Finding | Modeling Implication |
|---------|---------------------|
| Fat tails | Use Student-t innovations in GARCH; NN implicitly flexible |
| Negative skewness | Consider asymmetric models (GJR-GARCH); NN may capture if given signed returns |
| Volatility clustering | Both GARCH and GRU appropriate |
| Non-stationary higher moments | Fixed-parameter models may struggle during regime shifts |

---

## PART III: MODELING

### 3.1 Baseline: Naive Persistence Model

**Definition:**  
The naive forecast assumes tomorrow's volatility equals today's:
$$\hat{\sigma}_{t+1} = \sigma_t$$

**Justification:**  
This provides a minimal benchmark. Any useful model must outperform persistence, given the strong autocorrelation in volatility.

**Evidence:** Both `GRU-benchmark.ipynb` and `ARCH-benchmark.ipynb` compute naive baseline metrics for comparison.

### 3.2 GARCH(1,1) Benchmark

#### 3.2.1 Model Specification

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim t_\nu$$

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Interpretation |
|-----------|----------------|
| $\mu$ | Constant mean return |
| $\omega$ | Long-run variance floor |
| $\alpha$ | Shock impact (ARCH effect) |
| $\beta$ | Persistence (GARCH effect) |
| $\nu$ | Degrees of freedom (Student-t) |

**Evidence:** From `ARCH-benchmark.ipynb` (Cell 7):

```python
model = arch_model(r_daily, vol="Garch", p=1, q=1, dist="t", mean="constant")
res = model.fit(last_obs=split_date, disp="off")
```

#### 3.2.2 Design Decisions and Justifications

| Decision | Justification |
|----------|---------------|
| **GARCH(1,1)** | Parsimonious specification that captures first-order dynamics. Higher-order terms (p,q > 1) rarely improve out-of-sample performance and risk overfitting. The (1,1) specification is the workhorse of volatility modeling. |
| **Student-t innovations** | Addresses fat tails documented in exploratory analysis. Gaussian innovations systematically underestimate tail risk. The t-distribution adds one parameter (degrees of freedom) for substantial fit improvement. |
| **Constant mean** | Daily returns have near-zero expected value. Complex mean models (ARMA) add parameters without forecasting gain for volatility. Constant mean is standard practice. |
| **Input: Log returns × 100** | Standard scaling convention for GARCH. Ensures variance parameters are interpretable and comparable to literature values. |

#### 3.2.3 Model Assumptions

1. **Stationarity:** Requires $\alpha + \beta < 1$ for covariance stationarity
2. **No leverage effect:** Symmetric response to positive/negative shocks (GJR-GARCH would relax this)
3. **Parameter constancy:** Assumes structural stability over the estimation period
4. **No jumps:** Continuous diffusion assumption in conditional variance

#### 3.2.4 What GARCH Cannot Capture

- **Asymmetric volatility response** (leverage effect) without extension to GJR/EGARCH
- **Regime changes** without Markov-switching extension
- **Exogenous drivers** (VIX, volume, sentiment) without X-GARCH
- **Long memory** in volatility without FIGARCH

### 3.3 GRU Neural Network

#### 3.3.1 Model Architecture

**Evidence:** From `GRU-benchmark.ipynb` (Cell 14):

```python
model = Sequential([
    Input(shape=INPUT_SHAPE),
    GRU(units=64, return_sequences=False, activation="tanh", 
        recurrent_dropout=0.0),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(OUTPUT_SIZE, activation="linear"),
])
```

| Component | Specification | Justification |
|-----------|--------------|---------------|
| **Input** | (20, 1) | 20-day lookback, 1 feature (squared return) |
| **GRU Layer** | 64 units, tanh | Gated recurrent unit captures temporal dependencies; 64 units balances capacity vs. overfitting |
| **Dropout** | 0.3 | Regularization to prevent overfitting on training set |
| **Dense** | 32 units, ReLU | Non-linear transformation before output |
| **Output** | 1 unit, linear | Single-step variance forecast |

#### 3.3.2 Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Optimizer** | Adam | Adaptive learning rate; robust default for RNNs |
| **Loss** | MSE | Standard regression loss; directly optimizes squared error |
| **Batch size** | 32 | Balance between gradient noise and computation |
| **Epochs** | 100 (early stopping) | Train until convergence; early stopping prevents overfitting |
| **Patience** | 15 epochs | Sufficient to escape local plateaus |
| **Learning rate reduction** | Factor 0.5, patience 5 | Adaptive refinement when progress stalls |

**Evidence:** From `GRU-benchmark.ipynb` (Cells 15-16):

```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]
```

#### 3.3.3 Data Preprocessing

**Input Scaling:** StandardScaler applied to both features and targets:

```python
X_scaler = StandardScaler()
y_scaler = StandardScaler()
```

**Justification:** Neural networks are sensitive to input scale. Standardization (zero mean, unit variance) ensures stable gradients and comparable feature contributions.

**Inverse Transform:** Applied to predictions for evaluation in original units.

#### 3.3.4 Sequence Construction

**Evidence:** From `GRU-benchmark.ipynb` (Cell 11):

```python
HORIZON = 1
LOOKBACK = 20
feature_cols = ["Return",]  # Squared daily returns
predict_cols = ["Realized_Variance",]
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Lookback** | 20 days | Approximately one trading month; captures medium-term volatility dynamics without excessive memory burden |
| **Horizon** | 1 day | One-step-ahead forecast; directly comparable to GARCH |
| **Features** | Squared return | Minimal feature set; isolates learning capacity from feature engineering |

#### 3.3.5 Train/Test Split

**Evidence:** From `GRU-benchmark.ipynb`:

```python
ratio = .9
n_sep = int(ratio * X_raw.shape[0])
```

**Protocol:**

- 90% training / 10% test
- **No shuffling** (critical for time series)
- Temporal ordering preserved

**Justification:**  
Shuffling would introduce future information into training (look-ahead bias). The strict temporal split ensures valid out-of-sample evaluation.

#### 3.3.6 Model Assumptions

1. **Stationarity:** Input features assumed stationary (squared returns are stationary under typical conditions)
2. **Temporal structure:** Sequential dependencies captured by recurrent architecture
3. **Smooth target function:** The mapping from input sequences to variance is learnable

#### 3.3.7 What GRU Can and Cannot Capture

**Potential Advantages:**

- Non-linear relationships
- Complex temporal patterns
- Implicit handling of non-Gaussian distributions
- No parametric assumptions on volatility dynamics

**Limitations:**

- Requires substantial data for stable training
- Black-box nature reduces interpretability
- Hyperparameter sensitivity
- Current implementation uses only squared returns—no exogenous features

### 3.4 Feature Engineering

**Current Implementation:**

From `GRU-benchmark.ipynb` (Cell 10):

```python
rv["Return"] = r[rv.index].values ** 2 * 100
rv["Diff_Vol"] = diff_volume[rv.index].values
rv["Neg_Ret"] = neg_ret[rv.index].values
rv["Log_Var"] = np.log(target_variance[rv.index].values)
```

While multiple features were computed, **only squared returns were used as GRU input** in the evaluation:

```python
feature_cols = ["Return",]
```

**Implication:**  
The GRU model is handicapped by minimal feature input. This design choice isolates the comparison to learning capacity on the volatility proxy signal alone, but forfeits potential gains from richer feature sets.

**Features computed but not used:**

- Volume changes (`Diff_Vol`)
- Negative returns (`Neg_Ret`) — could capture asymmetry
- Log variance (`Log_Var`) — transformation for normality

> *The decision to use only squared returns as input is not explicitly justified in the notebooks.*

---

## PART IV: EVALUATION

### 4.1 Evaluation Protocol

#### 4.1.1 Out-of-Sample Design

| Aspect | Specification |
|--------|---------------|
| **Split ratio** | 90% train / 10% test |
| **Split type** | Temporal (no shuffling) |
| **Test period** | Approximately January 2020 – May 2021 |
| **Forecast horizon** | 1 day ahead |

**Critical Note:**  
The test period includes the COVID-19 market crash (March 2020), which represents extreme volatility conditions. This provides a stress test but may not reflect typical forecasting performance.

#### 4.1.2 GARCH Evaluation Protocol

**Evidence:** From `ARCH-benchmark.ipynb` (Cell 7):

```python
res = model.fit(last_obs=split_date, disp="off")
forecasts = res.forecast(start=split_date, method="analytic")
```

The GARCH model is:

1. Estimated on training data only (up to `split_date`)
2. Parameters fixed for out-of-sample forecasting
3. Forecasts generated from split date forward

This is a **fixed-window** evaluation (no re-estimation during test period).

### 4.2 Evaluation Metrics

#### 4.2.1 Mean Absolute Error (MAE)

$$MAE = \frac{1}{n} \sum_{t=1}^{n} |\hat{\sigma}_t - \sigma_t|$$

**Interpretation:** Average magnitude of forecast errors in original units (volatility). Robust to outliers relative to MSE.

**Justification:** Intuitive interpretation; penalizes all errors equally.

#### 4.2.2 Mean Squared Error (MSE)

$$MSE = \frac{1}{n} \sum_{t=1}^{n} (\hat{\sigma}_t - \sigma_t)^2$$

**Interpretation:** Average squared deviation. Heavily penalizes large errors.

**Justification:** Standard regression metric; corresponds to GRU training loss.

#### 4.2.3 QLIKE (Quasi-Likelihood)

$$QLIKE = \frac{1}{n} \sum_{t=1}^{n} \left( \frac{\sigma_t^2}{\hat{\sigma}_t^2} - \ln\left(\frac{\sigma_t^2}{\hat{\sigma}_t^2}\right) - 1 \right)$$

**Evidence:** From `GRU-benchmark.ipynb` (Cell 19):

```python
def Q_like(pred, true):
    div = true / pred
    ql = div - np.log(div) - 1
    return np.mean(ql)
```

**Interpretation:** Loss function derived from Gaussian log-likelihood for variance. Penalizes underestimation more heavily than overestimation (asymmetric).

**Justification:**  
QLIKE is the **economically appropriate loss function for variance forecasting** because:

1. It corresponds to the likelihood of observing realized variance given forecast
2. Underestimating volatility (and hence risk) is costlier than overestimation
3. Patton (2011) shows QLIKE ranks forecasts consistently even when realized volatility is measured with noise

### 4.3 Results

#### 4.3.1 GARCH(1,1) Performance

**Evidence:** From `ARCH-benchmark.ipynb` (Cell 12):

| Metric | Naive Baseline | GARCH(1,1) |
|--------|----------------|------------|
| **MAE** | 0.631 | **0.505** |
| **MSE** | 1.579 | **0.998** |
| **QLIKE** | 0.508 | **0.317** |

**Interpretation:**  
GARCH reduces MAE by 20% and QLIKE by 37% relative to naive persistence. This confirms substantial exploitable structure in volatility dynamics.

#### 4.3.2 GRU Performance

**Evidence:** From `GRU-benchmark.ipynb` (Cell 20):

| Metric | Naive Baseline | GRU |
|--------|----------------|-----|
| **MAE** | 0.633 | **0.543** |
| **MSE** | 1.588 | **1.121** |
| **QLIKE** | 0.498 | **0.395** |
| **R²** | 0.452 | **0.612** |

**Interpretation:**  
GRU improves substantially over naive baseline (14% MAE reduction, 21% QLIKE reduction) but underperforms GARCH on all metrics.

#### 4.3.3 Head-to-Head Comparison

| Metric | GARCH(1,1) | GRU | Winner |
|--------|------------|-----|--------|
| **MAE** | 0.505 | 0.543 | GARCH (+7%) |
| **MSE** | 0.998 | 1.121 | GARCH (+12%) |
| **QLIKE** | 0.317 | 0.395 | GARCH (+20%) |

**Key Finding:**  
GARCH(1,1) outperforms the GRU model across all metrics. The QLIKE advantage (20% lower) is particularly significant given its economic interpretation.

#### 4.3.4 Statistical Significance

> *Formal statistical tests (e.g., Diebold-Mariano test for equal predictive ability) were not conducted in the provided notebooks.*

**Limitation:** Without statistical testing, we cannot definitively conclude whether the GARCH advantage is significant or within sampling variation.

### 4.4 Failure Mode Analysis

#### 4.4.1 Visual Inspection

**Evidence:** Both notebooks include forecast vs. realized plots (Cells 18 and 10 respectively).

Qualitative observations:

1. **Both models track broad volatility dynamics** (spikes and troughs)
2. **Extreme spikes are underestimated** by both models (March 2020)
3. **Mean-reversion is captured**, but timing of peaks differs

#### 4.4.2 When Does Each Model Struggle?

| Regime | GARCH Behavior | GRU Behavior |
|--------|----------------|--------------|
| **Extreme spikes** | Underestimates (bounded by exponential decay) | Underestimates (limited training examples) |
| **Regime transitions** | Slow adaptation (fixed parameters) | Potentially faster (if seen in training) |
| **Low volatility** | Stable forecasts | May overfit to noise |

#### 4.4.3 Error Distribution

> *Formal analysis of residual properties (heteroskedasticity, autocorrelation) is not documented in the provided notebooks.*

**Recommendation:** Ljung-Box tests on squared forecast errors would reveal remaining predictable structure.

### 4.5 Robustness Considerations

#### 4.5.1 Regime Sensitivity

The test period (2020-2021) is dominated by:

- COVID-19 crash (March 2020)
- Recovery rally (April–December 2020)
- Meme stock volatility (early 2021)

This is **not representative of typical market conditions**. Both models may perform differently in low-volatility regimes.

#### 4.5.2 Split Sensitivity

> *Sensitivity to train/test split point was not analyzed in the provided notebooks.*

**Recommendation:** Rolling-window or expanding-window evaluation would provide robustness across regimes.

---

## PART V: CONCLUSIONS

### 5.1 Interpretation of Results

#### 5.1.1 Primary Finding

**GARCH(1,1) with Student-t innovations outperforms the GRU model** for one-day-ahead realized volatility forecasting on S&P 500 data under the current experimental design.

This result suggests that:

1. The volatility dynamics in this dataset are adequately captured by first-order autoregressive structure
2. The GRU's additional flexibility does not compensate for its training requirements
3. The minimal feature input (squared returns only) handicaps the neural network

#### 5.1.2 Why Deep Learning Did Not Add Value Here

| Factor | Explanation |
|--------|-------------|
| **Feature poverty** | GRU received only squared returns; GARCH implicitly uses the same information more efficiently |
| **Data efficiency** | GARCH has 4-5 parameters; GRU has thousands. More parameters require more data for stable estimation |
| **Volatility structure** | If true DGP is near-GARCH, parametric model will dominate |
| **Test regime** | COVID-19 period may favor models that mean-revert quickly (GARCH) over flexible learners |

#### 5.1.3 When Might GRU Outperform?

Hypothetically, GRU could dominate if:

- Richer features are provided (VIX, options-implied volatility, sentiment)
- Longer history with diverse regimes enables robust learning
- True dynamics involve complex non-linear patterns GARCH cannot capture
- Hyperparameter optimization is conducted systematically

### 5.2 Risk Considerations for Deployment

#### 5.2.1 Model Risk

| Risk | GARCH | GRU |
|------|-------|-----|
| **Specification error** | Known; bounded by parametric form | Unknown; black box |
| **Estimation error** | Low (few parameters) | High (many parameters) |
| **Interpretability** | High (economically meaningful parameters) | Low (hidden representations) |
| **Stability** | High (closed-form likelihood) | Medium (optimizer-dependent) |

#### 5.2.2 Operational Considerations

- **GARCH:** Simple to implement, well-understood failure modes, extensive literature
- **GRU:** Requires ML infrastructure, GPU resources for training, ongoing monitoring for drift

### 5.3 What Must Change Before Production Use

#### 5.3.1 For GARCH

1. Consider asymmetric variants (GJR-GARCH) to capture leverage effect
2. Implement rolling re-estimation for parameter adaptation
3. Add realized volatility as external regressor (RealGARCH)

#### 5.3.2 For GRU

1. **Expand feature set:** VIX, volume, sentiment, cross-asset correlations
2. **Hyperparameter optimization:** Systematic search over architecture and regularization
3. **Training regime diversity:** Ensure training data includes multiple market cycles
4. **Ensemble methods:** Combine multiple GRU specifications
5. **Uncertainty quantification:** Add prediction intervals

### 5.4 Next Steps

| Priority | Action | Rationale |
|----------|--------|-----------|
| **High** | Implement Diebold-Mariano test | Establish statistical significance of differences |
| **High** | Rolling-window evaluation | Assess robustness across regimes |
| **Medium** | Add features to GRU | Test whether deep learning gains value with richer inputs |
| **Medium** | Compare GJR-GARCH | Address asymmetry in volatility response |
| **Low** | Hybrid model (GARCH + NN) | Explore combining approaches |
| **Low** | Longer forecast horizons | Evaluate 5-day, 21-day forecasts |

---
