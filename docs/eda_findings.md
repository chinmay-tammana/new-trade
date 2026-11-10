# Exploratory Data Analysis (EDA) Findings and Statistical Insights

This document records findings from exploratory data analysis conducted **strictly on the Base training block** (and Base vs FT drift check). In compliance with IMPACT guidelines and test-set discipline, no data from the Validation or Test blocks was inspected during this phase.

---

## 1. Target Class Breakdown and Decisive Filtering
**Figure Reference**: `results/figures/eda_target_distribution.png`

- **Decisive Threshold**: Established at \(|fwd| > 0.0010\) (10 basis points).
- **Base Block Label Counts**:
  - Decisive Up moves (\(> +10\) bps): Represents approximately 39.5% of labelled candles.
  - Decisive Down moves (\(< -10\) bps): Represents approximately 38.8% of labelled candles.
  - Neutral / Indecisive moves (\(\le 10\) bps absolute move): Represents approximately 21.7% of candles.
- **Interpretation**: Intraday hourly returns are centered near zero with thick tails. Training models on uninformative noise candles (\(|fwd| \le 10\) bps) dilutes gradient optimization and encourages high churn. Restricting model training to decisive candles focuses the boosting process on economically meaningful price displacements, while all candles remain evaluated during validation and test scoring.

---

## 2. Stationary Feature Distributions
**Figure Reference**: `results/figures/eda_feature_distributions.png`

- `ret_1`: Centered symmetrically around 0.0003 with sample standard deviation ~0.0084 (84 bps/hour). Minimal skewness, leptokurtic distribution.
- `rsi` (Wilder 14): Bounded strictly in \([0, 100]\) with mean 50.8 and standard deviation 11.2; extreme values (\(<30\) or \(>70\)) occur in \(<4\%\) of hourly bars.
- `bb_pos` (%B): Bounded predominantly between 0.0 and 1.0, with mean 0.52. Mild boundary clipping occurs during intraday trend breakouts.
- `atr_n`: Normalized Average True Range fluctuates around 0.0075 (75 bps) during calm periods and spikes to >0.020 (200 bps) during earnings and macro announcements.
- `rvol`: Positively skewed relative volume ratio centered at 1.00; values exceeding 3.0 represent heavy institutional opening and closing auction volume.
- `rel_ret_1`: Excess return of GS relative to SPY; isolates idiosyncratic financial-firm alpha from broad macro beta.

---

## 3. Multicollinearity and Cross-Correlation
**Figure Reference**: `results/figures/eda_correlation_heatmap.png`

- High positive correlation observed among proximate return horizons (`ret_1`, `ret_2`, `ret_3`), as expected from overlapping moving windows.
- Moderate correlation between GS returns and context series (`spy_ret_1`, `xlf_ret_1`), reflecting Goldman Sachs' high beta (~1.2 to 1.4) to the financial sector and equity index.
- Relative return (`rel_ret_1`) is orthogonal to broad market returns, successfully isolating idiosyncratic single-stock alpha.
- Momentum (`rsi`, `macd_n`) and volatility (`atr_n`, `bb_width`) exhibit low pairwise correlation (\(<0.25\)), confirming complementary informational dimensions.

---

## 4. Return and Volatility Autocorrelation
**Figure Reference**: `results/figures/eda_autocorrelation.png`

- **Raw Return Autocorrelation (`ret_1`)**: Lies strictly within the 95% Bartlett confidence bands (\(\pm 1.96 / \sqrt{N}\)) across all lags from 1 to 15. This confirms that linear price predictability is virtually nonexistent in hourly equity returns, directly validating the weak-form Efficient Market Hypothesis.
- **Absolute Return Autocorrelation (`|ret_1|`)**: Exhibits strong, statistically significant persistence extending beyond 10 hourly lags (\(>0.15\) at lag 1, decaying slowly). This empirical finding validates the presence of volatility clustering in GS intraday returns and justifies the inclusion of `atr_n`, `vol_20`, and `bb_width`.

---

## 5. Diurnal and Weekly Seasonality
**Figure Reference**: `results/figures/eda_hourly_dow_uprate.png`

- **Hour of Day**:
  - The opening bar (09:30–10:30 ET) exhibits elevated volatility and an Up-rate near 52.4%.
  - Midday bars (12:30–14:30 ET) show compression towards 50.0% with low relative volume (lunch lull).
  - The penultimate afternoon session (14:30–15:30 ET) demonstrates modest directional momentum preceding closing imbalances.
- **Day of Week**: Up-rates remain balanced across Monday through Friday between 49.5% and 53.0%, with no single day exhibiting dominant drift in the Base era.

---

## 6. Base vs Fine-Tuning Era Distribution Drift
**Figure Reference**: `results/figures/eda_drift_ks_base_vs_ft.png`
**Data Table**: `results/tables/drift_ks_base_vs_ft.csv`

- A two-sample Kolmogorov-Smirnov test was executed across all features between the Base training block (~2022-2024) and the Year 3 Fine-Tuning block (~2024-2025).
- Features with lowest drift (KS \(< 0.06\)): `rsi`, `bb_pos`, `hour`, `dow`, `ret_1`. Bounded oscillators and return distributions remain highly stationary over time.
- Features with moderate drift (KS \(\approx 0.08 - 0.12\)): `atr_n`, `vol_20`, `rvol`. Driven by macro volatility cycle shifts between 2023 rate-hiking regimes and 2024-2025 equity rally conditions.
- **Decision Rationale**: Tree-based gradient boosting is invariant to monotonic feature transformations and handles moderate regime shifts well. In-fold Winsorization and standard scaling in linear models will protect against outlier distortion.

---

## 7. V2 Upgrade: 8-Year Pre-Holdout Feature Distributions & KS Drift
**Figure Reference**: `results/figures/eda_drift_ks_v2.png`, `results/figures/eda_correlation_matrix_v2.png`  
**Data Table**: `results/tables/drift_ks_v2.csv`

In accordance with strict test-set discipline, exploratory data analysis for the V2 feature expansion was conducted **strictly on the 12,004 pre-holdout candles** (October 2018 to October 2025). The Research Holdout (1,872 candles from October 2025 to November 2026) remains completely unobserved.

### 7.1 New Feature Group Distributions
1. **Banking Peers (`jpm_ret_1`, `ms_ret_1`, `bac_ret_1`, `gs_minus_peers`)**:
   - `GS` maintains an intraday return correlation of ~0.72 with `MS` and ~0.65 with `JPM`.
   - The idiosyncratic spread feature `gs_minus_peers` is zero-meaned ($\mu \approx -0.00004$) with standard deviation 0.0048, providing a clean relative-value signal purged of sector-wide beta.
2. **Macro Rates & Implied Volatility (`tlt_ret_1`, `vixy_ret_1`)**:
   - `tlt_ret_1`: Uncorrelated with raw GS hourly direction ($\rho \approx -0.04$), but large bond moves correlate with banking valuation repricing.
   - `vixy_ret_1`: Exhibits significant negative correlation with GS returns ($\rho \approx -0.42$), capturing broad risk-off flights.
3. **Intrabar Microstructure (`realized_vol_1m`, `up_minute_ratio`, `clv`, `signed_volume_ratio`, `last15_ret`)**:
   - `realized_vol_1m`: Highly correlated with candle range; captures sudden high-frequency bursts of intraday execution volume.
   - `clv`: Close Location Value ranges $[-1.0, +1.0]$ with mean $+0.03$; green bars close near highs, red bars near lows.
   - `signed_volume_ratio`: Measures net buying vs selling volume; provides a leading indicator for order flow imbalance (OFI).
4. **Levels & Multi-Day Trend (`dist_prev_close`, `dist_session_vwap`, `ret_20d`, `efficiency_ratio_20`)**:
   - `dist_session_vwap`: Stationary percentage deviation from intraday VWAP ($\sigma \approx 0.0062$); exhibits strong mean-reverting tendencies when exceeding $\pm 1.5\%$.
   - `efficiency_ratio_20`: Ranges $[0.0, 1.0]$ with median $0.28$; distinguishes directional trending regimes ($>0.45$) from horizontal chop ($<0.20$).

### 7.2 KS Drift Analysis (Early 4 Years vs Later 3 Years Pre-Holdout)
- Dividing the pre-holdout series at the median (6,002 bars each):
  - Highest drift observed in volatility metrics: `atr_n` ($\text{KS} = 0.181$), `vol_20` ($\text{KS} = 0.138$), and `bb_width` ($\text{KS} = 0.132$), reflecting the regime shift from the high-volatility COVID-19/2022 rate hike cycles to the lower-volatility 2023–2024 bull market.
  - Return distributions (`ret_1`, `ret_2`, `ret_3`) remained highly stationary ($\text{KS} < 0.04$), confirming the necessity of normalizing all level features by ATR or moving averages.
