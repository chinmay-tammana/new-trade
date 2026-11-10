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
