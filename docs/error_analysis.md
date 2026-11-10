# Error Analysis, Explainability, and Robustness

## 1. Diagnostic Visualizations
- **Representative Signal Execution** (`results/figures/signals_sample_chart.png`): Demonstrates the 1-bar execution cycle during choppy midday trading. Green triangles mark correct Up predictions; red crosses mark false breakouts where Goldman Sachs reversed immediately following candle entry.
- **Regime Error Breakdown** (`results/figures/error_breakdown.png`): Illustrates performance sensitivity across diurnal hours, days of the week, confidence buckets, and calendar quarters.
- **TreeSHAP Feature Attribution** (`results/figures/shap_summary_val.png`): Highlights the primary drivers of log-odds predictions on the Validation partition. **Crucial Scientific Caveat**: Feature attribution reflects statistical correlation within the ensemble trees and must not be conflated with financial causality.

## 2. Top Confident Erroneous Predictions on Test Block
| Timestamp (ET) | Close ($) | Forward Move | Label | Confidence | Realized Net Ret (5 bps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-08-18 09:30:00-04:00 | 1036.03 | -0.53% | 0 | 65.17% | -58.47 bps |
| 2026-06-26 13:30:00-04:00 | 1028.30 | -0.05% | 0 | 64.34% | -9.89 bps |
| 2026-09-01 09:30:00-04:00 | 1023.25 | -0.12% | 0 | 63.64% | -17.46 bps |
| 2026-10-30 12:30:00-04:00 | 885.72 | -0.50% | 0 | 63.51% | -54.68 bps |
| 2026-06-10 14:30:00-04:00 | 1002.17 | -0.14% | 0 | 63.43% | -18.57 bps |
| 2026-07-29 14:30:00-04:00 | 990.29 | -0.96% | 0 | 63.22% | -101.23 bps |
| 2026-04-29 09:30:00-04:00 | 915.22 | -0.71% | 0 | 62.68% | -75.86 bps |
| 2026-07-27 09:30:00-04:00 | 1059.17 | -0.93% | 0 | 62.60% | -97.95 bps |
| 2026-06-05 09:30:00-04:00 | 1067.27 | -0.06% | 0 | 62.22% | -11.47 bps |
| 2026-07-24 13:30:00-04:00 | 1064.19 | -0.30% | 0 | 62.22% | -34.88 bps |

## 3. Structural Limitations and Vulnerabilities
1. **Small Sample Size & Statistical Uncertainty**: With 188 emitted signals, the Wilson 95% score confidence interval spans `[39.30%, 53.41%]`, spanning across 50%. A point accuracy of 46.28% provides no demonstrable predictive edge over random guessing on held-out 2026 data.
2. **Market Regime Dependence & Asymmetric Bias**: The model selection on Validation chose an Up-only decision boundary because 2023–2025 featured strong equity upward drift. When the Test period encountered consolidation and mean-reverting market phases (overall Test Up-rate was 48.88%), the rigid Up-only rule suffered repeated false continuations.
3. **Transaction Costs & Microstructure Drag**: Intraday hourly returns have small median magnitude (~20–40 bps). A round-trip transaction friction of 5 bps imposes a severe hurdle, reducing gross returns from -2.24 bps to -7.24 bps per signal.
4. **Consolidated SIP vs Broker Feeds**: Features such as `rvol` depend on consolidated full-market volume. Live paper execution on broker-specific feeds (e.g. IEX) may experience volume mismatch.
5. **Near-Efficiency of Hourly Equity Prices**: In compliance with Fama (1970), hourly prices in mega-cap liquid equities like Goldman Sachs approach a martingales process where past OHLCV data alone carries negligible linear or non-linear directional predictability.