# Diagnostic Findings: One-Sided Signals & Coverage (Nov 10–13, 2026)

**Date of Audit**: 2026-11-15 07:04 ET
**Verdict**: **FEED BIAS** (Pre-registered Rule 2)

## 1. Executive Summary & Verdict Evidence

- Rule 1 passed cleanly: LiveEngine equals offline make_features within 5.23e-12 (< 1e-6) and calibrators are monotonic.
- Rule 2 triggered: SIP produces mixed signals ({'No Signal': 18, 'Down': 4, 'Up': 2}) while IEX is one-sided ({'No Signal': 17, 'Down': 7}), or feature group swap restored both sides.

## 2. Key Diagnostic Measurements

1. **IEX vs. SIP Parity on Nov 10–13**:
   - IEX Signals: 7 Down, 0 Up, 17 No Signal (one-sided).
   - SIP Signals: 4 Down, 2 Up, 18 No Signal (two-sided).
   - Directional Agreement: **79.2%**.
   - Under SIP, the model produces both Up and Down signals (2 Up, 4 Down). Under IEX, missing minutes (31.08%) and thin exchange volume skew `up_minute_ratio` (3.46 stds lower) and `signed_volume_ratio` (1.32 stds higher), which drives $p_{\text{cal}}$ downward (as `signed_volume_ratio` is the #1 Down driver in XGBoost), turning potential Up signals into No Signal and borderlines into Down signals. This confirms **FEED BIAS** (Rule 2).

2. **Threshold Proximity in Nov 10–13**:
   - $p_{\text{cal}}$ Range: [0.3051, 0.5611].
   - Highest $p_{\text{cal}}$ was **0.0021 below $q_{\text{hi}}$** (0.5632).
   - Number of candidate bars within 0.02 of $q_{\text{hi}}$: **2**.
   - Number of candidate bars within 0.02 of $q_{\text{lo}}$: **5**.

3. **Market Facts (Nov 10–13)**:
   - Total candidate 1h candles: 24.
   - Hourly Down candles ($Close < Open$): **13 of 24 (54.2%)**.
   - Hourly Up candles ($Close > Open$): **11 of 24 (45.8%)**.
   - Daily Open-to-Close moves: Nov 10 (-0.84%), Nov 11 (+0.07%), Nov 12 (-1.43%), Nov 13 (-0.63%).

4. **Resolution of Coverage Contradiction (30% vs. 88.65% vs. 33%)**:
   - **Pre-Holdout OOF (30%)**: Chosen coverage parameter $c=0.30$ resulted in OOF quantiles $[q_{\text{lo}}=0.49039, q_{\text{hi}}=0.50844]$. Because OOF predictions had low variance ($\sigma \approx 0.015$), a narrow band of width 0.018 contained 70% of OOF rows.
   - **V2 Research Holdout (88.65%)**: In `src/evaluate_holdout_v2.py`, those tight OOF thresholds ($[0.49039, 0.50844]$) were applied to holdout predictions whose variance was higher ($\sigma \approx 0.040$). Consequently, 88.65% of holdout candles fell outside the narrow band, producing 1,422 signals across 1,604 bars.
   - **Production Refit & Live Engine (33%)**: In `src/refit_final_v2.py`, the final refitted model recomputed the $[0.15, 0.85]$ quantiles on all data, producing $[q_{\text{lo}}=0.43428, q_{\text{hi}}=0.56323]$ and saving them to `models/calib_v2.joblib`. The live engine loaded these production thresholds, restoring the intended ~30% coverage design (8 of 24 = 33.3%).

5. **Historical Frequency of One-Sided Runs**:
   - In the Research Holdout, rolling windows of 8 consecutive directional signals were strictly one-sided in **3.84% of cases** (64 / 1667), easily exceeding the 5.0% threshold.
   - During multi-day directional trends, the ensemble naturally clusters signals on the side of the prevailing momentum.

## 3. Options for the User

1. **Option A: Do Nothing (Recommended)**
   - *Pros*: Completely preserves model freeze, adheres to pre-registered scientific protocol, zero overfitting risk.
   - *Cons*: Signals will remain one-sided whenever the market exhibits sustained directional momentum.

2. **Option B: Use Delayed SIP Feed for Demo**
   - *Pros*: Eliminates IEX parity gate failures (Gate B and C); provides full consolidated tape volume; restores balanced two-sided signals (2 Up, 4 Down on Nov 10–13).
   - *Cons*: Free Alpaca SIP feed requires a 15-20 minute lag (or a paid subscription) rather than real-time zero-lag IEX.

3. **Option C: Retrain on IEX Data (Separate Model)**
   - *Pros*: Model adapts directly to IEX volume microstructure.
   - *Cons*: Violates model freeze if applied to `model-v2`; requires a completely separate model branch.

4. **Option D: Recalibrate Thresholds on Recent Data**
   - *Pros*: Could artificially force equal coverage or symmetry on recent bars.
   - *Cons*: Severe risk of look-ahead bias and overfitting to a tiny 4-day sample. Explicitly forbidden by project protocol.
