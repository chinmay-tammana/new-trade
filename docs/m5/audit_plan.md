# Pre-Registered Integrity Audit Plan for M5 (5-Minute) System

**Document Status**: LOCKED & PRE-REGISTERED (2026-11-16)  
**Objective**: Rigorously investigate the 69.56% pooled OOF accuracy on 5-minute directional returns to detect potential data leakage, lookahead bias, or microstructure artifacts before modifying the signal coverage rule.

---

## 1. Pre-Registered Fail Gates & Decision Rules

### Gate 1: Shuffled-Label Null Test
- **Procedure**: Permute training labels randomly (`y_train_shuffled`) across each walk-forward fold, retrain the winning XGBoost depth-4 configuration, and evaluate out-of-fold accuracy on the unshuffled validation folds using the same $c=0.10$ threshold rule.
- **Fail Condition**: Signal accuracy outside the range **[48.0%, 52.0%]**.
- **Action on Failure**: Immediate STOP. Model is learning structural leakage rather than real financial dependencies.

### Gate 2: Time-Shift Lead/Lag Test
- **Procedure**: Shift the feature matrix by $+1$ bar (features from bar $t-1$ mapped to label $y_t$) and evaluate pooled OOF accuracy.
- **Fail Condition**: Pooled signal accuracy remains **$> 55.0\%$**.
- **Action on Failure**: Immediate STOP. Indicates lookahead leakage across bar boundaries.

### Gate 3: Single-Feature Information Leakage & Window Inspection
- **Procedure**:
  1. Compute ROC-AUC of each individual feature against binary direction $y$, and Spearman rank correlation $\rho$ with forward return $fwd$.
  2. Perform code inspection of feature windows: verify that `rvol_tod`, `session_vwap`, `opening_range`, `intrabar` features, and time-of-day statistics strictly use data $\le \text{close}[t]$ and zero future bars ($t+1$).
- **Fail Condition**: Any single feature exhibits $|\text{AUC} - 0.50| > 0.10$ (i.e., $\text{AUC} > 0.60$ or $\text{AUC} < 0.40$).
- **Action on Failure**: Immediate STOP. Disclose the offending feature and exact mathematical leakage mechanism.

### Gate 4: IEX Microstructure & Thin-Print Artifact Test
- **Procedure**:
  1. Build session-anchored 5-minute bars from cached consolidated SIP 1-minute data (`data/raw/alpaca_1min/GS/*.parquet`).
  2. Compute forward return $fwd_{\text{SIP}} = close_{\text{SIP}}[t+1] / close_{\text{SIP}}[t] - 1$.
  3. Score the exact signals generated from IEX features against $fwd_{\text{SIP}}$.
  4. Compute lag-1 return autocorrelation for both feeds: $\text{ACF}_1(r_{\text{IEX}})$ vs. $\text{ACF}_1(r_{\text{SIP}})$.
  5. Compute signal accuracy segmented by `n_minutes_with_trades` (Low: 1-2 trades/bar vs High: 4-5 trades/bar).
- **Classification Rule**:
  - If $\text{Acc}_{\text{SIP}} < \text{Acc}_{\text{IEX}} - 5.0\%$, OR $\text{ACF}_1(r_{\text{IEX}}) < -0.05$ while $\text{ACF}_1(r_{\text{SIP}}) \approx 0.0$, classify as **IEX PRINT ARTIFACT** (bid-ask bounce / thin-tape mean reversion unexecutable at consolidated market prices).
  - **Action**: Proceed with Section B, but mandate a prominent red warning banner in UI and disclosures.

### Gate 5: Subgroup Stability Audit
- **Procedure**: Compute signal accuracy and Wilson 95% confidence intervals broken down by:
  - Calendar Year (2022, 2023, 2024, 2025, 2026)
  - Time-of-Day Session Segment: Opening (09:30-11:00), Midday (11:00-14:00), Closing (14:00-15:55)
  - Directional Side (Up vs Down)

---

## 2. Hard Execution Rule
If **Gate 1, Gate 2, or Gate 3 FAILS**: STOP execution immediately. Write `docs/m5/audit_findings.md`, log all evidence, and report findings to the owner without proceeding to Section B.  
If **Gate 1, Gate 2, and Gate 3 PASS**, proceed to Section B.
