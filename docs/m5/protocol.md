# Pre-Registered Validation Protocol for 5-Minute Directional Indicator (M5)

**Document Status**: LOCKED & PRE-REGISTERED (2026-11-16)  
**Target Horizon**: Next 5-minute bar directional return ($H1_{5m}$, 5-minute forward candle return).  
**Primary Symbol**: GS (Goldman Sachs Group Inc.).  
**Feed**: IEX 1-minute aggregated bars (RAW adjustment, causal, zero lookahead).

---

## 1. Research Holdout Specification
- **Holdout Window**: Strictly the last 6 months of trading sessions immediately preceding `freeze_date` (2026-11-13). Specifically, from approximately mid-May 2026 through 2026-11-13.
- **Purge / Embargo Gap**: An 8-bar gap (40 minutes) is enforced between the training set and holdout / validation folds to prevent autocorrelation leakage.
- **Single Read Discipline**: The research holdout is completely excluded from:
  1. Feature selection and feature-group ablations.
  2. Model architecture selection and hyperparameter tuning.
  3. Calibrator fitting (Platt scaling).
  4. Signal threshold tuning ($q_{\text{lo}}$, $q_{\text{hi}}$, coverage $c$, and STRONG threshold).
- **Holdout Overlap Disclosure**: The holdout period overlaps calendar dates evaluated in the 1-hour model (V2). This is a completely distinct model architecture, sampling frequency (5-minute vs. 1-hour), feature set, and target label. This overlap is fully disclosed in all reports.

---

## 2. Walk-Forward Cross-Validation (Pre-Holdout OOF)
- **Initial Training Window**: Oldest 2 years of data (approx. Nov 2021 to Nov 2023).
- **Refit Frequency**: Expanding window refit every quarter (63 trading sessions).
- **Fold Gap**: 8-bar embargo (40 minutes) at the boundary of each training block and validation block.
- **Evaluation**: All out-of-fold (OOF) predictions are pooled across all pre-holdout quarters.
- **Candidates Evaluated**:
  1. Shallow XGBoost (`tree_method='hist'`, max_depth $\in \{3, 4, 5\}$, subsample $\in \{0.8, 1.0\}$, learning_rate $\in \{0.03, 0.05\}$, max 20 grid configurations).
  2. `HistGradientBoostingClassifier` (scikit-learn).
  3. Elastic-Net Logistic Regression (`SGDClassifier` / `LogisticRegression` with L1/L2 penalties and standard scaler).
  4. Simple average ensemble of calibrated probabilities from the candidate models.
- **Model Selection Rule**: Winner chosen strictly by lowest pooled OOF log-loss. Ties broken in favor of the simpler model.
- **Sample Weighting Half-Life**: Evaluated over $\{90\text{ days}, 180\text{ days}, 365\text{ days}, \infty\text{ (none)}\}$.

---

## 3. Calibration & Signal Rule
- **Calibrator**: Platt scaling (logistic sigmoid) fit on pooled pre-holdout OOF raw margins with class-balanced sample weights, ensuring $p_{\text{cal}} = 0.50$ represents an unbiased, neutral prior.
- **Signal Rule**:
  - $\text{Up}$ if $p_{\text{cal}} \ge q_{\text{hi}}$
  - $\text{Down}$ if $p_{\text{cal}} \le q_{\text{lo}}$
  - $\text{No Signal}$ if $q_{\text{lo}} < p_{\text{cal}} < q_{\text{hi}}$, or on the final bar of the trading day (15:55-16:00 close), or upon a data gap.
- **Threshold Selection**:
  - $q_{\text{lo}}$ and $q_{\text{hi}}$ are the $(c/2)$ and $(1 - c/2)$ empirical quantiles of pooled pre-holdout OOF $p_{\text{cal}}$.
  - Coverage parameter $c \in \{0.10, 0.15, 0.20, 0.25, 0.30, 0.40\}$ chosen strictly to maximize the Wilson 95% lower bound of pooled OOF signal accuracy, subject to a minimum of 500 directional signals per side ($N_{\text{up}} \ge 500, N_{\text{down}} \ge 500$).
- **Strength Tier**:
  - $\text{STRONG}$: Displayed confidence $\ge 75\text{th}$ percentile of OOF confidence among directional signals.
  - Displayed confidence: $p_{\text{cal}}$ for Up, $1 - p_{\text{cal}}$ for Down (always $\ge 50\%$).
- **One-Sidedness Safeguards**:
  - Both sides mandatory in OOF and Holdout ($N_{\text{up}} > 0, N_{\text{down}} > 0$).
  - A side may only be deactivated if its pooled OOF accuracy 95% upper bound is below 50% over $\ge 1,000$ signals.
  - Track run-length frequency of 8 consecutive same-side signals and Spearman rank correlation of $p_{\text{cal}}$ with 1-day and 5-day market returns.

---

## 4. Benchmark Baselines
Evaluated on identical bars:
1. **Constant Prior**: Predicts historical training base rate $\bar{y}$.
2. **Always-Up**: Unconditional Up prediction.
3. **Always-Down**: Unconditional Down prediction.
4. **Previous-Bar Persistence**: Predicts the direction of the immediate prior 5-minute bar.
5. **RSI(14) Rule**: Up if $\text{RSI} < 30$, Down if $\text{RSI} > 70$.
6. **Time-of-Day Drift**: Direction determined by historical mean return for that specific 5-minute slot.

---

## 5. Pre-Registered Success Criteria
Before holdout evaluation, the following criteria are locked:
- **Criterion A (OOF Skill)**: Pooled pre-holdout OOF log-loss must be lower than the constant-prior baseline, with a stationary block-bootstrap 95% confidence interval of the difference strictly excluding zero.
- **Criterion B (Holdout Count & Accuracy)**: Holdout must contain at least 300 directional signals per side ($N_{\text{up}} \ge 300, N_{\text{down}} \ge 300$) and achieve signal accuracy $\ge 52.0\%$ with permutation test $p < 0.05$ (10,000 shuffles preserving per-session marginals). If unfulfilled, the report must plainly state that no statistically significant directional edge was proven.
- **Criterion C (Base Rate Superiority)**: Both Up accuracy and Down accuracy on the holdout must exceed their respective market base rates, or limitations clearly recorded.
- **Criterion D (Calibration Quality)**: Expected Calibration Error ($\text{ECE}$) $\le 0.030$ on pooled OOF.
- **Criterion E (Data & Code Integrity)**: Truncation tests, calendar tests, duplicate bar idempotence, and hash-chain audits pass 100%.
- **Criterion F (Economic Reporting)**: Net return per signal at friction levels 0 bps, 1 bps, 2 bps, and 5 bps is honestly reported with block-bootstrap confidence intervals (informative reporting, not gating).

---

## 6. Post-Evaluation Model Refit
- After holdout evaluation is sealed, the selected configuration is refit on all available data up to `freeze_date` (including the holdout).
- The production calibrator and thresholds are re-derived from pooled walk-forward OOF predictions.
- The resulting production model is saved to `models/m5/model_m5.joblib` and `models/m5/calib_m5.json` and locked with cryptographic hashes in `models/m5/MODEL_SHA256_m5.json`.
