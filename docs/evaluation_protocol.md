# Evaluation Protocol (Frozen)

This document establishes and freezes the official evaluation protocol for the Goldman Sachs (`GS`) AI trading indicator project. It defines block partitions, performance metrics, selection rules, and pre-registered success criteria. **No model training may modify this protocol.**

---

## 1. Partition Architecture and Block Roles

The dataset spans 4+ years of session-anchored 1-hour regular-session candles (09:30 to 16:00 ET). Partitions are strictly chronological with 8-bar embargo gaps:

```
|---- Base: oldest ~2y ----|gap|--- FT: year 3 ---|gap|-- Val: 40% of year 4 --|gap|-- Test: 60% of year 4 --|
 oldest                                                                                           newest
```

| Partition | Temporal Span | Role & Constraints |
| :--- | :--- | :--- |
| **Base** | Oldest ~2 years (Oct 2022 – Nov 2024) | Model pre-training (S0) and fitting of in-fold preprocessors. Labelled decisive rows used for fitting; Base Up-rate defines the constant prior. |
| **Fine-Tuning (FT)** | Year 3 (Nov 2024 – Nov 2025) | Walk-forward cross-validation (4 chronological folds) for hyperparameter optimization and strategy evaluation (S1, S2, S3). |
| **Validation (Val)** | First 40% of Year 4 (~Nov 2025 – Apr 2026) | Probability calibration (Platt vs Isotonic), asymmetric signal threshold optimization \((d_{\text{up}}, d_{\text{down}})\), model selection, and ablations. Scored on **all** labelled candles. |
| **Test** | Final 60% of Year 4 (~Apr 2026 – Nov 2026) | **Held-out final evaluation evaluated strictly ONCE in Phase 7.** Never accessed during feature engineering, tuning, threshold selection, or model selection. |

---

## 2. Core Evaluation Metrics

1. **Probabilistic Scoring**:
   - **Binary Log-Loss**: Primary metric for model selection. Penalizes overconfident wrong predictions.
   - **Brier Score**: Quadratic accuracy of probabilistic forecasts \(\frac{1}{N} \sum (\hat{p}_i - y_i)^2\).
   - **Expected Calibration Error (ECE)**: 10 uniform probability bins measuring deviation between confidence and observed frequency.
2. **Directional Signal Metrics**:
   - **Signal Accuracy**: Percentage of correct directional forecasts on active candles (Up or Down).
   - **Wilson 95% Score Confidence Interval**: Exact binomial confidence interval incorporating sample size uncertainty.
   - **Coverage**: Proportion of eligible labelled candles for which an active signal is emitted.
   - **Precision, Recall, F1**: Per-class metrics for Up and Down.
3. **Economic Expectancy**:
   - **Simulated Net Return per Signal**: 1-bar holding period return with long on Up, short on Down:
     \[
     R_{\text{net}} = \text{Direction} \times fwd - \text{Cost}
     \]
     Evaluated at 2, 5, and 10 bps round-trip transaction costs; primary benchmark at **5 bps**.
   - **Block-Bootstrap 95% CI**: 2,000 resamples using 5-bar contiguous blocks to account for temporal dependence.

---

## 3. Baseline Benchmarks

All baselines are evaluated on the exact same labelled Validation set (and subsequently on the exact same Test set):
1. **Constant Prior**: Predicts constant probability \(\hat{p} = \text{Base Up-rate}\) on every candle. Provides a baseline log-loss and Brier reference.
2. **Always-Up**: Emits an Up signal on 100% of candles to capture prevailing equity upward drift.
3. **Majority Class of Base**: Emits the dominant class of the Base training period.
4. **RSI Heuristic Rule**: Up if \(\text{RSI}_{14} < 40\) (oversold mean reversion), Down if \(\text{RSI}_{14} > 60\) (overbought), No Signal otherwise.
5. **Previous-Bar Momentum / Persistence**: Up if \(\text{ret}_1 > 0\), Down if \(\text{ret}_1 \le 0\).

---

## 4. Model Selection and Deployment Rules

1. **XGBoost Strategy Selection**: Compare S0 (Base default), S1 (Base tuned), S2 (Base + FT refit), and S3 (Base + FT warm-start boosting) on Validation log-loss on all labelled rows. Prefer the lowest log-loss; ties within 0.001 favor simpler models (S1 < S2 < S3).
2. **Comparator Switch Rule**: The deployed model is the chosen XGBoost strategy UNLESS a conventional ML comparator (Logistic Regression, Random Forest, HistGradientBoosting) beats it on Validation log-loss by more than `switch_margin_logloss` (0.002).
3. **Asymmetric Signal Rule Search**:
   - Objective: Maximize the Wilson 95% lower bound of signal accuracy on Validation.
   - Constraints: Total active signals \(\ge 40\); active signals per enabled side \(\ge 25\); total coverage \(\ge 15\%\).
   - Fallback: If constraints cannot be satisfied, relax coverage first, then side counts, documenting the relaxation. If one side is disabled, classify the indicator as "Up-only" or "Down-only".

---

## 5. Pre-Registered Success Criteria (Fixed Protocol)

| Criterion | Target Threshold | Validation / Verification Method |
| :--- | :--- | :--- |
| **C1. Validation Log-Loss** | \(\le \text{Constant Prior Log-Loss} - 0.002\) | Evaluated on all labelled Val candles in Phase 5. |
| **C2. Test Signal Accuracy** | \(\ge 53.0\%\) with \(\ge 100\) signals & \(>\) Always-Up | Evaluated on unseen Test block in Phase 7. |
| **C3. Net Return Expectancy** | \(> 0.0\) bps after 5 bps costs | 1-bar holding simulation on Test block in Phase 7. |
| **C4. Probability Calibration**| \(\text{ECE} \le 0.05\) (10 bins) | Measured on Test block in Phase 7. |
| **C5. Directional Balance** | \(\ge 25\) Up and \(\ge 25\) Down signals | Measured on Test block; limitations disclosed if regime suppresses one side. |
| **C6. Integrity & Reproducibility**| 100% check passage | Shuffled label test (\(\approx 50\%\)), time-shift degradation, truncation invariance, secret check. |
