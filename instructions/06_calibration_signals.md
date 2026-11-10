# Phase 6: Calibration and Signal Rule

======================================================================
PHASE 6: CALIBRATION AND SIGNAL RULE (src/calibrate.py)
======================================================================
1. Raw p = P(Up). Calibrate on Val, ALL labelled rows (no decisive filter): Platt scaling (logistic regression on clipped logit(p)); compare isotonic; choose Platt unless isotonic is clearly better (Val is small). Reliability diagram, Brier score and ECE before and after. Do this for the deployed model and for each comparator.
2. Directional diagnostics (the earlier run produced zero Down signals): print the distribution of calibrated p on Val (mean, std, 5/25/50/75/95th percentiles), the Up-rate of labels for Base, FT and Val, and the mean predicted p for Base, FT and Val (never Test). Save results/tables/directional_diagnostics.csv.
3. Signal rule with separate margins: Up if p_cal >= 0.5 + d_up; Down if p_cal <= 0.5 - d_down; else No signal. Displayed confidence = p_cal for Up and 1 - p_cal for Down, so it is always above 50%. Choose (d_up, d_down) by a joint grid search on Val over margin_sweep, where each side may also be switched OFF. Constraints: each enabled side has >= min_signals_per_side signals; combined coverage >= min_coverage; combined signals >= min_signals. Objective: the Wilson 95% lower bound of combined signal accuracy. If no combination satisfies the constraints, relax in this order (min_coverage first, then min_signals_per_side), log it in progress.md, and report. If the chosen rule disables a side, state "Up-only" or "Down-only" everywhere (docs, user guide, report). Also report the best symmetric rule (d_up = d_down) for comparison.
4. Print Up / Down / No-signal counts on Val.
5. Save models/calib.joblib: calibrator, d_up, d_down (None if the side is off), enabled sides, FEATURES, label_threshold, horizon_bars, chosen strategy and model name.
CHECK: reliability plot exists; rule chosen on Val only; Up/Down/No-signal counts reported; at least min_signals signals on Val.
