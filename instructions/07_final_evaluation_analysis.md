# Phases 7 & 8: Final Evaluation and Error Analysis

======================================================================
PHASE 7: FINAL EVALUATION, ONCE (src/evaluate.py) (IMPACT Steps 9-10)
======================================================================
Run once on the Test block, all labelled rows. Evaluate the frozen deployed model, every comparator and every baseline on the same rows, in ONE run. Save results/tables/final_results.csv, results/logs/metrics.json, and the per-candle Test predictions (timestamp, calibrated p, signal, label, fwd, and the per-seed predictions) to results/tables/test_predictions.csv. After this phase NO code may read Test features or labels again; Phase 8 uses the saved predictions file.
- Result table (IMPACT recommended): Model | Validation score | Test score | Training time | Model size for the simple baselines, each conventional ML model and the proposed method. Primary score: log-loss (where a probability exists); also report accuracy.
- Classification metrics: precision, recall, F1 per class, ROC-AUC, PR-AUC, confusion matrix (Up/Down/No signal versus truth), signal accuracy with Wilson 95% CI, coverage, counts of Up/Down/No signal, log-loss, Brier, ECE.
- Always-Up accuracy and majority-class rate on the SAME rows, and always-Up accuracy restricted to the candles where the model signalled (to show whether the model beats drift).
- Trading simulation: long on Up, short on Down, 1-bar holding; net return per signal and cumulative return at 2, 5 and 10 bps; hit rate after costs; max drawdown; number of trades; block-bootstrap 95% CI for net return per signal.
- Variance: refit the frozen configuration with n_seeds_variance seeds (the frozen model remains seed 42) and report mean +/- std of Test metrics.
- Breakdown by hour of day, day of week, volatility regime (atr_n above/below the Base median) and confidence bucket; calibration on Test.
Statistics honesty: with about 150 signals a 55% accuracy has a CI of roughly +/-8 points. Do not claim an edge the CI does not support. Evaluate every pre-registered success criterion in docs/problem_definition.md and write the verdict (met / not met per criterion) into RESULTS.md. If the model fails, say so plainly; do not rerun with changes.
CHECK: metrics written; Test read exactly once; CIs present; verdict written.

======================================================================
PHASE 8: ERROR ANALYSIS, EXPLAINABILITY, ROBUSTNESS (src/analysis.py) (IMPACT Step 11)
======================================================================
Descriptive only; must not change the frozen model. Use results/tables/test_predictions.csv for Test-based analysis and Val for anything that needs features.
- Representative correct and incorrect signals: price chart with Up/Down markers and outcomes for sample windows; table of the 10 most confident wrong signals.
- Error by hour, day of week, volatility regime, confidence bucket, class (Up vs Down) and calendar quarter; confusion pattern; unstable predictions (signals that flip across seeds, from the per-seed predictions).
- Explainability on Val: XGBoost native contributions (pred_contribs=True) global summary and permutation importance; ablation recap. State that attribution is not causation.
- Robustness on Val: noise sensitivity (add Gaussian noise of 0.1 and 0.25 feature standard deviations, report the metric change); distribution shift (performance by era: FT versus Val versus Test using saved predictions); seed sensitivity (from Phase 7).
- docs/error_analysis.md: figures with captions and interpretations, plus a limitations paragraph: small number of signals, regime dependence, dividends/unadjusted prices, SIP versus live-feed volume, last bar of the day = No signal, hourly equity direction is near-efficient.
CHECK: figures with captions and interpretations exist in results/figures/ and docs/error_analysis.md; no Test features/labels were re-read.
