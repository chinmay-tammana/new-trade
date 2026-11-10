# Project Progress and Tracking

## Motivating Context: Earlier 2-Year Run
Earlier 2-year run: test set 587 eligible hourly candles; 114 Up signals, 0 Down signals, 473 No signal; signal accuracy 53.5% (about 61 of 114 correct); thresholds were p>=0.56 (Up) and p<=0.44 (Down). Likely causes: (a) the Down threshold was never reached, so the indicator was effectively Up-only and mostly rode upward drift; (b) 114 signals gives a 95% interval of roughly 44%-63%, i.e. no demonstrable edge; (c) only about 2 years of data.
*Disclosure Note*: In the final report, disclose that the new Test block overlaps the newest period of that earlier test set, so the live test is the only fully clean judge.

---

## Phase Checklist

- [x] **Phase 0: Setup**
  - [x] Repository layout created (`configs/`, `data/raw/`, `data/processed/`, `src/`, `results/tables/`, `results/figures/`, `results/logs/`, `models/`, `report/`, `slides/`, `docs/`, `tests/`, `instructions/`, `archive/`)
  - [x] Configuration created (`configs/config.py`)
  - [x] Git repository initialized, `.gitignore` created before first commit
  - [x] Secret detection tool created (`src/check_secrets.py`) and verified
  - [x] Virtual environment and dependencies installed (`requirements.txt`)
  - [x] Remote `origin` configured (`https://github.com/chinmay-tammana/new-trade.git`)
  - [x] Initial commit and push completed

- [x] **Phase 1: Problem Definition and Literature**
  - [x] `docs/problem_definition.md` created with pre-registered criteria
  - [x] `docs/literature_survey.md` created with verified citations
  - [x] Project title finalized: "Next-Hour Direction Prediction for Goldman Sachs Stock Using Calibrated XGBoost on Hourly OHLCV Data"
  - [x] Phase 1 committed and pushed before training code

- [x] **Phase 2: Data Collection and Audit**
  - [x] Alpaca credentials verified from `.env`
  - [x] Availability test on 1-min SIP bars completed (retrieved 330 1-min bars)
  - [x] 4 years + 30 days of 1-minute data for GS, SPY, XLF downloaded
  - [x] Session-anchored 1h bars constructed (09:30 to 16:00 ET, exactly 7,148 GS bars)
  - [x] Data quality audit completed (`results/logs/data_validation.txt`, `results/tables/audit_summary.csv`)
  - [x] Parity check with yfinance on recent ~60 days completed (corr = 0.999998, diff = 0.23 bps)
  - [x] `docs/dataset_card.md` and `docs/data_dictionary.md` written
  - [x] Audit figures generated in `results/figures/`
  - [x] Phase 2 committed and pushed

- [x] **Phase 3: EDA, Features, Labels, Preprocessing**
  - [x] Base-only EDA completed with figures in `results/figures/` and `docs/eda_findings.md`
  - [x] Drift check (Base vs FT Kolmogorov-Smirnov) completed (`results/tables/drift_ks_base_vs_ft.csv`)
  - [x] `src/features.py` implemented (`make_features`, 22 stationary causal features)
  - [x] `src/labels.py` implemented (fwd simple return, overnight NaN, decisive filtering)
  - [x] `src/preprocessing.py` implemented (fold-isolated InFoldWinsorizer and StandardScaler)
  - [x] Truncation test passed (50 cutoffs, max difference 0.00e+00, zero look-ahead confirmed)
  - [x] Pytest suite in `tests/` passed (9 passed)
  - [x] `data/processed/dataset.parquet` generated (7,118 rows)
  - [x] Phase 3 committed and pushed

- [x] **Phase 4: Split and Simple Baselines**
  - [x] Strict chronological split (Base, FT, Val, Test) with 8-bar gaps in `src/split.py`
  - [x] Split indices and boundaries saved to `results/logs/split.json`
  - [x] Evaluation protocol frozen in `docs/evaluation_protocol.md`
  - [x] Simple baselines evaluated on Val in `src/baselines.py`
  - [x] Baseline table generated in `results/tables/baselines_val.csv`
  - [x] Phase 4 committed and pushed

- [x] **Phase 5: Models, Training and Fine-Tuning**
  - [x] Stage A: Base model trained, CV sanity checked (CV loss: 0.7426), S0 evaluated on Val (0.70176)
  - [x] Stage B Part A: 4-fold walk-forward hyperparameter search on FT (40 configs, best CV loss: 0.69423)
  - [x] Stage B Part B: Comparison of strategies S0, S1, S2, S3 on Val (S2 won with 0.68953, beating constant prior by 0.00225)
  - [x] Stage C: Conventional ML comparators (Logistic Regression: 0.69109, Random Forest: 0.69092, HistGradientBoosting: 0.69276)
  - [x] Stage D: XGBoost ablations (time, context, rvol, class balance, label thresholds)
  - [x] Stage E: Feature importance (gain & permutation) on Val
  - [x] Final model chosen (XGBoost S2), saved to `models/model.joblib`, fit ledger verified leak-free
  - [x] `results/tables/finetune_report.md` written
  - [x] Phase 5 committed and pushed

- [x] **Phase 6: Calibration and Signal Rule**
  - [x] Platt scaling vs Isotonic calibration fitted on Val in `src/calibrate.py` (Platt chosen: ECE 0.0105, log-loss 0.68704)
  - [x] Reliability diagram and Brier / ECE computed before/after (`results/figures/calibration_reliability.png`)
  - [x] Directional diagnostics evaluated and saved (`results/tables/directional_diagnostics.csv`)
  - [x] Joint grid search for asymmetric margins `(d_up, d_down)` on Val (selected Up-only with d_up=0.06; best symmetric d_up=d_down=0.015 reported)
  - [x] Calibrator and signal rule frozen in `models/calib.joblib` (119 Up, 0 Down, 483 No-signal on Val; 61.34% accuracy)
  - [x] Phase 6 committed and pushed

- [x] **Phase 7: Final Evaluation (Once)**
  - [x] Held-out Test block read exactly ONCE in `src/evaluate.py`
  - [x] Deployed model, comparators, and baselines evaluated (`results/tables/final_results.csv`)
  - [x] Full classification metrics, confusion matrices, Wilson 95% CIs ([39.30%, 53.41%])
  - [x] Trading simulation (2, 5, 10 bps) with block-bootstrap CI ([-13.90, -0.02] bps)
  - [x] Variance analysis across 5 seeds (46.50% +/- 0.85%)
  - [x] Saved predictions to `results/tables/test_predictions.csv`
  - [x] Pre-registered criteria evaluated and verdict honestly recorded in `RESULTS.md` (C1, C4, C6 Met; C2, C3 Not Met; C5 Limitation Stated)
  - [x] Phase 7 committed and pushed

- [x] **Phase 8: Error Analysis, Explainability, Robustness**
  - [x] Test predictions analyzed (10 most confident errors, representative charts: `results/figures/signals_sample_chart.png`)
  - [x] Error breakdowns and seed flipping analysis in `src/analysis.py` (`results/figures/error_breakdown.png`)
  - [x] Global SHAP feature contributions on Val (`results/figures/shap_summary_val.png`)
  - [x] Robustness tests on Val (Gaussian noise sensitivity: +0.0032 log-loss at 25% noise)
  - [x] `docs/error_analysis.md` written with structural limitations
  - [x] Phase 8 committed and pushed

- [ ] **Phase 9: Integrity and Reproducibility**
  - [ ] Shuffled-label leak test passed (~50% accuracy on Val)
  - [ ] Time-shift test passed (performance degradation with shifted features)
  - [ ] Truncation test verified
  - [ ] Fit ledger audited (no Val/Test timestamps used in fits)
  - [ ] Live-compatibility feature test (200 & 500 bar windows vs full history)
  - [ ] End-to-end `run_all.sh` verified
  - [ ] Secret check on entire git tree and history verified
  - [ ] Phase 9 committed and pushed

- [ ] **Phase 10: Documentation, Report, Slides, Freeze**
  - [ ] Architecture diagram generated (`results/figures/architecture.png`)
  - [ ] Algorithm specification written (`docs/algorithm.md`)
  - [ ] Complete technical report written (`report/report.md`)
  - [ ] Slide deck outline written (`slides/outline.md`)
  - [ ] Trader user guide written (`docs/user_guide.md`)
  - [ ] Top-level `README.md` complete with reproduction commands
  - [ ] Artifacts frozen and committed
  - [ ] Release tagged as `model-v1` and tag pushed
  - [ ] Final summary printed

---

## Final IMPACT Completion Checklist

- [ ] 1. Problem and metrics defined (with pre-registered criteria)
- [ ] 2. Literature and gap documented with verified citations
- [ ] 3. Dataset source and licence recorded in dataset card
- [ ] 4. Data audit and EDA completed (Base-only EDA)
- [ ] 5. Leak-free split and pipeline verified (no look-ahead, 8-bar gaps)
- [ ] 6. Simple baselines established on identical validation set
- [ ] 7. Models tuned using walk-forward validation only
- [ ] 8. Held-out evaluation completed strictly once on Test
- [ ] 9. Results and failures honestly interpreted with statistical intervals
- [ ] 10. Seeds, versions, hardware, and exact commands recorded
- [ ] 11. Code, report, and slides organised without placeholders
- [ ] 12. Limitations, ethical considerations, and paper trading scope included

---

## Decisions and Assumptions

- *2026-11-10*: Project initialized under IMPACT Lab guidelines.
- *Workspace*: Fresh workspace at `/Users/chinmay/Desktop/trade final`.
- *Python Environment*: Target Python 3.11.18 located at `/opt/homebrew/bin/python3.11`.
- *Earlier 2-year run*: No prior files existed in directory; documented the motivating result from prior 2-year run per instructions.
