# Project Progress and Tracking

## Motivating Context: Earlier 2-Year Run
Earlier 2-year run: test set 587 eligible hourly candles; 114 Up signals, 0 Down signals, 473 No signal; signal accuracy 53.5% (about 61 of 114 correct); thresholds were p>=0.56 (Up) and p<=0.44 (Down). Likely causes: (a) the Down threshold was never reached, so the indicator was effectively Up-only and mostly rode upward drift; (b) 114 signals gives a 95% interval of roughly 44%-63%, i.e. no demonstrable edge; (c) only about 2 years of data.
*Disclosure Note*: In the final report, disclose that the new Test block overlaps the newest period of that earlier test set, so the live test is the only fully clean judge.

---

## Phase Checklist

- [ ] **Phase 0: Setup**
  - [ ] Repository layout created (`configs/`, `data/raw/`, `data/processed/`, `src/`, `results/tables/`, `results/figures/`, `results/logs/`, `models/`, `report/`, `slides/`, `docs/`, `tests/`, `instructions/`, `archive/`)
  - [ ] Configuration created (`configs/config.py`)
  - [ ] Git repository initialized, `.gitignore` created before first commit
  - [ ] Secret detection tool created (`src/check_secrets.py`) and verified
  - [ ] Virtual environment and dependencies installed (`requirements.txt`)
  - [ ] Remote `origin` configured (`https://github.com/chinmay-tammana/new-trade.git`)
  - [ ] Initial commit and push

- [ ] **Phase 1: Problem Definition and Literature**
  - [ ] `docs/problem_definition.md` created with pre-registered criteria
  - [ ] `docs/literature_survey.md` created with verified citations
  - [ ] Project title finalized
  - [ ] Phase 1 committed and pushed before training code

- [ ] **Phase 2: Data Collection and Audit**
  - [ ] Alpaca credentials verified from `.env`
  - [ ] Availability test on 1-min SIP bars completed
  - [ ] 4 years + 30 days of 1-minute data for GS, SPY, XLF downloaded
  - [ ] Session-anchored 1h bars constructed (09:30 to 16:00 ET)
  - [ ] Data quality audit completed (`results/logs/data_validation.txt`, `results/tables/audit_summary.csv`)
  - [ ] Parity check with yfinance on recent ~60 days completed
  - [ ] `docs/dataset_card.md` and `docs/data_dictionary.md` written
  - [ ] Audit figures generated in `results/figures/`
  - [ ] Phase 2 committed and pushed

- [ ] **Phase 3: EDA, Features, Labels, Preprocessing**
  - [ ] Base-only EDA completed with figures in `results/figures/` and `docs/eda_findings.md`
  - [ ] Drift check (Base vs FT Kolmogorov-Smirnov) completed
  - [ ] `src/features.py` implemented (`make_features`)
  - [ ] `src/labels.py` implemented (fwd simple return, overnight NaN, decisive filtering)
  - [ ] `src/preprocessing.py` implemented (fold-isolated transformers)
  - [ ] Truncation test passed (no look-ahead)
  - [ ] Pytest suite in `tests/` passed
  - [ ] `data/processed/dataset.parquet` generated
  - [ ] Phase 3 committed and pushed

- [ ] **Phase 4: Split and Simple Baselines**
  - [ ] Strict chronological split (Base, FT, Val, Test) with 8-bar gaps in `src/split.py`
  - [ ] Split indices and boundaries saved to `results/logs/split.json`
  - [ ] Evaluation protocol frozen in `docs/evaluation_protocol.md`
  - [ ] Simple baselines evaluated on Val in `src/baselines.py`
  - [ ] Baseline table generated in `results/tables/`
  - [ ] Phase 4 committed and pushed

- [ ] **Phase 5: Models, Training and Fine-Tuning**
  - [ ] Stage A: Base model trained, CV sanity checked, S0 evaluated on Val
  - [ ] Stage B Part A: 4-fold walk-forward hyperparameter search on FT (40 configs)
  - [ ] Stage B Part B: Comparison of strategies S0, S1, S2, S3 on Val
  - [ ] Stage C: Conventional ML comparators (Logistic Regression, Random Forest, HistGradientBoosting)
  - [ ] Stage D: XGBoost ablations (time, context, rvol, class balance, label thresholds)
  - [ ] Stage E: Feature importance (gain & permutation) on Val
  - [ ] Final model chosen, saved to `models/`, fit ledger updated
  - [ ] `results/tables/finetune_report.md` written
  - [ ] Phase 5 committed and pushed

- [ ] **Phase 6: Calibration and Signal Rule**
  - [ ] Platt scaling vs Isotonic calibration fitted on Val in `src/calibrate.py`
  - [ ] Reliability diagram and Brier / ECE computed before/after
  - [ ] Directional diagnostics evaluated and saved (`results/tables/directional_diagnostics.csv`)
  - [ ] Joint grid search for asymmetric margins `(d_up, d_down)` on Val
  - [ ] Calibrator and signal rule frozen in `models/calib.joblib`
  - [ ] Phase 6 committed and pushed

- [ ] **Phase 7: Final Evaluation (Once)**
  - [ ] Held-out Test block read exactly ONCE in `src/evaluate.py`
  - [ ] Deployed model, comparators, and baselines evaluated
  - [ ] Full classification metrics, confusion matrices, Wilson 95% CIs
  - [ ] Trading simulation (2, 5, 10 bps) with block-bootstrap CI
  - [ ] Variance analysis across 5 seeds
  - [ ] Regime breakdowns (hour, DOW, volatility, confidence)
  - [ ] Saved predictions to `results/tables/test_predictions.csv`
  - [ ] Pre-registered criteria evaluated and verdict recorded in `RESULTS.md`
  - [ ] Phase 7 committed and pushed

- [ ] **Phase 8: Error Analysis, Explainability, Robustness**
  - [ ] Test predictions analyzed (10 most confident errors, representative charts)
  - [ ] Error breakdowns and seed flipping analysis in `src/analysis.py`
  - [ ] Global SHAP / TreeSHAP contributions and permutation importance on Val
  - [ ] Robustness tests on Val (Gaussian noise, era shift)
  - [ ] `docs/error_analysis.md` written with limitations
  - [ ] Phase 8 committed and pushed

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
