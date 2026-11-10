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

- [x] **Phase 9: Integrity and Reproducibility**
  - [x] Shuffled-label leak test passed (~50% accuracy on Val: 50.50%, loss 0.69493)
  - [x] Time-shift test passed (performance degradation with shifted features: Delta log-loss +0.00450)
  - [x] Truncation test verified (50 cutoffs, max diff 0.00e+00)
  - [x] Fit ledger audited (19 entries, 0 fits touched Val or Test timestamps)
  - [x] Live-compatibility feature test (200 bar diff = 5.36e-05, 500 bar diff = 1.07e-12; >=300 bars recommended)
  - [x] End-to-end `run_all.sh` verified
  - [x] Secret check on entire git tree and history verified (0 secrets, 0 data files tracked)
  - [x] Phase 9 committed and pushed

- [x] **Phase 10: Documentation, Report, Slides, Freeze**
  - [x] Architecture diagram generated (`results/figures/architecture.png`)
  - [x] Algorithm specification written (`docs/algorithm.md`)
  - [x] Complete technical report written (`report/report.md`)
  - [x] Slide deck outline written (`slides/outline.md`)
  - [x] Trader user guide written (`docs/user_guide.md`)
  - [x] Top-level `README.md` complete with reproduction commands
  - [x] Artifacts frozen and committed
  - [x] Release tagged as `model-v1` and tag pushed
  - [x] Final summary printed

---

## V2 UPGRADE (instructions/10_v2_upgrade.md)

- [x] **V2.0: Setup & Archiving**
  - [x] Save prompt verbatim as `instructions/10_v2_upgrade.md`
  - [x] Update `AGENTS.md` (read `10_v2_upgrade.md` last, overrides earlier files)
  - [x] Tag commit `v1-negative-result` and push tag
  - [x] Copy v1 outputs to `archive/v1_four_year_run/`
  - [x] Record v1 result and causes in `progress.md`
  - [x] Commit and push V2.0 setup

- [x] **V2.1: Data Expansion & Ingestion**
  - [x] Print system date/time and diagnose why v1 ended at 2025-10-15 (System time 2026-11-10; v1 stopped at 2025-10 in docs due to project brief spec, but Alpaca SIP feed is fully available through 2026-11-10)
  - [x] Re-run Alpaca availability test on recent dates (Verified 736 bars on recent date)
  - [x] Extend history to 8 years (Oct 2018 onward, 30 buffer days, monthly resumable chunks; 14,101 GS 1h bars)
  - [x] Add context symbols: SPY, XLF, JPM, MS, BAC, TLT, VIXY; resample to 1h; drop symbols losing >2% GS bars (All 7 kept; max lost: MS 0.07%, BAC 0.04%, TLT 0.04%, VIXY 0.04%, SPY/XLF/JPM 0.00%)
  - [x] Drop incomplete bars; save `data_end` to `configs/data_freeze.json`; set `freeze_date` and `test_day` in `configs/config.py` (data_end: 2026-11-09 15:30:00-05:00, 14,084 synchronized bars)
  - [x] Re-run audit; update dataset card & data dictionary (All 8 symbols audited, 0 NaNs, 0 zero vol, yfinance parity confirmed)

- [ ] **V2.2: Rich Causal Feature Engineering**
  - [ ] Peers, rates, vol features (JPM, MS, BAC, TLT, VIXY returns, gs_minus_peers, xlf-relative)
  - [ ] Intrabar features from 1-min bars (realized_vol_1m, up_minute_ratio, clv, signed_volume_ratio, last15_ret)
  - [ ] Levels and trend features (dist_prev_close/high/low in ATR units, dist_session_vwap, ret_1d/5d/20d, ma_dist_50h/200h, efficiency_ratio_20, bars_since_open/close)
  - [ ] Organize `FEATURE_GROUPS` dict for group ablations
  - [ ] Document live 1m requirements and SIP vs IEX differences in `docs/`
  - [ ] Pass causal truncation test for all new features

- [ ] **V2.3: Multi-Horizon Targets & Insight Heads**
  - [ ] Direction heads: H1 (1h), H3 (3h same day), HD (rest of day); decisive thresholds scale with sqrt(bars)
  - [ ] Insight head: VOL (XGB regressor on log((high-low)/close), High/Normal/Low terciles vs persistence & EWMA)
  - [ ] Insight head: BAND (Quantile regression q=0.10, 0.90 for return, coverage & width vs constant-width baseline)
  - [ ] Insight head: REGIME (Uptrend/Downtrend/Chop from ret_20d & efficiency_ratio_20)
  - [ ] Pre-register primary direction horizon selection rule (highest Wilson 95% lower bound on pooled OOF with >=150 signals/side and >=20% coverage)

- [ ] **V2.4: Walk-Forward OOF Evaluation Protocol & Holdout**
  - [ ] Define Research Holdout: 2025-10-16 to freeze_date (8-bar gap after pre-holdout; read ONCE)
  - [ ] Implement quarterly walk-forward OOF (initial 2 years, refit every 63 trading days with 8-bar gap)
  - [ ] Hyperparameter search (n_iter 30) on pre-holdout OOF; recency half-life sweep {90, 180, 365 days, none}; warm-start 12m comparison
  - [ ] Candidate tournament on pre-holdout OOF: shallow XGBoost, HistGradientBoosting, Elastic-Net Logistic, Ensemble
  - [ ] Feature group ablations on pre-holdout OOF log-loss
  - [ ] Class-balanced Platt scaling on pre-holdout OOF; reliability diagram & ECE overall and per regime

- [ ] **V2.5: Mandatory Two-Sided Signal Rule**
  - [ ] Quantile thresholds q_lo and q_hi from pooled OOF p_cal over coverage c in {0.15, 0.20, 0.25, 0.30, 0.40}
  - [ ] Select c by Wilson 95% lower bound subject to >=150 signals on EACH side
  - [ ] Assert n_up >= 150 and n_down >= 150 on OOF; on holdout assert both > 0
  - [ ] Displayed confidence = p_cal for Up, 1 - p_cal for Down (always >= 50%)

- [ ] **V2.6: Honest Statistics & Pre-Registration**
  - [ ] Create `docs/problem_definition_v2.md` with pre-registered criteria BEFORE reading holdout
  - [ ] Compute all-candle always-Up/always-Down, drift-adjusted per side, balanced accuracy, 10k permutation test p-value, regime/calendar breakdowns, net return at 2/5/10 bps with block-bootstrap CI

- [ ] **V2.7: Final Model Refit, Day Replay & Insight Backend**
  - [ ] Single-read evaluation on Research Holdout
  - [ ] Refit selected configuration on ALL data up to freeze_date (including holdout); recompute calibrator and q_lo/q_hi; save `model_v2.joblib`, `calib_v2.joblib`
  - [ ] Build `src/predict_day.py --date YYYY-MM-DD` bar-by-bar day replay with table and plot output
  - [ ] Build `src/insights.py` reusable inference function

- [ ] **V2.8: Checks, Documentation & GitHub Release**
  - [ ] Rerun integrity checks: truncation, shuffled-label on OOF, time-shift, ledger audit, live compatibility, synthetic two-sided unit test
  - [ ] Update all docs, reports, and slides
  - [ ] Run secret check, commit per phase, push, tag `model-v2` and push tag
  - [ ] Print final comprehensive summary

---

## Final IMPACT Completion Checklist

- [x] 1. **Problem and metrics defined (with pre-registered criteria)**: Defined in `docs/problem_definition.md` with explicit criteria C1–C6 (Log-loss, Accuracy, Net return, ECE, Directional balance, Integrity).
- [x] 2. **Literature and gap documented with verified citations**: Documented in `docs/literature_survey.md` with real papers (Fama 1970, Malkiel 2003, Gu et al. 2020, De Prado 2018, Guo et al. 2017) and structured comparison table.
- [x] 3. **Dataset source and licence recorded in dataset card**: Documented in `docs/dataset_card.md` and `docs/data_dictionary.md` covering Alpaca SIP source, licensing terms, and schemas.
- [x] 4. **Data audit and EDA completed (Base-only EDA)**: Verified in `src/audit.py` (0 NaNs, 0 zero vol, 0.999998 yfinance parity corr) and `src/eda.py` (Base-only EDA plots, KS drift analysis in `docs/eda_findings.md`).
- [x] 5. **Leak-free split and pipeline verified (no look-ahead, 8-bar gaps)**: Implemented in `src/split.py` with 8-bar embargoes. Causal truncation test (50 cutoffs, max diff 0.00e+00) and in-fold Winsorizer/StandardScaler verified.
- [x] 6. **Simple baselines established on identical validation set**: Constant Prior (loss 0.69178), Always-Up (53.16%), RSI heuristic (49.83%), and Persistence (45.02%) evaluated in `src/baselines.py` and saved to `results/tables/baselines_val.csv`.
- [x] 7. **Models tuned using walk-forward validation only**: 4-fold walk-forward cross-validation (40 configs) executed on FT block in `src/finetune.py`. Best strategy S2 (Refit Base+FT) beat Constant Prior on Val (loss 0.68953 vs 0.69178).
- [x] 8. **Held-out evaluation completed strictly once on Test**: Executed in `src/evaluate.py`. Test block read strictly once with frozen models and thresholds; predictions locked to `results/tables/test_predictions.csv`.
- [x] 9. **Results and failures honestly interpreted with statistical intervals**: Reported in `RESULTS.md` and `report/report.md` with 95% Wilson CIs (accuracy 46.28% [39.30%, 53.41%]) and block-bootstrap CIs (net return -7.24 bps [-13.90, -0.02] bps).
- [x] 10. **Seeds, versions, hardware, and exact commands recorded**: Recorded in `results/logs/environment.json`, `configs/config.py` (seed 42), `requirements.txt`, and executable `run_all.sh`.
- [x] 11. **Code, report, and slides organised without placeholders**: Complete implementation with zero TODOs or placeholders across `src/`, `docs/`, `report/report.md` (10 sections), and `slides/outline.md` (15 slides).
- [x] 12. **Limitations, ethical considerations, and paper trading scope included**: Thoroughly documented in `report/report.md`, `docs/user_guide.md`, and `README.md` (educational scope, paper trading only, not financial advice, licensing compliance).

---

## Decisions and Assumptions

- *2026-11-10*: Project initialized under IMPACT Lab guidelines.
- *v1 result*: Val 119 Up / 0 Down, 61.34% (73/119); Test 188 Up / 0 Down / 708 No signal, 46.28% (87/188), Wilson 95% CI [39.30%, 53.41%], always-Up on all test candles 48.88%, net return -7.24 bps per signal at 5 bps. Causes: (a) the Val window (Nov 2024 to Mar 2025) was a rally, so the Val search switched the Down side off and v1 allowed it; (b) the rule was tuned on 602 candles in one regime (selection optimism); (c) 'always-Up on the same candles' is identical to the accuracy of an Up-only rule, so its +0.00% excess was meaningless; (d) data stopped at 2025-10-15.
- *Workspace*: Fresh workspace at `/Users/chinmay/Desktop/trade final`.
- *Python Environment*: Target Python 3.11.18 located at `/opt/homebrew/bin/python3.11`.
- *Phase 9 Integrity*: All 6 integrity tests passed. Shuffled-label null accuracy is 50.50% (proving zero label leakage). Lagged feature shift increases loss by +0.00450. Fit ledger confirms all 19 model fits completed prior to Val start timestamp (2025-11-10 11:30:00-05:00). Live window test confirms 300+ bars are sufficient for full EMA convergence (500 bars yields < 1e-11 diff). Git history scan confirms zero secrets or parquet files tracked.
- *Phase 10 Documentation & Freeze*: Architecture diagram generated at `results/figures/architecture.png`. Formal algorithm specification written in `docs/algorithm.md`. 10-section IMPACT report authored in `report/report.md`. 15-slide technical outline created in `slides/outline.md`. Complete trader user guide created in `docs/user_guide.md`. Top-level `README.md` complete with architecture diagram, results scorecard, and reproduction instructions. Model artifacts frozen and tagged as `model-v1`.
- *Earlier 2-year run*: No prior files existed in directory; documented the motivating result from prior 2-year run per instructions.
