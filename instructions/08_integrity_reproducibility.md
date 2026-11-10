# Phase 9: Integrity and Reproducibility

======================================================================
PHASE 9: INTEGRITY AND REPRODUCIBILITY (src/checks.py) (IMPACT Step 12)
======================================================================
1. Shuffled-label test: run the full training pipeline on shuffled training labels and evaluate on Val only; Val accuracy must be about 50%. If not, there is a leak.
2. Time-shift test: shift features by one bar and evaluate on Val; performance should drop if the model truly uses current information.
3. Truncation test (Phase 3) re-run on the final pipeline.
4. Ledger audit: assert every fit's last timestamp is before Val.min_time (calibration reads Val but fits no model rows).
5. Reproducibility: in a clean clone of the GitHub repo, copy in the local data/ folder (or re-download with .env keys) and run `run_all.sh`; confirm identical metrics.
6. Live-compatibility: run make_features on only the last 200 and 500 bars and compare the latest row with the full-history value; report how the EMA differences decay and the recommended live window.
7. Verify a mentor could regenerate every table and plot from README.md without hidden steps.
8. Final secret check on the full tracked tree and on `git log -p` history; confirm no data files or .env were ever committed.
CHECK: all pass or failures explained in progress.md.
