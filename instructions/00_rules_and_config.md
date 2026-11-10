# Phase 0: Rules, Architecture, and Configuration

You are a senior ML engineer. Build, validate, document and freeze a trained model for a Goldman Sachs (GS) AI trading indicator, following the IMPACT Lab (IIT Patna) End-to-End ML Project Guidelines. Work autonomously, phase by phase, in the current repository folder (the Codex project folder). Do NOT build the live backend or frontend; the task ends when the model, results, report draft and slides outline are frozen and pushed to GitHub.

GitHub repository (public, currently empty): https://github.com/chinmay-tammana/new-trade.git (branch: main)

======================================================================
GITHUB RULES
======================================================================
- Remote: https://github.com/chinmay-tammana/new-trade.git, branch main. The repository is PUBLIC, so anything pushed is visible to everyone.
- Setup (Phase 0): if no `.git` exists run `git init` and `git branch -M main`; otherwise inspect `git status`, `git log --oneline` and `git remote -v` first. Add `origin` with the URL above (if an `origin` with a different URL already exists, STOP and ask). Run `git ls-remote origin`: if the remote already has commits, STOP and ask; never overwrite it.
- If a local history with earlier commits already exists, inspect `git log --stat` before the first push. If any earlier commit contains `.env`, API keys, or data/parquet files, STOP and ask the user instead of pushing.
- If git user.name or user.email are not configured, ask the user; never invent an email address.
- NEVER force-push and never rewrite published history.
- NEVER commit: `.env`, API keys or tokens, raw or processed market data (data/ and any *.parquet), virtualenvs, caches. Market data is not committed because the data provider's terms may forbid redistribution; the README explains how to download it. DO commit: code, instructions/, docs/, configs/, tests/, results/tables, results/figures, results/logs, models (small), report/, slides/, README.md, requirements.txt, .env.example.
- .gitignore must exist BEFORE the first commit and include at least: `.env`, `.env.*` (but keep `!.env.example`), `data/`, `*.parquet`, `.venv/`, `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`. Verify with `git check-ignore -v .env`.
- Secret check: create `src/check_secrets.py` that loads the values of APCA_API_KEY_ID and APCA_API_SECRET_KEY from `.env` and fails (without printing the values) if either value, or the patterns `ghp_`, `github_pat_`, `APCA_API_SECRET_KEY=` followed by a non-empty value, appears in any tracked file or in the staged diff. Run it before EVERY commit. Also confirm `git ls-files` contains no `.env`, no *.parquet and nothing under data/.
- After each phase: run the phase CHECK, run the secret check, commit "phase N: <name>", then `git push -u origin main` (first push) / `git push` afterwards.
- If a push fails because of authentication or network: do NOT retry in a loop, do NOT ask for or store tokens, and NEVER put a token in the remote URL or any file. Keep committing locally, log it in progress.md, and tell the user to authenticate (for example `gh auth login`, GitHub Desktop, or the macOS keychain credential helper) and then run `git push` themselves.
- At the end: tag `model-v1` and push the tag.

======================================================================
GLOBAL RULES
======================================================================
- Asset: GS only. Bar size: 1h. Horizon: next 1h bar. Context series: SPY and XLF. Task type: binary classification (Up/Down) with abstention (No signal). Paper trading only.
- Python 3.10-3.12, fresh virtualenv. Install: pandas numpy scikit-learn xgboost pyarrow scipy joblib matplotlib pytest alpaca-py python-dotenv yfinance psutil. Pin installed versions in requirements.txt. No pandas-ta; compute every indicator in plain pandas/numpy so `make_features()` is reusable by a future live backend.
- No placeholders, TODOs or pseudocode in code. Every script runs end to end. Paths and settings configurable in configs/config.py. No hidden manual preprocessing.
- Seed 42 everywhere; final fit deterministic (random_state=42, n_jobs=1). Record software versions and hardware (OS, CPU, RAM) in results/logs/environment.json.
- Fit ledger: every model fit (CV folds, ablations, checks, final fits) appends to results/logs/fit_ledger.json: purpose, first and last timestamp of rows used, row count.
- Experiment log (IMPACT Step 8): results/tables/experiment_log.csv with columns run_id, split, seed, model, hyperparameters, cv_metric, val_metric, runtime_s. Every search configuration and baseline run gets a row (E01, E02, ...).
- Every figure: labelled axes with units, a caption, and a 1-3 sentence interpretation written under it in the docs/report. Save to results/figures/.
- TEST-SET DISCIPLINE (IMPACT stop sign): the Test block is read only in Phase 7, once. No tuning, thresholding, feature selection, normalisation fitting or model selection may use test information. Class balance of Val is first used in Phase 6 and of Test in Phase 7. All preprocessing (scaling, clipping, imputation, feature selection) is fitted inside training folds only (sklearn Pipelines/transformers).
- Write pytest tests in tests/ for features, labels, splits, Wilson interval, calibration and signal-rule helpers.
- Every methodological decision must be justified, recorded (progress.md) and verified (a check).

======================================================================
DATA DESIGN (4 YEARS)
======================================================================
Source: Alpaca Market Data (free Basic plan: full-market SIP bars older than 15 minutes; history since 2016). yfinance cannot supply this (1h capped at about 730 days) and is used only for a parity check.
Eras (chronological, never shuffled), defined from t_end = last GS timestamp, with era_days = 365:
- BASE (training): everything before y3_start = t_end - 2*era_days (about the oldest 2 years).
- FINE-TUNING (FT) block: from y3_start to y4_start (Year 3).
- YEAR 4 (newest ~12 months, from y4_start = t_end - era_days): VALIDATION = first 40% of Year-4 bars; TEST = last 60%.
- Leave a gap of gap_bars (8) bars between adjacent blocks: Base->FT, FT->Year 4, Val->Test.

|---- Base: oldest ~2y ----|gap|--- FT: year 3 ---|gap|-- Val: 40% of year 4 --|gap|-- Test: 60% of year 4 --|
 oldest                                                                                           newest

Roles: Base fits the base model. FT: hyperparameter search (walk-forward) and fine-tuning strategies. Val: calibration, signal rule, model/strategy selection, ablations. Test: final report, once.
Approximate labelled sizes: Base ~3,000; FT ~1,500; Val ~600; Test ~850.

======================================================================
REPO LAYOUT (IMPACT Step 12)
======================================================================
README.md  requirements.txt  .gitignore  .env.example  run_all.sh  AGENTS.md
configs/config.py
data/raw/  data/processed/        (git-ignored)
notebooks/                         (EDA only, optional)
src/__init__.py data_fetch.py audit.py eda.py features.py labels.py preprocessing.py split.py
    baselines.py models.py train_base.py finetune.py calibrate.py evaluate.py analysis.py
    checks.py diagrams.py check_secrets.py utils.py
results/tables/  results/figures/  results/logs/
models/  report/  slides/  docs/  tests/  instructions/  archive/
Run modules as `python3 -m src.<module>`; `run_all.sh` runs the whole pipeline in order.

======================================================================
configs/config.py (create exactly this; extend only if needed)
======================================================================
```python
CONFIG = {
    "ticker": "GS", "context_tickers": ["SPY", "XLF"],
    "interval": "1h", "tz": "America/New_York",
    "history_years": 4, "history_buffer_days": 30,
    "alpaca_feed": "sip",                 # "sip" (default) or "iex"
    "alpaca_end_lag_minutes": 20,         # free SIP needs data older than 15 min
    "session_open": "09:30", "session_close": "16:00",
    "seed": 42, "horizon_bars": 1, "gap_bars": 8, "warmup_rows": 30,
    "label_threshold": 0.0010,
    "label_threshold_candidates": [0.0005, 0.0010, 0.0015, 0.0020],
    "era_days": 365, "year4_val_frac": 0.40,
    "ft_cv_folds": 4,
    "n_iter_search": 40,
    "base_weight_options": [0.3, 0.6, 1.0],
    "warm_start_rounds_options": [50, 100, 200],
    "warm_start_lr_options": [0.01, 0.02, 0.05],
    "class_balance_weights": True,
    "n_seeds_variance": 5,
    "margin_sweep": [0.01, 0.15, 0.005],  # start, stop, step; used for d_up and d_down
    "min_coverage": 0.15, "min_signals": 40, "min_signals_per_side": 25,
    "switch_margin_logloss": 0.002,       # a comparator must beat XGBoost on Val by this to be deployed
    "costs_bps": [2, 5, 10], "primary_cost_bps": 5,
    "rsi_buy_below": 40, "rsi_sell_above": 60,
    "github_remote": "https://github.com/chinmay-tammana/new-trade.git",
    "paths": {"raw_dir": "data/raw", "processed": "data/processed/dataset.parquet",
              "models": "models", "tables": "results/tables", "figures": "results/figures",
              "logs": "results/logs", "split": "results/logs/split.json",
              "test_predictions": "results/tables/test_predictions.csv"},
}
```
XGBoost search space: max_depth 2-5, learning_rate 0.01-0.1, n_estimators 100-500, subsample 0.6-1.0, colsample_bytree 0.6-1.0, min_child_weight 1-20, reg_lambda 1-20, gamma 0-2, plus base_weight from config. Keep models shallow and regularized.

======================================================================
PHASE 0: SETUP
======================================================================
Create the layout, config, .gitignore, .env.example (APCA_API_KEY_ID=, APCA_API_SECRET_KEY=), requirements.txt, progress.md, src/check_secrets.py. Archive any earlier implementation (Step 0 item 5). Configure git and the remote exactly as in GITHUB RULES. Make the first commit and push.
CHECK: `python3 -c "from configs.config import CONFIG"` works; layout exists; `git check-ignore -v .env` confirms .env is ignored; secret check passes; first push succeeded or the failure is logged.
