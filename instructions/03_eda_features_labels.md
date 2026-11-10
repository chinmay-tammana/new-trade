# Phase 3: EDA, Features, Labels, Preprocessing

======================================================================
PHASE 3: EDA, FEATURES, LABELS, PREPROCESSING (IMPACT Steps 5-6)
======================================================================
EDA (src/eda.py): use ONLY the BASE block for data-dependent decisions. Produce: sample counts, missing-value summary, summary statistics; target class-count plot (Up/Down/decisive/neutral); feature distributions; correlation heatmap; autocorrelation of ret_1 and |ret_1|; mean Up-rate by hour of day and day of week; drift check comparing Base vs FT feature distributions (KS statistic per feature; do NOT look at Val/Test). Short interpretation under every figure; record insights in docs/eda_findings.md. Optional notebook for EDA only.

features.py exposes `make_features(df_gs, df_spy, df_xlf)` and FEATURES. Rules: stationary (returns, ratios, bounded oscillators; never raw dollar prices); causal (bar t uses only data up to and including the close of bar t); computable live from the last 200-500 closed bars. Compute SPY and XLF returns on their own series BEFORE merging.
- Returns: ret_1, ret_2, ret_3, ret_6, ret_14.
- Overnight: gap = open / previous close - 1.
- Momentum: rsi (Wilder 14), macd_n = MACD(12,26)/close, macd_hist_n = (MACD - signal(9))/close.
- Volatility: bb_pos (%B, 20, 2 std), bb_width, atr_n = ATR(14)/close, vol_20 = rolling std of ret_1 over 20 bars.
- Volume: rvol = volume / rolling 20-bar mean volume.
- Time (ET): hour = bar start hour + minute/60, dow, is_first_bar.
- Market context: spy_ret_1, spy_ret_3, xlf_ret_1, xlf_ret_3, rel_ret_1 = ret_1 - spy_ret_1.
Do not blind-dropna(). Drop the first warmup_rows rows; print NaN counts per column first.

preprocessing.py (IMPACT Step 6): XGBoost needs no scaling or clipping (tree-based). For logistic regression use a sklearn Pipeline [Winsorizer(0.5/99.5 percentiles, fitted in-fold) -> StandardScaler -> model]. The pattern raw data -> split -> fit transformer on the training fold -> transform validation/test is mandatory for every model. No target transforms.

labels.py:
- fwd = close[t+1] / close[t] - 1 (a simple return; call it that everywhere).
- fwd = NaN for the last bar of each trading day (found per ET date) and for any bar whose next bar is on a different date, so overnight moves are never labelled. Live, the indicator shows "No signal" on that bar.
- y = 1 if fwd > 0 else 0 (report the count of fwd == 0).
- decisive = |fwd| > label_threshold. Models TRAIN only on decisive rows. EVALUATION always uses ALL labelled rows (live, every candle is scored).
- Sample weights in training: sample_weight = base_weight * class_weight, where class_weight gives Up and Down equal total weight within the training rows (if class_balance_weights); use class_weight='balanced' for sklearn models that lack sample_weight.
- Print Up/Down/decisive shares for Base and FT only (not Val/Test).
Truncation test: for about 50 random bars t, recompute features on data cut at t and assert the row at t equals the full-data value (no look-ahead).
Save data/processed/dataset.parquet (features, fwd, y, decisive, block label).
CHECK: truncation test passes; no inf; no constant feature; no label on last-bar-of-day rows; EDA figures with interpretations exist; pytest passes.
