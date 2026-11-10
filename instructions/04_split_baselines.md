# Phase 4: Split and Simple Baselines

======================================================================
PHASE 4: SPLIT AND SIMPLE BASELINES (src/split.py, src/baselines.py) (IMPACT Steps 4, 7)
======================================================================
split.py (positions in BARS after warm-up; label-NaN rows kept; filter to labelled rows inside each block afterwards). N = number of bars; n4 = N - i4.
- Base = [0, i3 - gap); FT = [i3, i4 - gap); Val = [i4, i4 + int(0.40*n4)); Test = [val_end + gap, N).
- Assert Base.max_time < FT.min_time < FT.max_time < Val.min_time < Val.max_time < Test.min_time and every gap >= gap_bars bars.
- Save boundary dates, bar counts and labelled-row counts per block (counts only, no label shares) and the split indices to results/logs/split.json. FREEZE the protocol: write docs/evaluation_protocol.md before any training (role of each block, metrics, selection rules, pre-registered criteria reference).
Simple baselines, scored on the SAME Val rows (Test only in Phase 7), all labelled rows:
1. Constant prior: probability = Base Up-rate for every candle (provides a log-loss and Brier reference). 2. Always-Up. 3. Majority class of Base. 4. RSI rule (Up if rsi<40, Down if rsi>60, No signal otherwise; accuracy on signals plus coverage). 5. Previous-bar persistence (Up if ret_1>0 else Down).
For every baseline report accuracy, coverage, number of signals, precision/recall/F1 per class, net return per signal after costs (2, 5, 10 bps) and the Wilson 95% CI; log-loss and Brier only where a probability exists (constant prior); show "-" otherwise.
CHECK: split assertions pass; evaluation protocol committed; baseline table on Val exists.
