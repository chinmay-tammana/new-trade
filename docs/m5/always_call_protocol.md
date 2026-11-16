# Always-Call Signal Protocol (M5 5-Minute System)

**Date**: 2026-11-16  
**Status**: PRE-REGISTERED PRIOR TO EXECUTION  
**Reference**: `instructions/22_always_call.md` Section C

---

## 1. Owner Decision & Purpose

**Owner Decision (verbatim, dated 2026-11-16)**:
> *"There is still too much No signal. Keep No signal minimal, or at least reduce it a lot." Interpretation: every eligible 5-minute bar gets an Up or Down call with a confidence tier. No signal remains ONLY for: the end-of-day bar (no forward interval in the same session), a data gap (a required symbol has no minute data), market closed, or an exact p_cal tie at 0.5. This is a SIGNAL-RULE change, NOT a model change.*

**Observed Baseline Before Change (verbatim)**:
> OOF accuracy 62.2% at 30% coverage; holdout 59.82% over 1,827 signals with 19.3% coverage; and the inconsistent figures "No-Signal 63,041 bars (70.0%)" with 74,971 bars (that is 84.1%).

---

## 2. Signal Generation Logic

For every 5-minute bar at timestamp $t$:
1. **End-of-Day Bar**: If $t \ge 15:55$ ET (bar 15:55–16:00), there is no forward 5-minute interval within the same session. Emit `No signal (end of day)` (confidence = null, tier = null).
2. **Data Gap**: If any required instrument has missing minute data. Emit `No signal (data gap)`.
3. **Market Closed**: Market outside 09:30–16:00 ET. Emit `Standby / Market Closed`.
4. **Exact Tie**: If $p_{\text{cal}} == 0.500000$. Emit `No signal (exact tie)`.
5. **Directional Calls**:
   - If $p_{\text{cal}} > 0.50$: **`Up`**, displayed confidence $= p_{\text{cal}}$.
   - If $p_{\text{cal}} < 0.50$: **`Down`**, displayed confidence $= 1.0 - p_{\text{cal}}$.

---

## 3. Pre-Registered Confidence Tiers

Tier cutoffs are strictly computed from the pooled pre-holdout OOF confidence distribution ($\text{conf} = \max(p_{\text{cal}}, 1 - p_{\text{cal}})$) across all eligible validation bars:
- **`STRONG`**: $\text{Confidence} \ge \text{Quantile}(\text{conf}_{\text{OOF}}, 0.75)$ (Top quartile of conviction).
- **`NORMAL`**: $\text{Quantile}(\text{conf}_{\text{OOF}}, 0.25) < \text{Confidence} < \text{Quantile}(\text{conf}_{\text{OOF}}, 0.75)$ (Middle 50%).
- **`WEAK`**: $\text{Confidence} \le \text{Quantile}(\text{conf}_{\text{OOF}}, 0.25)$ ("close to a coin flip", bottom quartile).

---

## 4. Pre-Registered Acceptance Checks

All checks must be evaluated and reported as PASS or FAIL:
1. **Check (a) All-Bar Holdout Edge**: All-bar holdout accuracy must strictly exceed the larger empirical base rate ($\max(\text{Base}_{\text{Up}}, \text{Base}_{\text{Down}})$) with permutation $p < 0.05$. If not, state plainly that no skill is shown at full coverage.
2. **Check (b) Monotonic Tier Performance**: Directional accuracy must monotonically increase with conviction on both OOF and Holdout:
   $$\text{Accuracy}_{\text{WEAK}} < \text{Accuracy}_{\text{NORMAL}} < \text{Accuracy}_{\text{STRONG}}$$
   If not, state plainly that model confidence is not informative across tiers.
3. **Check (c) Probability Calibration**: Platt calibrator Expected Calibration Error $\text{ECE} \le 0.030$.
4. **Check (d) Two-Sided Presence**: Both Up and Down signals must be emitted with substantial volume ($n_{\text{up}} > 0, n_{\text{down}} > 0$).

---

## 5. Storage, Hash Lock & CLI Architecture

- Active rule configuration stored in `models/m5/rule_m5_v3.json` pointed to by `models/m5/active_rule.json`.
- `data/live/live_m5.db` signals table stores all re-derived signals under `rule_version = 3` (`late = 1`) with cryptographic hash chain verified.
- CLI: `python3 -m src.m5.apply_rule --mode always`.
