# Problem Definition and Pre-Registered Success Criteria (V2 Upgrade)

**Target Asset**: The Goldman Sachs Group, Inc. (`GS`, NYSE)  
**Document Status**: Pre-Registered and Frozen Prior to Research Holdout Access  
**Framework**: Walk-Forward Out-of-Fold (OOF) Modeling with Sealed Research Holdout  

---

## 1. Mathematical Formulation and Horizons

Let $C_t$ denote the closing price of session-anchored candle $t$. The model evaluates three directional prediction heads:
1. **Horizon H1 (Next 1-Hour Bar)**:
   $$F_t^{\text{H1}} = \frac{C_{t+1} - C_t}{C_t}, \quad \text{for } t \in \{0, 1, 2, 3, 4, 5\}$$
   Decisive threshold: $\delta_{\text{H1}} = 10\text{ bps} = 0.0010$. Censors final session bar (15:30) and overnight moves ($F = \text{NaN}$).
2. **Horizon H3 (Next 3-Hour Bars, strictly intra-day)**:
   $$F_t^{\text{H3}} = \frac{C_{t+3} - C_t}{C_t}, \quad \text{for } t \in \{0, 1, 2, 3\}$$
   Decisive threshold: $\delta_{\text{H3}} = 10 \cdot \sqrt{3} \approx 17.32\text{ bps}$.
3. **Horizon HD (Rest of Day Session Close)**:
   $$F_t^{\text{HD}} = \frac{C_{\text{day\_close}} - C_t}{C_t}, \quad \text{for } t \in \{0, 1, 2, 3, 4\}$$
   Decisive threshold: $\delta_{\text{HD}} = 10 \cdot \sqrt{6 - \text{session\_offset}}\text{ bps}$.

### Decisive Directional Binary Target
For each horizon $H \in \{\text{H1}, \text{H3}, \text{HD}\}$:
$$Y_t^H = \begin{cases} 1 & \text{if } F_t^H > +\delta_H \quad (\text{Up}) \\ 0 & \text{if } F_t^H < -\delta_H \quad (\text{Down}) \\ \text{Indecisive} & \text{if } |F_t^H| \le \delta_H \end{cases}$$

Training is conducted exclusively on decisive candles ($|F_t| > \delta$). Evaluation is conducted across all labelled candles where the signal rule emits an action ($s_t \ne 0$).

---

## 2. Insight Heads Formulation

1. **VOL (Next-Bar High-Low Range Prediction)**:
   - Target: $R_{t+1} = \log\left(\frac{High_{t+1} - Low_{t+1}}{Close_{t+1}}\right)$.
   - Output: Categorized into `High`, `Normal`, `Low` volatility terciles based on pre-holdout OOF distribution.
2. **BAND (Next-Bar Quantile Return Band)**:
   - Quantile regression models predicting the 10th percentile ($q = 0.10$) and 90th percentile ($q = 0.90$) of forward 1-hour return.
3. **REGIME (Macro Trend & Chop Classification)**:
   - Categorical tag $\in \{\text{Uptrend}, \text{Downtrend}, \text{Chop}\}$ based on the sign of trailing 20-day return (`ret_20d`) and market efficiency ratio (`efficiency_ratio_20`).

---

## 3. Pre-Registered Choice of Primary Direction Horizon

The primary direction horizon among H1, H3, and HD is selected by:
$$\arg\max_{H \in \{\text{H1}, \text{H3}, \text{HD}\}} \text{WilsonLower95}(\text{SignalAccuracy}_{\text{OOF}}^H)$$
subject to:
1. Coverage $c \ge 20.0\%$.
2. Both sides must be active with $n_{\text{up}} \ge 150$ and $n_{\text{down}} \ge 150$ on pooled pre-holdout OOF.

---

## 4. Pre-Registered Acceptance Criteria (V2 Upgrade)

These criteria are pre-registered prior to running final inference on the Research Holdout and **will never be altered post-hoc**:

- **Criterion A (Pooled OOF Log-Loss)**:
  Pooled OOF log-loss of the chosen model must improve upon the constant-prior baseline.
- **Criterion B (Holdout Signal Count & Accuracy)**:
  On the held-out Research Holdout:
  - $\ge 100$ signals per side ($n_{\text{up}} \ge 100$ and $n_{\text{down}} \ge 100$).
  - Signal accuracy $\ge 53.0\%$ with permutation test $p < 0.10$ (10,000 permutations), OR a plain statement that no statistical edge was shown.
- **Criterion C (Per-Side Drift-Adjusted Accuracy)**:
  Each active side must outperform its own period base rate on the holdout (Up accuracy $>$ Base Up-rate, Down accuracy $>$ Base Down-rate), or the directional limitation must be stated plainly.
- **Criterion D (Calibration Quality)**:
  Expected Calibration Error (ECE) over 10 bins on the holdout must satisfy $\text{ECE} \le 0.050$.
- **Criterion E (Data & Pipeline Integrity)**:
  All integrity checks must pass: causal truncation test discrepancy $\le 10^{-8}$, shuffled-label OOF accuracy within $[45\%, 55\%]$, time-shift degradation verified, fit ledger timestamp audit verified, and zero secrets committed.
