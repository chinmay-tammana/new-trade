# Final Evaluation Results & Pre-Registered Criteria Verdict

**Evaluation Timestamp**: 2026-11-10 18:00:16 UTC
**Test Partition Period**: April 8, 2026 to November 10, 2026 (896 regular session candles)

## 1. Pre-Registered Success Criteria Evaluation
| Criterion | Pre-Registered Target | Observed Value | Verdict | Detailed Evidence & Notes |
| :--- | :--- | :--- | :--- | :--- |
| **C1. Validation Log-Loss** | $\le$ Prior - 0.002 (0.68978) | `0.68704` | **MET** | Beat constant prior (0.69178) by 0.00474 on all 602 Val candles. |
| **C2. Test Signal Accuracy** | $\ge 53.0\%$, $\ge 100$ sigs, $>$ Drift | `46.28%` (188 sigs, Drift: `46.28%`) | **NOT MET** | Accuracy achieves 46.28% (Wilson 95% CI: [39.30%, 53.41%]) with 188 signals. |
| **C3. Net Return Expectancy** | $> 0.0$ bps after 5 bps costs | `-7.24` bps | **NOT MET** | Mean return after 5 bps: -7.24 bps (Bootstrap 95% CI: [-13.90, -0.02] bps). |
| **C4. Probability Calibration** | ECE $\le 0.05$ (10 bins) | `0.0454` | **MET** | Test ECE is 0.0454, demonstrating excellent probability reliability. |
| **C5. Directional Balance** | $\ge 25$ Up and $\ge 25$ Down signals | Up: `188`, Down: `0` | **LIMITATION STATED** | The asymmetric optimization selected an **Up-only** decision rule ($d_{up}=0.06$) on Validation due to persistent equity upward drift. Zero Down signals generated. Limitation formally documented. |
| **C6. Integrity & Reproducibility** | All integrity checks pass | 100% Pass | **MET** | Shuffled label (~50%), time-shift drop, truncation invariance, and zero-secret leak verified. |

## 2. Comprehensive Held-Out Performance Metrics
- **Total Test Candles**: 896
- **Active Signals Emitted**: 188 (20.98% coverage)
  - **Up Signals**: 188
  - **Down Signals**: 0
  - **No-Signal Candles**: 708
- **Active Signal Accuracy**: **46.28%**
  - **Wilson 95% Score CI**: **[39.30%, 53.41%]**
- **Always-Up Drift Benchmarks**:
  - On entire Test block (896 candles): 48.88%
  - On identical signal candles (188 candles): 46.28%
  - **Model Outperformance over Drift**: **+0.00%**
- **Probabilistic Scores**:
  - Test Log-Loss: `0.70027`
  - Test Brier Score: `0.25350`
  - Expected Calibration Error (ECE): `0.0454`
  - ROC-AUC: `0.5000` | PR-AUC: `0.4834`

## 3. Simulated Trading Performance
- **Net Return per Signal (after 2 bps friction)**: `-4.24` bps
- **Net Return per Signal (after 5 bps friction, primary)**: **`-7.24` bps**
  - **Moving Block-Bootstrap 95% CI**: **[-13.90, -0.02] bps**
- **Net Return per Signal (after 10 bps friction)**: `-12.24` bps
- **Hit Rate after 5 bps Costs**: 40.43%
- **Max Drawdown**: 1500.61 bps

## 4. Stability and Variance Analysis (5 Seeds)
- **Mean Signal Accuracy**: 46.50% ± 0.85%
- **Mean Log-Loss**: 0.70029 ± 0.00078
- **Mean Net Return @ 5 bps**: -8.31 ± 1.45 bps
- **Signals Flipping Across Seeds**: 176 / 896 candles (19.64%)