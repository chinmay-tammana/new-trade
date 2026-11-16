# Dataset Card: M5 5-Minute IEX Historical Dataset

## 1. Overview
- **Data Source**: Alpaca Historical Stock API (feed: `iex`, adjustment: `RAW`).
- **Span**: 2021-11-01 to 2026-11-13 (5 full calendar years, 1,263 trading sessions).
- **Sampling Frequency**: Raw 1-minute bars aggregated to regular-session 5-minute bars (09:30 to 16:00 ET).
- **Total GS 5-Minute Bars**: 87,109 bars.
- **Valid Model Features & Labels**: 84,443 rows after causal warmup.

## 2. Context Universe Audit & Selection
All symbols were downloaded monthly from 2021-11 to 2026-11. Under Section A.2, candidate symbols were evaluated against the 5% empty-bar rate threshold:

| Symbol | Role | 1-Min Bars | 5-Min Bars | Expected Bars | Empty Bar Rate | Audit Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GS** | Target Asset | 251,569 | 87,109 | 97,938 | 11.06% | Target |
| **SPY** | Macro Equity Index | 498,483 | 98,192 | 98,478 | 0.37% | Core Context |
| **XLF** | Financial Sector ETF | 459,211 | 98,122 | 98,190 | 0.15% | Core Context |
| **JPM** | Peer Bank | 408,209 | 97,350 | 98,118 | 0.78% | Peer Context |
| **MS** | Peer Bank (Lead) | 398,863 | 97,406 | 98,118 | 0.73% | Peer Context |
| **BAC** | Peer Bank | 477,789 | 98,040 | 98,118 | 0.08% | Peer Context |
| **TLT** | 20+ Year Treasury Bond | 438,299 | 97,814 | 98,370 | **0.64%** (< 5%) | **KEPT** |
| **VIXY** | Short-Term VIX Futures | 168,757 | 72,234 | 94,986 | **24.03%** (>= 5%) | **EXCLUDED** |

## 3. Data Integrity & Invariants
- Zero volume bars: 0.
- `high < low` anomalies: 0.
- Extreme returns (> 15% in 5 minutes): 0 for all kept symbols.
- Inner-join loss against GS: < 0.61% across all context symbols.
