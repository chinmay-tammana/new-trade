# V3.7 Revert Inventory: Audit of Partial State & Cancellation Record

**Date**: 2026-11-15  
**Trigger**: Owner Decision — *"I changed my mind. Revert to the previous rule."*  
**Operating Protocol**: The V3.7 run (`instructions/18_rule_47_53.md`) was cancelled by the owner while in progress. The active rule is restored to the frozen thresholds from `models/calib_live.json` ($q_{\text{lo}} \approx 0.434$, $q_{\text{hi}} \approx 0.563$), the original STRONG threshold ($\ge 0.60$), SIP-delayed signals, and the V3.6 layout.

---

## 1. Process Audit & Port Status
- **Running Processes Audited**:
  - Found stale server process PID `17902` running `/opt/homebrew/Cellar/python@3.11/3.11.18/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python -m src.live.server --port 8000`.
  - Stale process stopped cleanly with `kill 17902`.
  - Confirmed with `lsof -ti tcp:8000`: port 8000 is clean, free, and ready for single-instance restart.
- **Lock Files**:
  - Scanned repository for project lock files (`*.lock`, `*lock*`). Zero project lock files exist.

---

## 2. File Inventory (Files Created or Modified by Cancelled V3.7)

### A. Created Artifacts (To Be Moved to `archive/rule_47_53_cancelled/`)
1. `models/rule_v2.json`: Config file defining 47%/53% thresholds.
2. `results/tables/rule_47_53_coverage.csv`: Preliminary coverage measurements under 47%/53%.
3. `results/tables/rule_47_53_tier_accuracy.csv`: Preliminary tier accuracy table.
4. `results/tables/rule_47_53_signals_per_day.csv`: Daily signals under 47%/53%.
5. `docs/rule_47_53_measurements.md`: Written documentation of section A measurements.
6. `src/measure_rule_47_53.py`: Measurement script for cancelled rule.
7. `src/live/migrate_rule_v2.py`: Migration script for additive schema.
8. `src/live/backfill_rule2.py`: Backfill script for rule 2 signals.

### B. Modified Files (To Be Restored to Pre-V3.7 / V3.6 State)
1. `models/MODEL_SHA256.json`: Remove the `"rule_v2.json"` entry; keep the 3 original model files byte-identical.
2. `src/insights.py`: Revert threshold loading to `models/calib_live.json`; restore original thresholds and remove WEAK tier logic.
3. `src/live/aggregator.py`: Remove rule 2 tier parameters; restore standard calib_live inference.
4. `src/live/engine.py`: Revert live engine thresholds to `models/calib_live.json`; emit standard rule 1 signals.
5. `src/live/log_engine.py`: Revert thresholds to `models/calib_live.json`; filter queries with `(rule_version IS NULL OR rule_version = 1)`; hide rule 2 rows from summaries, table, multi-session balance, and scorecard; remove WEAK tier; remove `include_v1` toggle.
6. `src/live/server.py`: Remove `/api/rule` and `/api/tier_stats` endpoints; ensure all status, log, and replay endpoints filter `rule_version IS NULL OR rule_version = 1`.
7. `src/live/store.py`: Ensure default schema and queries filter out rule 2 rows while preserving additive columns; ensure cryptographic hash chain remains verified.
8. `static/index.html`: Revert hero threshold text to `needs ≥ 56.3% for Up, ≤ 43.4% for Down`; remove `#filter-v1-check`; restore Statistical Notice to pre-V3.7 wording; ensure no Confidence Tiers card or WEAK pills are present.
9. `static/style.css`: Remove any unneeded rule 2 styles.
10. `static/app.js`: Ensure threshold text reads from `calib_live.json` / status; ensure no WEAK pills or rule 2 markers are rendered; maintain V3.6 clean single-source-of-truth table.
11. `docs/user_guide.md`, `docs/live_expectations.md`, `docs/live_runbook.md`, `README.md`, `report/report.md`, `slides/outline.md`: Remove mentions of 47%/53% as active rule; restore original decision matrix; add required statement in `report/report.md`: *"A 47/53 no-signal band was considered and cancelled by the owner before any results were produced from it."*
12. `instructions/18_rule_47_53.md`: Kept strictly in place untouched; marked CANCELLED in `AGENTS.md` and `instructions/progress.md`.
13. `instructions/19_revert_rule.md`: Saved verbatim; active instruction.

---

## 3. Database State Inspection (`data/live/live.db`) Read-Only Audit

### A. Schema & Triggers
- **Tables**: `signals_raw_v1`, `outcomes`, `bars`, `outcomes_v2`, `signals_legacy_iex`, `signals`.
- **Triggers**: `prevent_signals_update`, `prevent_signals_delete` — **ACTIVE and intact**.
- **Views**: `signals_iex`.
- **Signals Table Columns**:
  `['bar_start', 'bar_end', 'horizon', 'direction', 'confidence', 'strong', 'vol_class', 'band_low', 'band_high', 'regime', 'hit_rate_hour_side', 'n_hour_side', 'model_sha256', 'config_sha256', 'git_commit', 'feed', 'generated_at', 'latency_s', 'late', 'sample', 'prev_hash', 'hash', 'delivered_at', 'delivery_delay_s', 'schema_v', 'rule_version', 'p_cal', 'q_lo', 'q_hi', 'tier']`.
  *(Additive columns remain in place; zero existing data was lost or truncated).*

### B. Row Counts by `rule_version`, Feed, and Sample
| `rule_version` | Feed | Sample Tag | Count | Status in V3.7-Revert |
| :--- | :--- | :--- | :--- | :--- |
| `1` (or NULL) | `iex` | `IN-SAMPLE` | 42 | Kept intact; hidden by default (IEX) |
| `1` (or NULL) | `iex` | `OUT-OF-SAMPLE` | 28 | Kept intact; hidden by default (IEX) |
| `1` (or NULL) | `sip_delayed` | `IN-SAMPLE` | 42 | **ACTIVE visible baseline** |
| `1` (or NULL) | `sip_delayed` | `OUT-OF-SAMPLE` | 28 | **ACTIVE visible baseline (Nov 10–13)** |
| `2` | `sip_delayed` | `IN-SAMPLE` | 42 | **HIDDEN by data layer** (`rule_version=2`) |
| `2` | `sip_delayed` | `OUT-OF-SAMPLE` | 28 | **HIDDEN by data layer** (`rule_version=2`) |
| **Total** | | | **210** | **Cryptographic chain valid: 100%** |

### C. Rule 1 Rows Integrity Verification
- **Total Rule 1 rows**: Exactly 140 rows (70 IEX, 70 SIP-delayed).
- **Integrity**: Every single row from the pre-V3.7 visible set is present, un-mutated, and in identical canonical hash order.
- **Rule 2 rows**: Exactly 70 rows have $q_{\text{lo}} = 0.47$ or $q_{\text{hi}} = 0.53$. Under the hard constraints, **zero rows will be deleted or updated**. The data layer will strictly filter `(rule_version IS NULL OR rule_version = 1)` everywhere signals are read.
- **Cryptographic Hash Chain**: Full table verification confirms `valid=True, count=210, err=None`.
