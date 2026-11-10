# Agent Instructions and Operating Protocol

Welcome. You are working on the Goldman Sachs (GS) AI trading indicator project following the IMPACT Lab (IIT Patna) End-to-End ML Project Guidelines.

## Operating Rules
1. **Read First**: Read `AGENTS.md` first, then every file in `instructions/` in numeric order (`00_rules_and_config.md` through `09_report_slides_freeze.md`) before writing any code. Read `instructions/10_v2_upgrade.md` last; it overrides earlier files.
2. **Work Phase by Phase**: Execute strictly phase by phase in sequential order. Do not skip or reorder phases.
3. **Phase Check & Logging**: After completing each phase, run its explicit CHECK, save execution logs into `results/logs/`, and run `src/check_secrets.py`.
4. **Git Commits & Pushes**: Commit after every phase with the message formatted exactly as `phase N: <name>` (e.g., `phase 0: setup`) and push to GitHub (`git push -u origin main` on first push, `git push` subsequently). Follow all GITHUB RULES.
5. **Progress Updates**: Update `instructions/progress.md` with checklist status, decisions, and any findings after each step. Never edit any other file in `instructions/`.
6. **Integrity on Failures**: If a verification check fails, fix it immediately or stop and log the failure. NEVER skip or bypass a check.
7. **Test-Set Discipline**: The Test block is read strictly ONCE in Phase 7 (`src/evaluate.py`). No code in any prior phase may access Test features or labels. After Phase 7, Phase 8 and subsequent analysis must only read `results/tables/test_predictions.csv`.
8. **No Placeholders**: No TODOs, placeholders, or pseudocode in any script. Every script must run end-to-end.
9. **Centralized Configuration**: All hyperparameters, thresholds, paths, and constants must reside in `configs/config.py`.
10. **Determinism**: Random seed 42 everywhere (`random_state=42`, `n_jobs=1` for final fits).
11. **Honest Reporting**: Report honest numbers with statistical intervals (Wilson CIs, bootstrap CIs). Never claim an edge that data does not support.
12. **Conservative Leak-Free Choices**: When any methodological question is ambiguous, always choose the more conservative, leak-free option and log the rationale in `instructions/progress.md`.
