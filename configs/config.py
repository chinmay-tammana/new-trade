"""Central configuration for Goldman Sachs (GS) AI trading indicator project (V2 Upgrade)."""

EVALUATION_DAY = "2026-11-16"

CONFIG = {
    "ticker": "GS",
    "context_tickers": ["SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"],
    "peer_tickers": ["JPM", "MS", "BAC"],
    "macro_tickers": ["SPY", "XLF", "TLT", "VIXY"],
    "interval": "1h",
    "tz": "America/New_York",
    "history_years": 8,
    "history_buffer_days": 30,
    "freeze_date": "2026-11-09",           # Last completed trading day before TA test day
    "test_day": "NOT_SPECIFIED",           # TA test day (default: freeze_date is most recent completed trading day)
    "research_holdout_start": "2025-10-16",# First day after v1 data
    "alpaca_feed": "sip",                 # "sip" (default) or "iex"
    "live_feed": "sip",                   # Live signal engine feed: "sip" (delayed >= 15 min)
    "live_display_feed": "iex",           # Price ticker & forming candle only: "iex" (real-time display)
    "alpaca_end_lag_minutes": 15,         # Free SIP needs query end at least 15 min old
    "session_open": "09:30",
    "session_close": "16:00",
    "seed": 42,
    "horizon_bars": 1,
    "horizons": ["H1", "H3", "HD"],
    "gap_bars": 8,
    "warmup_rows": 200,
    "label_threshold": 0.0010,
    "label_threshold_candidates": [0.0005, 0.0010, 0.0015, 0.0020],
    "quarter_days": 63,
    "initial_train_years": 2,
    "recency_halflives": [90, 180, 365, None],
    "coverage_sweep": [0.15, 0.20, 0.25, 0.30, 0.40],
    "costs_bps": [2, 5, 10],
    "primary_cost_bps": 5,
    "github_remote": "https://github.com/chinmay-tammana/new-trade.git",
    "paths": {
        "raw_dir": "data/raw",
        "processed": "data/processed/dataset_v2.parquet",
        "data_freeze": "configs/data_freeze.json",
        "models": "models",
        "tables": "results/tables",
        "figures": "results/figures",
        "logs": "results/logs",
        "holdout_predictions": "results/tables/holdout_predictions_v2.csv",
    },
}

CONFIG_M5 = {
    "bar_minutes": 5,
    "ticker": "GS",
    "context_tickers": ["SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"],
    "peer_tickers": ["JPM", "MS", "BAC"],
    "macro_tickers": ["SPY", "XLF", "TLT", "VIXY"],
    "feed": "iex",
    "tz": "America/New_York",
    "history_years": 5,
    "freeze_date": "2026-11-13",           # Last completed trading day before today
    "test_day": "NOT_SPECIFIED",           # TA test day (default: freeze_date is last completed trading day)
    "holdout_months": 6,                   # Last 6 months before freeze_date
    "session_open": "09:30",
    "session_close": "16:00",
    "bars_per_day": 78,                    # 09:30 to 15:55 (78 5-min bars)
    "seed": 42,
    "horizon_bars": 1,
    "gap_bars": 8,                         # 40-minute embargo
    "label_threshold_candidates": [0.0002, 0.0003, 0.0005], # 2, 3, 5 basis points
    "quarter_days": 63,
    "initial_train_years": 2,
    "recency_halflives": [90, 180, 365, None],
    "coverage_sweep": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "costs_bps": [0, 1, 2, 5],
    "primary_cost_bps": 2,
    "coverage_target_m5": 0.30,            # Target directional coverage (30% of bars directional)
    "min_margin_m5": 0.01,                 # Minimum directional margin guard (conf >= 51%)
    "evaluation_day": "2026-11-16",        # Evaluation day for live M5
    "max_nan_features": 0,                 # Refuse signal if NaN count exceeds this (feature gap)
    "max_context_ffill_bars": 1,           # Forward-fill context ticker missing bars at most 1 bar
    "watchdog_timeout_minutes": 7,         # Watchdog alert and recovery threshold if no signals emitted
    "paths": {
        "raw_dir": "data/raw/iex_1min",
        "processed_bars": "data/processed/m5/bars_5m.parquet",
        "processed_features": "data/processed/m5/features_m5.parquet",
        "data_freeze": "configs/data_freeze_m5.json",
        "models": "models/m5",
        "model_path": "models/m5/model_m5.joblib",
        "calib_path": "models/m5/calib_m5.json",
        "active_rule_path": "models/m5/active_rule.json",
        "sha_path": "models/m5/MODEL_SHA256_m5.json",
        "tables": "results/m5/tables",
        "figures": "results/m5/figures",
        "logs": "results/m5/logs",
        "db_path": "data/live/live_m5.db",
    },
}
