"""Central configuration for Goldman Sachs (GS) AI trading indicator project (V2 Upgrade)."""

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
    "live_feed": "iex",                   # Live feed: "iex" (real-time, zero delay)
    "alpaca_end_lag_minutes": 20,         # free SIP needs data older than 15 min
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
