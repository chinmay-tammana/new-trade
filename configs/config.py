"""Central configuration for Goldman Sachs (GS) AI trading indicator project."""

CONFIG = {
    "ticker": "GS",
    "context_tickers": ["SPY", "XLF"],
    "interval": "1h",
    "tz": "America/New_York",
    "history_years": 4,
    "history_buffer_days": 30,
    "alpaca_feed": "sip",                 # "sip" (default) or "iex"
    "alpaca_end_lag_minutes": 20,         # free SIP needs data older than 15 min
    "session_open": "09:30",
    "session_close": "16:00",
    "seed": 42,
    "horizon_bars": 1,
    "gap_bars": 8,
    "warmup_rows": 30,
    "label_threshold": 0.0010,
    "label_threshold_candidates": [0.0005, 0.0010, 0.0015, 0.0020],
    "era_days": 365,
    "year4_val_frac": 0.40,
    "ft_cv_folds": 4,
    "n_iter_search": 40,
    "base_weight_options": [0.3, 0.6, 1.0],
    "warm_start_rounds_options": [50, 100, 200],
    "warm_start_lr_options": [0.01, 0.02, 0.05],
    "class_balance_weights": True,
    "n_seeds_variance": 5,
    "margin_sweep": [0.01, 0.15, 0.005],  # start, stop, step; used for d_up and d_down
    "min_coverage": 0.15,
    "min_signals": 40,
    "min_signals_per_side": 25,
    "switch_margin_logloss": 0.002,       # a comparator must beat XGBoost on Val by this to be deployed
    "costs_bps": [2, 5, 10],
    "primary_cost_bps": 5,
    "rsi_buy_below": 40,
    "rsi_sell_above": 60,
    "github_remote": "https://github.com/chinmay-tammana/new-trade.git",
    "paths": {
        "raw_dir": "data/raw",
        "processed": "data/processed/dataset.parquet",
        "models": "models",
        "tables": "results/tables",
        "figures": "results/figures",
        "logs": "results/logs",
        "split": "results/logs/split.json",
        "test_predictions": "results/tables/test_predictions.csv",
    },
}
