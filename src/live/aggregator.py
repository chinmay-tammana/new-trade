"""Unified aggregation and feature extraction bridge for live engine and replay (V3 Upgrade).

Imports and calls:
- src.data_fetch.build_session_anchored_1h_bars
- src.features.make_features_v2
- src.insights.get_bar_insights
Zero duplicate logic.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from configs.config import CONFIG
from src.data_fetch import build_session_anchored_1h_bars
from src.features import FEATURES, make_features_v2
from src.insights import get_bar_insights, _load_production_models


def aggregate_minute_bars_to_1h(df_1m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Session-anchored 1h aggregation using exact offline protocol."""
    return build_session_anchored_1h_bars(df_1m, symbol=symbol, tz_str=CONFIG["tz"])


def compute_live_features_from_history(
    history_1h_dfs: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Compute stationary causal feature matrix from synchronized 1h history (>= 600 bars)."""
    df_gs = history_1h_dfs["GS"]
    context_dfs = {sym: history_1h_dfs[sym] for sym in CONFIG["context_tickers"] if sym in history_1h_dfs}
    
    # make_features_v2 with warmup_rows=0 returns full matrix
    df_features = make_features_v2(df_gs, context_dfs, warmup_rows=0)
    return df_features


def infer_latest_insights(
    df_features: pd.DataFrame,
    is_last_bar_of_day: bool = False,
    is_data_gap: bool = False
) -> Dict[str, Any]:
    """Call frozen model pipeline on the latest closed bar and return standard insight dict."""
    if is_data_gap:
        latest_row = df_features.iloc[-1]
        return {
            "direction": "No signal (data gap)",
            "confidence": 0.50,
            "horizon": "H1",
            "vol_class": "Normal",
            "band_low": None,
            "band_high": None,
            "regime": "Chop",
            "historical_hit_rate_for_this_hour_and_side": None,
            "n_hour_side": 0
        }

    latest_row = df_features.iloc[-1]
    insights = get_bar_insights(latest_row)

    if is_last_bar_of_day:
        insights["direction"] = "No signal (end of day)"

    return insights
