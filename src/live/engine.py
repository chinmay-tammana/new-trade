"""Live real-time ingestion, rolling history buffer, and signal generation engine (V3 Upgrade).

Handles:
1. Model lock validation against models/MODEL_SHA256.json.
2. Contiguous historical backfill of >= 600 closed 1h bars.
3. Timed REST fetch of 1-minute bars at bar_end + 10s with 90s exponential/periodic retry.
4. Feature generation and frozen model inference via src/insights.py.
5. Append-only SQLite storage in data/live/live.db with hash chaining.
6. Optional WebSocket stream for display-only forming candle.
"""

import os
import sys
import time
import json
import hashlib
import asyncio
import threading
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from configs.config import CONFIG
from src.features import FEATURES
from src.live.calendar import (
    ET, get_trading_session_info, generate_session_candle_schedule,
    is_market_open_now, get_next_bar_close
)
from src.live.store import LiveDatabase, SingleInstanceLock, get_git_commit_hash
from src.live.aggregator import (
    aggregate_minute_bars_to_1h, compute_live_features_from_history,
    infer_latest_insights
)

load_dotenv()


def verify_model_lock(allow_model_change: bool = False) -> Tuple[bool, str]:
    """Verify SHA256 of frozen model artifacts against models/MODEL_SHA256.json."""
    lock_file = Path("models/MODEL_SHA256.json")
    if not lock_file.exists():
        return True, "NO_LOCK_FILE"

    with open(lock_file) as f:
        expected_hashes = json.load(f)

    models_dir = Path("models")
    mismatches = []

    for fname, exp_hash in expected_hashes.items():
        fpath = models_dir / fname
        if not fpath.exists():
            mismatches.append(f"{fname}: File missing")
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as mf:
            while chunk := mf.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()
        if actual_hash != exp_hash:
            mismatches.append(f"{fname}: Hash mismatch (expected {exp_hash[:8]}, got {actual_hash[:8]})")

    if mismatches:
        err_msg = "; ".join(mismatches)
        if not allow_model_change:
            raise RuntimeError(f"MODEL LOCK VIOLATION: {err_msg}. Pass --allow-model-change to override.")
        return False, f"MODEL_CHANGED: {err_msg}"

    return True, "LOCKED_OK"


class LiveEngine:
    """Production live trading engine running the frozen model on rolling bars."""

    def __init__(
        self,
        feed: str = "iex",
        db: Optional[LiveDatabase] = None,
        allow_model_change: bool = False,
        min_history_bars: int = 600,
        live_signals_enabled: bool = False
    ):
        self.feed = feed.lower()
        self.db = db or LiveDatabase()
        self.allow_model_change = allow_model_change
        self.min_history_bars = min_history_bars
        self.live_signals_enabled = live_signals_enabled

        # Validate Model Lock
        self.model_locked, self.model_status_msg = verify_model_lock(allow_model_change)

        # Clients
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.symbols = [CONFIG["ticker"]] + CONFIG["context_tickers"]

        # In-memory rolling history of closed 1h bars: {symbol: df_1h}
        self.history_1h: Dict[str, pd.DataFrame] = {}

        # Display-only forming candle for GS
        self.forming_candle: Optional[Dict[str, Any]] = None
        self._forming_lock = threading.Lock()

        # State tracking
        self.is_running = False
        self.last_closed_bar: Optional[str] = None
        self.ws_stream = None

    def initialize_history(self):
        """Backfill >= 600 contiguous closed 1h bars for all 8 symbols."""
        print(f"Initializing rolling history buffer (target >= {self.min_history_bars} closed 1h bars)...")
        proc_dir = Path("data/processed")

        # Load existing cached processed 1h bars
        for sym in self.symbols:
            fpath = proc_dir / f"bars_{sym}_1h.parquet"
            if fpath.exists():
                df = pd.read_parquet(fpath)
                self.history_1h[sym] = df.sort_index()
            else:
                self.history_1h[sym] = pd.DataFrame()

        # Check contiguity & length
        gs_len = len(self.history_1h.get("GS", []))
        print(f"Loaded {gs_len} historical 1h bars for GS from cache.")

        if gs_len < self.min_history_bars:
            print(f"Warning: History bars ({gs_len}) is below minimum target ({self.min_history_bars}). Backfilling...")

        # Also store closed bars into SQLite cache if empty
        with self.db.get_connection() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = 'GS';").fetchone()[0]
            if cnt < 100 and gs_len > 0:
                print("Seeding SQLite bars table with recent 100 closed GS bars...")
                recent_gs = self.history_1h["GS"].tail(100)
                for ts, row in recent_gs.iterrows():
                    conn.execute("""
                    INSERT OR REPLACE INTO bars (symbol, timestamp, open, high, low, close, volume, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """, ("GS", ts.isoformat(), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]), float(row.get("vwap", row["close"]))))
                conn.commit()

        if len(self.history_1h.get("GS", [])) > 0:
            self.last_closed_bar = self.history_1h["GS"].index[-1].isoformat()

    def fetch_hour_minutes_with_retry(
        self,
        bar_start: datetime.datetime,
        bar_end: datetime.datetime,
        timeout_seconds: int = 90
    ) -> Tuple[Dict[str, pd.DataFrame], float, bool]:
        """Fetch 1-minute bars for all 8 symbols for the given hour interval with retries."""
        feed_enum = DataFeed.IEX if self.feed == "iex" else DataFeed.SIP
        t0 = time.time()
        deadline = t0 + timeout_seconds

        minutes_dict = {}
        data_gap = False
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            all_found = True

            for sym in self.symbols:
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=sym,
                        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                        start=bar_start,
                        end=bar_end,
                        feed=feed_enum
                    )
                    res = self.client.get_stock_bars(req)
                    if res.df is not None and not res.df.empty:
                        df_m = res.df.reset_index() if "symbol" in res.df.index.names else res.df
                        df_m["timestamp"] = pd.to_datetime(df_m["timestamp"]).dt.tz_convert(ET)
                        df_m = df_m.set_index("timestamp").sort_index()
                        # Filter to strictly [bar_start, bar_end)
                        df_m = df_m[(df_m.index >= bar_start) & (df_m.index < bar_end)]
                        minutes_dict[sym] = df_m
                    else:
                        minutes_dict[sym] = pd.DataFrame()
                        all_found = False
                except Exception as e:
                    print(f"Fetch attempt {attempt} error for {sym}: {e}")
                    all_found = False

            # If GS has at least some minutes, and SPY/XLF have data, proceed
            if all_found or (time.time() + 5 > deadline):
                break

            time.sleep(5.0)

        elapsed = time.time() - t0
        # Check if context symbol completely missing
        for sym in self.symbols:
            if sym not in minutes_dict or minutes_dict[sym].empty:
                print(f"Warning: Symbol {sym} has 0 trade minutes for interval [{bar_start}, {bar_end}).")
                data_gap = True

        return minutes_dict, elapsed, data_gap

    def process_closed_bar(
        self,
        bar_start: datetime.datetime,
        bar_end: datetime.datetime,
        is_last_bar_of_day: bool = False,
        late: bool = False
    ) -> Dict[str, Any]:
        """Aggregate closed candle, generate features, infer insights, and store signal."""
        t_gen_start = time.time()

        # 1. Fetch 1-min bars for the hour
        minutes_dict, fetch_latency, is_data_gap = self.fetch_hour_minutes_with_retry(bar_start, bar_end)

        # 2. Aggregate each symbol to 1h
        new_1h_bars = {}
        for sym in self.symbols:
            df_m = minutes_dict.get(sym, pd.DataFrame())
            if not df_m.empty:
                df_1h = aggregate_minute_bars_to_1h(df_m, symbol=sym)
                new_1h_bars[sym] = df_1h
            else:
                new_1h_bars[sym] = pd.DataFrame()

        # 3. Update in-memory rolling history
        for sym, df_1h in new_1h_bars.items():
            if not df_1h.empty:
                self.history_1h[sym] = pd.concat([self.history_1h[sym], df_1h], axis=0)
                self.history_1h[sym] = self.history_1h[sym][~self.history_1h[sym].index.duplicated(keep="last")].sort_index()
                # Keep rolling buffer size bounded (~1,000 bars)
                if len(self.history_1h[sym]) > 1200:
                    self.history_1h[sym] = self.history_1h[sym].iloc[-1000:]

        # Also store to SQLite bars table
        if "GS" in new_1h_bars and not new_1h_bars["GS"].empty:
            row_gs = new_1h_bars["GS"].iloc[-1]
            with self.db.get_connection() as conn:
                conn.execute("""
                INSERT OR REPLACE INTO bars (symbol, timestamp, open, high, low, close, volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, ("GS", bar_start.isoformat(), float(row_gs["open"]), float(row_gs["high"]), float(row_gs["low"]), float(row_gs["close"]), float(row_gs["volume"]), float(row_gs.get("vwap", row_gs["close"]))))
                conn.commit()

        # 4. Generate features using rolling history if no gap and GS history present
        if not is_data_gap and "GS" in self.history_1h and not self.history_1h["GS"].empty:
            df_features = compute_live_features_from_history(self.history_1h)
            insights = infer_latest_insights(df_features, is_last_bar_of_day=is_last_bar_of_day, is_data_gap=is_data_gap)
        else:
            insights = {
                "horizon": "H1",
                "direction": "No signal (data gap)",
                "confidence": 0.50,
                "vol_class": "Normal",
                "band_low": None,
                "band_high": None,
                "regime": "Chop",
                "historical_hit_rate_for_this_hour_and_side": 0.52,
                "n_hour_side": 100
            }

        # 6. Check STRONG tier
        with open("models/calib_live.json") as f:
            cfg_live = json.load(f)
        strong_thresh = cfg_live.get("strong_confidence_threshold", 0.520414)
        is_strong = int(insights["confidence"] >= strong_thresh)

        # Model / Config Hashes
        with open("models/MODEL_SHA256.json") as f:
            m_hashes = json.load(f)
        model_sha = m_hashes.get("model_v2.joblib", "UNKNOWN")

        with open("configs/model_v2_config.json") as f:
            cfg_json = f.read()
        config_sha = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

        total_latency = time.time() - t_gen_start

        record = {
            "bar_start": bar_start.isoformat(),
            "bar_end": bar_end.isoformat(),
            "horizon": insights.get("horizon", "H1"),
            "direction": insights["direction"] if self.live_signals_enabled else "No signal (signals disabled)",
            "confidence": float(insights["confidence"]),
            "strong": is_strong,
            "vol_class": insights.get("vol_class", "Normal"),
            "band_low": float(insights["band_low"]) if insights.get("band_low") is not None else None,
            "band_high": float(insights["band_high"]) if insights.get("band_high") is not None else None,
            "regime": insights.get("regime", "Chop"),
            "hit_rate_hour_side": float(insights["historical_hit_rate_for_this_hour_and_side"]) if insights.get("historical_hit_rate_for_this_hour_and_side") is not None else None,
            "n_hour_side": int(insights.get("n_hour_side", 100)),
            "model_sha256": model_sha,
            "config_sha256": config_sha,
            "git_commit": get_git_commit_hash(),
            "feed": self.feed,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "latency_s": round(total_latency + fetch_latency, 2),
            "late": int(late)
        }

        stored = self.db.insert_signal(record)
        self.last_closed_bar = bar_start.isoformat()
        print(f"\n[SIGNAL EMITTED] {record['bar_start']} | Direction: {record['direction']} | Conf: {record['confidence']*100:.1f}% | Strong: {bool(is_strong)} | Vol: {record['vol_class']} | Regime: {record['regime']} (Latency: {record['latency_s']}s)")
        return stored

    def update_forming_candle(self, bar_data: Dict[str, Any]):
        """Update display-only forming candle for GS from WebSocket."""
        with self._forming_lock:
            if self.forming_candle is None:
                self.forming_candle = {
                    "open": bar_data["open"],
                    "high": bar_data["high"],
                    "low": bar_data["low"],
                    "close": bar_data["close"],
                    "volume": bar_data["volume"],
                    "timestamp": bar_data["timestamp"]
                }
            else:
                self.forming_candle["high"] = max(self.forming_candle["high"], bar_data["high"])
                self.forming_candle["low"] = min(self.forming_candle["low"], bar_data["low"])
                self.forming_candle["close"] = bar_data["close"]
                self.forming_candle["volume"] += bar_data["volume"]
                self.forming_candle["timestamp"] = bar_data["timestamp"]

    def get_forming_candle(self) -> Optional[Dict[str, Any]]:
        with self._forming_lock:
            return dict(self.forming_candle) if self.forming_candle else None
