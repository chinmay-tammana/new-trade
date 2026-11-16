"""
Live 5-Minute Signal Engine (IEX Real-Time).
Fetches 1-minute IEX bars at bar close, aggregates causally via aggregate_1min_to_5min,
computes features via make_features_m5, stores raw minutes in raw_minutes,
and generates immutable 5-minute signals.
"""
import os
import sys
import time
import json
import hashlib
import datetime
import zoneinfo
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import joblib

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient

from configs.config import CONFIG_M5
from src.aggregate_m5 import aggregate_1min_to_5min, ET_TZ
from src.features_m5 import make_features_m5, load_and_build_features_m5, ALL_FEATURE_COLS
from src.live_m5.store import LiveM5Database

load_dotenv()


class LiveM5Engine:
    def __init__(self, db: Optional[LiveM5Database] = None):
        self.db = db or LiveM5Database()
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
        
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        
        self.model_bundle = None
        self.calib_config = None
        self.model_hash = "unverified"
        self.symbols = ["GS", "SPY", "XLF", "JPM", "MS", "BAC", "TLT"]
        self.active_rule_version = 5
        self.load_model()

    def load_model(self):
        model_path = CONFIG_M5["paths"]["model_path"]
        calib_path = CONFIG_M5["paths"]["calib_path"]
        sha_path = CONFIG_M5["paths"]["sha_path"]
        
        if not os.path.exists(model_path) or not os.path.exists(calib_path):
            print(f"[LiveM5Engine] Model files not found yet at {model_path}. Running in unverified standby.")
            return

        # Verify SHA256 hashes
        if os.path.exists(sha_path):
            with open(sha_path, "r") as sf:
                expected_hashes = json.load(sf)
            with open(model_path, "rb") as mf:
                actual_model_sha = hashlib.sha256(mf.read()).hexdigest()
            with open(calib_path, "rb") as cf:
                actual_calib_sha = hashlib.sha256(cf.read()).hexdigest()
            
            assert actual_model_sha == expected_hashes["model_m5.joblib"], "Model file hash mismatch!"
            assert actual_calib_sha == expected_hashes["calib_m5.json"], "Calib file hash mismatch!"
            self.model_hash = actual_model_sha[:8]
        else:
            self.model_hash = "m5_live"

        self.model_bundle = joblib.load(model_path)
        
        # Load active rule thresholds (Rule v4 uses identical thresholds to v3 Always-Call)
        active_ptr_path = "models/m5/active_rule.json"
        if os.path.exists(active_ptr_path):
            with open(active_ptr_path, "r") as f:
                ptr = json.load(f)
            rule_file = os.path.join("models/m5", ptr["active_rule_file"])
            if os.path.exists(rule_file):
                with open(rule_file, "r") as rf:
                    self.calib_config = json.load(rf)
            else:
                with open(calib_path, "r") as f:
                    self.calib_config = json.load(f)
        else:
            with open(calib_path, "r") as f:
                self.calib_config = json.load(f)
            
        print(f"[LiveM5Engine] Model verified and loaded (hash: {self.model_hash}, active rule v{self.active_rule_version})")

    def fetch_iex_minutes(self, symbol: str, start_dt: datetime.datetime, end_dt: datetime.datetime, retries: int = 3) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX,
            adjustment=Adjustment.RAW
        )
        for attempt in range(1, retries + 1):
            try:
                bars = self.data_client.get_stock_bars(req)
                df = bars.df
                if df.empty:
                    return pd.DataFrame()
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level="symbol")
                df = df.sort_index()
                return df
            except Exception as e:
                print(f"[fetch_iex_minutes] Attempt {attempt}/{retries} error fetching {symbol} [{start_dt} to {end_dt}]: {e}")
                if attempt < retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    print(f"[fetch_iex_minutes] Exhausted {retries} retries for {symbol}. Raising exception.")
                    raise e

    def process_closed_bar(self, bar_start: datetime.datetime, bar_end: datetime.datetime, is_live: bool = True) -> Optional[dict]:
        t_delivered = datetime.datetime.now(ET_TZ)
        delivery_delay = (t_delivered - bar_end).total_seconds()

        q_lo_val = float(self.calib_config.get("q_lo", 0.50)) if self.calib_config else 0.50
        q_hi_val = float(self.calib_config.get("q_hi", 0.50)) if self.calib_config else 0.50
        strong_thresh = float(self.calib_config.get("strong_threshold", 0.5811)) if self.calib_config else 0.5811
        weak_thresh = float(self.calib_config.get("weak_threshold", 0.5208)) if self.calib_config else 0.5208

        # 1. Fetch 1-minute bars with 10-minute late-trade settlement window across all symbols
        fetch_start = bar_end - datetime.timedelta(minutes=10)
        fetch_end = bar_end
        try:
            for sym in self.symbols:
                m_df = self.fetch_iex_minutes(sym, fetch_start, fetch_end)
                if not m_df.empty:
                    self.db.insert_raw_minutes(bar_start.isoformat(), sym, m_df)
        except Exception as e:
            print(f"[LiveM5Engine] Engine ingestion failure on bar {bar_start}: {e}")
            sig = {
                "bar_start": bar_start.isoformat(),
                "bar_end": bar_end.isoformat(),
                "horizon": "5m",
                "symbol": "GS",
                "feed": "iex",
                "direction": "No signal (engine error)",
                "confidence": 0.50,
                "p_cal": 0.50,
                "q_lo": q_lo_val,
                "q_hi": q_hi_val,
                "is_strong": 0,
                "tier": "NORMAL",
                "vol_class": None,
                "band_lo": None,
                "band_hi": None,
                "regime": "Chop",
                "delivered_at": t_delivered.isoformat(),
                "delivery_delay_s": delivery_delay,
                "sample": "LIVE" if is_live else "LIVE_REVISED",
                "late": 0 if is_live else 1,
                "rule_version": getattr(self, "active_rule_version", 5),
                "model_hash": self.model_hash
            }
            self.db.insert_signal(sig)
            return sig

        bar_start_utc = bar_start.astimezone(datetime.timezone.utc).isoformat()
        bar_end_utc = bar_end.astimezone(datetime.timezone.utc).isoformat()
        
        # Check GS minute data in DB for [bar_start, bar_end) after re-fetch
        with self.db.get_connection() as conn:
            cur = conn.execute(
                "SELECT timestamp, open, high, low, close, volume, vwap, trade_count FROM raw_minutes "
                "WHERE symbol = 'GS' AND timestamp >= ? AND timestamp < ? ORDER BY timestamp;",
                (bar_start_utc, bar_end_utc)
            )
            rows = cur.fetchall()

        if not rows:
            print(f"[LiveM5Engine] Alpaca returned 0 GS trades for {bar_start} after 10m re-fetch - feed gap.")
            direction = "No signal (feed gap)"
            confidence = 0.50
            p_cal = 0.50
            sig = {
                "bar_start": bar_start.isoformat(),
                "bar_end": bar_end.isoformat(),
                "horizon": "5m",
                "symbol": "GS",
                "feed": "iex",
                "direction": direction,
                "confidence": confidence,
                "p_cal": p_cal,
                "q_lo": q_lo_val,
                "q_hi": q_hi_val,
                "is_strong": 0,
                "tier": "NORMAL",
                "vol_class": None,
                "band_lo": None,
                "band_hi": None,
                "regime": "Chop",
                "delivered_at": t_delivered.isoformat(),
                "delivery_delay_s": delivery_delay,
                "sample": "LIVE" if is_live else "LIVE_REVISED",
                "late": 0 if is_live else 1,
                "rule_version": getattr(self, "active_rule_version", 5),
                "model_hash": self.model_hash
            }
            self.db.insert_signal(sig)
            return sig

        # 2. Aggregate 5-minute bars and cache in bars_5m for GS and context symbols
        cols = [d[0] for d in cur.description]
        gs_mins = pd.DataFrame(rows, columns=cols).set_index("timestamp")
        gs_mins.index = pd.to_datetime(gs_mins.index)
        bar_5m_gs = aggregate_1min_to_5min(gs_mins, symbol="GS")
        if not bar_5m_gs.empty:
            self.db.insert_bar_5m("GS", bar_start.isoformat(), bar_end.isoformat(), bar_5m_gs.iloc[-1].to_dict())

        for sym in self.symbols:
            if sym != "GS":
                with self.db.get_connection() as conn:
                    c_cur = conn.execute(
                        "SELECT timestamp, open, high, low, close, volume, vwap, trade_count FROM raw_minutes "
                        "WHERE symbol = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp;",
                        (sym, bar_start_utc, bar_end_utc)
                    )
                    c_rows = c_cur.fetchall()
                if c_rows:
                    c_df = pd.DataFrame(c_rows, columns=cols).set_index("timestamp")
                    c_df.index = pd.to_datetime(c_df.index)
                    b_ctx = aggregate_1min_to_5min(c_df, symbol=sym)
                    if not b_ctx.empty:
                        self.db.insert_bar_5m(sym, bar_start.isoformat(), bar_end.isoformat(), b_ctx.iloc[-1].to_dict())

        # 3. Check for End of Trading Day (15:55-16:00 close bar)
        if bar_start.time() >= datetime.time(15, 55):
            direction = "No signal (end of day)"
            confidence = 0.50
            p_cal = 0.50
            sig = {
                "bar_start": bar_start.isoformat(),
                "bar_end": bar_end.isoformat(),
                "horizon": "5m",
                "symbol": "GS",
                "feed": "iex",
                "direction": direction,
                "confidence": confidence,
                "p_cal": p_cal,
                "q_lo": q_lo_val,
                "q_hi": q_hi_val,
                "is_strong": 0,
                "tier": "NORMAL",
                "vol_class": None,
                "band_lo": None,
                "band_hi": None,
                "regime": "Chop",
                "delivered_at": t_delivered.isoformat(),
                "delivery_delay_s": delivery_delay,
                "sample": "LIVE" if is_live else "LIVE_REVISED",
                "late": 0 if is_live else 1,
                "rule_version": getattr(self, "active_rule_version", 5),
                "model_hash": self.model_hash
            }
            self.db.insert_signal(sig)
            return sig

        # 4. Compute features using identical multi-bar history pipeline as replay
        df_feat = load_and_build_features_m5(bar_start.date(), db_path=self.db.db_path)
        if df_feat.empty or bar_start not in df_feat.index:
            direction = "No signal (data gap)"
            confidence = 0.50
            p_cal = 0.50
            sig = {
                "bar_start": bar_start.isoformat(),
                "bar_end": bar_end.isoformat(),
                "horizon": "5m",
                "symbol": "GS",
                "feed": "iex",
                "direction": direction,
                "confidence": confidence,
                "p_cal": p_cal,
                "q_lo": q_lo_val,
                "q_hi": q_hi_val,
                "is_strong": 0,
                "tier": "NORMAL",
                "vol_class": None,
                "band_lo": None,
                "band_hi": None,
                "regime": "Chop",
                "delivered_at": t_delivered.isoformat(),
                "delivery_delay_s": delivery_delay,
                "sample": "LIVE" if is_live else "LIVE_REVISED",
                "late": 0 if is_live else 1,
                "rule_version": getattr(self, "active_rule_version", 5),
                "model_hash": self.model_hash
            }
            self.db.insert_signal(sig)
            return sig

        latest_row = df_feat.loc[bar_start]
        X_curr = latest_row[ALL_FEATURE_COLS].values.reshape(1, -1)
        nan_count = int(pd.isna(X_curr).sum())

        if nan_count > CONFIG_M5.get("max_nan_features", 0):
            print(f"[LiveM5Engine] Feature NaN count {nan_count} exceeds threshold - emitting feature gap.")
            direction = "No signal (feature gap)"
            confidence = 0.50
            p_cal = 0.50
            tier = "NORMAL"
            is_strong = 0
        else:
            if self.model_bundle is None:
                self.load_model()
            primary_clf = self.model_bundle["primary_model"]
            calibrator = self.model_bundle["calibrator"]
            
            raw_p = primary_clf.predict_proba(X_curr)[0, 1]
            p_cal = float(calibrator.predict_proba(np.array([raw_p]))[0])
            
            is_always_call = (self.calib_config.get("rule_type") == "always_call") or (abs(q_lo_val - 0.5) < 1e-6 and abs(q_hi_val - 0.5) < 1e-6)
            
            if is_always_call:
                if abs(p_cal - 0.5) < 1e-9:
                    direction = "No signal"
                    confidence = 0.50
                elif p_cal > 0.5:
                    direction = "Up"
                    confidence = p_cal
                else:
                    direction = "Down"
                    confidence = 1.0 - p_cal
            else:
                if p_cal >= q_hi_val:
                    direction = "Up"
                    confidence = max(p_cal, 0.51)
                elif p_cal <= q_lo_val:
                    direction = "Down"
                    confidence = max(1.0 - p_cal, 0.51)
                else:
                    direction = "No signal"
                    confidence = 0.50

            if direction in ["Up", "Down"]:
                if confidence >= strong_thresh:
                    tier = "STRONG"
                    is_strong = 1
                elif confidence <= weak_thresh:
                    tier = "WEAK"
                    is_strong = 0
                else:
                    tier = "NORMAL"
                    is_strong = 0
            else:
                tier = "NORMAL"
                is_strong = 0

        regime = latest_row.get("regime_tag", "Chop")
        sig = {
            "bar_start": bar_start.isoformat(),
            "bar_end": bar_end.isoformat(),
            "horizon": "5m",
            "symbol": "GS",
            "feed": "iex",
            "direction": direction,
            "confidence": confidence,
            "p_cal": p_cal,
            "q_lo": q_lo_val,
            "q_hi": q_hi_val,
            "is_strong": is_strong,
            "tier": tier,
            "vol_class": None,
            "band_lo": None,
            "band_hi": None,
            "regime": regime,
            "delivered_at": t_delivered.isoformat(),
            "delivery_delay_s": delivery_delay,
            "sample": "LIVE" if is_live else "LIVE_REVISED",
            "late": 0 if is_live else 1,
            "rule_version": getattr(self, "active_rule_version", 5),
            "model_hash": self.model_hash
        }
        self.db.insert_signal(sig)
        return sig

    def get_processed_bars_today(self, session_date: datetime.date) -> set:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT bar_start FROM signals WHERE bar_start LIKE ? AND rule_version = ?;",
                (f"{session_date.isoformat()}%", getattr(self, "active_rule_version", 5))
            )
            return {row["bar_start"] for row in cursor.fetchall()}

    def get_session_bars(self, session_date: datetime.date) -> list:
        bars = []
        cur = datetime.datetime(session_date.year, session_date.month, session_date.day, 9, 30, tzinfo=ET_TZ)
        for i in range(78):
            b_start = cur + datetime.timedelta(minutes=5 * i)
            b_end = b_start + datetime.timedelta(minutes=5)
            bars.append((b_start, b_end))
        return bars

    def catchup_today(self) -> int:
        now_et = datetime.datetime.now(ET_TZ)
        today = now_et.date()
        processed = self.get_processed_bars_today(today)
        all_bars = self.get_session_bars(today)
        emitted = 0
        for b_start, b_end in all_bars:
            # Settle delay 25s after bar close
            if b_end + datetime.timedelta(seconds=25) <= now_et:
                if b_start.isoformat() not in processed:
                    print(f"[LiveM5Engine] Processing closed bar: {b_start.strftime('%H:%M')} - {b_end.strftime('%H:%M')} ET")
                    self.process_closed_bar(b_start, b_end, is_live=True)
                    processed.add(b_start.isoformat())
                    emitted += 1
        return emitted

    def check_watchdog(self):
        now_et = datetime.datetime.now(ET_TZ)
        market_open_time = datetime.time(9, 35)
        market_close_time = datetime.time(16, 0)
        # Check during trading session
        if market_open_time <= now_et.time() <= market_close_time and now_et.weekday() < 5:
            with self.db.get_connection() as conn:
                cur = conn.execute(
                    "SELECT bar_end, created_at FROM signals WHERE rule_version = ? ORDER BY id DESC LIMIT 1;",
                    (getattr(self, "active_rule_version", 5),)
                )
                r = cur.fetchone()
            if r:
                latest_dt = datetime.datetime.fromisoformat(r["created_at"]).astimezone(ET_TZ)
                elapsed_s = (now_et - latest_dt).total_seconds()
                watchdog_limit_s = CONFIG_M5.get("watchdog_timeout_minutes", 7) * 60
                if elapsed_s > watchdog_limit_s:
                    print(f"[Watchdog] ALERT: No signal emitted for {elapsed_s:.1f}s (> {watchdog_limit_s}s limit) during market hours! Initiating recovery catchup...")
                    self.catchup_today()

    def run_daemon(self, poll_interval_s: int = 10):
        print(f"[LiveM5Engine] Starting live daemon loop with watchdog (poll interval {poll_interval_s}s)...")
        while True:
            try:
                self.catchup_today()
                self.check_watchdog()
            except Exception as e:
                print(f"[LiveM5Engine] Error in catchup loop: {e}")
            time.sleep(poll_interval_s)


if __name__ == "__main__":
    eng = LiveM5Engine()
    eng.run_daemon()

