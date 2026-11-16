"""
Immutable SQLite database store for M5 (5-minute) live signals and raw minutes.
Strict immutability triggers, cryptographic hash chain, and single-instance lock.
"""
import os
import sqlite3
import hashlib
import json
import datetime
import zoneinfo
from typing import Optional, List, Dict, Any

ET_TZ = zoneinfo.ZoneInfo("America/New_York")
DB_PATH = "data/live/live_m5.db"
LOCK_PATH = "data/live/live_m5.lock"


class SingleInstanceLock:
    def __init__(self, lock_path: str = LOCK_PATH):
        self.lock_path = lock_path
        self.pid = os.getpid()

    def acquire(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, "r") as f:
                    old_pid = int(f.read().strip())
                # Check if process exists
                try:
                    os.kill(old_pid, 0)
                    raise RuntimeError(f"Another live M5 process is already running with PID {old_pid}")
                except ProcessLookupError:
                    pass # Stale lock
            except (ValueError, IOError):
                pass
        with open(self.lock_path, "w") as f:
            f.write(str(self.pid))

    def release(self):
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, "r") as f:
                    p = int(f.read().strip())
                if p == self.pid:
                    os.remove(self.lock_path)
            except Exception:
                pass


class LiveM5Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            # Check schema of signals
            cursor = conn.execute("PRAGMA table_info(signals);")
            cols = [row["name"] for row in cursor.fetchall()]
            
            if cols and "rule_version" not in cols:
                # Additive schema evolution with composite UNIQUE(bar_start, horizon, rule_version)
                conn.execute("DROP TRIGGER IF EXISTS prevent_signals_update;")
                conn.execute("DROP TRIGGER IF EXISTS prevent_signals_delete;")
                conn.execute("ALTER TABLE signals RENAME TO signals_v1_legacy;")
                conn.execute("""
                CREATE TABLE signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bar_start TEXT NOT NULL,
                    bar_end TEXT NOT NULL,
                    horizon TEXT NOT NULL DEFAULT '5m',
                    symbol TEXT NOT NULL DEFAULT 'GS',
                    feed TEXT NOT NULL DEFAULT 'iex',
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    p_cal REAL NOT NULL,
                    q_lo REAL NOT NULL,
                    q_hi REAL NOT NULL,
                    is_strong INTEGER NOT NULL DEFAULT 0,
                    vol_class TEXT,
                    band_lo REAL,
                    band_hi REAL,
                    regime TEXT,
                    delivered_at TEXT NOT NULL,
                    delivery_delay_s REAL NOT NULL,
                    sample TEXT NOT NULL DEFAULT 'LIVE',
                    late INTEGER NOT NULL DEFAULT 0,
                    model_hash TEXT NOT NULL,
                    row_hash TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    rule_version INTEGER NOT NULL DEFAULT 1,
                    tier TEXT DEFAULT 'NORMAL',
                    UNIQUE(bar_start, horizon, rule_version)
                );
                """)
                conn.execute("""
                INSERT INTO signals (
                    id, bar_start, bar_end, horizon, symbol, feed, direction,
                    confidence, p_cal, q_lo, q_hi, is_strong, vol_class,
                    band_lo, band_hi, regime, delivered_at, delivery_delay_s,
                    sample, late, model_hash, row_hash, prev_hash, created_at,
                    rule_version, tier
                )
                SELECT 
                    id, bar_start, bar_end, horizon, symbol, feed, direction,
                    confidence, p_cal, q_lo, q_hi, is_strong, vol_class,
                    band_lo, band_hi, regime, delivered_at, delivery_delay_s,
                    sample, late, model_hash, row_hash, prev_hash, created_at,
                    1, CASE WHEN is_strong=1 THEN 'STRONG' ELSE 'NORMAL' END
                FROM signals_v1_legacy;
                """)
                conn.execute("DROP TABLE signals_v1_legacy;")
            elif not cols:
                # Create brand new signals table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bar_start TEXT NOT NULL,
                    bar_end TEXT NOT NULL,
                    horizon TEXT NOT NULL DEFAULT '5m',
                    symbol TEXT NOT NULL DEFAULT 'GS',
                    feed TEXT NOT NULL DEFAULT 'iex',
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    p_cal REAL NOT NULL,
                    q_lo REAL NOT NULL,
                    q_hi REAL NOT NULL,
                    is_strong INTEGER NOT NULL DEFAULT 0,
                    vol_class TEXT,
                    band_lo REAL,
                    band_hi REAL,
                    regime TEXT,
                    delivered_at TEXT NOT NULL,
                    delivery_delay_s REAL NOT NULL,
                    sample TEXT NOT NULL DEFAULT 'LIVE',
                    late INTEGER NOT NULL DEFAULT 0,
                    model_hash TEXT NOT NULL,
                    row_hash TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    rule_version INTEGER NOT NULL DEFAULT 1,
                    tier TEXT DEFAULT 'NORMAL',
                    UNIQUE(bar_start, horizon, rule_version)
                );
                """)

            # 2. Raw Minutes Table for Bit-for-Bit Replay
            conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_minutes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bar_start TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                trade_count REAL,
                vwap REAL,
                created_at TEXT NOT NULL,
                UNIQUE(bar_start, symbol, timestamp)
            );
            """)

            # 3. Bars cache table (for 5-minute bars)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS bars_5m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bar_start TEXT NOT NULL,
                bar_end TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                vwap REAL NOT NULL,
                trade_count REAL NOT NULL,
                n_minutes INTEGER NOT NULL,
                UNIQUE(symbol, bar_start)
            );
            """)

            # 4. Immutability Triggers on signals
            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_signals_update
            BEFORE UPDATE ON signals
            BEGIN
                SELECT RAISE(ABORT, 'Signals table is strictly immutable. UPDATE is forbidden.');
            END;
            """)

            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_signals_delete
            BEFORE DELETE ON signals
            BEGIN
                SELECT RAISE(ABORT, 'Signals table is strictly immutable. DELETE is forbidden.');
            END;
            """)

            # Indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_bar_start ON signals(bar_start);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_mins_bar_sym ON raw_minutes(bar_start, symbol);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bars5m_sym_start ON bars_5m(symbol, bar_start);")
            conn.commit()

    def get_latest_hash(self) -> str:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT row_hash FROM signals ORDER BY id DESC LIMIT 1;")
            row = cursor.fetchone()
            return row["row_hash"] if row else "genesis_m5_hash"

    def compute_row_hash(self, prev_hash: str, fields: dict) -> str:
        canonical_str = (
            f"{prev_hash}|{fields['bar_start']}|{fields['bar_end']}|{fields['horizon']}|"
            f"{fields['symbol']}|{fields['direction']}|{fields['confidence']:.6f}|"
            f"{fields['p_cal']:.6f}|{fields['delivered_at']}|{fields['model_hash']}"
        )
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def insert_signal(self, sig: dict) -> bool:
        prev_hash = self.get_latest_hash()
        row_hash = self.compute_row_hash(prev_hash, sig)
        now_str = datetime.datetime.now(ET_TZ).isoformat()
        rule_v = int(sig.get("rule_version", 1))
        tier_val = sig.get("tier", "STRONG" if sig.get("is_strong") else "NORMAL")
        
        with self.get_connection() as conn:
            try:
                conn.execute("""
                INSERT INTO signals (
                    bar_start, bar_end, horizon, symbol, feed, direction,
                    confidence, p_cal, q_lo, q_hi, is_strong, vol_class,
                    band_lo, band_hi, regime, delivered_at, delivery_delay_s,
                    sample, late, model_hash, row_hash, prev_hash, created_at,
                    rule_version, tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    sig["bar_start"], sig["bar_end"], sig.get("horizon", "5m"),
                    sig.get("symbol", "GS"), sig.get("feed", "iex"), sig["direction"],
                    float(sig["confidence"]), float(sig["p_cal"]), float(sig["q_lo"]),
                    float(sig["q_hi"]), int(sig.get("is_strong", 0)), sig.get("vol_class"),
                    sig.get("band_lo"), sig.get("band_hi"), sig.get("regime"),
                    sig["delivered_at"], float(sig["delivery_delay_s"]),
                    sig.get("sample", "LIVE"), int(sig.get("late", 0)),
                    sig["model_hash"], row_hash, prev_hash, now_str,
                    rule_v, tier_val
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    return False
                raise e

    def insert_raw_minutes(self, bar_start: str, symbol: str, minute_df) -> int:
        now_str = datetime.datetime.now(ET_TZ).isoformat()
        inserted = 0
        with self.get_connection() as conn:
            for idx, r in minute_df.iterrows():
                ts_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                try:
                    conn.execute("""
                    INSERT OR IGNORE INTO raw_minutes (
                        bar_start, symbol, timestamp, open, high, low, close,
                        volume, trade_count, vwap, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        bar_start, symbol, ts_str, float(r["open"]), float(r["high"]),
                        float(r["low"]), float(r["close"]), float(r["volume"]),
                        float(r.get("trade_count", 0)), float(r.get("vwap", r["close"])),
                        now_str
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        return inserted

    def insert_bar_5m(self, symbol: str, bar_start: str, bar_end: str, bar_data: dict):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO bars_5m (
                symbol, bar_start, bar_end, open, high, low, close, volume, vwap, trade_count, n_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                symbol, bar_start, bar_end, float(bar_data["open"]), float(bar_data["high"]),
                float(bar_data["low"]), float(bar_data["close"]), float(bar_data["volume"]),
                float(bar_data["vwap"]), float(bar_data["trade_count"]), int(bar_data["n_minutes_with_trades"])
            ))
            conn.commit()

    def verify_hash_chain(self) -> (bool, int, Optional[str]):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM signals ORDER BY id ASC;")
            rows = cursor.fetchall()
            if not rows:
                return (True, 0, None)
            
            expected_prev = "genesis_m5_hash"
            for r in rows:
                if r["prev_hash"] != expected_prev:
                    return (False, r["id"], f"Broken link at ID {r['id']}: expected prev {expected_prev}, got {r['prev_hash']}")
                
                sig_dict = {
                    "bar_start": r["bar_start"],
                    "bar_end": r["bar_end"],
                    "horizon": r["horizon"],
                    "symbol": r["symbol"],
                    "direction": r["direction"],
                    "confidence": r["confidence"],
                    "p_cal": r["p_cal"],
                    "delivered_at": r["delivered_at"],
                    "model_hash": r["model_hash"]
                }
                computed = self.compute_row_hash(expected_prev, sig_dict)
                if computed != r["row_hash"]:
                    return (False, r["id"], f"Hash mismatch at ID {r['id']}: computed {computed}, stored {r['row_hash']}")
                expected_prev = r["row_hash"]
                
        return (True, len(rows), None)
