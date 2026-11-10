"""Immutable SQLite storage, hash-chain auditing, and single-instance lock (V3 Upgrade).

Features:
1. SQLite WAL mode at data/live/live.db.
2. Immutable triggers: RAISE(ABORT) on UPDATE or DELETE of the signals table.
3. Cryptographic hash-chaining: hash = sha256(prev_hash + canonical_json_of_record).
4. Idempotent insert: repeated insert for the same bar_start returns existing record.
5. Outcomes stored in a separate table, recorded only after horizon candle closes.
6. Single-instance process lock on data/live/live.lock.
"""

import os
import json
import sqlite3
import hashlib
import fcntl
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import subprocess

DB_PATH = Path("data/live/live.db")
LOCK_PATH = Path("data/live/live.lock")


class LiveDatabase:
    """Encapsulates the immutable live SQLite database with audit triggers."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            # 1. Signals Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                bar_start TEXT PRIMARY KEY,
                bar_end TEXT NOT NULL,
                horizon TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                strong INTEGER NOT NULL,
                vol_class TEXT NOT NULL,
                band_low REAL,
                band_high REAL,
                regime TEXT NOT NULL,
                hit_rate_hour_side REAL,
                n_hour_side INTEGER,
                model_sha256 TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                git_commit TEXT NOT NULL,
                feed TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                latency_s REAL NOT NULL,
                late INTEGER NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            """)

            # 2. Immutable Triggers on Signals Table
            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_signals_update
            BEFORE UPDATE ON signals
            BEGIN
                SELECT RAISE(ABORT, 'Signals table is append-only: UPDATE forbidden');
            END;
            """)

            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_signals_delete
            BEFORE DELETE ON signals
            BEGIN
                SELECT RAISE(ABORT, 'Signals table is append-only: DELETE forbidden');
            END;
            """)

            # 3. Outcomes Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                bar_start TEXT PRIMARY KEY,
                realized_fwd_ret REAL NOT NULL,
                y_true INTEGER NOT NULL,
                correct INTEGER NOT NULL,
                recorded_at TEXT NOT NULL
            );
            """)

            # 4. Closed 1-Hour Bars Cache
            conn.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                vwap REAL,
                PRIMARY KEY (symbol, timestamp)
            );
            """)
            conn.commit()

    def get_latest_signal(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM signals ORDER BY bar_start DESC LIMIT 1;")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_signal_count(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM signals;")
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_signal_by_bar_start(self, bar_start: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM signals WHERE bar_start = ?;", (bar_start,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
            SELECT s.*, o.realized_fwd_ret, o.y_true, o.correct
            FROM signals s
            LEFT JOIN outcomes o ON s.bar_start = o.bar_start
            ORDER BY s.bar_start DESC
            LIMIT ?;
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def insert_signal(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a signal record into SQLite with cryptographic hash chaining."""
        # Check idempotency
        existing = self.get_signal_by_bar_start(record["bar_start"])
        if existing is not None:
            return existing

        with self.get_connection() as conn:
            # Find previous hash
            cursor = conn.execute("SELECT hash FROM signals ORDER BY bar_start DESC LIMIT 1;")
            prev_row = cursor.fetchone()
            prev_hash = prev_row["hash"] if prev_row else "GENESIS_0000000000000000000000000000000000000000000000000000000000000000"

            record["prev_hash"] = prev_hash

            # Compute sha256 of canonical record representation
            canonical_keys = [
                "bar_start", "bar_end", "horizon", "direction", "confidence",
                "strong", "vol_class", "band_low", "band_high", "regime",
                "hit_rate_hour_side", "n_hour_side", "model_sha256",
                "config_sha256", "git_commit", "feed", "generated_at",
                "latency_s", "late", "prev_hash"
            ]
            canonical_dict = {k: record.get(k) for k in canonical_keys}
            canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
            record_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
            record["hash"] = record_hash

            conn.execute("""
            INSERT INTO signals (
                bar_start, bar_end, horizon, direction, confidence, strong,
                vol_class, band_low, band_high, regime, hit_rate_hour_side,
                n_hour_side, model_sha256, config_sha256, git_commit, feed,
                generated_at, latency_s, late, prev_hash, hash
            ) VALUES (
                :bar_start, :bar_end, :horizon, :direction, :confidence, :strong,
                :vol_class, :band_low, :band_high, :regime, :hit_rate_hour_side,
                :n_hour_side, :model_sha256, :config_sha256, :git_commit, :feed,
                :generated_at, :latency_s, :late, :prev_hash, :hash
            );
            """, record)
            conn.commit()

        return record

    def insert_outcome(self, bar_start: str, realized_fwd_ret: float, y_true: int, correct: int):
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO outcomes (
                bar_start, realized_fwd_ret, y_true, correct, recorded_at
            ) VALUES (?, ?, ?, ?, ?);
            """, (bar_start, realized_fwd_ret, y_true, correct, now_str))
            conn.commit()

    def verify_hash_chain(self) -> Tuple[bool, int, Optional[str]]:
        """Verify the cryptographic integrity of the signals table from genesis."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM signals ORDER BY bar_start ASC;")
            rows = cursor.fetchall()

        expected_prev_hash = "GENESIS_0000000000000000000000000000000000000000000000000000000000000000"
        canonical_keys = [
            "bar_start", "bar_end", "horizon", "direction", "confidence",
            "strong", "vol_class", "band_low", "band_high", "regime",
            "hit_rate_hour_side", "n_hour_side", "model_sha256",
            "config_sha256", "git_commit", "feed", "generated_at",
            "latency_s", "late", "prev_hash"
        ]

        for i, row in enumerate(rows):
            d = dict(row)
            if d["prev_hash"] != expected_prev_hash:
                return False, i, f"Chain broken at bar {d['bar_start']}: prev_hash mismatch"

            canonical_dict = {k: d.get(k) for k in canonical_keys}
            canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
            computed_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

            if computed_hash != d["hash"]:
                return False, i, f"Hash tamper detected at bar {d['bar_start']}: expected {computed_hash}, found {d['hash']}"

            expected_prev_hash = d["hash"]

        return True, len(rows), None


class SingleInstanceLock:
    """Exclusive process lock to prevent concurrent live engine execution."""

    def __init__(self, lock_file: Path = LOCK_PATH):
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fd = None

    def acquire(self):
        self.fd = open(self.lock_file, "w")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(f"{os.getpid()}\n")
            self.fd.flush()
        except (IOError, BlockingIOError):
            raise RuntimeError(
                f"Another live engine process (PID in {self.lock_file}) is already running on this database! "
                "Refusing to start a second engine."
            )

    def release(self):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
            except Exception:
                pass
            self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"
