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
import datetime
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
            # 1. Signals Table with composite primary key (bar_start, feed, rule_version)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                bar_start TEXT NOT NULL,
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
                feed TEXT NOT NULL DEFAULT 'iex',
                generated_at TEXT NOT NULL,
                latency_s REAL NOT NULL,
                late INTEGER NOT NULL,
                sample TEXT NOT NULL DEFAULT 'LIVE',
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL,
                delivered_at TEXT,
                delivery_delay_s REAL,
                schema_v INTEGER DEFAULT 1,
                rule_version INTEGER DEFAULT 1,
                p_cal REAL,
                q_lo REAL DEFAULT 0.43428,
                q_hi REAL DEFAULT 0.56323,
                tier TEXT,
                PRIMARY KEY (bar_start, feed, rule_version)
            );
            """)

            # Add additive columns if missing
            for col_def in [
                "delivered_at TEXT",
                "delivery_delay_s REAL",
                "schema_v INTEGER DEFAULT 1",
                "rule_version INTEGER DEFAULT 1",
                "p_cal REAL",
                "q_lo REAL DEFAULT 0.43428",
                "q_hi REAL DEFAULT 0.56323",
                "tier TEXT"
            ]:
                try:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col_def};")
                except Exception:
                    pass

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

            # 3. View for legacy IEX signals
            conn.execute("CREATE VIEW IF NOT EXISTS signals_iex AS SELECT * FROM signals WHERE feed='iex';")

            # 4. Outcomes Table
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

    def get_latest_signal(self, feed: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            if feed:
                cursor = conn.execute("SELECT * FROM signals WHERE feed = ? AND (rule_version = 1 OR rule_version IS NULL) ORDER BY bar_start DESC LIMIT 1;", (feed,))
                row = cursor.fetchone()
                return dict(row) if row else None
            # Prioritize sip_delayed over iex
            cursor = conn.execute("SELECT * FROM signals WHERE feed = 'sip_delayed' AND (rule_version = 1 OR rule_version IS NULL) ORDER BY bar_start DESC LIMIT 1;")
            row = cursor.fetchone()
            if not row:
                cursor = conn.execute("SELECT * FROM signals WHERE (rule_version = 1 OR rule_version IS NULL) ORDER BY bar_start DESC LIMIT 1;")
                row = cursor.fetchone()
            return dict(row) if row else None

    def get_signal_count(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM signals WHERE (rule_version = 1 OR rule_version IS NULL);")
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_signal_by_bar_start(self, bar_start: str, feed: Optional[str] = None, rule_version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            if feed and rule_version:
                cursor = conn.execute("SELECT * FROM signals WHERE bar_start = ? AND feed = ? AND rule_version = ? LIMIT 1;", (bar_start, feed, rule_version))
            elif feed:
                cursor = conn.execute("SELECT * FROM signals WHERE bar_start = ? AND feed = ? AND (rule_version = 1 OR rule_version IS NULL) LIMIT 1;", (bar_start, feed))
            else:
                cursor = conn.execute("SELECT * FROM signals WHERE bar_start = ? AND (rule_version = 1 OR rule_version IS NULL) ORDER BY CASE WHEN feed = 'sip_delayed' THEN 1 ELSE 2 END ASC LIMIT 1;", (bar_start,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_signals(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
            SELECT s.*, o.realized_fwd_ret, o.y_true, o.correct
            FROM signals s
            LEFT JOIN outcomes_v2 o ON s.bar_start = o.bar_start
            WHERE (s.rule_version = 1 OR s.rule_version IS NULL)
            ORDER BY s.bar_start DESC
            LIMIT ? OFFSET ?;
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def insert_signal(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a signal record into SQLite with cryptographic hash chaining."""
        feed_val = record.get("feed", "sip_delayed")
        rule_v = int(record.get("rule_version", 1))
        existing = self.get_signal_by_bar_start(record["bar_start"], feed=feed_val, rule_version=rule_v)
        if existing is not None:
            return existing

        with self.get_connection() as conn:
            # Find previous hash (chronological insertion order by rowid)
            cursor = conn.execute("SELECT hash FROM signals ORDER BY rowid DESC LIMIT 1;")
            prev_row = cursor.fetchone()
            prev_hash = prev_row["hash"] if prev_row else "GENESIS_0000000000000000000000000000000000000000000000000000000000000000"

            record["prev_hash"] = prev_hash
            record["sample"] = record.get("sample", "LIVE")
            record["feed"] = feed_val
            schema_v = int(record.get("schema_v", 3 if rule_v == 2 else (2 if feed_val == "sip_delayed" else 1)))
            record["schema_v"] = schema_v
            record["rule_version"] = rule_v
            if "strong" in record and record["strong"] is not None:
                record["strong"] = int(record["strong"])
            if "late" in record and record["late"] is not None:
                record["late"] = int(record["late"])

            # Compute sha256 of canonical record representation
            canonical_dict = self._build_canonical_dict(record, schema_v)
            canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
            record_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
            record["hash"] = record_hash

            conn.execute("""
            INSERT INTO signals (
                bar_start, bar_end, horizon, direction, confidence, strong,
                vol_class, band_low, band_high, regime, hit_rate_hour_side,
                n_hour_side, model_sha256, config_sha256, git_commit, feed,
                generated_at, latency_s, late, sample, prev_hash, hash,
                delivered_at, delivery_delay_s, schema_v, rule_version,
                p_cal, q_lo, q_hi, tier
            ) VALUES (
                :bar_start, :bar_end, :horizon, :direction, :confidence, :strong,
                :vol_class, :band_low, :band_high, :regime, :hit_rate_hour_side,
                :n_hour_side, :model_sha256, :config_sha256, :git_commit, :feed,
                :generated_at, :latency_s, :late, :sample, :prev_hash, :hash,
                :delivered_at, :delivery_delay_s, :schema_v, :rule_version,
                :p_cal, :q_lo, :q_hi, :tier
            );
            """, {
                **record,
                "delivered_at": record.get("delivered_at"),
                "delivery_delay_s": record.get("delivery_delay_s"),
                "schema_v": schema_v,
                "rule_version": rule_v,
                "p_cal": record.get("p_cal"),
                "q_lo": record.get("q_lo", 0.43428),
                "q_hi": record.get("q_hi", 0.56323),
                "tier": record.get("tier")
            })
            conn.commit()

        return record

    def insert_outcome(self, bar_start: str, realized_fwd_ret: float, y_true: int, correct: Optional[int]):
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.get_connection() as conn:
            # Insert into outcomes_v2 (nullable correct)
            conn.execute("""
            INSERT OR REPLACE INTO outcomes_v2 (
                bar_start, realized_fwd_ret, y_true, correct, recorded_at
            ) VALUES (?, ?, ?, ?, ?);
            """, (bar_start, realized_fwd_ret, y_true, correct, now_str))
            # Also keep old outcomes table updated
            conn.execute("""
            INSERT OR REPLACE INTO outcomes (
                bar_start, realized_fwd_ret, y_true, correct, recorded_at
            ) VALUES (?, ?, ?, ?, ?);
            """, (bar_start, realized_fwd_ret, y_true, correct if correct is not None else 0, now_str))
            conn.commit()

    @staticmethod
    def _build_canonical_dict(record: Dict[str, Any], schema_v: int) -> Dict[str, Any]:
        if schema_v == 1:
            keys = [
                "bar_start", "bar_end", "horizon", "direction", "confidence",
                "strong", "vol_class", "band_low", "band_high", "regime",
                "hit_rate_hour_side", "n_hour_side", "model_sha256",
                "config_sha256", "git_commit", "feed", "generated_at",
                "latency_s", "late", "sample", "prev_hash"
            ]
        elif schema_v == 2:
            keys = [
                "bar_start", "bar_end", "horizon", "direction", "confidence",
                "strong", "vol_class", "band_low", "band_high", "regime",
                "hit_rate_hour_side", "n_hour_side", "model_sha256",
                "config_sha256", "git_commit", "feed", "generated_at",
                "latency_s", "late", "sample", "prev_hash",
                "delivered_at", "delivery_delay_s", "schema_v"
            ]
        else:
            keys = [
                "bar_start", "bar_end", "horizon", "direction", "confidence",
                "strong", "vol_class", "band_low", "band_high", "regime",
                "hit_rate_hour_side", "n_hour_side", "model_sha256",
                "config_sha256", "git_commit", "feed", "generated_at",
                "latency_s", "late", "sample", "prev_hash",
                "delivered_at", "delivery_delay_s", "schema_v",
                "rule_version", "p_cal", "q_lo", "q_hi", "tier"
            ]
        d = {}
        for k in keys:
            val = record.get(k)
            if k in ("strong", "late", "n_hour_side", "schema_v", "rule_version") and val is not None:
                val = int(val)
            elif k in ("confidence", "band_low", "band_high", "hit_rate_hour_side", "latency_s", "delivery_delay_s", "p_cal", "q_lo", "q_hi") and val is not None:
                val = float(val)
            d[k] = val
        return d

    def verify_hash_chain(self) -> Tuple[bool, int, Optional[str]]:
        """Verify the cryptographic integrity of the signals table from genesis."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM signals ORDER BY rowid ASC;")
            rows = cursor.fetchall()

        expected_prev_hash = "GENESIS_0000000000000000000000000000000000000000000000000000000000000000"

        for i, row in enumerate(rows):
            d = dict(row)
            if d["prev_hash"] != expected_prev_hash:
                return False, i, f"Chain broken at bar {d['bar_start']} (feed={d.get('feed')}): prev_hash mismatch"

            schema_v = int(d.get("schema_v") or 1)
            canonical_dict = self._build_canonical_dict(d, schema_v)
            canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
            computed_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

            if computed_hash != d["hash"]:
                return False, i, f"Hash tamper detected at bar {d['bar_start']} (feed={d.get('feed')}): expected {computed_hash}, found {d['hash']}"

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
