#!/usr/bin/env python3
"""
Additive Schema Migration for Rule V2 (No-Signal Band 47% to 53%).
Migrates signals table primary key to (bar_start, feed, rule_version)
and adds columns: rule_version, p_cal, q_lo, q_hi, tier.
Zero UPDATEs or DELETEs of existing data. Hash chain verification preserved.
"""

import sqlite3
import shutil
from pathlib import Path
from src.live.store import LiveDatabase

def migrate_db(db_path: Path = Path("data/live/live.db")):
    print(f"Starting schema migration on {db_path}...")
    backup_path = db_path.with_suffix(".db.bak_rule2")
    shutil.copy(db_path, backup_path)
    print(f"Created backup at {backup_path}")

    db = LiveDatabase(db_path)
    valid_orig, count_orig, err_orig = db.verify_hash_chain()
    print(f"Pre-migration hash chain check: valid={valid_orig}, count={count_orig}")
    assert valid_orig, f"Pre-migration hash chain broken: {err_orig}"

    with db.get_connection() as conn:
        conn.execute("DROP VIEW IF EXISTS signals_iex;")
        conn.execute("""
        CREATE TABLE signals_new (
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
        conn.execute("""
        INSERT INTO signals_new (
            bar_start, bar_end, horizon, direction, confidence, strong,
            vol_class, band_low, band_high, regime, hit_rate_hour_side,
            n_hour_side, model_sha256, config_sha256, git_commit, feed,
            generated_at, latency_s, late, sample, prev_hash, hash,
            delivered_at, delivery_delay_s, schema_v, rule_version,
            p_cal, q_lo, q_hi, tier
        )
        SELECT
            bar_start, bar_end, horizon, direction, confidence, strong,
            vol_class, band_low, band_high, regime, hit_rate_hour_side,
            n_hour_side, model_sha256, config_sha256, git_commit, feed,
            generated_at, latency_s, late, sample, prev_hash, hash,
            delivered_at, delivery_delay_s, schema_v, 1,
            NULL, 0.43428, 0.56323, CASE WHEN strong=1 THEN 'STRONG' ELSE 'NORMAL' END
        FROM signals;
        """)
        conn.execute("DROP TRIGGER IF EXISTS prevent_signals_update;")
        conn.execute("DROP TRIGGER IF EXISTS prevent_signals_delete;")
        conn.execute("DROP TABLE signals;")
        conn.execute("ALTER TABLE signals_new RENAME TO signals;")
        conn.execute("CREATE VIEW signals_iex AS SELECT * FROM signals WHERE feed='iex';")
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
        conn.commit()

    valid_post, count_post, err_post = db.verify_hash_chain()
    print(f"Post-migration hash chain check: valid={valid_post}, count={count_post}, err={err_post}")
    assert valid_post and count_post == count_orig, f"Post-migration check failed: {err_post}"
    print("Schema migration successfully completed!")

if __name__ == "__main__":
    migrate_db()
