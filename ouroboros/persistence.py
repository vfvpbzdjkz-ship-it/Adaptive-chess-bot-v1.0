"""Atomic saves, DB init, crash-safe helpers."""
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path("data/ouroboros.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS opponents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            is_bot INTEGER DEFAULT 0,
            title TEXT DEFAULT '',
            last_elo INTEGER DEFAULT 1500,
            games INTEGER DEFAULT 0,
            wins_vs_us INTEGER DEFAULT 0,
            losses_vs_us INTEGER DEFAULT 0,
            draws_vs_us INTEGER DEFAULT 0,
            determinism_score REAL DEFAULT 0.0,
            avg_move_time REAL DEFAULT 0.0,
            last_seen TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS opening_moves (
            opponent_id INTEGER NOT NULL,
            position_key TEXT NOT NULL,
            move_uci TEXT NOT NULL,
            times_played INTEGER DEFAULT 1,
            our_score_after REAL DEFAULT 0.5,
            last_played TEXT DEFAULT '',
            PRIMARY KEY (opponent_id, position_key, move_uci),
            FOREIGN KEY (opponent_id) REFERENCES opponents(id)
        );

        CREATE TABLE IF NOT EXISTS band_opening_moves (
            band INTEGER NOT NULL,
            position_key TEXT NOT NULL,
            move_uci TEXT NOT NULL,
            times_played INTEGER DEFAULT 1,
            our_score_after REAL DEFAULT 0.5,
            last_played TEXT DEFAULT '',
            PRIMARY KEY (band, position_key, move_uci)
        );

        CREATE TABLE IF NOT EXISTS exploit_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opponent_id INTEGER NOT NULL,
            line_uci TEXT NOT NULL,
            result REAL DEFAULT 1.0,
            length INTEGER DEFAULT 0,
            last_used TEXT DEFAULT '',
            times_used INTEGER DEFAULT 0,
            still_valid INTEGER DEFAULT 1,
            diverge_ply INTEGER DEFAULT -1,
            FOREIGN KEY (opponent_id) REFERENCES opponents(id)
        );

        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lichess_id TEXT UNIQUE,
            opponent_username TEXT,
            opponent_id INTEGER,
            our_color TEXT,
            result TEXT,
            pgn TEXT,
            clocks TEXT,
            timestamp TEXT,
            opponent_elo INTEGER,
            opponent_is_bot INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS ladder (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint TEXT,
            elo REAL DEFAULT 1500.0,
            games_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """)
    log.debug("Database initialized at %s", DB_PATH)


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def meta_get(key: str, default: str = "") -> str:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def meta_set(key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def atomic_json_write(path: Path, data: dict) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
