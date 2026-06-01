"""
storage/db.py – SQLite database bootstrap for AI Lab Studio.
Creates all required tables on first run.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ai_lab_studio.db")
DB_PATH = os.path.abspath(DB_PATH)


def get_db() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    cur = conn.cursor()

    # ── users ──────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,   -- bcrypt hash
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── experiments ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            algorithm       TEXT    NOT NULL,
            parameters      TEXT,           -- JSON blob
            results         TEXT,           -- JSON blob
            accuracy        REAL,
            cost            REAL,
            path_length     INTEGER,
            notes           TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # ── datasets ───────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            name            TEXT    NOT NULL,
            rows            INTEGER,
            cols            INTEGER,
            features        TEXT,           -- JSON list
            file_path       TEXT,           -- path to persisted CSV
            upload_date     TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # ── models ─────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            name            TEXT    NOT NULL,
            algorithm       TEXT    NOT NULL,
            accuracy        REAL,
            file_path       TEXT,           -- path to pickled model
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
