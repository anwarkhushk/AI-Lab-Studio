"""
storage/users.py – User CRUD and password utilities.
Uses hashlib (SHA-256 + salt) so no extra dependencies needed.
"""

import hashlib
import os
from .db import get_db


# ── Password hashing ──────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == h
    except Exception:
        return False


# ── Existence checks ──────────────────────────────────────────────────────────

def username_exists(username: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM users WHERE LOWER(username)=LOWER(?)", (username,)
    ).fetchone()
    conn.close()
    return row is not None


def email_exists(email: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM users WHERE LOWER(email)=LOWER(?)", (email,)
    ).fetchone()
    conn.close()
    return row is not None


# ── Create / Fetch ────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str) -> dict | None:
    """Insert new user; return user dict or None on failure."""
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), _hash_password(password)),
        )
        conn.commit()
        return {"id": cur.lastrowid, "username": username, "email": email}
    except Exception:
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE LOWER(username)=LOWER(?)", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE LOWER(email)=LOWER(?)", (email.lower(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
