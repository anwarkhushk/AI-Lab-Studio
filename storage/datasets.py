"""
storage/datasets.py – Dataset metadata save / list / delete.
Actual CSV files are stored on disk; metadata lives in SQLite.
"""

import json
import os
import shutil
from .db import get_db

DATASETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "user_datasets")
)
os.makedirs(DATASETS_DIR, exist_ok=True)


def save_dataset_meta(
    user_id: int,
    name: str,
    rows: int,
    cols: int,
    features: list,
    file_path: str = "",
) -> int:
    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO datasets (user_id, name, rows, cols, features, file_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, name, rows, cols, json.dumps(features), file_path),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_datasets(user_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM datasets WHERE user_id=? ORDER BY upload_date DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["features"] = json.loads(d["features"] or "[]")
        result.append(d)
    return result


def delete_dataset_meta(ds_id: int, user_id: int) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT file_path FROM datasets WHERE id=? AND user_id=?", (ds_id, user_id)
    ).fetchone()
    if row and row["file_path"] and os.path.exists(row["file_path"]):
        try:
            os.remove(row["file_path"])
        except OSError:
            pass
    conn.execute(
        "DELETE FROM datasets WHERE id=? AND user_id=?", (ds_id, user_id)
    )
    conn.commit()
    conn.close()
    return True


def dataset_storage_path(user_id: int, filename: str) -> str:
    """Return a path for storing a user's dataset file."""
    user_dir = os.path.join(DATASETS_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, filename)
