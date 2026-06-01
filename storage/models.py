"""
storage/models.py – Trained model save / list / delete.
Pickled model files stored on disk; metadata in SQLite.
"""

import os
import pickle
import joblib
from .db import get_db

MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "user_models")
)
os.makedirs(MODELS_DIR, exist_ok=True)


def _user_models_dir(user_id: int) -> str:
    d = os.path.join(MODELS_DIR, str(user_id))
    os.makedirs(d, exist_ok=True)
    return d


def save_model_meta(
    user_id: int,
    name: str,
    algorithm: str,
    accuracy: float = None,
    file_path: str = "",
) -> int:
    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO models (user_id, name, algorithm, accuracy, file_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, name, algorithm, accuracy, file_path),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_models(user_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM models WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_model_meta(model_id: int, user_id: int) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT file_path FROM models WHERE id=? AND user_id=?",
        (model_id, user_id),
    ).fetchone()
    if row and row["file_path"] and os.path.exists(row["file_path"]):
        try:
            os.remove(row["file_path"])
        except OSError:
            pass
    conn.execute(
        "DELETE FROM models WHERE id=? AND user_id=?", (model_id, user_id)
    )
    conn.commit()
    conn.close()
    return True


def persist_model(model, user_id: int, filename: str) -> str:
    """Pickle a trained model and return its storage path."""
    path = os.path.join(_user_models_dir(user_id), filename)
    joblib.dump(model, path)
    return path


def load_model(file_path: str):
    """Load a pickled model from disk."""
    return joblib.load(file_path)
