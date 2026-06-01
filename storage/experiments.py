"""
storage/experiments.py – Experiment save / list / delete operations.
"""

import json
from .db import get_db


def save_experiment(
    user_id: int,
    algorithm: str,
    parameters: dict = None,
    results: dict = None,
    accuracy: float = None,
    cost: float = None,
    path_length: int = None,
    notes: str = "",
) -> int:
    """Persist an experiment; returns the new row id."""
    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO experiments
            (user_id, algorithm, parameters, results, accuracy, cost, path_length, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            algorithm,
            json.dumps(parameters or {}),
            json.dumps(results or {}),
            accuracy,
            cost,
            path_length,
            notes,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_experiments(user_id: int) -> list[dict]:
    """Return all experiments for a user, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM experiments WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["parameters"] = json.loads(d["parameters"] or "{}")
        d["results"] = json.loads(d["results"] or "{}")
        result.append(d)
    return result


def delete_experiment(exp_id: int, user_id: int) -> bool:
    conn = get_db()
    conn.execute(
        "DELETE FROM experiments WHERE id=? AND user_id=?", (exp_id, user_id)
    )
    conn.commit()
    conn.close()
    return True
