import json
import sqlite3
from contextlib import closing
from datetime import datetime
from settings import DATA_DIR

DB_PATH = DATA_DIR / "returnwise_history.db"

# Columns added after the first release; init_db() migrates older databases in place.
_ADDED_COLUMNS = {"risk_probability": "REAL", "risk_level": "TEXT", "source": "TEXT",
                  "actual_outcome": "TEXT", "actual_at": "TEXT", "created_by": "TEXT"}
OUTCOMES = ("Returned", "Not returned")


def _connect():
    """Connection that is committed on success AND closed afterwards."""
    return closing(sqlite3.connect(DB_PATH))


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as con, con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL
        )
        """)
        existing = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
        for name, kind in _ADDED_COLUMNS.items():
            if name not in existing:
                con.execute(f"ALTER TABLE predictions ADD COLUMN {name} {kind}")


def save_prediction(inputs, prediction, probability=None, risk_probability=None, risk_level=None, source="single",
                    created_by=None):
    with _connect() as con, con:
        cur = con.execute(
            "INSERT INTO predictions (created_at, input_json, prediction, probability, risk_probability, risk_level, source,"
            " created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), json.dumps(inputs, default=str), str(prediction), probability, risk_probability, risk_level, source,
             created_by),
        )
        return cur.lastrowid


def save_predictions_bulk(rows, source="batch", created_by=None):
    """rows: iterable of dicts with keys inputs, prediction, risk_probability, risk_level."""
    data = [(_now(), json.dumps(r["inputs"], default=str), str(r["prediction"]), None,
             r.get("risk_probability"), r.get("risk_level"), source, created_by) for r in rows]
    with _connect() as con, con:
        con.executemany(
            "INSERT INTO predictions (created_at, input_json, prediction, probability, risk_probability, risk_level, source,"
            " created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data)
    return len(data)


def set_actual_outcome(pred_id, outcome):
    """Record what really happened (or None to clear). Returns True if a row was updated."""
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES} or None")
    with _connect() as con, con:
        cur = con.execute("UPDATE predictions SET actual_outcome = ?, actual_at = ? WHERE id = ?",
                          (outcome, _now() if outcome else None, int(pred_id)))
        return cur.rowcount > 0


def get_predictions():
    with _connect() as con:
        rows = con.execute(
            "SELECT id, created_at, input_json, prediction, probability, risk_probability, risk_level, source,"
            " actual_outcome, created_by FROM predictions ORDER BY id DESC").fetchall()
    result = []
    for r in rows:
        try:
            inputs = json.loads(r[2])
        except (TypeError, ValueError):
            inputs = {}
        result.append({"id": r[0], "created_at": r[1], "inputs": inputs, "prediction": r[3], "probability": r[4],
                       "risk_probability": r[5], "risk_level": r[6], "source": r[7], "actual_outcome": r[8],
                       "created_by": r[9]})
    return result


def clear_history():
    with _connect() as con, con:
        con.execute("DELETE FROM predictions")
