from datetime import datetime
from io import BytesIO
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from settings import DATA_DIR

ROOT = DATA_DIR            # dataset/, model/ and the database live under the data directory
MODEL_PATH = ROOT / "model" / "return_model.joblib"
META_PATH = ROOT / "model" / "model_metadata.json"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_BATCH_ROWS = 50_000


# ------------------------------------------------------------------ dataset
def _read_csv(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except UnicodeDecodeError:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        return pd.read_csv(path_or_buffer, encoding="latin-1")


def _candidates():
    return [
        ROOT / "dataset" / "online shopping returns.csv",
        ROOT / "dataset" / "online_shopping_returns.csv",
        ROOT / "online shopping returns.csv",
    ]


def default_dataset_path():
    return _candidates()[0]


def find_dataset():
    """Return (DataFrame, relative_path) for the first readable dataset, else (None, None)."""
    for path in _candidates():
        if path.exists():
            try:
                df = _read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
                continue
            return df, str(path.relative_to(ROOT))
    return None, None


def dataset_error():
    """Explain why an existing dataset file could not be read (None if fine / absent)."""
    for path in _candidates():
        if path.exists():
            try:
                _read_csv(path)
                return None
            except Exception as e:
                return f"{path.name}: {type(e).__name__}: {e}"
    return None


def dataset_summary(df):
    return {"rows": len(df), "columns": len(df.columns), "missing": int(df.isna().sum().sum())}


def parse_csv_bytes(raw, max_bytes=MAX_UPLOAD_BYTES):
    """Validate and parse an uploaded CSV. Raises ValueError with a readable message."""
    if not raw:
        raise ValueError("The file is empty.")
    if len(raw) > max_bytes:
        raise ValueError(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        df = _read_csv(BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Could not read the file as CSV ({type(e).__name__}: {e}).")
    if df.shape[1] < 2:
        raise ValueError("The CSV needs at least 2 columns (order details plus a returned/not-returned column).")
    if len(df) == 0:
        raise ValueError("The CSV has a header but no rows.")
    return df


def install_dataset(raw):
    """Save validated CSV bytes as the project dataset; the previous dataset is backed up."""
    parse_csv_bytes(raw)
    target = default_dataset_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        backup_dir = target.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"online_shopping_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        backup.write_bytes(target.read_bytes())
    target.write_bytes(raw)
    return backup


# -------------------------------------------------------------------- model
def load_model_bundle_checked():
    """Return (bundle, error). error is None when loaded or when the files simply don't exist."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        bundle = {"model": model, **meta}
        saved = meta.get("sklearn_version")
        if saved and saved != sklearn.__version__:
            bundle["version_warning"] = (f"Model was trained with scikit-learn {saved} but {sklearn.__version__} "
                                         "is installed. Retrain if predictions look wrong.")
        return bundle, None
    except Exception as e:
        return None, f"Model files exist but could not be loaded ({type(e).__name__}: {e}). Retrain the model."


def load_model_bundle():
    return load_model_bundle_checked()[0]


def clean_value(v):
    """Blank text inputs become missing values so the model's imputer handles them."""
    if isinstance(v, str) and not v.strip():
        return None
    return v


def _prep_frame(bundle, frame):
    """Align a raw frame with the training data types. Returns (frame, notes)."""
    frame = frame[bundle["feature_names"]].copy()
    numeric, notes = set(bundle["numeric_features"]), []
    options_map = bundle.get("feature_options", {})
    for name in bundle["feature_names"]:
        s = frame[name].astype(object)
        if name in numeric:
            present = s.notna() & (s.astype(str).str.strip() != "")
            conv = pd.to_numeric(s, errors="coerce")
            bad = int((present & conv.isna()).sum())
            if bad:
                notes.append(f"{bad} non-numeric value(s) in '{name}' were treated as missing.")
            frame[name] = conv
        else:
            known_opts = options_map.get(name, [])
            lookup = {str(opt).strip().lower(): str(opt).strip() for opt in known_opts}
            if {"yes", "no"} <= set(lookup.keys()):
                lookup.update({"true": "Yes", "t": "Yes", "1": "Yes", "false": "No", "f": "No", "0": "No"})

            def _clean_cat(v):
                if (not isinstance(v, str) and pd.isna(v)) or (isinstance(v, str) and not v.strip()):
                    return np.nan
                val_str = str(v).strip()
                return lookup.get(val_str.lower(), val_str)

            frame[name] = s.map(_clean_cat)
    return frame, notes


def risk_level(p, threshold=0.5):
    if p is None:
        return None
    if p >= threshold:
        return "High"
    if p >= threshold * 0.6:
        return "Medium"
    return "Low"


def _score(bundle, frame):
    """Return (proba matrix, classes list as str)."""
    model = bundle["model"]
    return model.predict_proba(frame), [str(c) for c in model.classes_]


def predict_one(bundle, values):
    raw = pd.DataFrame([{name: clean_value(values.get(name)) for name in bundle["feature_names"]}])
    frame, _ = _prep_frame(bundle, raw)
    proba, classes = _score(bundle, frame)
    class_probs = {c: float(p) for c, p in zip(classes, proba[0])}
    pos, thr = bundle.get("positive_class"), bundle.get("decision_threshold") or 0.5
    if pos is not None and pos in class_probs and len(classes) == 2:
        risk = class_probs[pos]
        other = next(c for c in classes if c != pos)
        label = pos if risk >= thr else other
        return {"label": label, "probability": class_probs[label], "risk_probability": risk,
                "risk_level": risk_level(risk, thr), "threshold": thr, "class_probabilities": class_probs}
    label = max(class_probs, key=class_probs.get)
    return {"label": label, "probability": class_probs[label], "risk_probability": None,
            "risk_level": None, "threshold": None, "class_probabilities": class_probs}


def predict_batch(bundle, df):
    """Score every row of df. Returns (scored DataFrame, notes). Raises ValueError on bad input."""
    missing = [f for f in bundle["feature_names"] if f not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if len(df) == 0:
        raise ValueError("The file has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"Too many rows ({len(df):,}); the limit is {MAX_BATCH_ROWS:,}.")
    frame, notes = _prep_frame(bundle, df)
    proba, classes = _score(bundle, frame)
    out = df.copy()
    pos, thr = bundle.get("positive_class"), bundle.get("decision_threshold") or 0.5
    if pos is not None and pos in classes and len(classes) == 2:
        risk = proba[:, classes.index(pos)]
        other = next(c for c in classes if c != pos)
        out["return_probability"] = np.round(risk, 4)
        out["risk_level"] = [risk_level(float(p), thr) for p in risk]
        out["predicted_outcome"] = np.where(risk >= thr, pos, other)
    else:
        out["predicted_outcome"] = [classes[i] for i in proba.argmax(axis=1)]
        out["confidence"] = np.round(proba.max(axis=1), 4)
    return out, notes


def explain_one(bundle, values, top=6):
    """What-if explanation: how much does each value move the return risk versus a typical order?

    Positive effect = this value raises the risk relative to the typical (median/most common) value.
    """
    pos = bundle.get("positive_class")
    defaults = bundle.get("feature_defaults") or {}
    if pos is None or not defaults:
        return None
    cleaned = {n: clean_value(values.get(n)) for n in bundle["feature_names"]}
    rows, changed = [cleaned], []
    for name in bundle["feature_names"]:
        base = defaults.get(name)
        if base is None or cleaned[name] is None or str(cleaned[name]) == str(base):
            continue
        row = dict(cleaned)
        row[name] = base
        rows.append(row)
        changed.append(name)
    frame, _ = _prep_frame(bundle, pd.DataFrame(rows))
    proba, classes = _score(bundle, frame)
    idx = classes.index(pos)
    risk = proba[:, idx]
    effects = [{"feature": n, "value": cleaned[n], "typical": defaults[n], "effect": float(risk[0] - risk[i + 1])}
               for i, n in enumerate(changed)]
    effects.sort(key=lambda d: abs(d["effect"]), reverse=True)
    return effects[:top]


def range_warnings(bundle, values):
    """Flag inputs the model has never seen, where predictions are less reliable."""
    msgs = []
    for name, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        v = values.get(name)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v < lo or v > hi:
            msgs.append(f"'{name}' = {v:g} is outside the training range ({lo:g} to {hi:g}).")
    for name, options in (bundle.get("feature_options") or {}).items():
        v = clean_value(values.get(name))
        if v is not None and options and len(options) < 40 and str(v) not in options:
            msgs.append(f"'{name}' = {v!r} was not seen in the training data.")
    return msgs


# ------------------------------------------------------------------- tables
def history_frame(history):
    """Flatten saved predictions into a table that displays and exports cleanly."""
    rows = []
    for h in history:
        row = {"id": h["id"], "created_at": h["created_at"], "source": h.get("source"), "created_by": h.get("created_by")}
        row.update({f"input: {k}": v for k, v in h["inputs"].items()})
        row["prediction"] = h["prediction"]
        row["return_probability"] = h.get("risk_probability")
        row["risk_level"] = h.get("risk_level")
        row["actual_outcome"] = h.get("actual_outcome")
        rows.append(row)
    return pd.DataFrame(rows)


def csv_safe(df):
    """Neutralise spreadsheet formula injection (cells starting with = + - @) before CSV export."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v and v[0] in "=+-@\t\r" else v)
    return out


def segment_rates(df, column, target, positive, bins=5):
    """Return rate per segment of `column`, ordered sensibly for charting.

    Numeric columns are split into quantile bins (kept in numeric order, never text order);
    categorical columns are sorted by return rate. Missing values form their own segment.
    """
    s = df[column]
    y = df[target].astype(str).str.strip() == str(positive)
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if numeric and s.nunique() > 12:
        try:
            seg = pd.qcut(s, q=bins, duplicates="drop")
        except ValueError:
            seg = s.astype(str)
        ordered = True
    elif numeric:
        seg, ordered = s, True
    else:
        seg, ordered = s.astype(object).where(s.notna(), "(missing)").astype(str), False
    frame = pd.DataFrame({"segment": seg, "returned": y})
    g = frame.groupby("segment", observed=True, dropna=False)["returned"].agg(["mean", "count"]).reset_index()
    g["segment"] = g["segment"].astype(str).replace({"nan": "(missing)"})
    g["return_rate_pct"] = g["mean"] * 100
    g = g.drop(columns="mean")
    if not ordered:
        g = g.sort_values("return_rate_pct", ascending=False).reset_index(drop=True)
    return g
