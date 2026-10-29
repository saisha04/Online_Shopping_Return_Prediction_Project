"""Training logic shared by the command line (train_model.py) and the Streamlit UI."""
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FILE = "return_model.joblib"
META_FILE = "model_metadata.json"
MIN_ROWS = 30
LEAK_HINTS = ("refund", "return_reason", "returned", "return_date", "return_status", "exchange")
POSITIVE_WORDS = {"yes", "y", "1", "1.0", "true", "t", "returned", "return"}


def guess_positive_class(classes, counts=None):
    """Pick the class that means 'returned'. Falls back to the rarest class."""
    classes = [str(c) for c in classes]
    for c in classes:
        if c.strip().lower() in POSITIVE_WORDS:
            return c
    if counts:
        return min(classes, key=lambda c: counts.get(c, 0))
    return classes[-1] if classes else None


def guess_target(df, candidates):
    """Pick the most likely 'returned / not returned' column among candidate columns."""
    exact = {"returned", "isreturned", "returnflag", "return", "returnstatus", "target", "label", "wasreturned"}
    binary_words = {"yes", "no", "y", "n", "true", "false", "0", "1", "returned", "kept", "notreturned"}

    def score(col):
        n = _norm(col)
        if n in exact:
            return 0
        if n.startswith("return") or n.endswith("returned"):
            return 1
        vals = {str(v).strip().lower().replace(" ", "") for v in df[col].dropna().unique()}
        if len(vals) == 2 and vals <= binary_words:
            return 2
        return 3

    return min(candidates, key=lambda c: (score(c), -list(candidates).index(c)))


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _best_f1_threshold(y_true, p):
    best_t, best_score = 0.50, -1.0
    for t in np.linspace(0.40, 0.60, 41):
        pred = p >= t
        acc = float(np.mean(pred == y_true))
        tp = int(np.sum(pred & y_true)); fp = int(np.sum(pred & ~y_true)); fn = int(np.sum(~pred & y_true))
        prec = (tp / (tp + fp)) if (tp + fp) else 0.0
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        score = acc + 0.5 * f1 + (0.1 if prec >= 0.5 else -0.2)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _atomic_save(model_dir, model, metadata):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    mp, jp = model_dir / MODEL_FILE, model_dir / META_FILE
    tmp_m, tmp_j = model_dir / (MODEL_FILE + ".tmp"), model_dir / (META_FILE + ".tmp")
    joblib.dump(model, tmp_m)
    tmp_j.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    os.replace(tmp_m, mp)
    os.replace(tmp_j, jp)


def train_from_dataframe(df, target, positive_class=None, model_dir=None):
    """Train, evaluate and save a model. Raises ValueError with a readable message on bad input."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    work = df.dropna(subset=[target]).copy()
    if len(work) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} labelled rows; found {len(work)}.")
    y = work[target].astype(str).str.strip()
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("Target column needs at least 2 different classes.")
    if len(counts) > 50:
        raise ValueError("Target has more than 50 distinct values - pick the column that says returned / not returned.")
    if counts.min() < 2:
        raise ValueError(f"Class '{counts.idxmin()}' has only {int(counts.min())} row(s); each class needs at least 2.")

    X = work.drop(columns=[target]).copy()
    warnings, dropped = [], {}

    def drop(cols, why):
        for c in cols:
            dropped[str(c)] = why
        return X.drop(columns=cols)

    X = drop([c for c in X.columns if X[c].isna().all()], "entirely empty")
    X = drop([c for c in X.columns if _norm(c) in {"id", "orderid", "customerid", "transactionid", "userid"}
              or (_norm(c).endswith("id") and X[c].nunique(dropna=True) > 0.9 * len(X))], "identifier")
    X = drop([c for c in X.columns if X[c].nunique(dropna=True) <= 1], "constant value")
    numeric_like = set(X.select_dtypes(include=["number", "bool"]).columns)
    X = drop([c for c in X.columns if c not in numeric_like and X[c].nunique(dropna=True) > 50
              and X[c].nunique(dropna=True) > 0.5 * len(X)], "text/date column with too many distinct values")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing identifiers/empty/constant columns.")
    for k, why in dropped.items():
        if why != "identifier":
            warnings.append(f"Column '{k}' was not used ({why}).")
    leaky = [c for c in X.columns if any(h in str(c).lower() for h in LEAK_HINTS)]
    if leaky:
        warnings.append(f"Possible target leakage: {leaky} may reveal the outcome. Remove them from the CSV if they are only known after a return.")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    for c in categorical_features:
        X[c] = X[c].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v).strip())

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                  ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                             categorical_features))
    model = Pipeline([("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
                      ("classifier", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=4,
                                                            random_state=42, class_weight="balanced_subsample"))])

    classes = sorted(counts.index.tolist())
    binary = len(classes) == 2
    pos = None
    if binary:
        pos = str(positive_class) if positive_class is not None else guess_positive_class(classes, counts.to_dict())
        if pos not in classes:
            raise ValueError(f"Positive class {pos!r} is not one of {classes}.")
    neg = next((c for c in classes if c != pos), None) if binary else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    threshold = 0.5
    if binary:
        n_splits = int(min(5, y_train.value_counts().min()))
        if n_splits >= 2 and y_train.nunique() == 2:
            oof = cross_val_predict(clone(model), X_train, y_train,
                                    cv=StratifiedKFold(n_splits, shuffle=True, random_state=42),
                                    method="predict_proba")[:, classes.index(pos)]
            threshold = _best_f1_threshold((y_train == pos).to_numpy(), oof)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    if binary:
        p_pos = proba[:, list(model.classes_).index(pos)]
        pred = np.where(p_pos >= threshold, pos, neg)
    else:
        pred = model.predict(X_test)

    metrics = {"accuracy": float(accuracy_score(y_test, pred)),
               "baseline_accuracy": float(y_test.value_counts(normalize=True).max())}
    if binary:
        metrics["precision"] = float(precision_score(y_test, pred, pos_label=pos, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, pred, pos_label=pos, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, pred, pos_label=pos, zero_division=0))
        try:
            metrics["roc_auc"] = float(roc_auc_score((y_test == pos).to_numpy(), p_pos))
        except ValueError:
            metrics["roc_auc"] = None
        if metrics["roc_auc"] is not None and metrics["roc_auc"] > 0.98 and leaky:
            warnings.append("Very high ROC AUC - check that no column leaks the outcome.")
    try:
        scoring = "roc_auc" if binary else "accuracy"
        pi = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=42)
    except Exception:
        pi = None
    importances = []
    if pi is not None:
        importances = sorted(({"feature": str(f), "importance": float(m), "std": float(s)}
                              for f, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)),
                             key=lambda d: d["importance"], reverse=True)

    feature_defaults, feature_ranges = {}, {}
    for c in numeric_features:
        col = pd.to_numeric(X[c], errors="coerce")
        feature_defaults[c] = float(col.median()) if col.notna().any() else 0.0
        feature_ranges[c] = [float(col.min()), float(col.max())] if col.notna().any() else [0.0, 0.0]
    for c in categorical_features:
        feature_defaults[c] = X[c].dropna().astype(str).mode().iloc[0] if X[c].notna().any() else ""
    integer_features = [c for c in numeric_features if c not in X.select_dtypes(include=["bool"]).columns
                        and (pd.to_numeric(X[c], errors="coerce").dropna() % 1 == 0).all()]

    metadata = {
        "target_column": str(target),
        "feature_names": list(X.columns),
        "numeric_features": numeric_features,
        "integer_features": integer_features,
        "bool_features": X.select_dtypes(include=["bool"]).columns.tolist(),
        "feature_types": {c: ("numeric" if c in numeric_features else "text") for c in X.columns},
        "feature_options": {c: X[c].dropna().astype(str).value_counts().index.tolist()[:40] for c in categorical_features},
        "feature_defaults": feature_defaults,
        "feature_ranges": feature_ranges,
        "classes": [str(v) for v in model.classes_],
        "positive_class": pos,
        "decision_threshold": threshold,
        "test_accuracy": metrics["accuracy"],
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "confusion_labels": classes,
        "classification_report_text": classification_report(y_test, pred, zero_division=0),
        "importances": importances,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "n_rows": int(len(work)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "dropped_columns": dropped,
        "warnings": warnings,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
    }
    if model_dir is not None:
        _atomic_save(model_dir, model, metadata)
    return metadata
