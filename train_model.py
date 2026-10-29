"""Command-line training. The same logic is available in the app under 'Data & Training'."""
import argparse
from pathlib import Path

from ml_utils import find_dataset
from trainer import train_from_dataframe

from settings import DATA_DIR

ROOT = DATA_DIR
MODEL_DIR = ROOT / "model"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the return-prediction model.")
    parser.add_argument("--target", help="exact name of the returned/not-returned column (skips the prompt)")
    parser.add_argument("--positive", help="label that means 'returned' (auto-detected if omitted)")
    args = parser.parse_args(argv)

    df, path = find_dataset()
    if df is None:
        raise SystemExit("Dataset not found. Put it in dataset/online shopping returns.csv")
    print(f"\nLoaded {path}: {df.shape[0]} rows x {df.shape[1]} columns\n\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col} (unique values: {df[col].nunique(dropna=True)})")

    raw = (args.target or input("\nType the exact target column name (the column indicating returned/not returned): ")).strip()
    matches = [c for c in df.columns if str(c).strip().lower() == raw.lower()]
    if not matches:
        raise SystemExit("Target column not found. Run again and type the exact column name shown above.")
    try:
        r = train_from_dataframe(df, matches[0], positive_class=args.positive, model_dir=MODEL_DIR)
    except ValueError as e:
        raise SystemExit(f"Training failed: {e}")

    for w in r["warnings"]:
        print("WARNING:", w)
    m = r["metrics"]
    print("\n===== MODEL EVALUATION (held-out test set) =====")
    print(f"Test accuracy: {m['accuracy']*100:.1f}%   (always guessing the majority class: {m['baseline_accuracy']*100:.1f}%)")
    if r["positive_class"]:
        print(f"'Returned' label: {r['positive_class']}   decision threshold: {r['decision_threshold']:.2f}")
        auc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a"
        print(f"Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | F1 {m['f1']:.3f} | ROC AUC {auc}")
    print("\nClassification report:\n" + r["classification_report_text"])
    print("Confusion matrix (rows=actual, cols=predicted; order:", r["confusion_labels"], ")")
    print(r["confusion_matrix"])
    print("\nSaved:\n-", MODEL_DIR / "return_model.joblib", "\n-", MODEL_DIR / "model_metadata.json")
    print("\nNote: accuracy alone is not enough; review precision, recall, and class balance too.")


if __name__ == "__main__":
    main()
