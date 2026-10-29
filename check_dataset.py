from ml_utils import find_dataset
df, path = find_dataset()
if df is None:
    print("Dataset not found.")
    print("Expected: dataset/online shopping returns.csv")
else:
    print("Dataset:", path)
    print("Rows:", df.shape[0], "| Columns:", df.shape[1])
    print("\nColumns:")
    for c in df.columns:
        print(f"- {c!r}: {df[c].nunique(dropna=True)} unique, {df[c].isna().sum()} missing")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
