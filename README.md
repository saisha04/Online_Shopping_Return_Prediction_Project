# ReturnWise — Online Shopping Return Prediction

Streamlit + scikit-learn + SQLite app that estimates how likely an order is to be returned.

## Quick start (local)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py        # or double-click run_app.bat (Windows) / ./run_app.sh
```
On first start the app asks you to create the **administrator** account (or set `RETURNWISE_ADMIN_USER` / `RETURNWISE_ADMIN_PASSWORD`). To skip login on your own PC: `RETURNWISE_AUTH=off`.

Then, inside the app (sidebar):
1. **Data & Training** → upload your order-history CSV (or `sample_data/sample_orders_SYNTHETIC.csv` to try it), choose the *returned* column, click **Train model**.
2. **Predict Return** → one order, with a risk gauge and a "why this score" chart.
3. **Batch Scoring** → upload a CSV of orders, get a ranked risk list to download.
4. **Prediction History** → record what really happened to measure real-world accuracy.
5. **Dashboard / Analytics** → model quality, guided setup, return rate by segment.

No dataset yet? `sample_data/sample_orders_SYNTHETIC.csv` is **randomly generated** test data. Never treat its numbers as real.

## Command-line alternative
Put your CSV at `dataset/online shopping returns.csv`, then:
```bash
python check_dataset.py
python train_model.py --target returned        # add --positive Yes to name the 'returned' label
```

## CSV requirements
One row per order; a column that says whether the order was returned (2 classes, e.g. Yes/No); at least 30 labelled rows and 2 rows per class. Identifier, empty, constant and date-like free-text columns are ignored automatically (listed in the training report). Do not include columns only known *after* a return (refund amount, return reason) — they leak the answer; the app warns about likely ones.

## How the model works
Random forest with median/most-frequent imputation. The decision threshold is tuned (cross-validation on the training split) to balance precision and recall, then evaluated once on a held-out 20% test split. The app shows accuracy next to the "always guess the majority" baseline, plus precision, recall, ROC AUC, a confusion matrix and permutation importance.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
Tests use small synthetic data created in a temporary folder. `tests/test_app_streamlit.py` opens every page with Streamlit's AppTest.

## Roles
- **admin** — everything: upload data, train, manage users, clear history.
- **analyst** — predict, batch score, review history and record outcomes.

## Deployment
See **DEPLOY.md** (Docker Compose, HTTPS, Render/Railway, backups).

## Notes
- Existing `returnwise_history.db` files are upgraded automatically (new columns are added; no data is lost).
- Uploaded datasets replace the current one after backing it up to `dataset/backups/`.
- Model files are Python pickles — only load models you trained yourself.
