# Deploying ReturnWise

ReturnWise is one Streamlit process plus a folder of data (SQLite database, uploaded dataset, trained model).
Run **one instance** and keep the data folder on **persistent storage**.

## 1. Docker Compose (any server with Docker) — recommended
```bash
cp .env.example .env            # edit: set RETURNWISE_ADMIN_USER / RETURNWISE_ADMIN_PASSWORD
docker compose up -d --build
```
Open http://SERVER:8501, sign in, then **Data & Training** → upload your CSV → **Train model**.
Data lives in the `returnwise_data` volume and survives restarts and rebuilds.

Backup:  `docker run --rm -v returnwise_data:/data -v "$PWD":/backup alpine tar czf /backup/returnwise-data.tgz -C /data .`

## 2. HTTPS (required for real use — passwords travel over the network)
Put a reverse proxy in front. Caddy example (automatic certificates):
```
returns.example.com {
    reverse_proxy localhost:8501
}
```
nginx must forward WebSocket headers (`Upgrade` / `Connection`) or the app will appear to hang.

## 3. Render / Railway / Fly.io (Docker or Procfile)
- Build from the `Dockerfile` (or use the `Procfile` start command).
- Attach a **persistent disk mounted at `/data`** and set `RETURNWISE_DATA_DIR=/data`.
- Set `RETURNWISE_ADMIN_USER` and `RETURNWISE_ADMIN_PASSWORD` as secrets.
- Do not scale beyond one instance (SQLite).

## 4. Streamlit Community Cloud
Works for a demo only: its disk is erased on restart, so users, uploaded data, the model and history disappear.

## Environment variables
| Variable | Purpose |
|---|---|
| `RETURNWISE_DATA_DIR` | folder for database, `dataset/`, `model/` (default: project folder; `/data` in Docker) |
| `RETURNWISE_ADMIN_USER` / `RETURNWISE_ADMIN_PASSWORD` | create the first administrator on first start (min. 10-character password) |
| `RETURNWISE_SETUP_TOKEN` | if set, the first-run setup screen requires it |
| `RETURNWISE_AUTH` | `off` disables login (local development only) |

## Before going live
- [ ] Change/remove the bootstrap password from your environment after first sign-in; use **Account** to set your own.
- [ ] HTTPS in front of the app.
- [ ] Persistent volume + a backup routine.
- [ ] Train with your **real** dataset (the bundled sample is synthetic).
- [ ] Review what the model is used for: scores are estimates, not guarantees, and have not been checked for bias.

## Operations
- Health check: `GET /_stcore/health` returns `ok`.
- Logs: standard output (errors include a reference ID shown to the user).
- Updating: `git pull && docker compose up -d --build` — the database upgrades itself.
- Lost admin password: stop the app, then run
  `python -c "import auth; auth.init_users(); auth.reset_password(1, 'NewLongPassword!')"` in the project/data context (user id 1 is the first admin).
