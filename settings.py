"""Deployment settings, read from environment variables (see .env.example)."""
import os
from pathlib import Path

APP_DIR = Path(__file__).parent

# Read .env file if present
_env_file = APP_DIR / ".env"
if _env_file.exists():
    try:
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip().strip("'\"")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass

# Where the dataset, trained model and SQLite database live. In Docker this is the mounted volume (/data).
DATA_DIR = Path(os.environ.get("RETURNWISE_DATA_DIR") or APP_DIR).expanduser()

# Login is ON by default. Only switch it off for local single-user development.
AUTH_ENABLED = os.environ.get("RETURNWISE_AUTH", "on").strip().lower() not in {"off", "0", "false", "no"}

# Optional: create the first administrator automatically on first start.
ADMIN_USER = os.environ.get("RETURNWISE_ADMIN_USER", "").strip()
ADMIN_PASSWORD = os.environ.get("RETURNWISE_ADMIN_PASSWORD", "")

# Optional: if set, the first-run "create administrator" screen asks for this token, so a stranger who
# reaches a brand-new deployment first cannot claim the admin account.
SETUP_TOKEN = os.environ.get("RETURNWISE_SETUP_TOKEN", "")
