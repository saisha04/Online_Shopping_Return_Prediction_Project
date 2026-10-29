"""User accounts, password hashing, roles and login lockout (SQLite, standard library only)."""
import base64
import hashlib
import hmac
import os
import re
from datetime import datetime, timedelta

import database
import settings

ROLES = ("admin", "analyst")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 200
MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 5
ITERATIONS = 390_000
_FMT = "%Y-%m-%d %H:%M:%S"
_DUMMY_HASH = None


# ---------------------------------------------------------------- hashing
def _b64(b):
    return base64.b64encode(b).decode("ascii")


def hash_password(password, iterations=ITERATIONS):
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(dk)}"


def verify_password(password, stored):
    try:
        scheme, iters, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(digest))
    except (ValueError, TypeError, AttributeError):
        return False


def _dummy_verify(password):
    """Spend the same time as a real check so unknown usernames can't be detected by timing."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("dummy-password-for-timing")
    verify_password(password, _DUMMY_HASH)


# ------------------------------------------------------------ validation
def normalize_username(username):
    return (username or "").strip().lower()


def validate_username(username):
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 characters: lowercase letters, digits, '.', '_' or '-'.")


def validate_password(password):
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    if password.strip() == "" or password.lower() in {"password123", "1234567890", "qwertyuiop"}:
        raise ValueError("Choose a less guessable password.")


# ------------------------------------------------------------- storage
def init_users():
    database.init_db()
    with database._connect() as con, con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT
        )""")


def _row_to_user(r):
    return {"id": r[0], "username": r[1], "role": r[2], "active": bool(r[3]), "created_at": r[4], "last_login": r[5]}


_COLS = "id, username, role, active, created_at, last_login"


def user_count():
    with database._connect() as con:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def list_users():
    with database._connect() as con:
        return [_row_to_user(r) for r in con.execute(f"SELECT {_COLS} FROM users ORDER BY username")]


def get_user(user_id):
    with database._connect() as con:
        r = con.execute(f"SELECT {_COLS} FROM users WHERE id = ?", (int(user_id),)).fetchone()
    return _row_to_user(r) if r else None


def _active_admin_count(con, excluding_id=None):
    q, args = "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1", []
    if excluding_id is not None:
        q += " AND id != ?"
        args.append(int(excluding_id))
    return con.execute(q, args).fetchone()[0]


def create_user(username, password, role="analyst"):
    username = normalize_username(username)
    validate_username(username)
    validate_password(password)
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")
    try:
        with database._connect() as con, con:
            cur = con.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                              (username, hash_password(password), role, datetime.now().strftime(_FMT)))
            return cur.lastrowid
    except database.sqlite3.IntegrityError:
        raise ValueError("That username is already taken.")


def bootstrap_from_env():
    """Create the first admin from RETURNWISE_ADMIN_USER / RETURNWISE_ADMIN_PASSWORD when no users exist."""
    if user_count() == 0 and settings.ADMIN_USER and settings.ADMIN_PASSWORD:
        create_user(settings.ADMIN_USER, settings.ADMIN_PASSWORD, "admin")
        return True
    return False


# --------------------------------------------------------- authentication
def authenticate(username, password):
    """Return (user, message). user is None on failure; message is safe to show to the visitor."""
    generic = "Invalid username or password."
    username = normalize_username(username)
    with database._connect() as con:
        r = con.execute(f"SELECT {_COLS}, password_hash, failed_attempts, locked_until FROM users WHERE username = ?",
                        (username,)).fetchone()
    if r is None:
        _dummy_verify(password or "")
        return None, generic
    user, pw_hash, failed, locked_until = _row_to_user(r[:6]), r[6], r[7], r[8]
    if locked_until:
        until = datetime.strptime(locked_until, _FMT)
        if until > datetime.now():
            minutes = int((until - datetime.now()).total_seconds() // 60) + 1
            return None, f"Too many failed attempts. Try again in about {minutes} minute(s)."
    ok = verify_password(password or "", pw_hash)
    if not ok or not user["active"]:
        if not ok:
            failed += 1
            lock = (datetime.now() + timedelta(minutes=LOCK_MINUTES)).strftime(_FMT) if failed >= MAX_FAILED_ATTEMPTS else None
            with database._connect() as con, con:
                con.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                            (0 if lock else failed, lock, user["id"]))
        return None, generic
    with database._connect() as con, con:
        con.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?",
                    (datetime.now().strftime(_FMT), user["id"]))
    return get_user(user["id"]), "ok"


# ------------------------------------------------------- administration
def set_active(user_id, active):
    with database._connect() as con, con:
        if not active and _active_admin_count(con, excluding_id=user_id) == 0:
            target = con.execute("SELECT role FROM users WHERE id=?", (int(user_id),)).fetchone()
            if target and target[0] == "admin":
                raise ValueError("You cannot disable the last active administrator.")
        return con.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, int(user_id))).rowcount > 0


def set_role(user_id, role):
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")
    with database._connect() as con, con:
        if role != "admin" and _active_admin_count(con, excluding_id=user_id) == 0:
            target = con.execute("SELECT role FROM users WHERE id=? AND active=1", (int(user_id),)).fetchone()
            if target and target[0] == "admin":
                raise ValueError("You cannot demote the last active administrator.")
        return con.execute("UPDATE users SET role = ? WHERE id = ?", (role, int(user_id))).rowcount > 0


def reset_password(user_id, new_password):
    validate_password(new_password)
    with database._connect() as con, con:
        return con.execute("UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
                           (hash_password(new_password), int(user_id))).rowcount > 0


def change_password(user_id, old_password, new_password):
    with database._connect() as con:
        r = con.execute("SELECT password_hash FROM users WHERE id = ?", (int(user_id),)).fetchone()
    if r is None or not verify_password(old_password or "", r[0]):
        raise ValueError("Current password is incorrect.")
    reset_password(user_id, new_password)
