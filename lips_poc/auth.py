"""
Authentication — the ONLY module that talks to the users store.

Mirrors the design of `tracking.py`: one thin wrapper, backend chosen by an env
var, so local dev and the deployed HF Space differ only by configuration.

- Users live in an `app_users` table. In production this is the SAME Neon
  Postgres instance MLflow uses, but a SEPARATE table — the two never touch each
  other's rows. Point `APP_DB_URI` at it (include `?sslmode=require` for Neon).
- Local dev (no `APP_DB_URI`) falls back to a sqlite file next to the repo, so
  the app runs with zero setup.
- Passwords are stored only as bcrypt hashes — never in plaintext.
- Roles are `admin` or `user`. Admins are seeded from the `ADMIN_EMAILS` env var
  (comma-separated). The check runs both at registration AND at login, so adding
  an email to `ADMIN_EMAILS` promotes that person on their next sign-in — no SQL.

This module is framework-agnostic (no Streamlit import): it takes/returns plain
dicts and raises ValueError with a human message on bad input. The Streamlit
login UI and session handling live in main.py.

DB URI resolution:
    APP_DB_URI env var  ->  used as-is (prod: the Neon Postgres URL)
    otherwise           ->  sqlite:///<repo>/app_users.db  (local dev)
"""

import os
import re
import uuid
from datetime import datetime

import bcrypt
from sqlalchemy import create_engine, text

# Local dev fallback: an absolute sqlite path so it resolves the same regardless
# of the working directory. Gitignored; created on first use.
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app_users.db")

# Emails that should be admins, seeded out-of-band via env. Compared lowercased.
_ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
}

# Input rules. Username is the upload identity (stamped on every HF model card),
# so keep it to a strict, stable slug: letters/digits/_/- only, 3–32 chars.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
MIN_PASSWORD_LEN = 8
# Human-readable rule shown in the register form.
USERNAME_RULE_TEXT = "3–32 characters · letters (a–z, A–Z), digits (0–9), and _ or - only"

_engine = None


def _db_uri() -> str:
    """The users DB URI: the env override, or the local sqlite dev DB."""
    return os.environ.get("APP_DB_URI", f"sqlite:///{_DEFAULT_DB}")


def _get_engine():
    """Lazily create the SQLAlchemy engine and ensure the table exists. Using
    SQLAlchemy Core means one code path serves both sqlite (dev) and Postgres
    (prod) — only the URI changes."""
    global _engine
    if _engine is None:
        _engine = create_engine(_db_uri(), future=True, pool_pre_ping=True)
        _init_schema(_engine)
    return _engine


def _init_schema(engine) -> None:
    """Create the app_users table if it doesn't exist. A TEXT uuid primary key
    (generated in Python) is used instead of SERIAL/AUTOINCREMENT so the exact
    same DDL works on both sqlite and Postgres."""
    ddl = """
    CREATE TABLE IF NOT EXISTS app_users (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE,
        username      TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'user',
        is_active     INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _hash_password(password: str) -> str:
    """bcrypt hash, stored as a utf-8 string."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _role_for(email: str) -> str:
    """'admin' if the email is seeded in ADMIN_EMAILS, else 'user'."""
    return "admin" if email.lower() in _ADMIN_EMAILS else "user"


def _row_to_user(row) -> dict:
    """A DB row -> the public user dict (never includes the password hash)."""
    return {
        "id": row.id,
        "email": row.email,
        "username": row.username,
        "role": row.role,
        "is_active": bool(row.is_active),
        "created_at": row.created_at,
    }


def email_format_error(email: str) -> "str | None":
    """None if the email is a valid format, else a short message. Cheap (no DB) —
    used for live form feedback and as the first guard in register()."""
    email = (email or "").strip().lower()
    if not email:
        return "Email is required."
    if not _EMAIL_RE.match(email):
        return "Enter a valid email address."
    return None


def username_format_error(username: str) -> "str | None":
    """None if the username satisfies the format rules, else a specific message.
    Cheap (no DB) — used for live form feedback and as a guard in register()."""
    username = (username or "").strip()
    if not username:
        return "Username is required."
    if len(username) < 3:
        return "Too short — minimum 3 characters."
    if len(username) > 32:
        return "Too long — maximum 32 characters."
    if not _USERNAME_RE.match(username):
        return "Only letters, digits, _ and - are allowed."
    return None


def email_taken(email: str) -> bool:
    """True if an account already uses this email (stored lowercased)."""
    email = (email or "").strip().lower()
    if not email:
        return False
    engine = _get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT 1 FROM app_users WHERE email = :e"), {"e": email}
        ).fetchone()
    return row is not None


def username_taken(username: str) -> bool:
    """True if the username is already used. Case-insensitive, so 'Alice' and
    'alice' are treated as the same name."""
    username = (username or "").strip()
    if not username:
        return False
    engine = _get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT 1 FROM app_users WHERE LOWER(username) = LOWER(:u)"),
            {"u": username},
        ).fetchone()
    return row is not None


def register(email: str, username: str, password: str) -> dict:
    """Create a new account. Validates format and uniqueness, hashes the
    password, assigns the role from ADMIN_EMAILS, and returns the public user
    dict. Raises ValueError (with a user-facing message) on any problem. This is
    the authoritative guard — the UI's live checks are only a convenience."""
    email = (email or "").strip().lower()
    username = (username or "").strip()

    err = email_format_error(email) or username_format_error(username)
    if err:
        raise ValueError(err)
    if len(password or "") < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")

    engine = _get_engine()
    with engine.begin() as conn:
        # Pre-check for friendly messages (the UNIQUE constraint is the real guard).
        # Username match is case-insensitive to block 'Alice' vs 'alice' dupes.
        clash = conn.execute(
            text("SELECT email, username FROM app_users "
                 "WHERE email = :e OR LOWER(username) = LOWER(:u)"),
            {"e": email, "u": username},
        ).fetchone()
        if clash:
            if clash.email == email:
                raise ValueError("An account with this email already exists.")
            raise ValueError("That username is taken — choose another.")

        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "username": username,
            "password_hash": _hash_password(password),
            "role": _role_for(email),
            "is_active": 1,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        conn.execute(
            text(
                "INSERT INTO app_users "
                "(id, email, username, password_hash, role, is_active, created_at) "
                "VALUES (:id, :email, :username, :password_hash, :role, :is_active, :created_at)"
            ),
            user,
        )
    user.pop("password_hash")
    user["is_active"] = True
    return user


def authenticate(email: str, password: str) -> "dict | None":
    """Verify credentials. Returns the public user dict on success, or None if
    the email is unknown, the password is wrong, or the account is disabled.

    Also re-applies ADMIN_EMAILS: if this email is now seeded as admin but the
    stored row still says 'user', it is promoted here (and vice-versa demoted),
    so ADMIN_EMAILS stays the single source of truth for who is an admin."""
    email = (email or "").strip().lower()
    engine = _get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT * FROM app_users WHERE email = :e"), {"e": email}
        ).fetchone()
        if row is None or not _verify_password(password, row.password_hash):
            return None
        if not row.is_active:
            return None

        expected_role = _role_for(email)
        if row.role != expected_role:
            conn.execute(
                text("UPDATE app_users SET role = :r WHERE id = :id"),
                {"r": expected_role, "id": row.id},
            )
            row = conn.execute(
                text("SELECT * FROM app_users WHERE id = :id"), {"id": row.id}
            ).fetchone()
        return _row_to_user(row)


def list_users() -> list:
    """All accounts, newest first — for the admin view. Never returns hashes."""
    engine = _get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT id, email, username, role, is_active, created_at "
                "FROM app_users ORDER BY created_at DESC"
            )
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def set_user_active(user_id: str, active: bool) -> None:
    """Enable or disable an account (admin soft delete). A disabled account is
    kept in the DB but rejected by authenticate(), so it loses platform access
    without losing its record; re-enabling restores access."""
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE app_users SET is_active = :a WHERE id = :id"),
            {"a": 1 if active else 0, "id": user_id},
        )
