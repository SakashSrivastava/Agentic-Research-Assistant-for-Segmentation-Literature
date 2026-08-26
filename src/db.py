"""User accounts + query history (SQLite, read-write).

Kept separate from app.db (the read-only corpus) so user data sits on a writable
volume and is never rebuilt alongside the corpus. Passwords are stored only as
salted hashes via werkzeug; plaintext is never persisted.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from src import config


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(config.USER_DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    con = _conn()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            email          TEXT UNIQUE NOT NULL,
            password_hash  TEXT NOT NULL,
            is_admin       INTEGER NOT NULL DEFAULT 0,
            email_verified INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS queries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            question   TEXT NOT NULL,
            answer     TEXT,
            steps      INTEGER,
            tokens     INTEGER,
            ok         INTEGER NOT NULL DEFAULT 1,
            own_key    INTEGER NOT NULL DEFAULT 0,
            cached     INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_queries_user ON queries(user_id, created_at DESC);

        -- Global semantic answer cache (shared across users): a past answer plus
        -- its question embedding and feedback tallies.
        CREATE TABLE IF NOT EXISTS answer_cache (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            steps      INTEGER,
            tokens     INTEGER,
            embedding  BLOB NOT NULL,
            up         INTEGER NOT NULL DEFAULT 0,
            down       INTEGER NOT NULL DEFAULT 0,
            hits       INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
    """)
    # Migrate older DBs that predate added columns.
    qcols = [r[1] for r in con.execute("PRAGMA table_info(queries)")]
    if "own_key" not in qcols:
        con.execute("ALTER TABLE queries ADD COLUMN own_key INTEGER NOT NULL DEFAULT 0")
    if "cached" not in qcols:
        con.execute("ALTER TABLE queries ADD COLUMN cached INTEGER NOT NULL DEFAULT 0")
    ucols = [r[1] for r in con.execute("PRAGMA table_info(users)")]
    if "email_verified" not in ucols:
        con.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
        con.execute("UPDATE users SET email_verified=1")  # grandfather pre-existing accounts
    con.commit()
    con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- users ----------------

def create_user(email: str, password: str) -> dict | None:
    """Insert a new REGULAR user. Admin is never granted through signup; it is
    granted server-side via `python -m src.manage_admin` (see set_admin).
    Returns the user, or None if the email is already registered."""
    email = email.strip().lower()
    con = _conn()
    try:
        cur = con.execute(
            "INSERT INTO users (email, password_hash, is_admin, created_at) VALUES (?,?,0,?)",
            (email, generate_password_hash(password), _now()))
        con.commit()
        uid = cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        con.close()
    return get_user(uid)


def set_admin(email: str, is_admin: bool = True) -> int:
    """Grant or revoke admin for an existing user. Returns rows changed (0 if no
    such user). Callable only from the server shell, never from the web app."""
    con = _conn()
    cur = con.execute("UPDATE users SET is_admin=? WHERE email=?",
                      (1 if is_admin else 0, email.strip().lower()))
    con.commit()
    n = cur.rowcount
    con.close()
    return n


def get_user(user_id) -> dict | None:
    con = _conn()
    row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    con = _conn()
    row = con.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    con.close()
    return dict(row) if row else None


def verify_password(user: dict, password: str) -> bool:
    return bool(user) and check_password_hash(user["password_hash"], password)


def set_email_verified(email: str) -> int:
    con = _conn()
    n = con.execute("UPDATE users SET email_verified=1 WHERE email=?",
                    (email.strip().lower(),)).rowcount
    con.commit()
    con.close()
    return n


def update_password(user_id, new_password: str) -> None:
    con = _conn()
    con.execute("UPDATE users SET password_hash=? WHERE id=?",
                (generate_password_hash(new_password), user_id))
    con.commit()
    con.close()


# ---------------- query history ----------------

def log_query(user_id, question, answer, steps, tokens, ok=True, own_key=False, cached=False) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO queries (user_id, question, answer, steps, tokens, ok, own_key, cached, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, question, answer, steps, tokens, 1 if ok else 0,
         1 if own_key else 0, 1 if cached else 0, _now()))
    con.commit()
    con.close()


def queries_today(user_id) -> int:
    """Successful SHARED-KEY, non-cached queries today (UTC), for free-tier gating.
    Own-key (own_key=1) and instant cache hits (cached=1) cost no shared budget, so
    neither counts against the free allowance."""
    today = datetime.now(timezone.utc).date().isoformat()
    con = _conn()
    n = con.execute("SELECT COUNT(*) FROM queries WHERE user_id=? AND ok=1 AND own_key=0 "
                    "AND cached=0 AND substr(created_at,1,10)=?", (user_id, today)).fetchone()[0]
    con.close()
    return n


# ---------------- semantic answer cache ----------------

def cache_candidates() -> list[dict]:
    """All cache rows (with raw embedding bytes) for similarity matching."""
    con = _conn()
    rows = con.execute("SELECT id, question, answer, steps, tokens, embedding, up, down, "
                       "created_at FROM answer_cache").fetchall()
    con.close()
    return [dict(r) for r in rows]


def cache_insert(question, answer, steps, tokens, embedding: bytes) -> int:
    con = _conn()
    cur = con.execute("INSERT INTO answer_cache (question, answer, steps, tokens, embedding, "
                      "created_at) VALUES (?,?,?,?,?,?)",
                      (question, answer, steps, tokens, embedding, _now()))
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def cache_bump_hit(cache_id) -> None:
    con = _conn()
    con.execute("UPDATE answer_cache SET hits=hits+1 WHERE id=?", (cache_id,))
    con.commit()
    con.close()


def cache_vote(cache_id, up: bool) -> None:
    # No string interpolation into SQL: fixed statements per branch.
    con = _conn()
    if up:
        con.execute("UPDATE answer_cache SET up = up + 1 WHERE id=?", (cache_id,))
    else:
        con.execute("UPDATE answer_cache SET down = down + 1 WHERE id=?", (cache_id,))
    con.commit()
    con.close()


def cache_stats() -> dict:
    con = _conn()
    one = lambda sql: con.execute(sql).fetchone()[0] or 0
    st = {"entries": one("SELECT COUNT(*) FROM answer_cache"),
          "hits": one("SELECT COALESCE(SUM(hits),0) FROM answer_cache"),
          "up": one("SELECT COALESCE(SUM(up),0) FROM answer_cache"),
          "down": one("SELECT COALESCE(SUM(down),0) FROM answer_cache")}
    con.close()
    return st


def user_history(user_id, limit=50) -> list[dict]:
    con = _conn()
    rows = con.execute(
        "SELECT * FROM queries WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ---------------- admin analytics ----------------

def admin_stats() -> dict:
    con = _conn()
    one = lambda sql: con.execute(sql).fetchone()[0] or 0
    stats = {
        "users": one("SELECT COUNT(*) FROM users"),
        "queries": one("SELECT COUNT(*) FROM queries"),
        "tokens": one("SELECT COALESCE(SUM(tokens),0) FROM queries"),
        "failed": one("SELECT COUNT(*) FROM queries WHERE ok=0"),
        "per_user": [dict(r) for r in con.execute(
            "SELECT u.email, u.is_admin, u.created_at, COUNT(q.id) AS n, "
            "COALESCE(SUM(q.tokens),0) AS toks, MAX(q.created_at) AS last "
            "FROM users u LEFT JOIN queries q ON q.user_id=u.id "
            "GROUP BY u.id ORDER BY n DESC").fetchall()],
        "recent": [dict(r) for r in con.execute(
            "SELECT q.question, q.tokens, q.ok, q.created_at, u.email "
            "FROM queries q JOIN users u ON u.id=q.user_id "
            "ORDER BY q.created_at DESC LIMIT 30").fetchall()],
    }
    con.close()
    return stats
