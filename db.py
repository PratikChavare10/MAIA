

import sqlite3

from config import APP_DB_PATH
from auth import hash_password, verify_password


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(APP_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            email          TEXT UNIQUE NOT NULL,
            password_hash  TEXT NOT NULL,
            city           TEXT DEFAULT 'Pune',
            language       TEXT DEFAULT 'English',
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            mode        TEXT NOT NULL,
            title       TEXT DEFAULT 'New Chat',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            role        TEXT NOT NULL,     -- 'user' | 'assistant'
            content     TEXT NOT NULL,
            mode        TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ════════════════════════════
# USERS
# ════════════════════════════
def register_user(name: str, email: str, password: str,
                   city: str = "Pune", language: str = "English") -> dict:
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return {"ok": False, "msg": "Email already registered."}

        conn.execute(
            "INSERT INTO users (name, email, password_hash, city, language) VALUES (?, ?, ?, ?, ?)",
            (name, email, hash_password(password), city, language),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def login_user(email: str, password: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return {"ok": False}
        return {"ok": True, "user": dict(row)}
    finally:
        conn.close()


# ════════════════════════════
# THREADS (sidebar)
# ════════════════════════════
def create_thread(user_id: int, mode: str, title: str = "New Chat") -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO threads (user_id, mode, title) VALUES (?, ?, ?)",
            (user_id, mode, title),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_threads(user_id: int) -> list:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT t.*,
                      (SELECT content FROM messages m
                       WHERE m.thread_id = t.id ORDER BY m.id ASC LIMIT 1) AS first_msg
               FROM threads t
               WHERE t.user_id = ?
               ORDER BY t.updated_at DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_thread_title(thread_id: int, title: str):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE threads SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, thread_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_thread(thread_id: int, user_id: int):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE id = ? AND user_id = ?", (thread_id, user_id))
        conn.commit()
    finally:
        conn.close()


# ════════════════════════════
# MESSAGES (what shows when you click a thread in the sidebar)
# ════════════════════════════
def save_message(thread_id: int, user_id: int, role: str, content: str, mode: str):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (thread_id, user_id, role, content, mode) VALUES (?, ?, ?, ?, ?)",
            (thread_id, user_id, role, content, mode),
        )
        conn.execute("UPDATE threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()


def get_messages(thread_id: int) -> list:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, mode FROM messages WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()