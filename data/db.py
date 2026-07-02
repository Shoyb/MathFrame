"""
data/db.py — Central SQLite connection and schema for MathFrame.

Replaces the JSON-file-backed persistence in ``data/quiz_store.py`` and
``data/permissions.py`` with a single SQLite database
(``data/mathframe.db``), accessed via ``aiosqlite`` so every query is a
real ``await`` inside the bot's existing event loop rather than a
blocking call — no thread-pool dispatch needed for simple reads/writes.

Usage
-----
Call :func:`init_db` once, early, before any cog touches the DB — wired
into ``main.py`` via ``bot.setup_hook``. Every other module in ``data/``
imports :func:`get_connection` to run its own queries against the shared
connection; nothing outside this module opens a second connection.

Concurrency model
------------------
A single ``aiosqlite.Connection`` is shared process-wide. SQLite itself
serializes writers; WAL mode (set in :func:`init_db`) lets readers proceed
without blocking on an in-progress write, which is what matters here since
``/quiz stats``, ``/quiz leaderboard`` etc. are far more frequent than
writes. An ``asyncio.Lock`` additionally serializes the handful of
multi-statement transactions (e.g. ``apply_battle_result`` touching two
users' rows) so they can't interleave.

Migration from JSON
--------------------
:func:`init_db` also runs :func:`_migrate_json_if_needed`, which is
idempotent and safe to leave in place permanently: it only imports from
``quiz_stats.json`` / ``quiz_daily.json`` / ``guild_permissions.json`` /
``guild_panic_backups.json`` the first time (when the corresponding SQLite
table is empty AND the JSON file exists). Once migrated, those JSON files
are left on disk untouched (not deleted) as a backup, but are never read
again after the first successful run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from urllib.parse import urlparse, unquote

import aiosqlite
import aiomysql

import config

log = logging.getLogger(__name__)

_DATA_DIR = os.path.dirname(__file__)
_DB_PATH = os.path.join(_DATA_DIR, "mathframe.db")

_QUIZ_JSON = os.path.join(_DATA_DIR, "quiz_stats.json")
_QUIZ_DAILY_JSON = os.path.join(_DATA_DIR, "quiz_daily.json")
_PERM_JSON = os.path.join(_DATA_DIR, "guild_permissions.json")
_PANIC_JSON = os.path.join(_DATA_DIR, "guild_panic_backups.json")

_connection: object | None = None
_write_lock = asyncio.Lock()

_MYSQL_DSN_RE = re.compile(r"^(?:jdbc:)?mysql://")

SUBJECTS = ("algebra", "calculus", "number_theory", "discrete")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_records (
    guild_id                INTEGER NOT NULL,
    user_id                 INTEGER NOT NULL,
    rating                  INTEGER NOT NULL DEFAULT 1200,
    solved                  INTEGER NOT NULL DEFAULT 0,
    wrong                   INTEGER NOT NULL DEFAULT 0,
    streak_current           INTEGER NOT NULL DEFAULT 0,
    streak_best              INTEGER NOT NULL DEFAULT 0,
    battle_wins              INTEGER NOT NULL DEFAULT 0,
    battle_losses            INTEGER NOT NULL DEFAULT 0,
    daily_streak_current      INTEGER NOT NULL DEFAULT 0,
    daily_streak_best         INTEGER NOT NULL DEFAULT 0,
    last_daily_date          TEXT,
    PRIMARY KEY (guild_id, user_id)
);

CREATE TABLE IF NOT EXISTS quiz_subject_stats (
    guild_id   INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    subject    TEXT NOT NULL,
    solved     INTEGER NOT NULL DEFAULT 0,
    wrong      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id, subject),
    FOREIGN KEY (guild_id, user_id) REFERENCES quiz_records(guild_id, user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quiz_achievements (
    guild_id        INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,
    achievement_id  TEXT NOT NULL,
    earned_at       TEXT NOT NULL,
    PRIMARY KEY (guild_id, user_id, achievement_id),
    FOREIGN KEY (guild_id, user_id) REFERENCES quiz_records(guild_id, user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quiz_daily_results (
    guild_id     INTEGER NOT NULL,
    user_id      INTEGER NOT NULL,
    date_str     TEXT NOT NULL,
    correct      INTEGER NOT NULL,
    answered_at  TEXT NOT NULL,
    PRIMARY KEY (guild_id, user_id, date_str)
);

CREATE TABLE IF NOT EXISTS guild_permissions (
    guild_id      INTEGER NOT NULL,
    channel_id    TEXT NOT NULL,   -- '__all__' sentinel included, matches old JSON scheme
    command_name  TEXT NOT NULL,   -- '__all__' sentinel included
    enabled       INTEGER NOT NULL,
    PRIMARY KEY (guild_id, channel_id, command_name)
);

CREATE TABLE IF NOT EXISTS guild_panic_backups (
    guild_id     INTEGER PRIMARY KEY,
    backup_json  TEXT NOT NULL   -- snapshot has variable shape; stored as a JSON blob rather than normalized
);
"""


class _DatabaseCursor:
    def __init__(self, cursor: object, backend: str) -> None:
        self._cursor = cursor
        self._backend = backend

    async def __aenter__(self) -> "_DatabaseCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._backend == "sqlite":
            await self._cursor.close()
        else:
            self._cursor.close()

    async def fetchall(self) -> list[dict[str, object]]:
        return await self._cursor.fetchall()

    async def fetchone(self) -> object | None:
        return await self._cursor.fetchone()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def __aiter__(self) -> "_DatabaseCursor":
        return self

    async def __anext__(self) -> object:
        row = await self.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row


class _DatabaseConnection:
    def __init__(self, conn: object, backend: str) -> None:
        self._conn = conn
        self._backend = backend

    async def execute(self, sql: str, params: tuple[object, ...] | list[object] | None = None) -> _DatabaseCursor:
        params = params or ()
        if self._backend == "mysql":
            sql = _sqlite_to_mysql_sql(sql)
            cur = await self._conn.cursor(aiomysql.DictCursor)
            await cur.execute(sql, params)
            return _DatabaseCursor(cur, self._backend)

        cursor = await self._conn.execute(sql, params)
        return _DatabaseCursor(cursor, self._backend)

    async def executescript(self, script: str) -> None:
        if self._backend == "sqlite":
            await self._conn.executescript(script)
            return

        for statement in (stmt.strip() for stmt in script.split(";") if stmt.strip()):
            await self.execute(statement)

    async def commit(self) -> None:
        await self._conn.commit()

    async def close(self) -> None:
        if self._backend == "sqlite":
            await self._conn.close()
        else:
            self._conn.close()


def _sqlite_to_mysql_sql(sql: str) -> str:
    sql = sql.replace("?", "%s")
    sql = sql.replace("INSERT OR IGNORE INTO", "INSERT IGNORE INTO")
    sql = re.sub(r"ON CONFLICT\([^\)]*\)\s+DO UPDATE SET\s+", "ON DUPLICATE KEY UPDATE ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"excluded\.([a-zA-Z_][a-zA-Z0-9_]*)", r"VALUES(\1)", sql, flags=re.IGNORECASE)
    return sql


def _parse_database_url(url: str) -> dict[str, object]:
    if url.startswith("jdbc:"):
        url = url[len("jdbc:"):]

    parsed = urlparse(url)
    if parsed.scheme != "mysql":
        raise ValueError("Unsupported DATABASE_URL scheme: %s" % parsed.scheme)

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    host = parsed.hostname or "localhost"
    port = parsed.port or 3306
    database = parsed.path.lstrip("/")
    if not database:
        raise ValueError("DATABASE_URL must include a database name")

    charset = "utf8mb4"
    if parsed.query:
        for part in parsed.query.split("&"):
            key, _, value = part.partition("=")
            if key.lower() == "charset" and value:
                charset = unquote(value)

    return {
        "backend": "mysql",
        "host": host,
        "port": port,
        "user": username,
        "password": password,
        "database": database,
        "charset": charset,
    }


async def init_db() -> None:
    """
    Open the shared connection, create tables if absent, and run the
    one-time JSON migration. Call once from ``main.py``'s ``setup_hook``
    before any cog can touch the database.
    """
    global _connection
    if _connection is not None:
        return  # already initialized (e.g. a stray double-call) — no-op

    if config.DATABASE_URL is None:
        sqlite_conn = await aiosqlite.connect(_DB_PATH)
        sqlite_conn.row_factory = aiosqlite.Row
        _connection = _DatabaseConnection(sqlite_conn, "sqlite")
        await _connection.execute("PRAGMA journal_mode = WAL;")
        await _connection.execute("PRAGMA foreign_keys = ON;")
        await _connection.executescript(_SCHEMA)
        await _connection.commit()
        log.info("SQLite DB ready at %s", _DB_PATH)
        await _migrate_json_if_needed()
        return

    db_info = _parse_database_url(config.DATABASE_URL)
    raw_conn = await aiomysql.connect(
        host=db_info["host"],
        port=db_info["port"],
        user=db_info["user"],
        password=db_info["password"],
        db=db_info["database"],
        charset=db_info["charset"],
        autocommit=False,
        cursorclass=aiomysql.DictCursor,
    )
    _connection = _DatabaseConnection(raw_conn, "mysql")
    await _connection.executescript(_SCHEMA)
    await _connection.commit()
    log.info("MySQL DB ready at %s", config.DATABASE_URL)


def get_connection() -> _DatabaseConnection:
    """Return the shared connection. Raises if :func:`init_db` hasn't run yet."""
    if _connection is None:
        raise RuntimeError("data.db.init_db() must be awaited before the database is used.")
    return _connection


def get_write_lock() -> asyncio.Lock:
    """Shared lock for multi-statement transactions that must not interleave."""
    return _write_lock


async def close_db() -> None:
    """Close the shared connection. Call from a shutdown hook if you add one."""
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None


# ---------------------------------------------------------------------------
# One-time JSON -> SQLite migration
# ---------------------------------------------------------------------------


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        log.warning("Couldn't read %s during migration — skipping it.", path)
        return {}


async def _migrate_json_if_needed() -> None:
    """
    Import legacy JSON data into SQLite exactly once. Guarded per-table:
    if a table already has rows, that table's migration is skipped (so
    this is safe to call on every startup, forever).
    """
    conn = get_connection()

    async with conn.execute("SELECT COUNT(*) FROM quiz_records") as cur:
        (quiz_count,) = await cur.fetchone()
    if quiz_count == 0:
        await _migrate_quiz_json(conn)

    async with conn.execute("SELECT COUNT(*) FROM quiz_daily_results") as cur:
        (daily_count,) = await cur.fetchone()
    if daily_count == 0:
        await _migrate_quiz_daily_json(conn)

    async with conn.execute("SELECT COUNT(*) FROM guild_permissions") as cur:
        (perm_count,) = await cur.fetchone()
    if perm_count == 0:
        await _migrate_permissions_json(conn)

    async with conn.execute("SELECT COUNT(*) FROM guild_panic_backups") as cur:
        (panic_count,) = await cur.fetchone()
    if panic_count == 0:
        await _migrate_panic_json(conn)


async def _migrate_quiz_json(conn: aiosqlite.Connection) -> None:
    data = _read_json(_QUIZ_JSON)
    if not data:
        return
    imported = 0
    for key, record in data.items():
        try:
            guild_id_s, user_id_s = key.split(":", 1)
            guild_id, user_id = int(guild_id_s), int(user_id_s)
        except ValueError:
            log.warning("Skipping malformed quiz_stats.json key: %r", key)
            continue

        await conn.execute(
            """
            INSERT OR REPLACE INTO quiz_records
                (guild_id, user_id, rating, solved, wrong, streak_current, streak_best,
                 battle_wins, battle_losses, daily_streak_current, daily_streak_best, last_daily_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guild_id, user_id,
                record.get("rating", 1200),
                record.get("solved", 0),
                record.get("wrong", 0),
                record.get("streak_current", 0),
                record.get("streak_best", 0),
                record.get("battle_wins", 0),
                record.get("battle_losses", 0),
                record.get("daily_streak_current", 0),
                record.get("daily_streak_best", 0),
                record.get("last_daily_date"),
            ),
        )
        for subject, s in record.get("subject_stats", {}).items():
            await conn.execute(
                """
                INSERT OR REPLACE INTO quiz_subject_stats (guild_id, user_id, subject, solved, wrong)
                VALUES (?, ?, ?, ?, ?)
                """,
                (guild_id, user_id, subject, s.get("solved", 0), s.get("wrong", 0)),
            )
        for aid in record.get("achievements", []):
            await conn.execute(
                """
                INSERT OR IGNORE INTO quiz_achievements (guild_id, user_id, achievement_id, earned_at)
                VALUES (?, ?, ?, ?)
                """,
                (guild_id, user_id, aid, "unknown"),  # JSON never stored a timestamp
            )
        imported += 1

    await conn.commit()
    log.info("Migrated %d quiz record(s) from quiz_stats.json into SQLite.", imported)


async def _migrate_quiz_daily_json(conn: aiosqlite.Connection) -> None:
    data = _read_json(_QUIZ_DAILY_JSON)
    if not data:
        return
    imported = 0
    for date_str, day in data.items():
        for key, entry in day.items():
            try:
                guild_id_s, user_id_s = key.split(":", 1)
                guild_id, user_id = int(guild_id_s), int(user_id_s)
            except ValueError:
                continue
            await conn.execute(
                """
                INSERT OR REPLACE INTO quiz_daily_results (guild_id, user_id, date_str, correct, answered_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (guild_id, user_id, date_str, int(bool(entry.get("correct"))), entry.get("answered_at", "")),
            )
            imported += 1
    await conn.commit()
    log.info("Migrated %d daily-challenge result(s) from quiz_daily.json into SQLite.", imported)


async def _migrate_permissions_json(conn: aiosqlite.Connection) -> None:
    data = _read_json(_PERM_JSON)
    if not data:
        return
    imported = 0
    for gid_s, channels in data.items():
        guild_id = int(gid_s)
        for channel_id, commands_ in channels.items():
            for command_name, enabled in commands_.items():
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO guild_permissions (guild_id, channel_id, command_name, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (guild_id, channel_id, command_name, int(bool(enabled))),
                )
                imported += 1
    await conn.commit()
    log.info("Migrated %d permission rule(s) from guild_permissions.json into SQLite.", imported)


async def _migrate_panic_json(conn: aiosqlite.Connection) -> None:
    data = _read_json(_PANIC_JSON)
    if not data:
        return
    imported = 0
    for gid_s, backup in data.items():
        await conn.execute(
            "INSERT OR REPLACE INTO guild_panic_backups (guild_id, backup_json) VALUES (?, ?)",
            (int(gid_s), json.dumps(backup)),
        )
        imported += 1
    await conn.commit()
    log.info("Migrated %d panic backup(s) from guild_panic_backups.json into SQLite.", imported)
