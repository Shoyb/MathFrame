"""
data/permissions.py — Guild-level command permission store (SQLite-backed).

Migrated from the JSON-file version to ``data/db.py``'s shared SQLite
connection. Same table shape as the old nested JSON
(guild -> channel|'__all__' -> command|'__all__' -> enabled), same
specificity-ordered lookup, same public API — every function is now a
coroutine, so callers (``main.py``, ``cogs/admin.py``) need ``await`` at
each call site.

Lookup order (most-specific wins), unchanged from the JSON version
---------------------------------------------------------------------
1. guild -> channel -> command
2. guild -> __all__ -> command
3. guild -> channel -> __all__
4. guild -> __all__ -> __all__
5. Not present -> allowed (fail-open)

Public API
----------
is_command_allowed(guild_id, channel_id, command_name) -> bool
set_permission(guild_id, channel_id, command_name, enabled) -> None
clear_permission(guild_id, channel_id, command_name) -> bool
get_guild_status(guild_id) -> list[dict]
panic_lock(guild_id) -> None
panic_unlock(guild_id) -> bool
"""

from __future__ import annotations

import json

from data.db import get_connection, get_write_lock

_ALL = "__all__"


def _row_get(rows, channel_id: str, command_name: str) -> bool | None:
    for r in rows:
        if r["channel_id"] == channel_id and r["command_name"] == command_name:
            return bool(r["enabled"])
    return None


async def is_command_allowed(guild_id: int, channel_id: int, command_name: str) -> bool:
    """Return True if *command_name* is allowed in *channel_id* of *guild_id* (fail-open)."""
    cid = str(channel_id)
    cmd = command_name.lower()
    conn = get_connection()

    async with conn.execute(
        "SELECT channel_id, command_name, enabled FROM guild_permissions WHERE guild_id = ?",
        (guild_id,),
    ) as cur:
        rows = await cur.fetchall()
    if not rows:
        return True

    for check_cid, check_cmd in ((cid, cmd), (_ALL, cmd), (cid, _ALL), (_ALL, _ALL)):
        val = _row_get(rows, check_cid, check_cmd)
        if val is not None:
            return val

    return True


async def set_permission(guild_id: int, channel_id: int | None, command_name: str | None, enabled: bool) -> None:
    """Set (or overwrite) a permission rule."""
    cid = str(channel_id) if channel_id is not None else _ALL
    cmd = command_name.lower() if command_name is not None else _ALL
    conn = get_connection()

    async with get_write_lock():
        await conn.execute(
            """
            INSERT INTO guild_permissions (guild_id, channel_id, command_name, enabled)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(guild_id, channel_id, command_name) DO UPDATE SET enabled = excluded.enabled
            """,
            (guild_id, cid, cmd, int(enabled)),
        )
        await conn.commit()


async def clear_permission(guild_id: int, channel_id: int | None, command_name: str | None) -> bool:
    """Remove an existing rule. Returns True if a rule was actually removed."""
    cid = str(channel_id) if channel_id is not None else _ALL
    cmd = command_name.lower() if command_name is not None else _ALL
    conn = get_connection()

    async with get_write_lock():
        cur = await conn.execute(
            "DELETE FROM guild_permissions WHERE guild_id = ? AND channel_id = ? AND command_name = ?",
            (guild_id, cid, cmd),
        )
        await conn.commit()
        return cur.rowcount > 0


async def get_guild_status(guild_id: int) -> list[dict]:
    """All permission rules for *guild_id* as a flat list of {channel, command, enabled} dicts."""
    conn = get_connection()
    async with conn.execute(
        "SELECT channel_id, command_name, enabled FROM guild_permissions WHERE guild_id = ? "
        "ORDER BY channel_id, command_name",
        (guild_id,),
    ) as cur:
        rows = await cur.fetchall()

    result = []
    for r in rows:
        channel_label = "All channels" if r["channel_id"] == _ALL else f"<#{r['channel_id']}>"
        cmd_label = "All commands" if r["command_name"] == _ALL else f"/{r['command_name']}"
        result.append({"channel": channel_label, "command": cmd_label, "enabled": bool(r["enabled"])})
    return result


# ---------------------------------------------------------------------------
# Panic lockdown / restore
# ---------------------------------------------------------------------------


async def panic_lock(guild_id: int) -> None:
    """Snapshot the guild's current ruleset, then deny every command everywhere."""
    conn = get_connection()
    async with get_write_lock():
        async with conn.execute(
            "SELECT channel_id, command_name, enabled FROM guild_permissions WHERE guild_id = ?",
            (guild_id,),
        ) as cur:
            rows = await cur.fetchall()
        backup = [{"channel_id": r["channel_id"], "command_name": r["command_name"], "enabled": bool(r["enabled"])} for r in rows]

        await conn.execute(
            "INSERT INTO guild_panic_backups (guild_id, backup_json) VALUES (?, ?) "
            "ON CONFLICT(guild_id) DO UPDATE SET backup_json = excluded.backup_json",
            (guild_id, json.dumps(backup)),
        )
        await conn.execute("DELETE FROM guild_permissions WHERE guild_id = ?", (guild_id,))
        await conn.execute(
            "INSERT INTO guild_permissions (guild_id, channel_id, command_name, enabled) VALUES (?, ?, ?, 0)",
            (guild_id, _ALL, _ALL),
        )
        await conn.commit()


async def panic_unlock(guild_id: int) -> bool:
    """Restore the most recent panic_lock snapshot. Returns False if none existed."""
    conn = get_connection()
    async with get_write_lock():
        async with conn.execute(
            "SELECT backup_json FROM guild_panic_backups WHERE guild_id = ?", (guild_id,)
        ) as cur:
            row = await cur.fetchone()

        await conn.execute("DELETE FROM guild_permissions WHERE guild_id = ?", (guild_id,))

        if row is None:
            await conn.commit()
            return False

        backup = json.loads(row["backup_json"])
        for entry in backup:
            await conn.execute(
                "INSERT INTO guild_permissions (guild_id, channel_id, command_name, enabled) VALUES (?, ?, ?, ?)",
                (guild_id, entry["channel_id"], entry["command_name"], int(entry["enabled"])),
            )
        await conn.execute("DELETE FROM guild_panic_backups WHERE guild_id = ?", (guild_id,))
        await conn.commit()
        return True
