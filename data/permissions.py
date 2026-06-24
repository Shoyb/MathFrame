"""
data/permissions.py — Guild-level command permission store.

Schema
------
The JSON file (``data/guild_permissions.json``) stores a nested dict:

    {
        "<guild_id>": {
            "<channel_id | '__all__'>": {
                "<command_name | '__all__'>": true | false
            }
        }
    }

Lookup order (most-specific wins)
----------------------------------
1. ``guild → channel → command``   (channel + command specific)
2. ``guild → __all__ → command``   (guild-wide for this command)
3. ``guild → channel → __all__``   (all commands in this channel)
4. ``guild → __all__ → __all__``   (entire guild default)
5. Not present → allowed (fail-open; admins must explicitly disable)

Public API
----------
is_command_allowed(guild_id, channel_id, command_name) -> bool
set_permission(guild_id, channel_id, command_name, enabled) -> None
get_guild_status(guild_id) -> list[dict]
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------

_DATA_DIR   = os.path.join(os.path.dirname(__file__))
_PERM_FILE  = os.path.join(_DATA_DIR, "guild_permissions.json")

# In-memory cache — loaded once and written back on every change.
_data: dict[str, Any] = {}
_lock = threading.Lock()

# Sentinel string used as the wildcard channel / command key.
_ALL = "__all__"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> None:
    """(Re-)read the JSON file into ``_data``. Called once at import time."""
    global _data
    if os.path.exists(_PERM_FILE):
        try:
            with open(_PERM_FILE, "r", encoding="utf-8") as fh:
                _data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            _data = {}
    else:
        _data = {}


def _save() -> None:
    """Write ``_data`` back to disk. Must be called under ``_lock``."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_PERM_FILE, "w", encoding="utf-8") as fh:
        json.dump(_data, fh, indent=2)


# Load on import.
_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_command_allowed(guild_id: int, channel_id: int, command_name: str) -> bool:
    """
    Return ``True`` if *command_name* is allowed in *channel_id* of *guild_id*.

    The lookup follows the specificity order documented in the module docstring.
    When no rule matches, the command is considered **allowed** (fail-open).
    """
    gid = str(guild_id)
    cid = str(channel_id)
    cmd = command_name

    with _lock:
        guild = _data.get(gid, {})
        if not guild:
            return True  # no rules for this guild → allow

        # 1. channel + command
        val = guild.get(cid, {}).get(cmd)
        if val is not None:
            return bool(val)

        # 2. guild-wide + command
        val = guild.get(_ALL, {}).get(cmd)
        if val is not None:
            return bool(val)

        # 3. channel + all commands
        val = guild.get(cid, {}).get(_ALL)
        if val is not None:
            return bool(val)

        # 4. guild-wide + all commands
        val = guild.get(_ALL, {}).get(_ALL)
        if val is not None:
            return bool(val)

    return True  # no matching rule → allow


def set_permission(
    guild_id: int,
    channel_id: int | None,
    command_name: str | None,
    enabled: bool,
) -> None:
    """
    Set or clear a permission rule.

    Parameters
    ----------
    guild_id:
        The Discord guild (server) snowflake ID.
    channel_id:
        The channel snowflake ID, or ``None`` to apply guild-wide (``__all__``).
    command_name:
        The slash command name (without ``/``), or ``None`` for all commands.
    enabled:
        ``True`` to allow, ``False`` to deny.
    """
    gid = str(guild_id)
    cid = str(channel_id) if channel_id is not None else _ALL
    cmd = command_name if command_name is not None else _ALL

    with _lock:
        guild = _data.setdefault(gid, {})
        channel = guild.setdefault(cid, {})
        channel[cmd] = enabled
        _save()


def clear_permission(
    guild_id: int,
    channel_id: int | None,
    command_name: str | None,
) -> bool:
    """
    Remove an existing permission rule.

    Returns ``True`` if a rule was removed, ``False`` if none was found.
    """
    gid = str(guild_id)
    cid = str(channel_id) if channel_id is not None else _ALL
    cmd = command_name if command_name is not None else _ALL

    with _lock:
        guild = _data.get(gid, {})
        channel = guild.get(cid, {})
        if cmd in channel:
            del channel[cmd]
            # Prune empty dicts to keep the file tidy
            if not channel:
                del guild[cid]
            if not guild:
                del _data[gid]
            _save()
            return True
    return False


def get_guild_status(guild_id: int) -> list[dict]:
    """
    Return all permission rules for *guild_id* as a flat list of dicts.

    Each dict has keys: ``channel``, ``command``, ``enabled``.
    """
    gid = str(guild_id)
    rows: list[dict] = []

    with _lock:
        guild = _data.get(gid, {})
        for cid, commands in guild.items():
            channel_label = "All channels" if cid == _ALL else f"<#{cid}>"
            for cmd, enabled in commands.items():
                cmd_label = "All commands" if cmd == _ALL else f"/{cmd}"
                rows.append({
                    "channel": channel_label,
                    "command": cmd_label,
                    "enabled": enabled,
                })

    return rows
