"""
data/history.py — In-memory per-user calculation history.

Like :mod:`data.cache`, this is **process-memory only** — nothing is written
to disk or any database. History is lost on restart, and each user's log is
capped at :data:`_MAX_ENTRIES_PER_USER` entries (oldest entries are dropped
automatically), so memory usage stays bounded for the lifetime of the bot.

Usage
-----
::

    from data.history import save_history, get_history, clear_history

    save_history(interaction.user.id, "circle_area", "radius=3", "9*pi")
    entries = get_history(interaction.user.id, limit=20)
    clear_history(interaction.user.id)
"""

import threading
from collections import deque
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Module-level singleton store
# ---------------------------------------------------------------------------

_MAX_ENTRIES_PER_USER = 20

_histories: dict[int, deque["HistoryEntry"]] = {}

# A plain lock is enough — entries are small and operations are O(1)/O(n)
# on a tiny per-user deque, so contention is a non-issue at bot scale.
_lock = threading.Lock()


class HistoryEntry:
    """
    One logged command invocation.

    Attributes
    ----------
    command:
        The command name, e.g. ``"circle_area"``.
    input:
        A short human-readable summary of the inputs, e.g. ``"radius=3"``.
    result:
        A short human-readable summary of the result, e.g. ``"9*pi"``.
    timestamp:
        UTC :class:`~datetime.datetime` of when the entry was recorded.
    """

    __slots__ = ("command", "input", "result", "timestamp")

    def __init__(self, command: str, input_str: str, result: str, timestamp: datetime | None = None) -> None:
        self.command = command
        self.input = input_str
        self.result = result
        self.timestamp = timestamp or datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_history(user_id: int, command: str, input_str: str, result: str) -> None:
    """
    Record one command invocation for *user_id*.

    Entries are stored newest-first. If the user's history is already at
    :data:`_MAX_ENTRIES_PER_USER`, the oldest entry is dropped automatically.

    Parameters
    ----------
    user_id:
        The Discord user's numeric ID (``interaction.user.id``).
    command:
        The command name, e.g. ``"circle_area"``.
    input_str:
        A short human-readable summary of the inputs.
    result:
        A short human-readable summary of the result.
    """
    with _lock:
        dq = _histories.setdefault(user_id, deque(maxlen=_MAX_ENTRIES_PER_USER))
        dq.appendleft(HistoryEntry(command, input_str, result))


def get_history(user_id: int, limit: int = 20) -> list[HistoryEntry]:
    """
    Return *user_id*'s most recent entries, newest first.

    Parameters
    ----------
    user_id:
        The Discord user's numeric ID.
    limit:
        Maximum number of entries to return (default ``20``).

    Returns
    -------
    list[HistoryEntry]
        Possibly empty if the user has no recorded history.
    """
    with _lock:
        dq = _histories.get(user_id)
        if not dq:
            return []
        return list(dq)[:limit]


def clear_history(user_id: int) -> None:
    """Discard all stored history for *user_id*."""
    with _lock:
        _histories.pop(user_id, None)