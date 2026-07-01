"""
data/csv_session.py — In-memory per-user CSV session store.

Pattern mirrors :mod:`data.history` exactly: in-memory dict, threading.Lock,
bounded size.  No database, no disk writes.  Sessions expire lazily on access
after :data:`_SESSION_TTL_MINUTES` minutes of inactivity.

Usage
-----
::

    from data.csv_session import (
        store_session,
        get_session,
        clear_session,
        get_numeric_column,
        get_column_names,
    )

    session = store_session(user_id, "sales.csv", columns, rows)
    session = get_session(user_id)          # None if expired / not found
    arr     = get_numeric_column(session, "revenue")
    names   = get_column_names(user_id)     # [] if no active session
    clear_session(user_id)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants / limits
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES   = 25 * 1024 * 1024  # 25 MB (Discord free-tier upload cap)
_MAX_ROWS         = 10_000
_MAX_COLUMNS      = 50
_SESSION_TTL_MINUTES = 30

# ---------------------------------------------------------------------------
# Module-level singleton store
# ---------------------------------------------------------------------------

_sessions: dict[int, "CSVSession"] = {}

# A plain lock suffices — individual sessions are small and all operations
# are O(1) / O(n) on tiny per-user objects.
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CSVSession:
    """
    One user's active CSV upload.

    Attributes
    ----------
    user_id:
        Discord user's numeric ID.
    filename:
        Original filename, shown in embeds.
    columns:
        Header row, order preserved.
    rows:
        Raw string data; values are converted on demand by
        :func:`get_numeric_column`.
    uploaded_at:
        UTC timestamp; used for TTL expiry check in :func:`get_session`.
    row_count:
        Cached ``len(rows)`` so callers need not re-evaluate.
    """

    user_id:     int
    filename:    str
    columns:     list[str]
    rows:        list[dict[str, str]]
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    row_count:   int = field(init=False)

    def __post_init__(self) -> None:
        self.row_count = len(self.rows)

    def is_expired(self) -> bool:
        """Return True if the session is older than the TTL."""
        age = datetime.now(tz=timezone.utc) - self.uploaded_at
        return age > timedelta(minutes=_SESSION_TTL_MINUTES)

    def minutes_remaining(self) -> int:
        """Return whole minutes left before expiry (0 if already expired)."""
        age = datetime.now(tz=timezone.utc) - self.uploaded_at
        remaining = timedelta(minutes=_SESSION_TTL_MINUTES) - age
        return max(0, int(remaining.total_seconds() // 60))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_session(
    user_id:  int,
    filename: str,
    columns:  list[str],
    rows:     list[dict[str, str]],
) -> CSVSession:
    """
    Create (or replace) the CSV session for *user_id*.

    Any existing session for this user is silently discarded.

    Parameters
    ----------
    user_id:
        Discord user's numeric ID.
    filename:
        Original filename for display purposes.
    columns:
        Ordered list of column names from the CSV header row.
    rows:
        List of ``{column_name: raw_string_value}`` dicts — one per data row.

    Returns
    -------
    CSVSession
        The newly created session.
    """
    session = CSVSession(
        user_id=user_id,
        filename=filename,
        columns=columns,
        rows=rows,
    )
    with _lock:
        _sessions[user_id] = session
    return session


def get_session(user_id: int) -> Optional[CSVSession]:
    """
    Return the active session for *user_id*, or ``None`` if it has expired
    or never existed.

    Expiry is checked lazily on every access; stale sessions are removed
    from the store at that point.

    Parameters
    ----------
    user_id:
        Discord user's numeric ID.
    """
    with _lock:
        session = _sessions.get(user_id)
        if session is None:
            return None
        if session.is_expired():
            del _sessions[user_id]
            return None
        return session


def clear_session(user_id: int) -> None:
    """
    Discard the session for *user_id*.  No-op if no session exists.

    Parameters
    ----------
    user_id:
        Discord user's numeric ID.
    """
    with _lock:
        _sessions.pop(user_id, None)


def get_numeric_column(session: CSVSession, col: str) -> np.ndarray:
    """
    Extract *col* from *session* as a ``float64`` NumPy array.

    Parameters
    ----------
    session:
        An active :class:`CSVSession`.
    col:
        Column name — must exist in ``session.columns``.

    Returns
    -------
    np.ndarray
        1-D array of ``float64`` values, one per data row.

    Raises
    ------
    ValueError
        If *col* is not in the session, or if any non-empty cell cannot be
        converted to a float (the first offending cell is shown in the message).
    """
    if col not in session.columns:
        available = ", ".join(f"`{c}`" for c in session.columns)
        raise ValueError(
            f"Column `{col}` not found.\n"
            f"Available columns: {available}"
        )

    values: list[float] = []
    for i, row in enumerate(session.rows, start=1):
        raw = row.get(col, "").strip()
        if not raw:
            continue  # skip blanks rather than erroring
        try:
            values.append(float(raw))
        except ValueError:
            raise ValueError(
                f"Column `{col}` contains a non-numeric value at row {i}: `{raw}`.\n"
                "This command requires a numeric column."
            )

    if not values:
        raise ValueError(f"Column `{col}` has no numeric values.")

    return np.array(values, dtype=np.float64)


def get_column_names(user_id: int) -> list[str]:
    """
    Return the column names from the active session for *user_id*.

    Returns an empty list if no session exists or the session has expired.
    Used by the autocomplete callback so it never raises.

    Parameters
    ----------
    user_id:
        Discord user's numeric ID.
    """
    session = get_session(user_id)
    if session is None:
        return []
    return session.columns