"""
data/quiz_store.py — Persistent quiz statistics store.

Unlike ``data/memory.py``, ``data/history.py``, and ``data/csv_session.py``
(all in-memory only, wiped on restart), quiz ratings/streaks need to
survive a bot restart, so this module follows ``data/permissions.py``'s
JSON-file-backed pattern instead — the one existing precedent for
persistence in this codebase, per the plan doc
(RANDOM_PROBABILITY_QUIZ_PLAN.md, Phase 4).

Schema
------
The JSON file (``data/quiz_stats.json``) stores a flat dict:

    {
        "<guild_id>:<user_id>": {
            "rating": 1200,
            "solved": 0,
            "wrong": 0,
            "streak_current": 0,
            "streak_best": 0,
            "subject_stats": {
                "algebra":        {"solved": 0, "wrong": 0},
                "calculus":       {"solved": 0, "wrong": 0},
                "number_theory":  {"solved": 0, "wrong": 0},
                "discrete":       {"solved": 0, "wrong": 0}
            },
            "last_daily_date": null,
            "achievements": []
        }
    }

Public API
----------
get_record(guild_id, user_id) -> dict          Read-only copy of a user's record (creates default if absent, but does NOT persist the default until first write).
record_result(guild_id, user_id, subject, correct) -> dict   Update stats after one answered question; returns the updated record.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__))
_QUIZ_FILE = os.path.join(_DATA_DIR, "quiz_stats.json")

_data: dict[str, Any] = {}
_lock = threading.Lock()

SUBJECTS = ("algebra", "calculus", "number_theory", "discrete")

_DEFAULT_RECORD_TEMPLATE: dict[str, Any] = {
    "rating": 1200,
    "solved": 0,
    "wrong": 0,
    "streak_current": 0,
    "streak_best": 0,
    "subject_stats": {s: {"solved": 0, "wrong": 0} for s in SUBJECTS},
    "last_daily_date": None,
    "achievements": [],
}


def _default_record() -> dict[str, Any]:
    """Return a fresh default record (deep-copied, never shared/mutated in place)."""
    return {
        "rating": _DEFAULT_RECORD_TEMPLATE["rating"],
        "solved": 0,
        "wrong": 0,
        "streak_current": 0,
        "streak_best": 0,
        "subject_stats": {s: {"solved": 0, "wrong": 0} for s in SUBJECTS},
        "last_daily_date": None,
        "achievements": [],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load() -> None:
    """(Re-)read the JSON file into ``_data``. Called once at import time."""
    global _data
    if os.path.exists(_QUIZ_FILE):
        try:
            with open(_QUIZ_FILE, "r", encoding="utf-8") as fh:
                _data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            _data = {}
    else:
        _data = {}


def _save() -> None:
    """Write ``_data`` back to disk. Must be called under ``_lock``."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_QUIZ_FILE, "w", encoding="utf-8") as fh:
        json.dump(_data, fh, indent=2)


def _key(guild_id: int, user_id: int) -> str:
    return f"{guild_id}:{user_id}"


# Load on import.
_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_record(guild_id: int, user_id: int) -> dict[str, Any]:
    """
    Return a copy of *user_id*'s quiz record in *guild_id*.

    If the user has no record yet, returns a fresh default WITHOUT writing
    it to disk — the record is only persisted once the user actually
    answers a question, via :func:`record_result`. This keeps the JSON
    file free of empty entries for users who merely ran ``/quiz stats``
    out of curiosity without ever practicing.
    """
    key = _key(guild_id, user_id)
    with _lock:
        record = _data.get(key)
        if record is None:
            return _default_record()
        # Defensive copy so callers can't mutate our in-memory cache directly.
        return json.loads(json.dumps(record))


def record_result(guild_id: int, user_id: int, subject: str, correct: bool) -> dict[str, Any]:
    """
    Update *user_id*'s stats in *guild_id* after answering one *subject*
    question, and persist to disk.

    Parameters
    ----------
    subject:
        One of :data:`SUBJECTS`.
    correct:
        Whether the answer was correct.

    Returns
    -------
    dict
        A copy of the updated record.

    Raises
    ------
    ValueError
        If *subject* isn't a recognized subject.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")

    key = _key(guild_id, user_id)
    with _lock:
        record = _data.setdefault(key, _default_record())
        # Guard against a record from an older/partial schema (e.g. a
        # subject added after some users already have records on disk).
        record.setdefault("subject_stats", {})
        record["subject_stats"].setdefault(subject, {"solved": 0, "wrong": 0})

        if correct:
            record["solved"] += 1
            record["subject_stats"][subject]["solved"] += 1
            record["streak_current"] += 1
            record["streak_best"] = max(record["streak_best"], record["streak_current"])
        else:
            record["wrong"] += 1
            record["subject_stats"][subject]["wrong"] += 1
            record["streak_current"] = 0

        _save()
        return json.loads(json.dumps(record))


def get_leaderboard(guild_id: int, subject: str | None = None, limit: int = 10) -> list[dict]:
    """
    Return the top *limit* users in *guild_id* by rating, scoped to this
    guild only (never leaks across servers — matches the ``<guild_id>:``
    key prefix).

    Parameters
    ----------
    subject:
        If given, ranks by that subject's solved count instead of overall
        rating (kept for forward-compatibility with Phase 5's subject
        leaderboards; unused by Phase 4).

    Returns
    -------
    list[dict]
        Each entry has ``user_id`` (int) plus the full record fields,
        sorted best-first.
    """
    prefix = f"{guild_id}:"
    with _lock:
        entries = [
            {"user_id": int(key.split(":", 1)[1]), **json.loads(json.dumps(rec))}
            for key, rec in _data.items()
            if key.startswith(prefix)
        ]

    if subject is not None:
        if subject not in SUBJECTS:
            raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")
        entries.sort(key=lambda e: e["subject_stats"].get(subject, {}).get("solved", 0), reverse=True)
    else:
        entries.sort(key=lambda e: e["rating"], reverse=True)

    return entries[:limit]
