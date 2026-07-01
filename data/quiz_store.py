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

Phase 5 additions (RANDOM_PROBABILITY_QUIZ_PLAN.md, Phase 5 — social layer)
----------------------------------------------------------------------------
apply_battle_result(guild_id, winner_id, loser_id) -> tuple[dict, dict, int, int]
    Apply an ELO-style rating delta + win/loss tally to both battle
    participants. Deliberately does NOT touch solved/wrong/subject_stats —
    call ``record_result`` separately for that, since a battle loser may
    have answered correctly-but-late (still a "solved", just not a win) or
    not answered at all (no stats change either way).
apply_hint_cost(guild_id, user_id) -> dict
    Deduct a flat rating cost for using ``/quiz hint``, clamped at 0.
get_leaderboard(guild_id, subject=None, limit=10) -> list[dict]
    Already present from Phase 4 (built ahead of need); used by
    ``/quiz leaderboard`` starting Phase 5.
get_daily_seed(date_str=None) -> int
    Deterministic seed for "the same question for everyone today".
has_answered_daily(guild_id, user_id, date_str=None) -> bool
record_daily_result(guild_id, user_id, subject, correct, date_str=None) -> dict
    Records one user's daily-challenge attempt (once per day, enforced
    here), updates the daily streak, and folds the result into the same
    solved/wrong/subject_stats counters practice questions use.
get_daily_leaderboard(guild_id, date_str=None, limit=10) -> list[dict]
    Users who answered *today's* challenge correctly, fastest first.

This module still follows ``data/permissions.py``'s JSON-file +
``threading.Lock`` pattern — no database is introduced for Phase 5 either;
everything above is just more shape layered onto the same flat file plus
one small sibling file (``quiz_daily.json``) for the day-scoped leaderboard
data, which doesn't belong in the per-user record schema.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__))
_QUIZ_FILE = os.path.join(_DATA_DIR, "quiz_stats.json")
_DAILY_FILE = os.path.join(_DATA_DIR, "quiz_daily.json")

_data: dict[str, Any] = {}
_daily_data: dict[str, Any] = {}
_lock = threading.Lock()

SUBJECTS = ("algebra", "calculus", "number_theory", "discrete")

# Phase 5 tuning constants.
_ELO_K = 32           # standard chess-style K-factor (open decision #2, default per plan doc)
_HINT_COST = 15        # rating points deducted per /quiz hint use
HINT_COST = _HINT_COST  # public alias — cogs/quiz.py displays this in the hint embed footer

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
        # Phase 5 fields. Existing on-disk records predate these — every
        # accessor below uses .get()/.setdefault() so old records upgrade
        # in place on first write rather than needing a migration script.
        "battle_wins": 0,
        "battle_losses": 0,
        "daily_streak_current": 0,
        "daily_streak_best": 0,
    }


# ---------------------------------------------------------------------------
# Achievements (Phase 5) — cheap rule-based checks, evaluated after every
# stats-mutating call while the record is already open under the lock.
# ---------------------------------------------------------------------------

_ACHIEVEMENTS: dict[str, tuple[str, Callable[[dict], bool]]] = {
    "week_streak":     ("🔥 Week Streak — 7-question correct streak",       lambda r: r.get("streak_current", 0) >= 7),
    "century_solver":  ("💯 Century Solver — 100 questions solved",         lambda r: r.get("solved", 0) >= 100),
    "battle_champion": ("⚔️ Battle Champion — 10 battle wins",              lambda r: r.get("battle_wins", 0) >= 10),
    "daily_devotee":   ("📅 Daily Devotee — 7-day daily-challenge streak",  lambda r: r.get("daily_streak_current", 0) >= 7),
}

ACHIEVEMENT_LABELS: dict[str, str] = {aid: label for aid, (label, _cond) in _ACHIEVEMENTS.items()}


def _check_achievements(record: dict[str, Any]) -> None:
    """
    Mutate *record*'s ``achievements`` list in place, adding any newly
    earned achievement IDs. Must be called while holding ``_lock``, before
    ``_save()``. Callers that want to announce newly-earned achievements to
    the user should snapshot ``record["achievements"]`` via ``get_record``
    *before* the mutating call and diff it against the returned record's
    list afterward — kept as a caller-side diff rather than a return-value
    change so this stays a drop-in addition to every existing call site.
    """
    held = set(record.get("achievements", []))
    for aid, (_label, cond) in _ACHIEVEMENTS.items():
        if aid not in held and cond(record):
            held.add(aid)
    record["achievements"] = sorted(held)


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


def _load_daily() -> None:
    """(Re-)read the daily-leaderboard JSON file into ``_daily_data``."""
    global _daily_data
    if os.path.exists(_DAILY_FILE):
        try:
            with open(_DAILY_FILE, "r", encoding="utf-8") as fh:
                _daily_data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            _daily_data = {}
    else:
        _daily_data = {}


def _save_daily() -> None:
    """Write ``_daily_data`` back to disk. Must be called under ``_lock``."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_DAILY_FILE, "w", encoding="utf-8") as fh:
        json.dump(_daily_data, fh, indent=2)


def _key(guild_id: int, user_id: int) -> str:
    return f"{guild_id}:{user_id}"


# Load on import.
_load()
_load_daily()


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

        _check_achievements(record)
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


# ---------------------------------------------------------------------------
# Phase 5 — battles (ELO rating + win/loss)
# ---------------------------------------------------------------------------


def apply_battle_result(
    guild_id: int, winner_id: int, loser_id: int
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """
    Apply a standard chess-style ELO update (K=32, starting rating 1200 —
    the plan doc's default assumption) to both battle participants, plus a
    win/loss tally used by the ``battle_champion`` achievement.

    Deliberately does NOT touch ``solved``/``wrong``/``subject_stats`` —
    the caller (``cogs/quiz.py``) calls :func:`record_result` separately
    for each participant based on what they actually submitted, since
    "lost the race" and "answered incorrectly" are different things (a
    correct-but-late answer should still count as solved, just not a win).

    Returns
    -------
    tuple
        ``(winner_record, loser_record, winner_delta, loser_delta)`` —
        the two record copies plus the signed rating change applied to
        each, handy for a "+14 / -14" display in the battle-result embed.
    """
    wkey, lkey = _key(guild_id, winner_id), _key(guild_id, loser_id)
    with _lock:
        winner = _data.setdefault(wkey, _default_record())
        loser = _data.setdefault(lkey, _default_record())
        winner.setdefault("battle_wins", 0)
        loser.setdefault("battle_losses", 0)

        expected_winner = 1 / (1 + 10 ** ((loser["rating"] - winner["rating"]) / 400))
        expected_loser = 1 - expected_winner

        delta_w = round(_ELO_K * (1 - expected_winner))
        delta_l = round(_ELO_K * (0 - expected_loser))

        winner["rating"] = max(0, winner["rating"] + delta_w)
        loser["rating"] = max(0, loser["rating"] + delta_l)
        winner["battle_wins"] += 1
        loser["battle_losses"] += 1

        _check_achievements(winner)
        _check_achievements(loser)
        _save()

        return (
            json.loads(json.dumps(winner)),
            json.loads(json.dumps(loser)),
            delta_w,
            delta_l,
        )


def apply_hint_cost(guild_id: int, user_id: int) -> dict[str, Any]:
    """
    Deduct :data:`_HINT_COST` rating points from *user_id* for using
    ``/quiz hint``, clamped so rating never goes negative. Returns a copy
    of the updated record.
    """
    key = _key(guild_id, user_id)
    with _lock:
        record = _data.setdefault(key, _default_record())
        record["rating"] = max(0, record["rating"] - _HINT_COST)
        _save()
        return json.loads(json.dumps(record))


# ---------------------------------------------------------------------------
# Phase 5 — daily challenge
# ---------------------------------------------------------------------------


def get_daily_date_str() -> str:
    """Return today's UTC date as ``YYYY-MM-DD`` — the daily challenge's day boundary."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def get_daily_seed(date_str: str | None = None) -> int:
    """
    Deterministic seed derived from *date_str* (defaults to today, UTC).

    Passed straight into ``utils.quiz_generator.generate_question(seed=...)``
    so every player in every guild gets the exact same question on the
    same day — the whole point of "daily challenge".
    """
    date_str = date_str or get_daily_date_str()
    return int(date_str.replace("-", ""))


def has_answered_daily(guild_id: int, user_id: int, date_str: str | None = None) -> bool:
    """Return whether *user_id* has already submitted an attempt at today's (or *date_str*'s) daily challenge."""
    date_str = date_str or get_daily_date_str()
    key = _key(guild_id, user_id)
    with _lock:
        return key in _daily_data.get(date_str, {})


def record_daily_result(
    guild_id: int, user_id: int, subject: str, correct: bool, date_str: str | None = None
) -> dict[str, Any]:
    """
    Record *user_id*'s one-and-only attempt at the daily challenge for
    *date_str* (defaults to today), update the daily streak, and fold the
    result into the same ``solved``/``wrong``/``subject_stats`` counters
    practice questions use (it's still a solved-or-not math question).

    Raises
    ------
    ValueError
        If *subject* isn't recognized, or *user_id* already has a
        recorded attempt for *date_str* (daily challenge is once-per-day,
        enforced here rather than trusting the caller).
    """
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")

    date_str = date_str or get_daily_date_str()
    key = _key(guild_id, user_id)

    with _lock:
        day = _daily_data.setdefault(date_str, {})
        if key in day:
            raise ValueError("You've already answered today's daily challenge — come back tomorrow!")

        day[key] = {
            "correct": correct,
            "answered_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        _save_daily()

        record = _data.setdefault(key, _default_record())
        record.setdefault("subject_stats", {})
        record["subject_stats"].setdefault(subject, {"solved": 0, "wrong": 0})

        yesterday = (
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        streak_continues = record.get("last_daily_date") == yesterday

        if correct:
            record["solved"] += 1
            record["subject_stats"][subject]["solved"] += 1
            record["daily_streak_current"] = (
                record.get("daily_streak_current", 0) + 1 if streak_continues else 1
            )
            record["daily_streak_best"] = max(
                record.get("daily_streak_best", 0), record["daily_streak_current"]
            )
        else:
            record["wrong"] += 1
            record["subject_stats"][subject]["wrong"] += 1
            record["daily_streak_current"] = 0

        record["last_daily_date"] = date_str

        _check_achievements(record)
        _save()
        return json.loads(json.dumps(record))


def get_daily_leaderboard(guild_id: int, date_str: str | None = None, limit: int = 10) -> list[dict]:
    """
    Return users in *guild_id* who answered *date_str*'s (defaults to
    today's) daily challenge correctly, fastest-first.

    Returns
    -------
    list[dict]
        Each entry has ``user_id`` (int), ``correct`` (always True — wrong
        attempts are excluded), and ``answered_at`` (ISO timestamp).
    """
    date_str = date_str or get_daily_date_str()
    prefix = f"{guild_id}:"
    with _lock:
        day = _daily_data.get(date_str, {})
        entries = [
            {"user_id": int(k.split(":", 1)[1]), **v}
            for k, v in day.items()
            if k.startswith(prefix) and v.get("correct")
        ]

    entries.sort(key=lambda e: e["answered_at"])
    return entries[:limit]
