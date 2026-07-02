"""
data/quiz_store.py — Persistent quiz statistics store (SQLite-backed).

Migrated from the JSON-file + threading.Lock version to ``data/db.py``'s
shared SQLite connection. Every public function below is now a coroutine
(``await`` it) but the returned record shape is unchanged, so callers only
need to add ``await`` at each call site — see ``cogs/quiz.py``.

Returned record shape (identical to the old JSON version)
-----------------------------------------------------------
    {
        "rating": 1200,
        "solved": 0,
        "wrong": 0,
        "streak_current": 0,
        "streak_best": 0,
        "battle_wins": 0,
        "battle_losses": 0,
        "daily_streak_current": 0,
        "daily_streak_best": 0,
        "last_daily_date": None,
        "subject_stats": {
            "algebra":       {"solved": 0, "wrong": 0},
            "calculus":      {"solved": 0, "wrong": 0},
            "number_theory": {"solved": 0, "wrong": 0},
            "discrete":      {"solved": 0, "wrong": 0},
        },
        "achievements": [],
    }

Public API
----------
Unchanged from the JSON version, just async now:
    get_record, record_result, get_leaderboard, apply_battle_result,
    apply_hint_cost, get_daily_date_str, get_daily_seed,
    has_answered_daily, record_daily_result, get_daily_leaderboard
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from data.db import get_connection, get_write_lock, SUBJECTS

_ELO_K = 32
_HINT_COST = 15
HINT_COST = _HINT_COST

_ACHIEVEMENTS: dict[str, tuple[str, Callable[[dict], bool]]] = {
    "week_streak":     ("🔥 Week Streak — 7-question correct streak",       lambda r: r.get("streak_current", 0) >= 7),
    "century_solver":  ("💯 Century Solver — 100 questions solved",         lambda r: r.get("solved", 0) >= 100),
    "battle_champion": ("⚔️ Battle Champion — 10 battle wins",              lambda r: r.get("battle_wins", 0) >= 10),
    "daily_devotee":   ("📅 Daily Devotee — 7-day daily-challenge streak",  lambda r: r.get("daily_streak_current", 0) >= 7),
}

ACHIEVEMENT_LABELS: dict[str, str] = {aid: label for aid, (label, _cond) in _ACHIEVEMENTS.items()}


def _default_record() -> dict[str, Any]:
    return {
        "rating": 1200,
        "solved": 0,
        "wrong": 0,
        "streak_current": 0,
        "streak_best": 0,
        "battle_wins": 0,
        "battle_losses": 0,
        "daily_streak_current": 0,
        "daily_streak_best": 0,
        "last_daily_date": None,
        "subject_stats": {s: {"solved": 0, "wrong": 0} for s in SUBJECTS},
        "achievements": [],
    }


async def _fetch_record(guild_id: int, user_id: int) -> dict[str, Any] | None:
    """Read one record + its subject stats + achievements. None if the user has no row yet."""
    conn = get_connection()
    async with conn.execute(
        "SELECT * FROM quiz_records WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None

    record = _default_record()
    for field in (
        "rating", "solved", "wrong", "streak_current", "streak_best",
        "battle_wins", "battle_losses", "daily_streak_current",
        "daily_streak_best", "last_daily_date",
    ):
        record[field] = row[field]

    async with conn.execute(
        "SELECT subject, solved, wrong FROM quiz_subject_stats WHERE guild_id = ? AND user_id = ?",
        (guild_id, user_id),
    ) as cur:
        async for s_row in cur:
            record["subject_stats"][s_row["subject"]] = {"solved": s_row["solved"], "wrong": s_row["wrong"]}

    async with conn.execute(
        "SELECT achievement_id FROM quiz_achievements WHERE guild_id = ? AND user_id = ? ORDER BY achievement_id",
        (guild_id, user_id),
    ) as cur:
        record["achievements"] = [r["achievement_id"] async for r in cur]

    return record


async def _ensure_row(guild_id: int, user_id: int) -> None:
    """INSERT OR IGNORE a default row so subsequent UPDATEs have something to touch."""
    conn = get_connection()
    await conn.execute(
        "INSERT OR IGNORE INTO quiz_records (guild_id, user_id) VALUES (?, ?)", (guild_id, user_id)
    )


async def _save_achievements(guild_id: int, user_id: int, record: dict[str, Any]) -> None:
    """Insert any achievement in record['achievements'] not already stored (idempotent)."""
    conn = get_connection()
    now = datetime.now(tz=timezone.utc).isoformat()
    for aid in record["achievements"]:
        await conn.execute(
            "INSERT OR IGNORE INTO quiz_achievements (guild_id, user_id, achievement_id, earned_at) VALUES (?, ?, ?, ?)",
            (guild_id, user_id, aid, now),
        )


def _check_achievements(record: dict[str, Any]) -> None:
    """Mutate record['achievements'] in place, same rule-based logic as before."""
    held = set(record.get("achievements", []))
    for aid, (_label, cond) in _ACHIEVEMENTS.items():
        if aid not in held and cond(record):
            held.add(aid)
    record["achievements"] = sorted(held)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_record(guild_id: int, user_id: int) -> dict[str, Any]:
    """Return a copy of *user_id*'s quiz record in *guild_id* (defaults if none exists yet)."""
    record = await _fetch_record(guild_id, user_id)
    return record if record is not None else _default_record()


async def record_result(guild_id: int, user_id: int, subject: str, correct: bool) -> dict[str, Any]:
    """Update stats after one answered *subject* question and persist."""
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")

    conn = get_connection()
    async with get_write_lock():
        await _ensure_row(guild_id, user_id)
        record = await _fetch_record(guild_id, user_id) or _default_record()

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

        await conn.execute(
            """
            UPDATE quiz_records
            SET solved = ?, wrong = ?, streak_current = ?, streak_best = ?
            WHERE guild_id = ? AND user_id = ?
            """,
            (record["solved"], record["wrong"], record["streak_current"], record["streak_best"], guild_id, user_id),
        )
        await conn.execute(
            """
            INSERT INTO quiz_subject_stats (guild_id, user_id, subject, solved, wrong)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(guild_id, user_id, subject) DO UPDATE SET solved = excluded.solved, wrong = excluded.wrong
            """,
            (guild_id, user_id, subject, record["subject_stats"][subject]["solved"], record["subject_stats"][subject]["wrong"]),
        )
        await _save_achievements(guild_id, user_id, record)
        await conn.commit()
        return record


async def get_leaderboard(guild_id: int, subject: str | None = None, limit: int = 10) -> list[dict]:
    """Top *limit* users in *guild_id* by rating, or by a subject's solved count."""
    conn = get_connection()
    if subject is not None:
        if subject not in SUBJECTS:
            raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")
        async with conn.execute(
            """
            SELECT r.user_id, COALESCE(s.solved, 0) AS solved
            FROM quiz_records r
            LEFT JOIN quiz_subject_stats s
                ON s.guild_id = r.guild_id AND s.user_id = r.user_id AND s.subject = ?
            WHERE r.guild_id = ?
            ORDER BY solved DESC
            LIMIT ?
            """,
            (subject, guild_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        results = []
        for row in rows:
            record = await get_record(guild_id, row["user_id"])
            results.append({"user_id": row["user_id"], **record})
        return results

    async with conn.execute(
        "SELECT user_id FROM quiz_records WHERE guild_id = ? ORDER BY rating DESC LIMIT ?",
        (guild_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    results = []
    for row in rows:
        record = await get_record(guild_id, row["user_id"])
        results.append({"user_id": row["user_id"], **record})
    return results


# ---------------------------------------------------------------------------
# Battles (ELO rating + win/loss)
# ---------------------------------------------------------------------------


async def apply_battle_result(
    guild_id: int, winner_id: int, loser_id: int
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Apply a chess-style ELO update (K=32) + win/loss tally to both battle participants."""
    conn = get_connection()
    async with get_write_lock():
        await _ensure_row(guild_id, winner_id)
        await _ensure_row(guild_id, loser_id)
        winner = await _fetch_record(guild_id, winner_id) or _default_record()
        loser = await _fetch_record(guild_id, loser_id) or _default_record()

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

        await conn.execute(
            "UPDATE quiz_records SET rating = ?, battle_wins = ? WHERE guild_id = ? AND user_id = ?",
            (winner["rating"], winner["battle_wins"], guild_id, winner_id),
        )
        await conn.execute(
            "UPDATE quiz_records SET rating = ?, battle_losses = ? WHERE guild_id = ? AND user_id = ?",
            (loser["rating"], loser["battle_losses"], guild_id, loser_id),
        )
        await _save_achievements(guild_id, winner_id, winner)
        await _save_achievements(guild_id, loser_id, loser)
        await conn.commit()

        return winner, loser, delta_w, delta_l


async def apply_hint_cost(guild_id: int, user_id: int) -> dict[str, Any]:
    """Deduct HINT_COST rating points, clamped at 0."""
    conn = get_connection()
    async with get_write_lock():
        await _ensure_row(guild_id, user_id)
        record = await _fetch_record(guild_id, user_id) or _default_record()
        record["rating"] = max(0, record["rating"] - _HINT_COST)
        await conn.execute(
            "UPDATE quiz_records SET rating = ? WHERE guild_id = ? AND user_id = ?",
            (record["rating"], guild_id, user_id),
        )
        await conn.commit()
        return record


# ---------------------------------------------------------------------------
# Daily challenge
# ---------------------------------------------------------------------------


def get_daily_date_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def get_daily_seed(date_str: str | None = None) -> int:
    date_str = date_str or get_daily_date_str()
    return int(date_str.replace("-", ""))


async def has_answered_daily(guild_id: int, user_id: int, date_str: str | None = None) -> bool:
    date_str = date_str or get_daily_date_str()
    conn = get_connection()
    async with conn.execute(
        "SELECT 1 FROM quiz_daily_results WHERE guild_id = ? AND user_id = ? AND date_str = ?",
        (guild_id, user_id, date_str),
    ) as cur:
        return (await cur.fetchone()) is not None


async def record_daily_result(
    guild_id: int, user_id: int, subject: str, correct: bool, date_str: str | None = None
) -> dict[str, Any]:
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")

    date_str = date_str or get_daily_date_str()
    conn = get_connection()

    async with get_write_lock():
        if await has_answered_daily(guild_id, user_id, date_str):
            raise ValueError("You've already answered today's daily challenge — come back tomorrow!")

        await conn.execute(
            "INSERT INTO quiz_daily_results (guild_id, user_id, date_str, correct, answered_at) VALUES (?, ?, ?, ?, ?)",
            (guild_id, user_id, date_str, int(correct), datetime.now(tz=timezone.utc).isoformat()),
        )

        await _ensure_row(guild_id, user_id)
        record = await _fetch_record(guild_id, user_id) or _default_record()

        yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        streak_continues = record.get("last_daily_date") == yesterday

        if correct:
            record["solved"] += 1
            record["subject_stats"][subject]["solved"] += 1
            record["daily_streak_current"] = record["daily_streak_current"] + 1 if streak_continues else 1
            record["daily_streak_best"] = max(record["daily_streak_best"], record["daily_streak_current"])
        else:
            record["wrong"] += 1
            record["subject_stats"][subject]["wrong"] += 1
            record["daily_streak_current"] = 0

        record["last_daily_date"] = date_str
        _check_achievements(record)

        await conn.execute(
            """
            UPDATE quiz_records
            SET solved = ?, wrong = ?, daily_streak_current = ?, daily_streak_best = ?, last_daily_date = ?
            WHERE guild_id = ? AND user_id = ?
            """,
            (
                record["solved"], record["wrong"], record["daily_streak_current"],
                record["daily_streak_best"], record["last_daily_date"], guild_id, user_id,
            ),
        )
        await conn.execute(
            """
            INSERT INTO quiz_subject_stats (guild_id, user_id, subject, solved, wrong)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(guild_id, user_id, subject) DO UPDATE SET solved = excluded.solved, wrong = excluded.wrong
            """,
            (guild_id, user_id, subject, record["subject_stats"][subject]["solved"], record["subject_stats"][subject]["wrong"]),
        )
        await _save_achievements(guild_id, user_id, record)
        await conn.commit()
        return record


async def get_daily_leaderboard(guild_id: int, date_str: str | None = None, limit: int = 10) -> list[dict]:
    date_str = date_str or get_daily_date_str()
    conn = get_connection()
    async with conn.execute(
        """
        SELECT user_id, answered_at FROM quiz_daily_results
        WHERE guild_id = ? AND date_str = ? AND correct = 1
        ORDER BY answered_at ASC
        LIMIT ?
        """,
        (guild_id, date_str, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [{"user_id": row["user_id"], "correct": True, "answered_at": row["answered_at"]} for row in rows]
