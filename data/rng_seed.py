"""
data/rng_seed.py — Per-user reproducible-randomness seed store.

Mirrors :mod:`data.memory`'s pattern exactly: thread-safe, in-memory,
keyed by ``(guild_id, user_id)``. Deliberately NOT persisted to disk —
unlike quiz ratings (Phase 4/5), a session seed only needs to survive a
single live session ("regenerate the same battle question"), not a bot
restart, so the lighter in-memory-only pattern (matching memory.py and
history.py rather than permissions.py) is the right fit here.

Usage::

    from data.rng_seed import seed_store

    seed_store.set(guild_id, user_id, 42)
    seed = seed_store.get(guild_id, user_id)   # 42, or None if unset
    seed_store.clear(guild_id, user_id)
"""

from __future__ import annotations

import threading


class SeedStore:
    """Thread-safe per-user reproducible-randomness seed store."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], int] = {}
        self._lock = threading.Lock()

    def set(self, guild_id: int, user_id: int, seed: int) -> None:
        """Set a reproducible seed for this user's session."""
        with self._lock:
            self._data[(guild_id, user_id)] = seed

    def get(self, guild_id: int, user_id: int) -> int | None:
        """Return this user's seed, or ``None`` if unset."""
        with self._lock:
            return self._data.get((guild_id, user_id))

    def clear(self, guild_id: int, user_id: int) -> bool:
        """
        Clear this user's seed.

        Returns
        -------
        bool
            ``True`` if a seed was present and removed, ``False`` if there
            was nothing to clear.
        """
        with self._lock:
            return self._data.pop((guild_id, user_id), None) is not None


seed_store = SeedStore()
