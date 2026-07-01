"""
utils/rng.py — Shared randomness helpers.

Used directly by ``cogs/random_tools.py`` (Phase 1), and will be reused
without modification by the probability cog (Phase 3) and the quiz
question generator (Phase 4/5), so that every part of the bot that needs
"a random thing" goes through one source of truth rather than each cog
calling the stdlib ``random`` module independently with subtly different
conventions.

Two concerns are deliberately kept separate here:

``get_rng`` / ``get_user_rng``
    Ordinary pseudo-randomness for dice rolls, shuffles, quiz question
    generation, etc. Uses :mod:`random`, optionally seeded for
    reproducibility within a session (e.g. "regenerate the same battle
    question for both players").

``secure_token``
    Anything security-sensitive (tokens/passwords). Uses :mod:`secrets`,
    never :mod:`random` — ``random.Random`` is not cryptographically
    secure and must never be used for this purpose.
"""

from __future__ import annotations

import random
import secrets

from data.rng_seed import seed_store

# ---------------------------------------------------------------------------
# Ordinary randomness
# ---------------------------------------------------------------------------


def get_rng(seed: int | None = None) -> random.Random:
    """
    Return a fresh ``random.Random`` instance.

    If *seed* is given, the instance is deterministic — calling this twice
    with the same seed produces the same sequence of outputs. Used for
    reproducible quiz questions (so a battle's two participants can be
    proven to have received the identical problem) and for debugging.
    """
    return random.Random(seed) if seed is not None else random.Random()


def get_user_rng(guild_id: int, user_id: int) -> random.Random:
    """
    Return a ``random.Random`` instance honoring this user's session seed,
    if they've set one via ``/rand seed``. Falls back to fresh randomness
    if no seed is set.
    """
    seed = seed_store.get(guild_id, user_id)
    return get_rng(seed)


# ---------------------------------------------------------------------------
# Security-sensitive randomness
# ---------------------------------------------------------------------------


def secure_token(length: int) -> str:
    """
    Return a cryptographically secure URL-safe token of roughly *length*
    characters.

    Always uses :mod:`secrets`. Never routed through ``get_rng``/
    ``random.Random``, which are explicitly NOT suitable for anything
    security-sensitive (predictable given enough output, seedable).
    """
    return secrets.token_urlsafe(length)
