"""
config.py — Bot-wide configuration loaded from environment variables.

Constants
---------
DISCORD_TOKEN : str
    Your bot's secret token from the Discord Developer Portal.
    Loaded from the DISCORD_TOKEN environment variable (set in .env).
    Never commit this value to version control.

PREFIX : str
    The prefix for traditional prefix-based commands (e.g. "!help").
    Slash commands do not use this, but discord.py's Bot requires it.

MAX_EXPR_LENGTH : int
    Maximum number of characters accepted in a math expression string.
    Inputs longer than this are rejected before any parsing begins,
    preventing memory abuse from absurdly large inputs.

COMPUTE_TIMEOUT : int | float
    Seconds allowed for a single computation before it is cancelled.
    SymPy can hang indefinitely on pathological inputs (e.g. symbolic
    integrals with no closed form), so this acts as a hard safety net.

CACHE_TTL : int
    Time-to-live in seconds for entries in the in-memory result cache.
    After this many seconds a cached result is considered stale and
    will be recomputed on the next request. 300 s = 5 minutes.

CACHE_MAXSIZE : int
    Maximum number of entries the TTL cache may hold at once.
    Once the cache is full, the least-recently-used entry is evicted
    to make room for a new one.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN")
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
PREFIX: str = "!"

MAX_EXPR_LENGTH: int = 500

COMPUTE_TIMEOUT: int = 3

CACHE_TTL: int = 300

CACHE_MAXSIZE: int = 256