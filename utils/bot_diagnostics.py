"""
utils/bot_diagnostics.py — Shared bot health/log introspection helpers.

Backs ``/admin diagnostics`` and ``/admin logs`` (see ``cogs/admin.py``).

``LOG_FILE`` is the single source of truth for where the bot's rotating
log lives: ``main.py`` writes to it (via a ``RotatingFileHandler``
configured with this same path) and ``cogs/admin.py`` reads from it.
Defining the path in exactly one place avoids the two ever drifting apart.

Process vs. system stats
-------------------------
``get_process_stats()`` reports both the bot's own process usage (CPU%,
RSS memory, thread count — "is *this bot* healthy?") and host-level
system usage (overall CPU/memory/disk — "is the *machine* under
pressure?"). Both are useful for different questions an admin might be
asking, so both are included rather than picking one.
"""

from __future__ import annotations

import os

import psutil

# ---------------------------------------------------------------------------
# Log file location
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOG_FILE = os.path.join(LOG_DIR, "bot.log")


def tail_log(n: int = 50) -> list[str]:
    """
    Return the last *n* lines of the log file, oldest first.

    Returns an empty list if the log file doesn't exist yet (e.g. the bot
    was just started and hasn't logged anything, or is running with a
    different working directory than expected).
    """
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return [ln.rstrip("\n") for ln in lines[-n:]]


# ---------------------------------------------------------------------------
# Resource usage
# ---------------------------------------------------------------------------


def get_process_stats() -> dict[str, float]:
    """
    Return a snapshot of process- and system-level resource usage.

    Note: ``cpu_percent(interval=0.1)`` blocks for ~100ms to measure CPU
    usage over that window — psutil's very first call otherwise returns a
    meaningless ``0.0`` (it has no prior sample to compare against). A
    100ms blocking call is a reasonable cost for an infrequently-run
    admin diagnostics command; it would NOT be reasonable to call this on
    every message or every command.
    """
    proc = psutil.Process(os.getpid())
    with proc.oneshot():
        process_cpu_percent = proc.cpu_percent(interval=0.1)
        mem = proc.memory_info()
        process_memory_percent = proc.memory_percent()
        thread_count = proc.num_threads()

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(LOG_DIR if os.path.isdir(LOG_DIR) else "/")

    return {
        "process_cpu_percent": process_cpu_percent,
        "process_memory_mb": mem.rss / (1024**2),
        "process_memory_percent": process_memory_percent,
        "thread_count": thread_count,
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_memory_percent": vm.percent,
        "system_memory_used_gb": vm.used / (1024**3),
        "system_memory_total_gb": vm.total / (1024**3),
        "disk_percent": disk.percent,
        "disk_used_gb": disk.used / (1024**3),
        "disk_total_gb": disk.total / (1024**3),
    }


def get_data_dir_size_mb() -> float:
    """Total size in MB of files directly inside ``data/`` — the JSON stores plus the log file(s)."""
    total = 0
    if os.path.isdir(LOG_DIR):
        for fname in os.listdir(LOG_DIR):
            fpath = os.path.join(LOG_DIR, fname)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
    return total / (1024**2)


def format_uptime(total_seconds: float) -> str:
    """Render a duration in seconds as ``"1d 2h 3m 4s"``, omitting leading zero units."""
    total = int(total_seconds)
    days, r = divmod(total, 86400)
    hours, r = divmod(r, 3600)
    minutes, seconds = divmod(r, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)
