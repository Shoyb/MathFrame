"""
data/cache.py — In-memory TTL result cache.

A module-level singleton TTLCache shared across all cogs.
Entries expire automatically after config.CACHE_TTL seconds and
the cache never holds more than config.CACHE_MAXSIZE entries at once
(LRU eviction when full).

Usage
-----
    from data.cache import get, set, cache_key

    key = cache_key("simplify", "x**2 + 2*x + 1")
    result = get(key)
    if result is None:
        result = expensive_computation()
        set(key, result)
"""

import threading

from cachetools import TTLCache

import config

# ---------------------------------------------------------------------------
# Singleton cache instance
# ---------------------------------------------------------------------------

_cache: TTLCache = TTLCache(
    maxsize=config.CACHE_MAXSIZE,
    ttl=config.CACHE_TTL,
)

# TTLCache is not thread-safe by default; a lock makes concurrent cog
# calls safe without any performance hit for a bot-scale workload.
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_key(*args: object) -> str:
    """
    Build a cache key by joining all arguments with ``|``.

    Examples
    --------
    >>> cache_key("simplify", "x**2 + 1")
    'simplify|x**2 + 1'
    >>> cache_key("diff", "sin(x)", "x", "2")
    'diff|sin(x)|x|2'
    """
    return "|".join(str(a) for a in args)


def get(key: str) -> object:
    """
    Return the cached value for *key*, or ``None`` if absent / expired.

    Parameters
    ----------
    key:
        A string key, typically produced by :func:`cache_key`.
    """
    with _lock:
        return _cache.get(key)          # returns None on missing key


def set(key: str, value: object) -> None:
    """
    Store *value* under *key*.

    If the cache is full the least-recently-used entry is evicted
    automatically by TTLCache before the new entry is inserted.

    Parameters
    ----------
    key:
        A string key, typically produced by :func:`cache_key`.
    value:
        Any picklable Python object (discord.Embed, str, list, …).
    """
    with _lock:
        _cache[key] = value


def clear() -> None:
    """
    Remove all entries from the cache.

    Intended for use in tests and for ``/cache_clear`` admin commands.
    """
    with _lock:
        _cache.clear()


def info() -> dict:
    """
    Return a snapshot of current cache statistics.

    Returns
    -------
    dict with keys ``size``, ``maxsize``, ``ttl``, ``currsize``.
    Useful for a ``/debug`` or ``/about`` admin command.
    """
    with _lock:
        return {
            "currsize": _cache.currsize,
            "maxsize":  _cache.maxsize,
            "ttl":      _cache.ttl,
        }
