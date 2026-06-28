"""
data/memory.py — Per-user in-memory variable store.

Each user gets an isolated namespace keyed by (guild_id, user_id).
Guild-ID 0 is used for DMs.

Stored types
------------
NUMBER      A SymPy expression with no free symbols (pure numeric value).
            Examples: 3.14, sqrt(2), 1/3, pi
EXPRESSION  A SymPy expression with at least one free symbol.
            Examples: x^2 + 2*x, sin(t) * exp(-t)
MATRIX      A sympy.Matrix object.
            Examples: [[1,2],[3,4]], [[1],[0],[0]]

$-reference syntax
------------------
In any expression string, ``$name`` is replaced with the stored value
before parsing.  This keeps stored names from ever clashing with SymPy's
free-symbol parsing — bare ``x`` in ``/solve x^2 = 0`` is always a free
symbol regardless of what is stored under the name ``x``.

    /mem set  k  3.14
    /diff     $k * x^2   →  parse_expression("(3.14) * x^2")

Matrix entries cannot be inlined with ``$`` (they raise ValueError) but can
be retrieved via ``/mem get`` or used in matrix-specific commands.
"""

from __future__ import annotations

import ast
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ENTRIES  = 50
MAX_NAME_LEN = 32

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class MemType(str, Enum):
    NUMBER     = "number"
    EXPRESSION = "expression"
    MATRIX     = "matrix"


_EMOJI = {
    MemType.NUMBER:     "🔢",
    MemType.EXPRESSION: "🔣",
    MemType.MATRIX:     "📐",
}


@dataclass
class MemEntry:
    """A single stored memory entry."""
    name:     str
    mem_type: MemType
    value:    Any    # sympy.Expr | sympy.Matrix
    raw:      str    # original string the user supplied

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @property
    def emoji(self) -> str:
        return _EMOJI[self.mem_type]

    @property
    def type_label(self) -> str:
        return self.mem_type.value.capitalize()

    def display_str(self) -> str:
        """Human-readable value (pretty for matrices, str() for scalars)."""
        if self.mem_type == MemType.MATRIX:
            return sympy.pretty(self.value, use_unicode=True)
        # Apply readable formatting: ** → ^, remove redundant *
        s = str(self.value)
        s = s.replace("**", "^")
        import re as _re
        s = _re.sub(r'(?<=[a-zA-Z\d\)])\*(?=[a-zA-Z\(])', '', s)
        return s

    def short_display(self) -> str:
        """One-line preview for /mem list (max 60 chars)."""
        s = str(self.value)
        return s[:57] + "…" if len(s) > 60 else s

    def sympy_str(self) -> str:
        """
        SymPy-parseable string for inline $-substitution.
        Matrices are excluded (callers should check mem_type before calling).
        """
        return str(self.value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r'^[A-Za-z_]\w*$')
_MEM_REF = re.compile(r'\$([A-Za-z_]\w*)')

# These names shadow SymPy built-ins — we allow storage but warn via the cog.
SYMPY_BUILTIN_NAMES: frozenset[str] = frozenset({
    "E", "I", "pi", "oo", "zoo", "nan",
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "S", "N", "O", "Q",
})


def validate_name(name: str) -> None:
    """Raise ValueError if *name* is not a valid memory variable name."""
    if not name:
        raise ValueError("Variable name cannot be empty.")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(
            f"Name `{name}` is too long ({len(name)} chars, max {MAX_NAME_LEN})."
        )
    if not _NAME_RE.match(name):
        raise ValueError(
            f"`{name}` is not a valid variable name. "
            "Use only letters, digits, and underscores; must start with a letter."
        )


# ---------------------------------------------------------------------------
# Matrix parsing
# ---------------------------------------------------------------------------

def parse_matrix(raw: str) -> sympy.Matrix:
    """
    Parse ``[[a, b], [c, d]]`` (or ``[a, b, c]`` for a row-vector) notation
    into a :class:`sympy.Matrix`.

    Only numeric Python literals are accepted (int, float).  Each element
    is passed through ``sympy.sympify`` so fractions and floats work correctly.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a nested list of numbers.
    """
    try:
        nested = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse matrix: {exc}") from exc

    if not isinstance(nested, list) or not nested:
        raise ValueError("Expected a non-empty list, e.g. [[1,2],[3,4]].")

    # 1-D list → row vector
    if not isinstance(nested[0], list):
        nested = [nested]

    try:
        mat = sympy.Matrix([
            [sympy.sympify(x) for x in row]
            for row in nested
        ])
    except Exception as exc:
        raise ValueError(f"Could not build matrix from values: {exc}") from exc

    return mat


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Thread-safe per-user memory store.

    Usage::

        from data.memory import memory

        memory.set(guild_id, user_id, "k", entry)
        entry = memory.get(guild_id, user_id, "k")
        resolved = memory.resolve(guild_id, user_id, "$k * x^2")
    """

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], dict[str, MemEntry]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ns(self, guild_id: int, user_id: int) -> dict[str, MemEntry]:
        """Return the mutable namespace dict (must be called under lock)."""
        key = (guild_id, user_id)
        if key not in self._data:
            self._data[key] = {}
        return self._data[key]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set(
        self,
        guild_id: int,
        user_id: int,
        name: str,
        entry: MemEntry,
    ) -> None:
        """
        Store *entry* under *name*.

        Raises
        ------
        ValueError
            If the name is invalid or the namespace is full.
        """
        validate_name(name)
        entry.name = name
        with self._lock:
            ns = self._ns(guild_id, user_id)
            if name not in ns and len(ns) >= MAX_ENTRIES:
                raise ValueError(
                    f"Memory is full ({MAX_ENTRIES} entries). "
                    "Delete some with `/mem del`."
                )
            ns[name] = entry

    def get(self, guild_id: int, user_id: int, name: str) -> MemEntry | None:
        """Return the entry for *name*, or ``None`` if not found."""
        with self._lock:
            return self._ns(guild_id, user_id).get(name)

    def delete(self, guild_id: int, user_id: int, name: str) -> bool:
        """Delete *name*. Returns ``True`` if it existed."""
        with self._lock:
            ns = self._ns(guild_id, user_id)
            if name in ns:
                del ns[name]
                return True
            return False

    def list_all(self, guild_id: int, user_id: int) -> dict[str, MemEntry]:
        """Return a snapshot of the full namespace (safe to iterate)."""
        with self._lock:
            return dict(self._ns(guild_id, user_id))

    def clear(self, guild_id: int, user_id: int) -> int:
        """Delete all entries for this user. Returns the count deleted."""
        with self._lock:
            ns = self._ns(guild_id, user_id)
            count = len(ns)
            ns.clear()
            return count

    def count(self, guild_id: int, user_id: int) -> int:
        """Return how many entries this user has stored."""
        with self._lock:
            return len(self._ns(guild_id, user_id))

    # ------------------------------------------------------------------
    # $-reference resolution
    # ------------------------------------------------------------------

    def has_refs(self, raw: str) -> bool:
        """Return ``True`` if *raw* contains any ``$name`` tokens."""
        return "$" in raw and bool(_MEM_REF.search(raw))

    def resolve(self, guild_id: int, user_id: int, raw: str) -> str:
        """
        Replace every ``$name`` token in *raw* with the stored value's
        SymPy-parseable string, wrapped in parentheses for safe inlining.

        If *raw* contains no ``$`` tokens the string is returned unchanged
        with zero overhead.

        Parameters
        ----------
        guild_id, user_id:
            The user's namespace coordinates.
        raw:
            The raw expression string from a Discord slash-command argument.

        Returns
        -------
        str
            The expression with all ``$name`` tokens substituted.

        Raises
        ------
        ValueError
            * A ``$name`` token references a name not in memory.
            * A ``$name`` token references a MATRIX entry (matrices cannot be
              inlined into scalar expressions).
        """
        if "$" not in raw:
            return raw

        # Snapshot the namespace once to avoid repeated lock acquisitions.
        with self._lock:
            ns = dict(self._ns(guild_id, user_id))

        def _sub(m: re.Match) -> str:
            token = m.group(1)
            entry = ns.get(token)
            if entry is None:
                raise ValueError(
                    f"Memory variable `${token}` is not defined. "
                    "Use `/mem list` to see what's stored, or "
                    f"`/mem set {token} <value>` to define it."
                )
            if entry.mem_type == MemType.MATRIX:
                raise ValueError(
                    f"`${token}` is a matrix and cannot be inlined in a scalar "
                    f"expression.  Use `/mem get {token}` to view it."
                )
            return f"({entry.sympy_str()})"

        return _MEM_REF.sub(_sub, raw)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

memory: MemoryStore = MemoryStore()