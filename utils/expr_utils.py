"""Shared SymPy expression parsing helpers for plot rendering."""

from __future__ import annotations

import re

import sympy

# Import the same validation guard used by parse_expression() so that all
# expressions entering the plot engine are subject to the same length cap
# and forbidden-keyword filter.
from utils.parser import _validate_raw  # noqa: PLC2701 (private import is intentional)

_ASSIGNMENT_PREFIX_RE = re.compile(
    r"^\s*[A-Za-z_]\w*\s*(?:\([^)]*\))?\s*=(?!=)\s*(.+)$"
)


def _clean_sympy_expr(s: str) -> str:
    """Normalize common user expression syntax before SymPy parsing."""
    if not s:
        return ""
    s = s.strip()
    m = _ASSIGNMENT_PREFIX_RE.match(s)
    if m:
        s = m.group(1).strip()
    s = s.replace("^", "**")
    s = re.sub(r"\be\^", "exp", s)
    s = re.sub(r"\be\*\*", "exp", s)
    if ":" in s and "|" in s:
        return _clean_piecewise_expr(s)
    return s


def _clean_piecewise_expr(s: str) -> str:
    pieces = []
    for part in s.split("|"):
        if ":" not in part:
            raise ValueError(
                "Piecewise entries must look like 'condition: expression'."
            )
        cond, expr = part.split(":", 1)
        pieces.append(f"({_clean_sympy_expr(expr)}, {_clean_sympy_expr(cond)})")
    return f"Piecewise({', '.join(pieces)})"


def _sympy_expr(s: str, *syms: sympy.Symbol) -> sympy.Expr:
    # Validate before calling sympify so that length abuse and forbidden
    # keywords are caught here, not only in the async parse_expression() path.
    _validate_raw(s)
    try:
        local = {str(sym): sym for sym in syms}
        local["Piecewise"] = sympy.Piecewise
        return sympy.sympify(s, locals=local)
    except Exception as exc:
        raise ValueError(f"Cannot parse expression `{s}`: {exc}") from exc


__all__ = ["_clean_sympy_expr", "_sympy_expr"]
