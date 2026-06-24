"""
utils/parser.py — Math expression parser.

This is the **only** module in the project that calls ``parse_expr`` or
``latex2sympy``.  Every cog must go through :func:`parse_expression` and
work with the returned ``sympy.Expr`` — never call the underlying parsers
directly.

Supported input formats (auto-detected)
----------------------------------------
latex   : ``\\frac{1}{2}``, ``\\int x\\,dx``, ``\\sqrt{x}``, ``\\sin(x)``,
          ``x^{2}``, ``\\alpha``, ``\\pi``, ``\\cdot``, and many more.
          Triggered by a leading backslash, any recognised LaTeX macro,
          or braced exponents like ``x^{2}``.  Parsed via latex2sympy2.
          If latex2sympy2 fails, a second attempt is made using SymPy's
          own ``parse_expr`` pipeline (plain/implicit-multiplication mode)
          before the error is surfaced to the user.
python  : ``x**2 + 2*x``, ``math.sin(x)``.  Carets are still normalised
          so mixed inputs like ``x^2 + math.sin(x)`` work correctly.
natural : ``x squared plus 2 times x``
plain   : ``x^2 + 2x``  -- default; handles caret and implicit multiplication
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import config

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS: list[str] = [
    "__", "import", "exec", "eval", "open", "os", "sys", "subprocess",
]

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# One executor shared for the lifetime of the process.
# max_workers=4 caps concurrent heavy parses without starving the event loop.
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_raw(expr: str) -> None:
    """
    Raise :class:`ValueError` if *expr* is obviously unsafe or too long.

    Checks (in order)
    -----------------
    1. Length against ``config.MAX_EXPR_LENGTH``.
    2. Presence of any string in :data:`FORBIDDEN_KEYWORDS`.
    """
    if len(expr) > config.MAX_EXPR_LENGTH:
        raise ValueError(
            f"Expression is too long ({len(expr)} chars). "
            f"Maximum allowed length is {config.MAX_EXPR_LENGTH} characters."
        )
    for kw in FORBIDDEN_KEYWORDS:
        if kw in expr:
            raise ValueError(
                f"Expression contains a forbidden keyword: `{kw}`"
            )

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Natural-language words that unambiguously signal a prose-style expression.
_NATURAL_PATTERN = re.compile(
    r"\b(squared|cubed|plus|minus|times|divided\s+by|sqrt\s+of|square\s+root\s+of)\b",
    re.IGNORECASE,
)

# LaTeX control sequences that unambiguously signal LaTeX input.
# Covers common functions, operators, Greek letters, and environments.
_LATEX_KEYWORDS: tuple[str, ...] = (
    # Structures
    r"\frac", r"\dfrac", r"\tfrac",
    r"\int", r"\iint", r"\iiint", r"\oint",
    r"\sum", r"\prod", r"\lim",
    r"\sqrt", r"\root",
    r"\left", r"\right",
    r"\begin", r"\end",
    # Trig / functions
    r"\sin", r"\cos", r"\tan",
    r"\csc", r"\sec", r"\cot",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\sinh", r"\cosh", r"\tanh",
    r"\ln", r"\log", r"\exp",
    r"\lim", r"\max", r"\min",
    r"\gcd", r"\det", r"\deg",
    # Operators / relations
    r"\cdot", r"\times", r"\div",
    r"\leq", r"\geq", r"\neq",
    r"\infty", r"\partial", r"\nabla",
    r"\pm", r"\mp",
    # Greek letters (common ones)
    r"\alpha", r"\beta", r"\gamma", r"\delta",
    r"\epsilon", r"\theta", r"\lambda", r"\mu",
    r"\pi", r"\sigma", r"\phi", r"\omega",
    # Misc
    r"\over", r"\choose",
)

# Regex that matches LaTeX-style braced exponents like x^{2} or e^{-x}
_LATEX_BRACE_EXP = re.compile(r"\^\s*\{")


def _detect_format(expr: str) -> str:
    """
    Heuristically classify *expr* as one of ``"latex"``, ``"python"``,
    ``"natural"``, or ``"plain"``.

    Detection priority (most-specific → least-specific)
    ---------------------------------------------------
    1. **LaTeX** — starts with ``\\``, contains any known LaTeX keyword,
       or uses braced exponents like ``x^{2}``.
    2. **Python** — contains ``**`` or ``math.`` prefix.  Note: expressions
       with ``^`` (plain caret) fall through to "plain", not "python", so
       ``x^2`` is always handled by the correct normaliser.
    3. **Natural** — contains prose keywords like "squared", "plus", etc.
    4. **Plain** — default; handles carets and implicit multiplication.

    Parameters
    ----------
    expr:
        Raw user input, stripped but otherwise unmodified.

    Returns
    -------
    str
        One of ``"latex"``, ``"python"``, ``"natural"``, ``"plain"``.
    """
    stripped = expr.strip()

    # LaTeX: leading backslash, any known macro, or braced exponent x^{n}
    if (
        stripped.startswith("\\")
        or any(kw in stripped for kw in _LATEX_KEYWORDS)
        or _LATEX_BRACE_EXP.search(stripped)
    ):
        return "latex"

    # Python-style: double-star exponentiation or math module prefix.
    # Plain carets (x^2) must NOT match here — they belong to "plain".
    if "**" in stripped or "math." in stripped:
        return "python"

    # Natural language: prose keywords
    if _NATURAL_PATTERN.search(stripped):
        return "natural"

    # Default: caret / implicit-multiplication notation (e.g. "x^2 + 2x")
    return "plain"

# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------

def _normalize_plain(expr: str) -> sympy.Expr:
    """
    Parse a *plain* expression such as ``x^2 + 2x``.

    Transformations applied before handing off to SymPy
    ----------------------------------------------------
    * ``^``  →  ``**``   (caret exponentiation)
    * ``2x`` →  ``2*x``  (digit immediately followed by a letter)

    Then ``parse_expr`` is called with
    ``implicit_multiplication_application`` so that ``sin(x)cos(x)``
    and similar forms also work.
    """
    expr = re.sub(r"\^", "**", expr)
    # Insert explicit * between a digit and a letter: 2x → 2*x, 3xy → 3*xy
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)
    return parse_expr(expr, transformations=_TRANSFORMATIONS)


def _normalize_natural(expr: str) -> sympy.Expr:
    """
    Translate natural-language keywords to SymPy-parseable syntax, then
    delegate to :func:`_normalize_plain`.

    Substitution map (case-insensitive, longest match first)
    ---------------------------------------------------------
    ``squared``         → ``**2``
    ``cubed``           → ``**3``
    ``plus``            → ``+``
    ``minus``           → ``-``
    ``times``           → ``*``
    ``divided by``      → ``/``
    ``square root of``  → ``sqrt``
    ``sqrt of``         → ``sqrt``
    """
    substitutions: list[tuple[str, str]] = [
        # Multi-word patterns first (avoid partial matches)
        (r"\bsquare\s+root\s+of\b", "sqrt"),
        (r"\bdivided\s+by\b",        "/"),
        (r"\bsqrt\s+of\b",           "sqrt"),
        # Single-word patterns
        (r"\bsquared\b",             "**2"),
        (r"\bcubed\b",               "**3"),
        (r"\bplus\b",                "+"),
        (r"\bminus\b",               "-"),
        (r"\btimes\b",               "*"),
    ]
    for pattern, replacement in substitutions:
        expr = re.sub(pattern, replacement, expr, flags=re.IGNORECASE)
    return _normalize_plain(expr)

# ---------------------------------------------------------------------------
# Blocking dispatcher (runs inside the thread-pool executor)
# ---------------------------------------------------------------------------

def _parse_blocking(raw: str) -> sympy.Expr:
    """
    Detect the input format and dispatch to the correct normaliser.

    This function runs **synchronously** inside a
    :class:`~concurrent.futures.ThreadPoolExecutor` worker so that heavy
    SymPy work cannot block the asyncio event loop.

    Parameters
    ----------
    raw:
        Validated, user-supplied expression string.

    Returns
    -------
    sympy.Expr
        The parsed symbolic expression.

    Raises
    ------
    Exception
        Any exception from the underlying parser is propagated as-is;
        :func:`parse_expression` translates it into a user-friendly
        :class:`ValueError`.
    """
    fmt = _detect_format(raw)
    _log = logging.getLogger(__name__)

    if fmt == "latex":
        from latex2sympy2 import latex2sympy  # lazy import — only when needed
        try:
            return latex2sympy(raw)
        except Exception as latex_exc:
            # latex2sympy2 can fail on edge-case syntax or after library updates.
            # Log the failure for debugging and attempt a plain-notation fallback
            # before surfacing an error to the user.
            _log.warning(
                "latex2sympy2 failed to parse %r (%s); trying plain-notation fallback.",
                raw,
                latex_exc,
            )
            try:
                return _normalize_plain(raw)
            except Exception as fallback_exc:
                raise ValueError(
                    f"LaTeX parse failed ({latex_exc}). "
                    f"Plain-notation fallback also failed ({fallback_exc}). "
                    "Try rewriting the expression in plain notation (e.g. x^2 + 2*x)."
                ) from fallback_exc

    if fmt == "python":
        # Route through _normalize_plain so that carets and implicit
        # multiplication are handled even in python-style expressions.
        return _normalize_plain(raw)

    if fmt == "natural":
        return _normalize_natural(raw)

    # "plain" — default
    return _normalize_plain(raw)

# ---------------------------------------------------------------------------
# Public async entry-point
# ---------------------------------------------------------------------------

async def parse_expression(raw: str) -> sympy.Expr:
    """
    Parse a user-supplied math expression into a :class:`sympy.Expr`.

    This is the **only** function cogs should call.  It is async-safe and
    guarded by ``config.COMPUTE_TIMEOUT`` so a pathological expression
    cannot hang the bot.

    Parameters
    ----------
    raw:
        The raw string from a Discord slash-command argument.

    Returns
    -------
    sympy.Expr

    Raises
    ------
    ValueError
        With a user-friendly message if the expression is too long,
        contains forbidden keywords, times out, or cannot be parsed.

    Examples
    --------
    >>> import asyncio
    >>> asyncio.run(parse_expression("x^2 + 2x"))        # plain
    x**2 + 2*x
    >>> asyncio.run(parse_expression("x**2 + 2*x"))      # python
    x**2 + 2*x
    >>> asyncio.run(parse_expression("x squared plus 2x")) # natural
    x**2 + 2*x
    >>> asyncio.run(parse_expression(r"\\frac{1}{x}"))   # latex
    1/x
    >>> asyncio.run(parse_expression(r"\\sin(x)"))       # latex (trig macro)
    sin(x)
    >>> asyncio.run(parse_expression(r"x^{2} + 2x"))     # latex (braced exp)
    x**2 + 2*x
    """
    _validate_raw(raw)

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _parse_blocking, raw),
            timeout=config.COMPUTE_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        raise ValueError(
            f"Computation timed out after {config.COMPUTE_TIMEOUT}s. "
            "Try a simpler expression."
        )
    except ValueError:
        raise   # re-raise user-friendly ValueErrors from _validate_raw
    except Exception as exc:
        raise ValueError(f"Couldn't parse expression: {exc}") from exc
