"""
cogs/discrete.py — Discrete math slash commands for the math bot.

Commands
--------
/permutation     n r                         nPr = n! / (n − r)!
/combination     n r                         nCr = n! / (r!(n − r)!)
/truth_table     expression                  Truth table for a boolean expression in A–D.
/set_ops         set_a set_b operation       Union, intersection, difference, symmetric difference.
/binomial_coeff  n                           The nth row of Pascal's triangle (n ≤ 20).

All commands defer immediately, validate their inputs, and surface errors
through a consistent red error embed.
"""

import itertools
import math
import re

from discord import app_commands
from discord.ext import commands
import discord

from utils.formatter import math_embed, error_embed
from utils.paginator import send_paginated

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Above this, n! has so many digits that it would overflow an embed field —
# see _format_large_int(). 1000! is ~2568 digits but math.perm/comb of
# numbers this size still complete instantly; the *display* is what's capped.
_MAX_N = 1000

# Truth tables larger than this many variables (2^5 = 32 rows) are refused
# outright rather than paginated into oblivion.
_MAX_TRUTH_TABLE_VARS = 4

# Discord embed field value limit (used for truth-table pagination).
_FIELD_LIMIT = 1024


def _format_large_int(value: int, max_chars: int = 900) -> str:
    """
    Stringify *value*, falling back to scientific notation if it would be
    too wide for a Discord embed field.

    Parameters
    ----------
    value:
        The integer to display (typically the result of ``math.perm`` or
        ``math.comb``).
    max_chars:
        Threshold above which the full decimal expansion is replaced with
        scientific notation (default ``900``, comfortably under the 1024
        character embed field limit).

    Returns
    -------
    str
        Either the exact decimal string, or ``"≈ 1.234560e+299  (300 digits)"``
        for very large values.
    """
    s = str(value)
    if len(s) <= max_chars:
        return s
    return f"≈ {value:.6e}   ({len(s)} digits)"


# ---------------------------------------------------------------------------
# Boolean-expression parser (for /truth_table)
# ---------------------------------------------------------------------------
#
# Grammar (lowest to highest precedence), all binary operators left-assoc
# except IMPLIES which is right-assoc:
#
#   expr     := implies
#   implies  := xor_expr ( "implies" implies )?
#   xor_expr := or_expr  ( "xor" or_expr )*
#   or_expr  := and_expr ( "or" and_expr )*
#   and_expr := not_expr ( "and" not_expr )*
#   not_expr := "not" not_expr | atom
#   atom     := VARIABLE | "(" expr ")"

_BOOL_KEYWORDS = {"and", "or", "not", "xor", "implies"}
_TOKEN_RE = re.compile(r"\(|\)|[A-Za-z]+")


def _tokenize_boolean(expr: str) -> list[str]:
    """
    Tokenize a boolean expression into operators, parentheses, and
    single-letter variable names.

    Keywords (``and``, ``or``, ``not``, ``xor``, ``implies``) are
    normalised to lowercase; variables are normalised to uppercase.

    Raises
    ------
    ValueError
        If the expression is empty, or contains a word that is neither a
        recognised keyword nor a single letter.
    """
    tokens = _TOKEN_RE.findall(expr)
    if not tokens:
        raise ValueError("Expression cannot be empty.")

    out: list[str] = []
    for tok in tokens:
        if tok in ("(", ")"):
            out.append(tok)
        elif tok.lower() in _BOOL_KEYWORDS:
            out.append(tok.lower())
        elif len(tok) == 1 and tok.isalpha():
            out.append(tok.upper())
        else:
            raise ValueError(
                f"Unrecognised token `{tok}`. Variables must be single "
                "letters (A, B, C, …) and operators must be one of "
                "`and`, `or`, `not`, `xor`, `implies`."
            )
    return out


class _BoolParser:
    """
    Recursive-descent parser turning a token list into a nested-tuple AST.

    AST node shapes
    ----------------
    * ``"X"``                  — a variable (single uppercase letter)
    * ``("NOT", node)``
    * ``("AND", left, right)``
    * ``("OR", left, right)``
    * ``("XOR", left, right)``
    * ``("IMPLIES", left, right)``
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str | None:
        tok = self._peek()
        self._pos += 1
        return tok

    def _expect(self, tok: str) -> None:
        if self._peek() != tok:
            raise ValueError(
                f"Expected `{tok}` but found "
                f"{'end of expression' if self._peek() is None else f'`{self._peek()}`'}."
            )
        self._advance()

    def parse(self) -> tuple | str:
        """Parse the full token stream and return the AST root."""
        node = self._parse_implies()
        if self._pos != len(self._tokens):
            raise ValueError(f"Unexpected token `{self._peek()}` in expression.")
        return node

    def _parse_implies(self) -> tuple | str:
        left = self._parse_xor()
        if self._peek() == "implies":
            self._advance()
            right = self._parse_implies()  # right-associative
            return ("IMPLIES", left, right)
        return left

    def _parse_xor(self) -> tuple | str:
        node = self._parse_or()
        while self._peek() == "xor":
            self._advance()
            node = ("XOR", node, self._parse_or())
        return node

    def _parse_or(self) -> tuple | str:
        node = self._parse_and()
        while self._peek() == "or":
            self._advance()
            node = ("OR", node, self._parse_and())
        return node

    def _parse_and(self) -> tuple | str:
        node = self._parse_not()
        while self._peek() == "and":
            self._advance()
            node = ("AND", node, self._parse_not())
        return node

    def _parse_not(self) -> tuple | str:
        if self._peek() == "not":
            self._advance()
            return ("NOT", self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> tuple | str:
        tok = self._peek()
        if tok == "(":
            self._advance()
            node = self._parse_implies()
            self._expect(")")
            return node
        if tok is not None and len(tok) == 1 and tok.isalpha():
            self._advance()
            return tok
        raise ValueError(
            f"Unexpected token "
            f"{'end of expression' if tok is None else f'`{tok}`'} in expression."
        )


def _evaluate_boolean(node: tuple | str, values: dict[str, bool]) -> bool:
    """Recursively evaluate an AST produced by :class:`_BoolParser`."""
    if isinstance(node, str):
        return values[node]

    op = node[0]
    if op == "NOT":
        return not _evaluate_boolean(node[1], values)

    left = _evaluate_boolean(node[1], values)
    right = _evaluate_boolean(node[2], values)
    if op == "AND":
        return left and right
    if op == "OR":
        return left or right
    if op == "XOR":
        return left != right
    if op == "IMPLIES":
        return (not left) or right

    raise ValueError(f"Unknown operator `{op}`.")  # pragma: no cover — defensive


def _build_truth_table_lines(variables: list[str], rows: list[tuple[bool, ...]]) -> tuple[str, str, list[str]]:
    """
    Render the header, separator, and data rows of a truth table.

    Parameters
    ----------
    variables:
        Ordered list of variable names (column headers, excluding "Result").
    rows:
        Each tuple is ``(val_for_var_1, ..., val_for_var_n, result)`` of
        booleans, in the order the table should be displayed.

    Returns
    -------
    tuple[str, str, list[str]]
        ``(header_line, separator_line, data_lines)``. Each is a single
        plain-text row using ``T`` / ``F`` for truth values, padded so all
        columns line up in a monospace font.
    """
    headers = variables + ["Result"]
    col_widths = [max(len(h), 1) for h in headers]

    def fmt(cells: list[str]) -> str:
        return " | ".join(cell.center(w) for cell, w in zip(cells, col_widths))

    header_line = fmt(headers)
    separator_line = "-+-".join("-" * w for w in col_widths)
    data_lines = [fmt(["T" if v else "F" for v in row]) for row in rows]

    return header_line, separator_line, data_lines


def _paginate_table(header_line: str, separator_line: str, data_lines: list[str]) -> list[str]:
    """
    Split a truth table's data rows into chunks that each fit, with the
    header repeated, inside a single Discord embed field
    (:data:`_FIELD_LIMIT` characters once wrapped in a code block).

    Returns
    -------
    list[str]
        One or more ready-to-embed table strings (no surrounding code
        fences — :func:`utils.formatter.math_embed` adds those).
    """
    fence_overhead = 6  # ``` ... ``` adds 3 backticks on each side
    header_block = f"{header_line}\n{separator_line}"
    base_len = fence_overhead + len(header_block) + 1  # +1 for the newline before data

    pages: list[str] = []
    current: list[str] = []
    current_len = base_len

    for line in data_lines:
        line_len = len(line) + 1  # +1 for its newline
        if current and current_len + line_len > _FIELD_LIMIT:
            pages.append(header_block + "\n" + "\n".join(current))
            current = []
            current_len = base_len
        current.append(line)
        current_len += line_len

    if current:
        pages.append(header_block + "\n" + "\n".join(current))

    return pages or [header_block]


# ---------------------------------------------------------------------------
# Set helpers (for /set_ops)
# ---------------------------------------------------------------------------

def _parse_set_element(token: str) -> int | float | str:
    """
    Convert one comma-separated element to ``int``, then ``float``, then
    fall back to a stripped string — so ``"1, 2, apple"`` produces the set
    ``{1, 2, "apple"}``.
    """
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _parse_set(raw: str, name: str) -> set:
    """
    Parse a comma-separated string such as ``"1, 2, 3, 4"`` into a Python
    ``set``.

    Raises
    ------
    ValueError
        If *raw* is empty/whitespace, or contains an empty element
        (e.g. a trailing comma: ``"1, 2,"``).
    """
    if not raw.strip():
        raise ValueError(f"`{name}` cannot be empty.")

    pieces = raw.split(",")
    if any(not piece.strip() for piece in pieces):
        raise ValueError(
            f"`{name}` contains an empty element — check for stray or "
            "trailing commas."
        )

    return {_parse_set_element(piece) for piece in pieces}


def _format_set(s: set) -> str:
    """
    Render a set as ``{1, 2, 3}``, sorted for stable output.

    Mixed-type sets (numbers and strings together) can't be sorted with
    ``<``; in that case elements are sorted by their string form instead.
    """
    if not s:
        return "∅  (empty set)"
    try:
        items = sorted(s)
    except TypeError:
        items = sorted(s, key=str)
    return "{" + ", ".join(str(x) for x in items) + "}"


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class DiscreteCog(commands.Cog, name="Discrete Math"):
    """Discrete math commands: counting, logic, sets, and Pascal's triangle."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /permutation
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="permutation",
        description="Compute nPr — the number of ways to arrange r items chosen from n.",
    )
    @app_commands.describe(
        n="Total number of items",
        r="Number of items to arrange (must be ≤ n)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def permutation(
        self,
        interaction: discord.Interaction,
        n: int,
        r: int,
    ) -> None:
        """Compute ``nPr = n! / (n − r)!`` with step-by-step working."""
        await interaction.response.defer()

        try:
            if n < 0 or r < 0:
                raise ValueError("`n` and `r` must both be non-negative.")
            if r > n:
                raise ValueError(f"`r` must be ≤ `n` (got r = {r}, n = {n}).")
            if n > _MAX_N:
                raise ValueError(f"`n` is capped at {_MAX_N} (got {n}).")

            result = math.perm(n, r)
            result_str = _format_large_int(result)

            steps = [
                ("Formula", "nPr = n! / (n − r)!"),
                ("Substitute n and r", f"{n}P{r} = {n}! / ({n} − {r})! = {n}! / {n - r}!"),
                ("Result", result_str),
            ]

            embed = math_embed(
                title=f"Permutation — {n}P{r}",
                result=result_str,
                steps=steps,
                footer=f"Number of ordered arrangements of {r} item(s) chosen from {n}",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /combination
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="combination",
        description="Compute nCr — the number of ways to choose r items from n (order doesn't matter).",
    )
    @app_commands.describe(
        n="Total number of items",
        r="Number of items to choose (must be ≤ n)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def combination(
        self,
        interaction: discord.Interaction,
        n: int,
        r: int,
    ) -> None:
        """Compute ``nCr = n! / (r!(n − r)!)`` with step-by-step working."""
        await interaction.response.defer()

        try:
            if n < 0 or r < 0:
                raise ValueError("`n` and `r` must both be non-negative.")
            if r > n:
                raise ValueError(f"`r` must be ≤ `n` (got r = {r}, n = {n}).")
            if n > _MAX_N:
                raise ValueError(f"`n` is capped at {_MAX_N} (got {n}).")

            result = math.comb(n, r)
            result_str = _format_large_int(result)

            steps = [
                ("Formula", "nCr = n! / ( r! (n − r)! )"),
                (
                    "Substitute n and r",
                    f"{n}C{r} = {n}! / ( {r}! · ({n} − {r})! ) = {n}! / ( {r}! · {n - r}! )",
                ),
                ("Result", result_str),
            ]

            embed = math_embed(
                title=f"Combination — {n}C{r}",
                result=result_str,
                steps=steps,
                footer=f"Number of ways to choose {r} item(s) from {n}, order doesn't matter",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /truth_table
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="truth_table",
        description="Generate a truth table for a boolean expression.",
    )
    @app_commands.describe(
        expression="Boolean expression using A, B, C, D and and/or/not/xor/implies, e.g. 'A and not B'",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def truth_table(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        """
        Build the truth table for *expression*.

        Variables are auto-detected from the single uppercase letters used
        (A, B, C, …); up to :data:`_MAX_TRUTH_TABLE_VARS` are supported.
        Operators ``and``, ``or``, ``not``, ``xor``, and ``implies`` are
        recognised (``A implies B`` ≡ ``not A or B``).
        """
        await interaction.response.defer()

        try:
            tokens = _tokenize_boolean(expression)

            variables = sorted({
                tok for tok in tokens
                if tok not in _BOOL_KEYWORDS and tok not in ("(", ")")
            })

            if not variables:
                raise ValueError(
                    "Expression must contain at least one variable (a single "
                    "letter such as A, B, or C)."
                )
            if len(variables) > _MAX_TRUTH_TABLE_VARS:
                raise ValueError(
                    f"Expression uses {len(variables)} variables "
                    f"({', '.join(variables)}); a maximum of "
                    f"{_MAX_TRUTH_TABLE_VARS} is supported "
                    f"(2^{_MAX_TRUTH_TABLE_VARS} = {2 ** _MAX_TRUTH_TABLE_VARS} rows)."
                )

            ast = _BoolParser(tokens).parse()

            rows: list[tuple[bool, ...]] = []
            for combo in itertools.product([True, False], repeat=len(variables)):
                values = dict(zip(variables, combo))
                result = _evaluate_boolean(ast, values)
                rows.append(combo + (result,))

            header_line, separator_line, data_lines = _build_truth_table_lines(variables, rows)
            pages_text = _paginate_table(header_line, separator_line, data_lines)

            title = "Truth Table"
            footer = (
                f"{expression}   |   variables: {', '.join(variables)}   |   "
                f"{2 ** len(variables)} row(s)"
            )

            pages = [
                math_embed(title=title, result=text, footer=footer)
                for text in pages_text
            ]

            await send_paginated(interaction, pages)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /set_ops
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="set_ops",
        description="Perform a set operation on two comma-separated sets.",
    )
    @app_commands.describe(
        set_a="First set, comma-separated, e.g. 1, 2, 3, 4",
        set_b="Second set, comma-separated, e.g. 3, 4, 5",
        operation="Set operation to perform",
    )
    @app_commands.choices(operation=[
        app_commands.Choice(name="Union (A ∪ B)", value="union"),
        app_commands.Choice(name="Intersection (A ∩ B)", value="intersection"),
        app_commands.Choice(name="Difference (A − B)", value="difference"),
        app_commands.Choice(name="Symmetric Difference (A △ B)", value="symmetric_difference"),
    ])
    @app_commands.checks.cooldown(1, 2.0)
    async def set_ops(
        self,
        interaction: discord.Interaction,
        set_a: str,
        set_b: str,
        operation: app_commands.Choice[str],
    ) -> None:
        """Compute the union, intersection, difference, or symmetric difference of two sets."""
        await interaction.response.defer()

        try:
            a = _parse_set(set_a, "set_a")
            b = _parse_set(set_b, "set_b")

            op = operation.value
            if op == "union":
                result, symbol = a | b, "A ∪ B"
            elif op == "intersection":
                result, symbol = a & b, "A ∩ B"
            elif op == "difference":
                result, symbol = a - b, "A − B"
            elif op == "symmetric_difference":
                result, symbol = a ^ b, "A △ B"
            else:  # pragma: no cover — app_commands.choices guarantees membership
                raise ValueError(f"Unknown operation `{op}`.")

            result_str = _format_set(result)

            steps = [
                ("A", _format_set(a)),
                ("B", _format_set(b)),
                (symbol, result_str),
            ]

            embed = math_embed(
                title=f"Set {operation.name}",
                result=result_str,
                steps=steps,
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /binomial_coeff
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="binomial_coeff",
        description="Show the nth row of Pascal's triangle.",
    )
    @app_commands.describe(n="Row number, 0-indexed (capped at 20)")
    @app_commands.checks.cooldown(1, 2.0)
    async def binomial_coeff(
        self,
        interaction: discord.Interaction,
        n: int,
    ) -> None:
        """Display ``C(n, 0), C(n, 1), …, C(n, n)`` — row *n* of Pascal's triangle."""
        await interaction.response.defer()

        try:
            if n < 0:
                raise ValueError(f"`n` must be non-negative (got {n}).")
            if n > 20:
                raise ValueError(f"`n` is capped at 20 (got {n}).")

            row = [math.comb(n, k) for k in range(n + 1)]
            row_str = " ".join(str(v) for v in row)

            embed = math_embed(
                title=f"Pascal's Triangle — Row {n}",
                result=row_str,
                footer=f"C({n}, k) for k = 0, 1, …, {n}",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the DiscreteCog into *bot*."""
    await bot.add_cog(DiscreteCog(bot))