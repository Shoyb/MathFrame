"""
cogs/equations.py — Equation solving slash commands.

Commands
--------
/solve_sim  equations [variables]   Solve a system of simultaneous equations.

This cog complements ``cogs/arithmetic.py``'s ``/solve_system`` command.
While ``/solve_system`` accepts general-purpose equation strings separated
by semicolons, ``/solve_sim`` is designed to mirror the 991CW's dedicated
*simultaneous equation* mode: clean ``x = …, y = …, z = …`` output for
2×2 and 3×3 linear systems, with an automatic fallback to SymPy's full
nonlinear solver for harder cases.
"""

from __future__ import annotations

import asyncio
import re

import sympy
import discord
from discord import app_commands
from discord.ext import commands

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches a bare '=' that is not part of '==', '<=', '>=', '!='
_EQ_SPLIT_RE = re.compile(r"(?<![<>!=])=(?!=)")

_MAX_EQUATIONS = 6
_MAX_SOLUTIONS = 5

# ---------------------------------------------------------------------------
# Reserved-name handling (BUG-001 fix)
# ---------------------------------------------------------------------------

# SymPy's default parse_expr namespace contains several single-letter names
# that collide with common variable choices:
#
#   E  → sympy.E  (Euler's number ≈ 2.718) — silently absorbed, wrong result
#   I  → sympy.I  (imaginary unit)          — silently absorbed, wrong result
#   N  → N()      (numerical evaluator fn)  — TypeError crash
#   O  → Order    (Big-O class)             — TypeError crash
#   S  → S        (singleton registry)      — TypeError crash
#   Q  → Q        (assumption keys)         — TypeError crash
#
# We pre-scan the raw equation strings, identify any of these letters used as
# variable candidates, and pass them as local_dict overrides to parse_expr so
# they are treated as plain Symbol objects instead.
_SYMPY_RESERVED: frozenset[str] = frozenset({"E", "I", "N", "O", "S", "Q"})

# Tokens that are function or constant names — never variable candidates.
# Keeps e.g. "pi" in "pi*x + y = 1" from being flagged as a missing variable.
_KNOWN_NON_VARS: frozenset[str] = frozenset({
    # Trig / inverse trig
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "arcsin", "arccos", "arctan",
    # Hyperbolic
    "sinh", "cosh", "tanh", "coth",
    # Exponential / logarithm / roots
    "exp", "log", "ln", "sqrt", "cbrt", "root",
    # Misc functions
    "abs", "sign", "floor", "ceiling", "conjugate", "arg",
    "Re", "Im", "Abs", "Max", "Min",
    "factorial", "binomial", "gamma", "beta",
    # SymPy class names sometimes typed by users
    "Rational", "Integer", "Float", "Piecewise",
    # Constants
    "pi", "oo", "zoo", "nan",
})

_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z_]\w*")


def _extract_variable_candidates(raw_strs: list[str]) -> set[str]:
    """
    Scan *raw_strs* for identifier tokens that are plausible variable names.

    Tokens listed in :data:`_KNOWN_NON_VARS` (function/constant names) are
    skipped.  Everything else — including tokens that collide with SymPy's
    reserved names — is included as a candidate.
    """
    candidates: set[str] = set()
    for raw in raw_strs:
        for token in _TOKEN_RE.findall(raw):
            if token not in _KNOWN_NON_VARS:
                candidates.add(token)
    return candidates


def _build_local_dict(candidates: set[str]) -> dict[str, sympy.Symbol]:
    """
    Build a ``local_dict`` override for :func:`~utils.parser.parse_expression`.

    For every candidate that collides with a name in :data:`_SYMPY_RESERVED`,
    the returned dict maps that name to a plain :class:`sympy.Symbol` so
    ``parse_expr`` treats it as a variable, not a built-in constant or object.
    """
    return {
        name: sympy.Symbol(name)
        for name in candidates
        if name in _SYMPY_RESERVED
    }


async def _parse_equation(raw: str, local_dict: dict | None = None) -> sympy.Expr:
    """
    Parse a single equation string into the SymPy expression that equals zero.

    Supports:
    * **Explicit equals** ``"x + y = 5"`` — returns ``lhs - rhs``.
    * **Implicit zero**   ``"x + y - 5"`` — returned as-is.

    Parameters
    ----------
    local_dict:
        Symbol overrides forwarded to :func:`~utils.parser.parse_expression`.
        Used to prevent SymPy reserved names (``E``, ``I``, ``N``, …) from
        being rewritten as built-in constants when the user intends them as
        variable names.  Built by :func:`_build_local_dict`.
    """
    raw = raw.strip()
    parts = _EQ_SPLIT_RE.split(raw, maxsplit=1)
    if len(parts) == 2:
        lhs, rhs = await asyncio.gather(
            parse_expression(parts[0].strip(), local_dict=local_dict),
            parse_expression(parts[1].strip(), local_dict=local_dict),
        )
        return lhs - rhs
    return await parse_expression(raw, local_dict=local_dict)


def _split_equations(raw: str) -> list[str]:
    """
    Split the user input string into individual equation strings.

    Semicolons are preferred as separators (they can't appear in a math
    expression); commas are used as a fallback only when no semicolon is
    present.
    """
    if ";" in raw:
        parts = raw.split(";")
    else:
        parts = raw.split(",")
    return [p.strip() for p in parts if p.strip()]


def _format_solution(var_syms: list[sympy.Symbol], sol_tuple) -> str:
    """Render a solution tuple (from linsolve) as ``x = val`` lines."""
    return "\n".join(
        f"{var}  =  {sympy.simplify(val)}"
        for var, val in zip(var_syms, sol_tuple)
    )


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class EquationsCog(commands.Cog, name="Equations"):
    """Equation-solving commands: simultaneous linear and non-linear systems."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /solve_sim
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="solve_sim",
        description="Solve simultaneous equations — get clean x = …, y = …, z = … output.",
    )
    @app_commands.describe(
        equations=(
            'Equations separated by semicolons (preferred) or commas. '
            'Supports explicit = e.g. "2x+y=5; x-y=1" or implicit = 0 e.g. "2x+y-5; x-y-1".'
        ),
        variables=(
            "Variables to solve for, comma-separated (leave blank to auto-detect). "
            'E.g. "x,y" or "a,b,c".'
        ),
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def solve_sim(
        self,
        interaction: discord.Interaction,
        equations: str,
        variables: str = "",
    ) -> None:
        """
        Solve a system of simultaneous equations.

        Mirrors the 991CW simultaneous equation mode.  The solver:

        1. Parses each equation (explicit ``lhs = rhs`` or implicit ``= 0``).
        2. Auto-detects free variables when none are supplied.
        3. Tries ``sympy.linsolve`` first for exact, clean output on linear
           systems (2×2, 3×3, or larger).
        4. Falls back to ``sympy.solve`` for non-linear or mixed systems.
        """
        await interaction.response.defer()

        try:
            # ---- Split & validate ----------------------------------------
            raw_eqs = _split_equations(equations)

            if len(raw_eqs) < 2:
                raise ValueError(
                    "Please supply at least **2** equations, separated by "
                    "semicolons (`;`) or commas (`,`).\n"
                    "Example:  `2x + y = 5 ; x - y = 1`"
                )
            if len(raw_eqs) > _MAX_EQUATIONS:
                raise ValueError(
                    f"At most **{_MAX_EQUATIONS}** equations are supported at once."
                )

            # ---- Parse ---------------------------------------------------
            # Pre-scan raw strings to detect any variable names that clash
            # with SymPy's reserved namespace (E, I, N, O, S, Q).  Build a
            # local_dict override so parse_expr treats them as Symbols.
            candidates = _extract_variable_candidates(raw_eqs)
            local_dict = _build_local_dict(candidates)

            parsed: list[sympy.Expr] = list(
                await asyncio.gather(
                    *(_parse_equation(eq, local_dict=local_dict) for eq in raw_eqs)
                )
            )

            # ---- Determine variables -------------------------------------
            if variables.strip():
                raw_var_names = [v.strip() for v in variables.split(",") if v.strip()]
                # Validate: each token must be a legal Python identifier so that
                # space-separated input like "x y" is caught immediately instead
                # of creating a Symbol named "x y" that matches nothing.
                invalid_names = [v for v in raw_var_names if not v.isidentifier()]
                if invalid_names:
                    raise ValueError(
                        f"Invalid variable name(s): "
                        f"{', '.join(f'`{v}`' for v in invalid_names)}.\n"
                        "Variable names must be comma-separated identifiers, "
                        "e.g. `x, y` or `a, b, c`."
                    )
                var_syms: list[sympy.Symbol] = [
                    sympy.Symbol(v) for v in raw_var_names
                ]
            else:
                free: set[sympy.Symbol] = set()
                for expr in parsed:
                    free.update(expr.free_symbols)
                var_syms = sorted(free, key=lambda s: s.name)

            if not var_syms:
                raise ValueError("No free variables detected in the equations.")

            n_eq  = len(raw_eqs)
            n_var = len(var_syms)

            # ---- Solve ---------------------------------------------------
            solution_str: str | None = None
            method_note = ""

            # --- 1. linsolve (linear systems — exact, clean) --------------
            try:
                lin_result = sympy.linsolve(parsed, var_syms)
                if lin_result:
                    sol_tuple = next(iter(lin_result))
                    solution_str = _format_solution(var_syms, sol_tuple)
                    method_note = "linear"
            except Exception:
                pass  # not linear → fall through

            # --- 2. sympy.solve (non-linear / underdetermined) ------------
            if solution_str is None:
                raw_sols = sympy.solve(parsed, var_syms, dict=True)
                if not raw_sols:
                    raise ValueError(
                        "No solutions found.\n"
                        "The system may be inconsistent, underdetermined, or have "
                        "no closed-form solution."
                    )

                shown = raw_sols[:_MAX_SOLUTIONS]
                method_note = "non-linear"

                if len(shown) == 1:
                    solution_str = "\n".join(
                        f"{var}  =  {sympy.simplify(val)}"
                        for var, val in shown[0].items()
                    )
                else:
                    blocks: list[str] = []
                    for i, sol in enumerate(shown, 1):
                        lines = [
                            f"{var}  =  {sympy.simplify(val)}"
                            for var, val in sol.items()
                        ]
                        blocks.append(f"Solution {i}:\n" + "\n".join(lines))
                    solution_str = "\n\n".join(blocks)

                if len(raw_sols) > _MAX_SOLUTIONS:
                    method_note += f"  |  {len(raw_sols)} branches (showing first {_MAX_SOLUTIONS})"
                else:
                    method_note += f"  |  {len(raw_sols)} solution branch(es)"

            # ---- Build embed ---------------------------------------------
            eq_display = "\n".join(f"  {eq}" for eq in raw_eqs)
            var_display = ",  ".join(str(v) for v in var_syms)

            steps = [
                ("System",    eq_display),
                ("Variables", var_display),
                ("Solution",  solution_str),
            ]
            footer_parts = [
                f"{n_eq} equation(s)",
                f"{n_var} variable(s)",
            ]
            if method_note:
                footer_parts.append(method_note)

            embed = math_embed(
                title="Simultaneous Equations",
                result=solution_str,
                steps=steps,
                footer="  |  ".join(footer_parts),
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Polynomial error: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this system.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the EquationsCog into *bot*."""
    await bot.add_cog(EquationsCog(bot))
