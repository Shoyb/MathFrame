"""
cogs/algebra.py — Algebra group: arithmetic, equations, and inequalities.

Merged from:  arithmetic.py  +  equations.py  +  inequalities.py

Commands  (all under the /alg group)
-------------------------------------
/alg simplify     expression           Simplify a mathematical expression.
/alg solve        expression [var]     Solve a polynomial equation, step-by-step.
/alg expand       expression           Expand (distribute) an expression.
/alg factor       expression           Factor an expression, step-by-step.
/alg solve_sys    equations [vars]     Solve a system of simultaneous equations.
/alg table        expression ...       Generate a value table for f(x) over a range.
/alg poly_div     dividend divisor     Polynomial division (quotient & remainder).
/alg verify       expr_a expr_b        Check if two expressions are equivalent.
/alg compare      expr_f expr_g        Side-by-side comparison of two functions.
/alg solve_sim    equations [vars]     Solve simultaneous equations (clean x=…, y=… output).
/alg ineq         expression [var]     Solve a single inequality.
/alg ineq_sys     expressions [vars]   Solve a system of inequalities.
"""

from __future__ import annotations

import asyncio
import re

import sympy
import discord
from discord import app_commands
from discord.ext import commands
import numpy as np

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed, to_readable_text
from data.memory     import memory
from utils.paginator import send_paginated
from utils.solver    import solve_quadratic_steps, solve_cubic_steps, solve_quartic_steps, factor_steps
from data.cache      import get, set, cache_key

# ---------------------------------------------------------------------------
# Module-level helpers (from arithmetic.py)
# ---------------------------------------------------------------------------

# Regex: a bare '=' that is not part of '==', '<=', '>=', '!='
_EQ_SPLIT_RE = re.compile(r"(?<![<>!=])=(?!=)")


async def _parse_equation(raw: str) -> sympy.Expr:
    """
    Parse a single equation string into a SymPy expression that equals zero.

    Supports two forms:
    * **Explicit equals** (``"x + y = 5"``) — split on ``=``, parse both
      sides, return ``lhs − rhs``.
    * **Implicit equals zero** (``"x + y − 5"``) — parse as-is.
    """
    raw = raw.strip()
    parts = _EQ_SPLIT_RE.split(raw, maxsplit=1)
    if len(parts) == 2:
        lhs_expr, rhs_expr = await asyncio.gather(
            parse_expression(parts[0].strip()),
            parse_expression(parts[1].strip()),
        )
        return lhs_expr - rhs_expr
    return await parse_expression(raw)

# ---------------------------------------------------------------------------
# Helpers (from equations.py) — for /alg solve_sim
# ---------------------------------------------------------------------------

_MAX_EQUATIONS = 6
_MAX_SOLUTIONS = 5

_SYMPY_RESERVED: frozenset[str] = frozenset({"E", "I", "N", "O", "S", "Q"})

_KNOWN_NON_VARS: frozenset[str] = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "coth",
    "exp", "log", "ln", "sqrt", "cbrt", "root",
    "abs", "sign", "floor", "ceiling", "conjugate", "arg",
    "Re", "Im", "Abs", "Max", "Min",
    "factorial", "binomial", "gamma", "beta",
    "Rational", "Integer", "Float", "Piecewise",
    "pi", "oo", "zoo", "nan",
})

_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z_]\w*")


def _extract_variable_candidates(raw_strs: list[str]) -> set[str]:
    candidates: set[str] = set()
    for raw in raw_strs:
        for token in _TOKEN_RE.findall(raw):
            if token not in _KNOWN_NON_VARS:
                candidates.add(token)
    return candidates


def _build_local_dict(candidates: set[str]) -> dict[str, sympy.Symbol]:
    return {
        name: sympy.Symbol(name)
        for name in candidates
        if name in _SYMPY_RESERVED
    }


async def _parse_equation_sim(raw: str, local_dict: dict | None = None) -> sympy.Expr:
    """Parse equation for solve_sim (supports local_dict overrides)."""
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
    if ";" in raw:
        parts = raw.split(";")
    else:
        parts = raw.split(",")
    return [p.strip() for p in parts if p.strip()]


def _format_solution(var_syms: list[sympy.Symbol], sol_tuple) -> str:
    return "\n".join(
        f"{var}  =  {sympy.simplify(val)}"
        for var, val in zip(var_syms, sol_tuple)
    )

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class AlgebraCog(commands.Cog, name="Algebra"):
    """Algebra commands: simplify, solve, expand, factor, inequalities, and more."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    alg = app_commands.Group(name="alg", description="Algebra and equation solving.")

    # ──────────────────── from arithmetic.py ────────────────────────────────

    @alg.command(
        name="simplify",
        description="Simplify a mathematical expression.",
    )
    @app_commands.describe(expression="The expression to simplify, e.g. x**2 + 2*x + 1")
    @app_commands.checks.cooldown(1, 2.0)
    async def simplify(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expression = memory.resolve(interaction.guild_id or 0, interaction.user.id, expression)
            key    = cache_key("simplify", expression)
            cached = get(key)
            if cached is not None:
                await interaction.followup.send(embed=cached)
                return
            expr   = await parse_expression(expression)
            result = sympy.simplify(expr)
            embed = math_embed(
                title="Simplify",
                result=str(result),
                footer="Tip: try /alg expand or /alg factor for other forms",
            )
            set(key, embed)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="solve",
        description="Solve a polynomial equation (= 0) for a variable.",
    )
    @app_commands.describe(
        expression="The expression to set equal to zero, e.g. x**2 - 5*x + 6",
        variable="Variable to solve for (default: x)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def solve(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
    ) -> None:
        await interaction.response.defer()
        try:
            expression = memory.resolve(interaction.guild_id or 0, interaction.user.id, expression)
            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)
            solutions = sympy.solve(expr, var)
            if not solutions:
                await interaction.followup.send(embed=error_embed("No solutions found for the given expression."))
                return
            steps: list | None = None
            footer_parts: list[str] = []
            try:
                poly = sympy.Poly(sympy.expand(expr), var)
                deg  = poly.degree()
                if deg == 2:
                    raw = solve_quadratic_steps(expr, var)
                    steps = None if (raw and raw[0][0] == "Error") else raw
                elif deg == 3:
                    raw = solve_cubic_steps(expr, var)
                    steps = None if (raw and raw[0][0] == "Error") else raw
                elif deg == 4:
                    raw = solve_quartic_steps(expr, var)
                    steps = None if (raw and raw[0][0] == "Error") else raw
                else:
                    footer_parts.append("Step-by-step working is only shown for degree 2, 3, and 4 polynomials")
            except sympy.PolynomialError:
                footer_parts.append("Step-by-step working requires a polynomial expression")
            has_complex = any(not s.is_real for s in solutions)
            sol_str     = ",   ".join(f"{variable} = {s}" for s in solutions)
            if steps is None and not footer_parts:
                footer_parts.append("Step-by-step working is only shown for degree 2, 3, and 4 polynomials")
            if has_complex:
                footer_parts.append("solutions include complex numbers")
            footer = "  |  ".join(footer_parts)
            embed = math_embed(title=f"Solve for {variable}", result=sol_str, steps=steps, footer=footer)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="expand",
        description="Expand (distribute) a mathematical expression.",
    )
    @app_commands.describe(expression="The expression to expand, e.g. (x+1)**3")
    @app_commands.checks.cooldown(1, 2.0)
    async def expand(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expression = memory.resolve(interaction.guild_id or 0, interaction.user.id, expression)
            key    = cache_key("expand", expression)
            cached = get(key)
            if cached is not None:
                await interaction.followup.send(embed=cached)
                return
            expr   = await parse_expression(expression)
            result = sympy.expand(expr)
            embed = math_embed(title="Expand", result=str(result))
            set(key, embed)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="factor",
        description="Factor a mathematical expression.",
    )
    @app_commands.describe(expression="The expression to factor, e.g. x**2 - 5*x + 6")
    @app_commands.checks.cooldown(1, 2.0)
    async def factor(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expression = memory.resolve(interaction.guild_id or 0, interaction.user.id, expression)
            key    = cache_key("factor", expression)
            cached = get(key)
            if cached is not None:
                await interaction.followup.send(embed=cached)
                return
            expr    = await parse_expression(expression)
            steps   = factor_steps(expr)
            result  = sympy.factor(expr)
            embed = math_embed(title="Factor", result=str(result), steps=steps)
            set(key, embed)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="solve_sys",
        description="Solve a system of simultaneous equations for multiple variables.",
    )
    @app_commands.describe(
        equations='Equations separated by semicolons or commas, e.g. "x+y=5; x-y=1"',
        variables="Variables to solve for, comma-separated (leave blank to auto-detect)",
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def solve_system(
        self,
        interaction: discord.Interaction,
        equations: str,
        variables: str = "",
    ) -> None:
        await interaction.response.defer()
        try:
            if ";" in equations:
                raw_eqs = [e.strip() for e in equations.split(";") if e.strip()]
            else:
                raw_eqs = [e.strip() for e in equations.split(",") if e.strip()]
            if len(raw_eqs) < 2:
                raise ValueError("Please provide at least **two** equations, separated by semicolons (`;`) or commas (`,`).")
            if len(raw_eqs) > 6:
                raise ValueError("At most 6 equations are supported at once.")
            parsed: list[sympy.Expr] = list(await asyncio.gather(*(_parse_equation(eq) for eq in raw_eqs)))
            if variables.strip():
                var_syms: list[sympy.Symbol] = [sympy.Symbol(v.strip()) for v in variables.split(",") if v.strip()]
            else:
                all_free: set[sympy.Symbol] = set()
                for expr in parsed:
                    all_free.update(expr.free_symbols)
                var_syms = sorted(all_free, key=lambda s: s.name)
            if not var_syms:
                raise ValueError("No free variables detected in the equations.")
            solution_str: str | None = None
            method_note: str = ""
            try:
                lin_result = sympy.linsolve(parsed, var_syms)
                if lin_result:
                    sol_tuple = next(iter(lin_result))
                    lines = [f"{var}  =  {val}" for var, val in zip(var_syms, sol_tuple)]
                    solution_str = "\n".join(lines)
                    method_note = "linear system"
            except Exception:
                pass
            if solution_str is None:
                solutions = sympy.solve(parsed, var_syms, dict=True)
                if not solutions:
                    raise ValueError("No solutions found. The system may be inconsistent or have no closed-form solution.")
                shown = solutions[:5]
                if len(shown) == 1:
                    lines = [f"{var}  =  {val}" for var, val in shown[0].items()]
                    solution_str = "\n".join(lines)
                else:
                    blocks = []
                    for i, sol in enumerate(shown, 1):
                        block_lines = [f"{var}  =  {val}" for var, val in sol.items()]
                        blocks.append(f"Solution {i}:\n" + "\n".join(block_lines))
                    solution_str = "\n\n".join(blocks)
                method_note = f"non-linear  |  {len(solutions)} solution branch(es)"
                if len(solutions) > 5:
                    method_note += "  |  showing first 5"
            eq_display = "\n".join(f"  {eq}" for eq in raw_eqs)
            steps = [
                ("System",    eq_display),
                ("Variables", ",  ".join(str(v) for v in var_syms)),
                ("Solution",  solution_str),
            ]
            footer = (
                f"{len(raw_eqs)} equation(s)  |  {len(var_syms)} variable(s)"
                + (f"  |  {method_note}" if method_note else "")
            )
            embed = math_embed(title="Solve System", result=solution_str, steps=steps, footer=footer)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this system."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="table",
        description="Generate a value table for f(x) over a range with a configurable step.",
    )
    @app_commands.describe(
        expression="The expression to evaluate, e.g. x**2 - 5*x + 6",
        start="Start value for x (default: -5.0)",
        end="End value for x (default: 5.0)",
        step="Step size (default: 1.0)",
        variable="Variable to evaluate over (default: x)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def table(
        self,
        interaction: discord.Interaction,
        expression: str,
        start: float = -5.0,
        end: float = 5.0,
        step: float = 1.0,
        variable: str = "x",
    ) -> None:
        await interaction.response.defer()
        try:
            if step == 0:
                raise ValueError("Step cannot be zero.")
            if (end - start) * step < 0:
                raise ValueError("Step direction does not match start/end range.")
            MAX_ROWS = 200
            num_rows = int(abs((end - start) / step)) + 1
            if num_rows > MAX_ROWS:
                raise ValueError(f"Table would generate {num_rows} rows. Please adjust your range or step to keep it under {MAX_ROWS} rows.")
            expr = await parse_expression(expression)
            var = sympy.Symbol(variable)
            free_vars = expr.free_symbols
            if len(free_vars) > 1 or (len(free_vars) == 1 and var not in free_vars):
                raise ValueError(f"Expression contains free variables other than `{variable}`. Table mode only supports single-variable functions.")
            epsilon = step * 0.0001 if step > 0 else step * -0.0001
            x_vals = np.arange(start, end + epsilon, step)
            f = sympy.lambdify(var, expr, "numpy")
            try:
                y_vals = f(x_vals)
                if np.isscalar(y_vals):
                    y_vals = np.full_like(x_vals, y_vals)
            except Exception as e:
                raise ValueError(f"Evaluation failed: {e}")
            rows = []
            for x, y in zip(x_vals, y_vals):
                x_str = f"{x:g}"
                if isinstance(y, complex) or np.iscomplexobj(y):
                    if np.isnan(y).any():
                        y_str = "Undefined"
                    else:
                        y_str = f"{y.real:g} + {y.imag:g}i" if y.imag >= 0 else f"{y.real:g} - {abs(y.imag):g}i"
                elif np.isnan(y):
                    y_str = "Undefined"
                elif np.isinf(y):
                    y_str = "∞" if y > 0 else "-∞"
                else:
                    y_str = f"{float(y):g}"
                rows.append(f"`{x_str.ljust(10)} | {y_str}`")
            pages: list[discord.Embed] = []
            ROWS_PER_PAGE = 20
            for i in range(0, len(rows), ROWS_PER_PAGE):
                chunk = rows[i:i + ROWS_PER_PAGE]
                embed = math_embed(
                    title=f"Table Mode: f({variable}) = {expr}",
                    result=f"`{variable.ljust(10)} | f({variable})`\n" + "`" + "-" * 25 + "`\n" + "\n".join(chunk),
                )
                pages.append(embed)
            await send_paginated(interaction, pages)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="poly_div",
        description="Divide two polynomials to find the quotient and remainder.",
    )
    @app_commands.describe(
        dividend="The polynomial to divide, e.g. x**3 - 2*x + 4",
        divisor="The polynomial to divide by, e.g. x - 1",
        variable="Variable to divide with respect to (leave blank to auto-detect)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def poly_div(
        self,
        interaction: discord.Interaction,
        dividend: str,
        divisor: str,
        variable: str = "",
    ) -> None:
        await interaction.response.defer()
        try:
            expr_n = await parse_expression(dividend)
            expr_d = await parse_expression(divisor)
            if variable:
                var = sympy.Symbol(variable)
            else:
                free_vars = expr_n.free_symbols | expr_d.free_symbols
                if not free_vars:
                    var = sympy.Symbol("x")
                else:
                    var = sorted(free_vars, key=lambda s: s.name)[0]
            quotient, remainder = sympy.div(expr_n, expr_d, var)
            steps = [
                ("Dividend",     to_readable_text(expr_n)),
                ("Divisor",      to_readable_text(expr_d)),
                ("Verification", f"({to_readable_text(expr_d)}) * ({to_readable_text(quotient)}) + ({to_readable_text(remainder)})"),
            ]
            result_str = f"Quotient:   {to_readable_text(quotient)}\nRemainder:  {to_readable_text(remainder)}"
            embed = math_embed(title="Polynomial Division", result=result_str, steps=steps, footer=f"Divided with respect to {var}")
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Polynomial error: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="verify",
        description="Check if two mathematical expressions are equivalent.",
    )
    @app_commands.describe(expr_a="First expression", expr_b="Second expression")
    @app_commands.checks.cooldown(1, 4.0)
    async def verify(
        self,
        interaction: discord.Interaction,
        expr_a: str,
        expr_b: str,
    ) -> None:
        await interaction.response.defer()
        try:
            a = await parse_expression(expr_a)
            b = await parse_expression(expr_b)
            diff = sympy.simplify(a - b)
            if diff == 0:
                embed = math_embed(
                    title="✅ Equivalent",
                    result=f"`{expr_a}`  =  `{expr_b}`",
                    steps=[("Expression A", str(a)), ("Expression B", str(b))],
                    footer="Equivalence checked via symbolic simplification.",
                )
                embed.color = discord.Color.green()
            else:
                embed = math_embed(
                    title="❌ Not Equivalent",
                    result=f"`{expr_a}`  ≠  `{expr_b}`",
                    steps=[
                        ("Expression A", str(a)),
                        ("Expression B", str(b)),
                        ("Simplified Difference (A - B)", str(diff)),
                    ],
                    footer="Equivalence checking uses symbolic simplification and may fail on complex transcendentals.",
                )
                embed.color = discord.Color.red()
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="compare",
        description="Side-by-side comparison of two functions.",
    )
    @app_commands.describe(
        expr_f="First function, e.g. x**2 - 1",
        expr_g="Second function, e.g. (x - 1)*(x + 1)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def compare(
        self,
        interaction: discord.Interaction,
        expr_f: str,
        expr_g: str,
    ) -> None:
        await interaction.response.defer()
        try:
            f = await parse_expression(expr_f)
            g = await parse_expression(expr_g)
            loop = asyncio.get_running_loop()
            def _analyze() -> tuple[sympy.Expr, sympy.Expr, sympy.Expr, bool]:
                f_simp = sympy.simplify(f)
                g_simp = sympy.simplify(g)
                diff = sympy.simplify(f - g)
                equiv = (diff == 0)
                return f_simp, g_simp, diff, equiv
            f_simp, g_simp, diff, equiv = await loop.run_in_executor(None, _analyze)
            embed = discord.Embed(
                title="Function Comparison",
                color=discord.Color.green() if equiv else discord.Color.blue()
            )
            embed.add_field(name="Function f", value=f"`{f}`", inline=True)
            embed.add_field(name="Function g", value=f"`{g}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=False)
            embed.add_field(name="Simplified f", value=f"`{f_simp}`", inline=True)
            embed.add_field(name="Simplified g", value=f"`{g_simp}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=False)
            if not equiv:
                embed.add_field(name="Difference (f - g)", value=f"`{diff}`", inline=False)
            embed.add_field(
                name="Equivalence",
                value="✅ Equivalent (f = g)" if equiv else "❌ Not Equivalent (f ≠ g)",
                inline=False,
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # ──────────────────── from equations.py ─────────────────────────────────

    @alg.command(
        name="solve_sim",
        description="Solve simultaneous equations — get clean x = …, y = …, z = … output.",
    )
    @app_commands.describe(
        equations=(
            'Equations separated by semicolons (preferred) or commas. '
            'Supports explicit = e.g. "2x+y=5; x-y=1" or implicit = 0 e.g. "2x+y-5; x-y-1".'
        ),
        variables=(
            'Variables to solve for, comma-separated (leave blank to auto-detect). '
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
        await interaction.response.defer()
        try:
            raw_eqs = _split_equations(equations)
            if len(raw_eqs) < 2:
                raise ValueError(
                    "Please supply at least **2** equations, separated by "
                    "semicolons (`;`) or commas (`,`).\n"
                    "Example:  `2x + y = 5 ; x - y = 1`"
                )
            if len(raw_eqs) > _MAX_EQUATIONS:
                raise ValueError(f"At most **{_MAX_EQUATIONS}** equations are supported at once.")
            candidates = _extract_variable_candidates(raw_eqs)
            local_dict = _build_local_dict(candidates)
            parsed: list[sympy.Expr] = list(
                await asyncio.gather(*(_parse_equation_sim(eq, local_dict=local_dict) for eq in raw_eqs))
            )
            if variables.strip():
                raw_var_names = [v.strip() for v in variables.split(",") if v.strip()]
                invalid_names = [v for v in raw_var_names if not v.isidentifier()]
                if invalid_names:
                    raise ValueError(
                        f"Invalid variable name(s): {', '.join(f'`{v}`' for v in invalid_names)}.\n"
                        "Variable names must be comma-separated identifiers, e.g. `x, y` or `a, b, c`."
                    )
                var_syms: list[sympy.Symbol] = [sympy.Symbol(v) for v in raw_var_names]
            else:
                free: set[sympy.Symbol] = set()
                for expr in parsed:
                    free.update(expr.free_symbols)
                var_syms = sorted(free, key=lambda s: s.name)
            if not var_syms:
                raise ValueError("No free variables detected in the equations.")
            n_eq  = len(raw_eqs)
            n_var = len(var_syms)
            solution_str: str | None = None
            method_note = ""
            try:
                lin_result = sympy.linsolve(parsed, var_syms)
                if lin_result:
                    sol_tuple = next(iter(lin_result))
                    solution_str = _format_solution(var_syms, sol_tuple)
                    method_note = "linear"
            except Exception:
                pass
            if solution_str is None:
                raw_sols = sympy.solve(parsed, var_syms, dict=True)
                if not raw_sols:
                    raise ValueError(
                        "No solutions found.\n"
                        "The system may be inconsistent, underdetermined, or have no closed-form solution."
                    )
                shown = raw_sols[:_MAX_SOLUTIONS]
                method_note = "non-linear"
                if len(shown) == 1:
                    solution_str = "\n".join(f"{var}  =  {sympy.simplify(val)}" for var, val in shown[0].items())
                else:
                    blocks: list[str] = []
                    for i, sol in enumerate(shown, 1):
                        lines = [f"{var}  =  {sympy.simplify(val)}" for var, val in sol.items()]
                        blocks.append(f"Solution {i}:\n" + "\n".join(lines))
                    solution_str = "\n\n".join(blocks)
                if len(raw_sols) > _MAX_SOLUTIONS:
                    method_note += f"  |  {len(raw_sols)} branches (showing first {_MAX_SOLUTIONS})"
                else:
                    method_note += f"  |  {len(raw_sols)} solution branch(es)"
            eq_display = "\n".join(f"  {eq}" for eq in raw_eqs)
            var_display = ",  ".join(str(v) for v in var_syms)
            steps = [
                ("System",    eq_display),
                ("Variables", var_display),
                ("Solution",  solution_str),
            ]
            footer_parts = [f"{n_eq} equation(s)", f"{n_var} variable(s)"]
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
            await interaction.followup.send(embed=error_embed(f"Polynomial error: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't find a closed form for this system."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # ──────────────────── from inequalities.py ──────────────────────────────

    @alg.command(
        name="ineq",
        description="Solve a single inequality, returning the solution set as an interval.",
    )
    @app_commands.describe(
        expression="The inequality to solve, e.g. x**2 - 3*x + 2 < 0",
        variable="Variable to solve for (leave blank to auto-detect)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def solve_ineq(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            if not isinstance(expr, (sympy.StrictGreaterThan, sympy.StrictLessThan, sympy.GreaterThan, sympy.LessThan)):
                raise ValueError("Expression must be an inequality (e.g. use <, >, <=, >=).")
            free_vars = expr.free_symbols
            if not free_vars:
                result = str(expr)
                embed = math_embed(title="Evaluate Inequality", result=result, steps=[("Original", str(expr))])
                await interaction.followup.send(embed=embed)
                return
            if variable:
                var = sympy.Symbol(variable)
            else:
                var = sorted(free_vars, key=lambda s: s.name)[0]
            solution = sympy.solve_univariate_inequality(expr, var, relational=False)
            try:
                result_str = sympy.pretty(solution, use_unicode=False)
            except Exception:
                result_str = str(solution)
            embed = math_embed(
                title=f"Solve Inequality for {var}",
                result=result_str,
                steps=[("Original", str(expr))],
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't solve this inequality."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    @alg.command(
        name="ineq_sys",
        description="Solve a system of inequalities.",
    )
    @app_commands.describe(
        expressions="Inequalities separated by semicolons, e.g. x > 0; x < 5",
        variables="Variables to solve for, comma-separated (leave blank to auto-detect)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def solve_ineq_system(
        self,
        interaction: discord.Interaction,
        expressions: str,
        variables: str = "",
    ) -> None:
        await interaction.response.defer()
        try:
            if ";" in expressions:
                raw_eqs = [e.strip() for e in expressions.split(";") if e.strip()]
            else:
                raw_eqs = [e.strip() for e in expressions.split(",") if e.strip()]
            if len(raw_eqs) < 2:
                raise ValueError("Please provide at least two inequalities, separated by semicolons.")
            parsed = list(await asyncio.gather(*(parse_expression(eq) for eq in raw_eqs)))
            for eq in parsed:
                if not isinstance(eq, (sympy.StrictGreaterThan, sympy.StrictLessThan, sympy.GreaterThan, sympy.LessThan)):
                    raise ValueError(f"Expression `{eq}` is not an inequality.")
            if variables:
                var_syms = [sympy.Symbol(v.strip()) for v in variables.split(",")]
            else:
                free_vars = set()
                for eq in parsed:
                    free_vars.update(eq.free_symbols)
                var_syms = sorted(free_vars, key=lambda s: s.name)
            if not var_syms:
                raise ValueError("No free variables found.")
            solution = sympy.reduce_inequalities(parsed, var_syms)
            try:
                result_str = sympy.pretty(solution, use_unicode=False)
            except Exception:
                result_str = str(solution)
            steps = [
                ("System", "\n".join(str(eq) for eq in parsed)),
                ("Variables", ", ".join(str(v) for v in var_syms)),
            ]
            embed = math_embed(title="Solve System of Inequalities", result=result_str, steps=steps)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't solve this system of inequalities."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the AlgebraCog into *bot*."""
    await bot.add_cog(AlgebraCog(bot))
