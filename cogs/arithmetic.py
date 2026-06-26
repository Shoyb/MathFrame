"""
cogs/arithmetic.py — Core arithmetic and algebra slash commands.

Commands
--------
/simplify     expression           Simplify a mathematical expression.
/solve        expression [var]     Solve a polynomial equation, with step-by-step working.
/expand       expression           Expand (distribute) an expression.
/factor       expression           Factor an expression, with step-by-step working.
/solve_system equations [vars]     Solve a system of simultaneous equations.
/table        expression ...       Generate a value table for f(x) over a range.
/poly_div     dividend divisor     Polynomial division (quotient & remainder).
/verify       expr_a expr_b        Check if two expressions are mathematically equivalent.

All commands defer immediately, pull from / write to the in-memory TTL cache,
and surface errors through a consistent red error embed.
"""

import asyncio
import re

import sympy
from discord import app_commands
from discord.ext import commands
import discord
import numpy as np

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed
from utils.paginator import send_paginated
from utils.solver    import solve_quadratic_steps, solve_cubic_steps, solve_quartic_steps, factor_steps
from data.cache      import get, set, cache_key

# ---------------------------------------------------------------------------
# Module-level helpers
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
# Cog
# ---------------------------------------------------------------------------

class ArithmeticCog(commands.Cog, name="Arithmetic"):
    """Algebra and arithmetic commands: simplify, solve, expand, factor."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /simplify
    # -----------------------------------------------------------------------

    @app_commands.command(
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
        """Simplify *expression* and display the result."""
        await interaction.response.defer()

        try:
            # Cache check ------------------------------------------------
            key    = cache_key("simplify", expression)
            cached = get(key)
            if cached is not None:
                await interaction.followup.send(embed=cached)
                return

            # Compute ----------------------------------------------------
            expr   = await parse_expression(expression)
            result = sympy.simplify(expr)

            embed = math_embed(
                title="Simplify",
                result=str(result),
                footer="Tip: try /expand or /factor for other forms",
            )

            # Cache and respond ------------------------------------------
            set(key, embed)
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /solve
    # -----------------------------------------------------------------------

    @app_commands.command(
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
        """Solve *expression* = 0 for *variable*, showing quadratic steps."""
        await interaction.response.defer()

        try:
            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)

            # Actual solutions (SymPy handles any degree)
            solutions = sympy.solve(expr, var)

            if not solutions:
                await interaction.followup.send(
                    embed=error_embed("No solutions found for the given expression.")
                )
                return

            # Step-by-step working — dispatch by polynomial degree.
            # Degrees 2, 3, 4 each have a dedicated step-builder.
            # Higher degrees: SymPy still solves but no detailed walk-through.
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
                    footer_parts.append(
                        "Step-by-step working is only shown for degree 2, 3, and 4 polynomials"
                    )
            except sympy.PolynomialError:
                footer_parts.append(
                    "Step-by-step working requires a polynomial expression"
                )

            has_complex = any(not s.is_real for s in solutions)
            sol_str     = ",   ".join(f"{variable} = {s}" for s in solutions)

            if steps is None and not footer_parts:
                footer_parts.append("Step-by-step working is only shown for degree 2, 3, and 4 polynomials")
            if has_complex:
                footer_parts.append("solutions include complex numbers")
            footer = "  |  ".join(footer_parts)

            embed = math_embed(
                title=f"Solve for {variable}",
                result=sol_str,
                steps=steps,
                footer=footer,
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /expand
    # -----------------------------------------------------------------------

    @app_commands.command(
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
        """Fully expand *expression* and display the result."""
        await interaction.response.defer()

        try:
            # Cache check ------------------------------------------------
            key    = cache_key("expand", expression)
            cached = get(key)
            if cached is not None:
                await interaction.followup.send(embed=cached)
                return

            # Compute ----------------------------------------------------
            expr   = await parse_expression(expression)
            result = sympy.expand(expr)

            embed = math_embed(
                title="Expand",
                result=str(result),
            )

            set(key, embed)
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /factor
    # -----------------------------------------------------------------------

    @app_commands.command(
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
        """Factor *expression* and display the result with step-by-step working."""
        await interaction.response.defer()

        try:
            # Cache check ------------------------------------------------
            key    = cache_key("factor", expression)
            cached = get(key)
            if cached is not None:
                await interaction.followup.send(embed=cached)
                return

            # Compute ----------------------------------------------------
            expr    = await parse_expression(expression)
            steps   = factor_steps(expr)
            result  = sympy.factor(expr)

            embed = math_embed(
                title="Factor",
                result=str(result),
                steps=steps,
            )

            set(key, embed)
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /solve_system
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="solve_system",
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
        """
        Solve a system of equations.

        Uses ``sympy.linsolve`` for linear systems (exact fractions, clean
        output) and falls back to ``sympy.solve`` for non-linear systems.
        """
        await interaction.response.defer()

        try:
            # ---- Split equations ----------------------------------------
            # Prefer semicolons as separators; fall back to commas only when
            # no semicolon is present (commas can appear inside expressions).
            if ";" in equations:
                raw_eqs = [e.strip() for e in equations.split(";") if e.strip()]
            else:
                raw_eqs = [e.strip() for e in equations.split(",") if e.strip()]

            if len(raw_eqs) < 2:
                raise ValueError(
                    "Please provide at least **two** equations, "
                    "separated by semicolons (`;`) or commas (`,`)."
                )
            if len(raw_eqs) > 6:
                raise ValueError("At most 6 equations are supported at once.")

            # ---- Parse equations ----------------------------------------
            parsed: list[sympy.Expr] = list(
                await asyncio.gather(*(_parse_equation(eq) for eq in raw_eqs))
            )

            # ---- Determine variables ------------------------------------
            if variables.strip():
                var_syms: list[sympy.Symbol] = [
                    sympy.Symbol(v.strip()) for v in variables.split(",") if v.strip()
                ]
            else:
                all_free: set[sympy.Symbol] = set()
                for expr in parsed:
                    all_free.update(expr.free_symbols)
                var_syms = sorted(all_free, key=lambda s: s.name)

            if not var_syms:
                raise ValueError("No free variables detected in the equations.")

            # ---- Solve --------------------------------------------------
            solution_str: str | None = None
            method_note: str = ""

            # 1. Try linsolve (best for linear systems — exact, clean output)
            try:
                lin_result = sympy.linsolve(parsed, var_syms)
                if lin_result:
                    sol_tuple = next(iter(lin_result))
                    lines = [
                        f"{var}  =  {val}"
                        for var, val in zip(var_syms, sol_tuple)
                    ]
                    solution_str = "\n".join(lines)
                    method_note = "linear system"
            except Exception:
                pass  # not linear or linsolve failed — fall through

            # 2. Fall back to solve() for non-linear / underdetermined systems
            if solution_str is None:
                solutions = sympy.solve(parsed, var_syms, dict=True)
                if not solutions:
                    raise ValueError(
                        "No solutions found. The system may be inconsistent "
                        "or have no closed-form solution."
                    )
                shown = solutions[:5]  # cap display at 5 branches
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

            # ---- Build embed --------------------------------------------
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
            embed = math_embed(
                title="Solve System",
                result=solution_str,
                steps=steps,
                footer=footer,
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this system.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /table
    # -----------------------------------------------------------------------

    @app_commands.command(
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
        """Evaluate *expression* over a range and display the result as a paginated table."""
        await interaction.response.defer()

        try:
            if step == 0:
                raise ValueError("Step cannot be zero.")
            if (end - start) * step < 0:
                raise ValueError("Step direction does not match start/end range.")

            # Cap the number of rows to avoid memory/embed flooding
            MAX_ROWS = 200
            num_rows = int(abs((end - start) / step)) + 1
            if num_rows > MAX_ROWS:
                raise ValueError(
                    f"Table would generate {num_rows} rows. "
                    f"Please adjust your range or step to keep it under {MAX_ROWS} rows."
                )

            expr = await parse_expression(expression)
            var = sympy.Symbol(variable)

            # Ensure the expression doesn't have multiple free variables we aren't iterating over
            free_vars = expr.free_symbols
            if len(free_vars) > 1 or (len(free_vars) == 1 and var not in free_vars):
                raise ValueError(
                    f"Expression contains free variables other than `{variable}`. "
                    "Table mode only supports single-variable functions."
                )

            # Evaluate over the range using lambdify and numpy
            # Adding a tiny amount to 'end' ensures floating point drift doesn't drop the last value
            epsilon = step * 0.0001 if step > 0 else step * -0.0001
            x_vals = np.arange(start, end + epsilon, step)
            
            f = sympy.lambdify(var, expr, "numpy")
            
            try:
                y_vals = f(x_vals)
                # If f(x) is a constant (e.g. `f(x) = 5`), lambdify might return a scalar.
                if np.isscalar(y_vals):
                    y_vals = np.full_like(x_vals, y_vals)
            except Exception as e:
                 raise ValueError(f"Evaluation failed: {e}")

            # Format rows
            rows = []
            for x, y in zip(x_vals, y_vals):
                x_str = f"{x:g}"
                
                if isinstance(y, complex) or np.iscomplexobj(y):
                    # For complex results, format real and imag parts
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

            # Paginate
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
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )


    # -----------------------------------------------------------------------
    # /poly_div
    # -----------------------------------------------------------------------

    @app_commands.command(
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
                ("Dividend", str(expr_n)),
                ("Divisor", str(expr_d)),
                ("Verification", f"({str(expr_d)}) * ({str(quotient)}) + ({str(remainder)})")
            ]
            
            result_str = f"**Quotient:** {quotient}\n**Remainder:** {remainder}"
            
            embed = math_embed(
                title="Polynomial Division",
                result=result_str,
                steps=steps,
                footer=f"Divided with respect to {var}"
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(embed=error_embed(f"Polynomial error: {exc}"))
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /verify
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="verify",
        description="Check if two mathematical expressions are equivalent.",
    )
    @app_commands.describe(
        expr_a="First expression",
        expr_b="Second expression",
    )
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
                        ("Simplified Difference (A - B)", str(diff))
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
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(
                embed=error_embed("SymPy couldn't find a closed form for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the ArithmeticCog into *bot*."""
    await bot.add_cog(ArithmeticCog(bot))