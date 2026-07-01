"""
cogs/calculus.py — Calculus slash commands for the math bot.

Commands
--------
/diff       expression [variable] [order]               Differentiate an expression.
/integrate  expression [variable] [lower] [upper]       Definite or indefinite integral.
/limit          expression [variable] [point] [direction]   Evaluate a limit.
/series         expression [variable] [point] [terms]       Taylor / Maclaurin series.
/sum_series     expression variable [lower] [upper]         Evaluate a summation (Σ).
/product_series expression variable [lower] [upper]         Evaluate a product (Π).
/ode            expression [initial_conditions]             Solve a differential equation.
/ode_numeric    expression initial_conditions x_start x_end  Numerically integrate an ODE.
/gradient       expression [variables]                      Compute the gradient ∇f.
/divergence     expression [variables]                      Compute the divergence ∇·F.
/curl           expression [variables]                      Compute the curl ∇×F.

All commands defer immediately and surface errors through a consistent
red error embed.  Computation-heavy calls run through the async parser
so the event loop is never blocked.

Note: plotting is handled entirely by cogs/plot_engine.py (/plot, /quickplot,
/multiplot).  Do not add a /plot command here.
"""

import asyncio
import io

import numpy as np
import sympy
from discord import app_commands
from discord.ext import commands
import discord
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")  # headless — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from utils.parser    import parse_expression, _validate_raw  # noqa: PLC2701
from utils.formatter import math_embed, error_embed, to_readable_text
from data.memory     import memory
from utils.solver    import differentiate_steps, integrate_steps
from utils.renderer  import result_to_image
from utils.ode_utils import (
    parse_ode,
    parse_symbolic_ics,
    ode_order,
    extract_numeric_ics,
    build_numeric_rhs,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ORDINALS = {
    1: "1st", 2: "2nd", 3: "3rd",
}


def _ordinal(n: int) -> str:
    """Return "1st", "2nd", "3rd", "4th", … for a positive integer *n*."""
    return _ORDINALS.get(n, f"{n}th")


def _parse_point(point_str: str) -> sympy.Basic:
    """
    Convert a user-supplied limit/series point string to a SymPy value.

    Recognises ``"oo"`` and ``"+oo"`` as ``sympy.oo`` and ``"-oo"`` as
    ``-sympy.oo``; everything else is parsed as a SymPy expression so
    that fractions, ``pi``, ``E``, etc. all work naturally.

    Parameters
    ----------
    point_str:
        Raw string from a Discord slash-command argument.

    Returns
    -------
    sympy.Basic

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    stripped = point_str.strip()
    if stripped in ("oo", "+oo", "inf", "+inf"):
        return sympy.oo
    if stripped in ("-oo", "-inf"):
        return -sympy.oo
    # Apply the same forbidden-keyword and length guard used by parse_expression()
    # before handing off to sympify, which has no such protections.
    _validate_raw(stripped)
    try:
        return sympy.sympify(stripped)
    except sympy.SympifyError as exc:
        raise ValueError(f"Cannot parse point `{point_str}`: {exc}") from exc


def _ode_numeric_plot_bytes(
    t: np.ndarray,
    y: np.ndarray,
    func_name: str,
    order: int,
) -> io.BytesIO:
    """
    Render a numeric ODE trajectory to a PNG and return it as a BytesIO.

    Plots the solution itself; for order ≥ 2 also overlays each derivative
    up to (order - 1) on the same axes so e.g. a 2nd-order solve shows both
    y(x) and y'(x).

    Parameters
    ----------
    t:
        1-D array of independent-variable sample points (``solve_ivp``'s
        ``.t``).
    y:
        2-D array of shape ``(order, len(t))`` — ``solve_ivp``'s ``.y``,
        one row per state component ``[y, y', y'', ...]``.
    func_name:
        Name of the dependent function, e.g. ``"y"``, for axis labels.
    order:
        Number of state components in *y* (the ODE's order).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    primes = ""
    for k in range(order):
        label = f"{func_name}{primes}(x)" if k > 0 else f"{func_name}(x)"
        ax.plot(t, y[k], linewidth=2, label=label)
        primes += "\u2032"

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel(func_name, fontsize=10)
    ax.set_title(f"Numeric solution of the ODE for {func_name}(x)", fontsize=12, pad=6)
    if order > 1:
        ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class CalculusCog(commands.Cog, name="Calculus"):
    """Calculus commands: differentiation, integration, limits, and series."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    calc = app_commands.Group(name="calc", description="Calculus operations.")


    # -----------------------------------------------------------------------
    # /diff
    # -----------------------------------------------------------------------

    @calc.command(
        name="diff",
        description="Differentiate an expression with respect to a variable.",
    )
    @app_commands.describe(
        expression="Expression to differentiate, e.g. sin(x)*x**2",
        variable="Variable of differentiation (default: x)",
        order="How many times to differentiate (default: 1)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def diff(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
        order: int = 1,
    ) -> None:
        """Compute the *order*-th derivative of *expression* w.r.t. *variable*."""
        await interaction.response.defer()

        try:
            if order < 1:
                raise ValueError("Differentiation order must be at least 1.")

            expression = memory.resolve(interaction.guild_id or 0, interaction.user.id, expression)
            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)

            steps  = differentiate_steps(expr, var, order)
            result = sympy.diff(expr, var, order)
            result = sympy.simplify(result)

            footer = f"{_ordinal(order)} derivative with respect to {variable}"

            embed = math_embed(
                title=f"d{'ⁿ' if order > 1 else ''}/d{variable}{'ⁿ' if order > 1 else ''}  [{expression}]",
                result=to_readable_text(result),
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
    # /integrate
    # -----------------------------------------------------------------------

    @calc.command(
        name="integrate",
        description="Compute the definite or indefinite integral of an expression.",
    )
    @app_commands.describe(
        expression="Integrand, e.g. x**2 + sin(x)",
        variable="Variable of integration (default: x)",
        lower="Lower bound for a definite integral (leave blank for indefinite)",
        upper="Upper bound for a definite integral (leave blank for indefinite)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def integrate(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
        lower: str = "",
        upper: str = "",
    ) -> None:
        """Integrate *expression* — definite if both bounds are given, otherwise indefinite."""
        await interaction.response.defer()

        try:
            expression = memory.resolve(interaction.guild_id or 0, interaction.user.id, expression)
            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)

            definite = lower.strip() and upper.strip()

            if definite:
                a = _parse_point(lower)
                b = _parse_point(upper)
                result = sympy.integrate(expr, (var, a, b))
                result = sympy.simplify(result)

                title  = f"∫ from {lower} to {upper}  [{expression}]  d{variable}"
                footer = f"Definite integral on [{lower}, {upper}]"
                steps  = None   # bounds make steps less meaningful; skip them
            else:
                steps  = integrate_steps(expr, var)
                result = sympy.integrate(expr, var)

                title  = f"∫ [{expression}]  d{variable}"
                footer = "+ C  (constant of integration omitted from result)"

            # Warn if SymPy could not evaluate the integral
            if result.has(sympy.Integral):
                footer = "No elementary antiderivative found — result shown in integral form."

            embed = math_embed(
                title=title,
                result=to_readable_text(result),
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
                embed=error_embed("SymPy couldn't find a closed form for this integral.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /limit
    # -----------------------------------------------------------------------

    @calc.command(
        name="limit",
        description="Evaluate the limit of an expression as a variable approaches a point.",
    )
    @app_commands.describe(
        expression="Expression to evaluate, e.g. sin(x)/x",
        variable="Variable that approaches the point (default: x)",
        point="Value the variable approaches; use 'oo' or '-oo' for infinity (default: 0)",
        direction="Approach direction: + (right), - (left), or +- (two-sided, default: +)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def limit(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
        point: str = "0",
        direction: str = "+",
    ) -> None:
        """Compute lim_{variable → point} expression."""
        await interaction.response.defer()

        try:
            if direction not in ("+", "-", "+-"):
                raise ValueError(
                    f"Direction must be `+`, `-`, or `+-` (got `{direction}`)."
                )

            expr  = await parse_expression(expression)
            var   = sympy.Symbol(variable)
            pt    = _parse_point(point)

            result = sympy.limit(expr, var, pt, direction)

            # Human-readable notation for the title
            dir_symbol = {"+": "⁺", "-": "⁻", "+-": ""}.get(direction, "")
            title = f"lim  {variable} → {point}{dir_symbol}  [{expression}]"

            embed = math_embed(
                title=title,
                result=to_readable_text(result),
                footer=f"Limit as {variable} → {point} from the "
                       + {"+": "right", "-": "left", "+-": "both sides"}[direction],
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
                embed=error_embed("SymPy couldn't compute this limit.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )

    # -----------------------------------------------------------------------
    # /series
    # -----------------------------------------------------------------------

    @calc.command(
        name="series",
        description="Expand an expression as a Taylor / Maclaurin series.",
    )
    @app_commands.describe(
        expression="Expression to expand, e.g. sin(x) or exp(x)",
        variable="Expansion variable (default: x)",
        point="Point around which to expand; use 'oo' for Laurent series (default: 0)",
        terms="Number of terms to compute (default: 6)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def series(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
        point: str = "0",
        terms: int = 6,
    ) -> None:
        """Compute the Taylor series of *expression* around *point*."""
        await interaction.response.defer()

        try:
            if terms < 1:
                raise ValueError("Number of terms must be at least 1.")
            if terms > 20:
                raise ValueError("Number of terms is capped at 20 to avoid overly long output.")

            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)
            pt   = _parse_point(point)

            # sympy.series returns an Add with a trailing O(x**n) term
            raw_series  = sympy.series(expr, var, pt, terms)
            result_no_O = raw_series.removeO()

            embed = math_embed(
                title=f"Series  [{expression}]  around {variable} = {point}",
                result=to_readable_text(raw_series),   # keep O() for mathematical honesty
                steps=[
                    ("Full series (with remainder)", to_readable_text(raw_series)),
                    ("Without O() term",             to_readable_text(result_no_O)),
                ],
                footer=f"Taylor series of {expression} around {variable} = {point}  |  {terms} terms",
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
                embed=error_embed("SymPy couldn't find a closed-form series expansion for this.")
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"An unexpected error occurred: {exc}")
            )


    # -----------------------------------------------------------------------
    # /sum_series
    # -----------------------------------------------------------------------

    @calc.command(
        name="sum",
        description="Evaluate a summation (Σ) over a range.",
    )
    @app_commands.describe(
        expression="The expression to sum, e.g. 1/n**2",
        variable="Index variable, e.g. n",
        lower="Lower bound (default: 1)",
        upper="Upper bound, e.g. 10 or oo for infinity (default: oo)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def sum_series(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str,
        lower: str = "1",
        upper: str = "oo",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var = sympy.Symbol(variable)
            
            # Use 'oo' for infinity
            lower_expr = sympy.oo if lower.lower() == "oo" else await parse_expression(lower)
            upper_expr = sympy.oo if upper.lower() == "oo" else await parse_expression(upper)
            
            sum_result = sympy.summation(expr, (var, lower_expr, upper_expr))
            
            try:
                exact_str = str(sympy.simplify(sum_result))
            except Exception:
                exact_str = str(sum_result)
                
            steps = [
                ("Summation", f"Σ ({expression}) from {variable}={lower} to {upper}")
            ]
            
            try:
                if sum_result.is_number and not sum_result.has(sympy.oo) and not sum_result.has(sympy.nan):
                    num_val = sympy.N(sum_result, 6)
                    num_str = str(num_val)
                    if num_str != exact_str and "e" not in exact_str and exact_str != str(sum_result):
                        exact_str = f"{exact_str}  ≈  {num_str}"
            except Exception:
                pass
                
            embed = math_embed(
                title="Summation Σ",
                result=exact_str,
                steps=steps
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
    # /product_series
    # -----------------------------------------------------------------------

    @calc.command(
        name="product",
        description="Evaluate a product (Π) over a range.",
    )
    @app_commands.describe(
        expression="The expression to multiply, e.g. n/(n+1)",
        variable="Index variable, e.g. n",
        lower="Lower bound (default: 1)",
        upper="Upper bound, e.g. 10 or oo for infinity",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def product_series(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str,
        lower: str = "1",
        upper: str = "10",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var = sympy.Symbol(variable)
            
            lower_expr = sympy.oo if lower.lower() == "oo" else await parse_expression(lower)
            upper_expr = sympy.oo if upper.lower() == "oo" else await parse_expression(upper)
            
            prod_result = sympy.product(expr, (var, lower_expr, upper_expr))
            
            try:
                exact_str = str(sympy.simplify(prod_result))
            except Exception:
                exact_str = str(prod_result)
                
            steps = [
                ("Product", f"Π ({expression}) from {variable}={lower} to {upper}")
            ]
            
            try:
                if prod_result.is_number and not prod_result.has(sympy.oo) and not prod_result.has(sympy.nan):
                    num_val = sympy.N(prod_result, 6)
                    num_str = str(num_val)
                    if num_str != exact_str and "e" not in exact_str and exact_str != str(prod_result):
                        exact_str = f"{exact_str}  ≈  {num_str}"
            except Exception:
                pass
                
            embed = math_embed(
                title="Product Π",
                result=exact_str,
                steps=steps
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
    # /ode
    # -----------------------------------------------------------------------

    @calc.command(
        name="ode",
        description="Solve a differential equation.",
    )
    @app_commands.describe(
        expression="The ODE, e.g. y'' + y = 0 or f'(x) = x",
        initial_conditions="Optional initial conditions, e.g. y(0)=1, y'(0)=0",
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def ode(
        self,
        interaction: discord.Interaction,
        expression: str,
        initial_conditions: str = "",
    ) -> None:
        await interaction.response.defer()
        try:
            eq, func_to_solve = await parse_ode(expression)
            ics = await parse_symbolic_ics(initial_conditions, func_to_solve)

            # Solve ODE
            loop = asyncio.get_running_loop()
            def _do_dsolve():
                return sympy.dsolve(eq, func_to_solve, ics=ics if ics else None)

            solution = await loop.run_in_executor(None, _do_dsolve)

            embed = math_embed(
                title="Differential Equation Solution",
                result="Solution attached as image",
                footer=f"Parsed ODE: {eq}"
            )
            embed.set_image(url="attachment://formula.png")

            file = await result_to_image(solution)

            await interaction.followup.send(embed=embed, file=file)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except sympy.PolynomialError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Expression couldn't be treated as a polynomial: {exc}")
            )
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed(
                "SymPy could not find a closed-form solution to this ODE. "
                "Try `/calc ode_numeric` for a numeric trajectory instead "
                "(requires initial conditions and an integration range)."
            ))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /ode_numeric
    # -----------------------------------------------------------------------

    @calc.command(
        name="ode_numeric",
        description="Numerically integrate a differential equation that has no closed-form solution.",
    )
    @app_commands.describe(
        expression="The ODE, e.g. y' = sin(x*y) or y'' + y = 0",
        initial_conditions="Required, given at x_start, e.g. y(0)=1, y'(0)=0",
        x_start="Start of the integration interval — must match the initial condition's x point",
        x_end="End of the integration interval",
        steps="Number of output points to plot (10-2000, default 200)",
        method="Integration method (default RK45; try Radau/BDF for stiff systems)",
    )
    @app_commands.choices(method=[
        app_commands.Choice(name="RK45 (default, non-stiff)", value="RK45"),
        app_commands.Choice(name="RK23 (lower order, non-stiff)", value="RK23"),
        app_commands.Choice(name="Radau (implicit, stiff)", value="Radau"),
        app_commands.Choice(name="BDF (implicit, stiff)", value="BDF"),
        app_commands.Choice(name="LSODA (auto-switching)", value="LSODA"),
    ])
    @app_commands.checks.cooldown(1, 5.0)
    async def ode_numeric(
        self,
        interaction: discord.Interaction,
        expression: str,
        initial_conditions: str,
        x_start: float,
        x_end: float,
        steps: app_commands.Range[int, 10, 2000] = 200,
        method: str = "RK45",
    ) -> None:
        await interaction.response.defer()
        try:
            if x_end <= x_start:
                raise ValueError("`x_end` must be greater than `x_start`.")

            eq, func_to_solve = await parse_ode(expression)
            order = ode_order(eq)
            ics = await parse_symbolic_ics(initial_conditions, func_to_solve)
            x0, y0 = extract_numeric_ics(ics, order)

            if not (x_start <= x0 <= x_end):
                raise ValueError(
                    f"Initial condition is given at x={x0:g}, which lies outside "
                    f"the integration range [{x_start:g}, {x_end:g}]."
                )
            if abs(x0 - x_start) > 1e-9:
                raise ValueError(
                    f"The initial condition is given at x={x0:g}, but `x_start` is "
                    f"{x_start:g}. Numeric integration requires the initial condition "
                    f"to be at x_start — set `x_start={x0:g}`, or split the problem "
                    "into two calls (one integrating backward to your earlier bound, "
                    "one forward to your later bound)."
                )

            indep_var = func_to_solve.args[0] if func_to_solve.args else sympy.Symbol("x")

            loop = asyncio.get_running_loop()

            def _do_solve():
                rhs = build_numeric_rhs(eq, func_to_solve, indep_var, order)
                t_eval = np.linspace(x_start, x_end, steps)
                return solve_ivp(rhs, [x_start, x_end], y0, method=method, t_eval=t_eval)

            result = await loop.run_in_executor(None, _do_solve)

            if not result.success:
                raise ValueError(
                    f"Integration failed: {result.message}  "
                    "Try a different method (Radau/BDF often help with stiff systems) "
                    "or a smaller interval."
                )

            func_name = str(func_to_solve.func)

            def _do_plot():
                return _ode_numeric_plot_bytes(result.t, result.y, func_name, order)

            buf = await loop.run_in_executor(None, _do_plot)

            final_state = ", ".join(
                f"{func_name}{chr(0x2032) * k}({x_end:g}) ≈ {result.y[k][-1]:.6g}"
                for k in range(order)
            )

            embed = math_embed(
                title="Numeric ODE Solution",
                result=f"Integration complete over [{x_start:g}, {x_end:g}]\n\n{final_state}",
                footer=f"Parsed ODE: {eq}  |  method: {method}  |  {steps} points",
            )
            embed.set_image(url="attachment://ode_numeric.png")
            file = discord.File(buf, filename="ode_numeric.png")

            await interaction.followup.send(embed=embed, file=file)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except sympy.SympifyError as exc:
            await interaction.followup.send(embed=error_embed(f"Could not parse expression: {exc}"))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /gradient
    # -----------------------------------------------------------------------

    @calc.command(
        name="gradient",
        description="Compute the gradient of a scalar field ∇f.",
    )
    @app_commands.describe(
        expression="Scalar function, e.g. x**2 + y**2",
        variables="Comma-separated variables, e.g. x,y,z (default: x, y, z)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def gradient(
        self,
        interaction: discord.Interaction,
        expression: str,
        variables: str = "x, y, z",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            vars_list = [sympy.Symbol(v.strip()) for v in variables.split(',')]
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: [sympy.diff(expr, v) for v in vars_list]
            )
            
            res_str = "[" + ", ".join(str(r) for r in result) + "]"
            
            embed = math_embed(
                title="Gradient ∇f",
                result=res_str,
                steps=[
                    ("Scalar Field f", str(expr)),
                    ("Variables", ", ".join(v.name for v in vars_list)),
                ]
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /divergence
    # -----------------------------------------------------------------------

    @calc.command(
        name="divergence",
        description="Compute the divergence of a vector field ∇·F.",
    )
    @app_commands.describe(
        expression="Vector field as a list, e.g. [x*y, y*z, z*x]",
        variables="Comma-separated variables, e.g. x,y,z (default: x, y, z)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def divergence(
        self,
        interaction: discord.Interaction,
        expression: str,
        variables: str = "x, y, z",
    ) -> None:
        await interaction.response.defer()
        try:
            expr_list = await parse_expression(expression)
            if not isinstance(expr_list, list):
                raise ValueError("Expression must be a vector (e.g. enclosed in square brackets `[ ... ]`).")
                
            vars_list = [sympy.Symbol(v.strip()) for v in variables.split(',')]
            
            if len(expr_list) != len(vars_list):
                raise ValueError(f"Number of vector components ({len(expr_list)}) must match number of variables ({len(vars_list)}).")
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: sum(sympy.diff(comp, var) for comp, var in zip(expr_list, vars_list))
            )
            
            embed = math_embed(
                title="Divergence ∇·F",
                result=str(result),
                steps=[
                    ("Vector Field F", "[" + ", ".join(str(c) for c in expr_list) + "]"),
                    ("Variables", ", ".join(v.name for v in vars_list)),
                ]
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /curl
    # -----------------------------------------------------------------------

    @calc.command(
        name="curl",
        description="Compute the curl of a 3D vector field ∇×F.",
    )
    @app_commands.describe(
        expression="3D Vector field as a list, e.g. [x*y, y*z, z*x]",
        variables="Comma-separated variables, e.g. x,y,z (default: x, y, z)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def curl(
        self,
        interaction: discord.Interaction,
        expression: str,
        variables: str = "x, y, z",
    ) -> None:
        await interaction.response.defer()
        try:
            expr_list = await parse_expression(expression)
            if not isinstance(expr_list, list):
                raise ValueError("Expression must be a vector (e.g. enclosed in square brackets `[ ... ]`).")
                
            vars_list = [sympy.Symbol(v.strip()) for v in variables.split(',')]
            
            if len(expr_list) != 3 or len(vars_list) != 3:
                raise ValueError("Curl is only defined for 3-dimensional vector fields with 3 variables.")
            
            P, Q, R = expr_list
            x, y, z = vars_list
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: [
                    sympy.diff(R, y) - sympy.diff(Q, z),
                    sympy.diff(P, z) - sympy.diff(R, x),
                    sympy.diff(Q, x) - sympy.diff(P, y)
                ]
            )
            
            res_str = "[" + ", ".join(str(r) for r in result) + "]"
            
            embed = math_embed(
                title="Curl ∇×F",
                result=res_str,
                steps=[
                    ("Vector Field F", "[" + ", ".join(str(c) for c in expr_list) + "]"),
                    ("Variables", ", ".join(v.name for v in vars_list)),
                ]
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the CalculusCog into *bot*."""
    await bot.add_cog(CalculusCog(bot))