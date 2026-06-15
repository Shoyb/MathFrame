"""
cogs/calculus.py — Calculus slash commands for the math bot.

Commands
--------
/diff       expression [variable] [order]               Differentiate an expression.
/integrate  expression [variable] [lower] [upper]       Definite or indefinite integral.
/limit      expression [variable] [point] [direction]   Evaluate a limit.
/series     expression [variable] [point] [terms]       Taylor / Maclaurin series.
/plot       expression [variable] [xmin] [xmax]         Plot a function as a PNG image.

All commands defer immediately and surface errors through a consistent
red error embed.  Computation-heavy calls run through the async parser
so the event loop is never blocked.
"""

import sympy
from discord import app_commands
from discord.ext import commands
import discord

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed, to_readable_text
from utils.solver    import differentiate_steps, integrate_steps
from utils.plotter   import plot_function
from utils.renderer  import result_to_image

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
    try:
        return sympy.sympify(stripped)
    except sympy.SympifyError as exc:
        raise ValueError(f"Cannot parse point `{point_str}`: {exc}") from exc


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class CalculusCog(commands.Cog, name="Calculus"):
    """Calculus commands: differentiation, integration, limits, series, and plotting."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /diff
    # -----------------------------------------------------------------------

    @app_commands.command(
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

    # -----------------------------------------------------------------------
    # /integrate
    # -----------------------------------------------------------------------

    @app_commands.command(
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

    # -----------------------------------------------------------------------
    # /limit
    # -----------------------------------------------------------------------

    @app_commands.command(
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
            dir_symbol = {"+" : "⁺", "-": "⁻", "+-": ""}.get(direction, "")
            title = f"lim  {variable} → {point}{dir_symbol}  [{expression}]"

            embed = math_embed(
                title=title,
                result=to_readable_text(result),
                footer=f"Limit as {variable} → {point} from the "
                       + {"+" : "right", "-": "left", "+-": "both sides"}[direction],
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /series
    # -----------------------------------------------------------------------

    @app_commands.command(
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

    # -----------------------------------------------------------------------
    # /plot
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="plot",
        description="Plot a function over a specified x range.",
    )
    @app_commands.describe(
        expression="Function to plot, e.g. sin(x)/x or x**3 - x",
        variable="Independent variable (default: x)",
        xmin="Left boundary of the plot domain (default: -10)",
        xmax="Right boundary of the plot domain (default: 10)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def plot(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
        xmin: float = -10.0,
        xmax: float = 10.0,
    ) -> None:
        """Render a PNG plot of *expression* and send it as a file attachment."""
        await interaction.response.defer()

        try:
            if xmin >= xmax:
                raise ValueError(
                    f"`xmin` ({xmin}) must be strictly less than `xmax` ({xmax})."
                )

            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)

            # Render the expression as a LaTeX thumbnail and the function as a plot
            latex_file = await result_to_image(expr)
            plot_file  = await plot_function(
                expr,
                var,
                x_min=xmin,
                x_max=xmax,
                title=f"f({variable}) = {expression}",
            )

            embed = discord.Embed(
                title=f"Plot of  {expression}",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name="Expression",
                value=f"```{expression}```",
                inline=False,
            )
            embed.add_field(
                name="Domain",
                value=f"`{variable}` ∈ [{xmin}, {xmax}]",
                inline=True,
            )
            embed.set_footer(text="Values outside ±1 000 000 are clipped for readability.")

            # Main image: the function plot; thumbnail: the rendered LaTeX formula
            embed.set_image(url="attachment://plot.png")
            embed.set_thumbnail(url="attachment://formula.png")

            await interaction.followup.send(
                embed=embed,
                files=[plot_file, latex_file],
            )

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the CalculusCog into *bot*."""
    await bot.add_cog(CalculusCog(bot))