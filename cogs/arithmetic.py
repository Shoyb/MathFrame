"""
cogs/arithmetic.py — Core arithmetic and algebra slash commands.

Commands
--------
/simplify expression        Simplify a mathematical expression.
/solve    expression [var]  Solve a polynomial equation, with step-by-step working.
/expand   expression        Expand (distribute) an expression.
/factor   expression        Factor an expression, with step-by-step working.

All commands defer immediately, pull from / write to the in-memory TTL cache,
and surface errors through a consistent red error embed.
"""

import sympy
from discord import app_commands
from discord.ext import commands
import discord

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed
from utils.solver    import solve_quadratic_steps, factor_steps
from data.cache      import get, set, cache_key

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

            # Step-by-step working — only meaningful for quadratics.
            # solve_quadratic_steps returns [("Error", "...")] for non-quadratics;
            # discard those steps so the embed doesn't show a broken steps field.
            raw_steps = solve_quadratic_steps(expr, var)
            steps = None if (raw_steps and raw_steps[0][0] == "Error") else raw_steps

            # Warn if any solution is complex
            has_complex = any(not s.is_real for s in solutions)
            sol_str     = ",   ".join(f"{variable} = {s}" for s in solutions)

            footer_parts = []
            if steps is None:
                footer_parts.append("Step-by-step working is only shown for quadratic equations")
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


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the ArithmeticCog into *bot*."""
    await bot.add_cog(ArithmeticCog(bot))