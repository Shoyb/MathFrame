"""
cogs/transforms.py — Integral transforms for the math bot.

Commands
--------
/laplace          expression  [t]  [s]        Laplace transform.
/inv_laplace      expression  [s]  [t]        Inverse Laplace transform.
/fourier          expression  [x]  [k]        Fourier transform.
/inv_fourier      expression  [k]  [x]        Inverse Fourier transform.
"""

import asyncio

import discord
import sympy
from discord import app_commands
from discord.ext import commands

from utils.formatter import error_embed, math_embed
from utils.parser import parse_expression


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class TransformsCog(commands.Cog, name="Transforms"):
    """Integral transforms: Laplace, Fourier, and their inverses."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /laplace
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="laplace",
        description="Compute the unilateral Laplace transform L{f(t)}(s).",
    )
    @app_commands.describe(
        expression="Function of time, e.g. t**2 or sin(t)",
        t="Time domain variable (default: t)",
        s="Frequency domain variable (default: s)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def laplace(
        self,
        interaction: discord.Interaction,
        expression: str,
        t: str = "t",
        s: str = "s",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var_t = sympy.Symbol(t)
            var_s = sympy.Symbol(s)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: sympy.laplace_transform(expr, var_t, var_s, noconds=True)
            )

            embed = math_embed(
                title="Laplace Transform",
                result=str(result),
                steps=[
                    ("Function f(t)", str(expr)),
                    ("Transform F(s)", str(result)),
                ],
                footer="Computed without convergence conditions",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /inv_laplace
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="inv_laplace",
        description="Compute the inverse Laplace transform L⁻¹{F(s)}(t).",
    )
    @app_commands.describe(
        expression="Function of frequency, e.g. 1/(s**2 + 1)",
        s="Frequency domain variable (default: s)",
        t="Time domain variable (default: t)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def inv_laplace(
        self,
        interaction: discord.Interaction,
        expression: str,
        s: str = "s",
        t: str = "t",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var_s = sympy.Symbol(s)
            var_t = sympy.Symbol(t, positive=True) # Usually inverse Laplace evaluates for t > 0

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: sympy.inverse_laplace_transform(expr, var_s, var_t)
            )

            embed = math_embed(
                title="Inverse Laplace Transform",
                result=str(result),
                steps=[
                    ("Function F(s)", str(expr)),
                    ("Inverse f(t)", str(result)),
                ],
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /fourier
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="fourier",
        description="Compute the continuous Fourier transform F{f(x)}(k).",
    )
    @app_commands.describe(
        expression="Function of space/time, e.g. exp(-x**2)",
        x="Spatial/Time domain variable (default: x)",
        k="Frequency domain variable (default: k)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def fourier(
        self,
        interaction: discord.Interaction,
        expression: str,
        x: str = "x",
        k: str = "k",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var_x = sympy.Symbol(x)
            var_k = sympy.Symbol(k)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: sympy.fourier_transform(expr, var_x, var_k)
            )

            embed = math_embed(
                title="Fourier Transform",
                result=str(result),
                steps=[
                    ("Function f(x)", str(expr)),
                    ("Transform F(k)", str(result)),
                ],
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /inv_fourier
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="inv_fourier",
        description="Compute the inverse continuous Fourier transform F⁻¹{F(k)}(x).",
    )
    @app_commands.describe(
        expression="Function of frequency, e.g. sqrt(pi)*exp(-pi**2*k**2)",
        k="Frequency domain variable (default: k)",
        x="Spatial/Time domain variable (default: x)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def inv_fourier(
        self,
        interaction: discord.Interaction,
        expression: str,
        k: str = "k",
        x: str = "x",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var_k = sympy.Symbol(k)
            var_x = sympy.Symbol(x)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: sympy.inverse_fourier_transform(expr, var_k, var_x)
            )

            embed = math_embed(
                title="Inverse Fourier Transform",
                result=str(result),
                steps=[
                    ("Function F(k)", str(expr)),
                    ("Inverse f(x)", str(result)),
                ],
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the TransformsCog into *bot*."""
    await bot.add_cog(TransformsCog(bot))
