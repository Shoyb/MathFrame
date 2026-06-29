"""
cogs/complex.py — Complex number operations.

Commands
--------
/complex_calc      expression     Evaluate any complex expression.
/complex_polar     expression     Convert to polar form (modulus + argument).
/complex_rect      r theta        Convert polar (r, theta) to rectangular form.
/complex_conjugate expression     Return the complex conjugate.
/complex_modulus   expression     Return the absolute value |z|.
"""

from __future__ import annotations

import sympy
import discord
from discord import app_commands
from discord.ext import commands

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _parse_complex(raw: str) -> sympy.Expr:
    """
    Parse an expression and aggressively substitute the symbol 'i' and 'j' with
    SymPy's imaginary unit `sympy.I`.
    """
    expr = await parse_expression(raw)
    
    # Substitute both 'i' and 'j' to I, as users might use either
    i_sym = sympy.Symbol('i')
    j_sym = sympy.Symbol('j')
    
    expr = expr.subs({i_sym: sympy.I, j_sym: sympy.I})
    return sympy.simplify(expr)

def _format_complex(z: sympy.Expr) -> str:
    """Format a complex number nicely for Discord embeds."""
    # Try to simplify and expand complex to a + bi
    expanded = sympy.expand_complex(z)
    
    # If the exact form is too complex, we might also show the numeric evaluation
    exact = str(expanded).replace('I', 'i')
    
    try:
        if not expanded.has(sympy.Symbol):
            numeric = sympy.N(expanded, 6)
            num_str = str(numeric).replace('I', 'i')
            if exact != num_str and "e" not in exact and exact != str(expanded):
                 return f"{exact}  ≈  {num_str}"
    except Exception:
        pass
        
    return exact

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class ComplexCog(commands.Cog, name="Complex"):
    """Complex number operations: rectangular/polar conversion, modulus, conjugate."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    cx = app_commands.Group(name="cx", description="Complex number commands.")


    # -----------------------------------------------------------------------
    # /complex_calc
    # -----------------------------------------------------------------------

    @cx.command(
        name="calc",
        description="Evaluate a complex arithmetic expression (e.g. (2+3i)*(1-i)).",
    )
    @app_commands.describe(expression="The expression to evaluate.")
    @app_commands.checks.cooldown(1, 2.0)
    async def complex_calc(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await _parse_complex(expression)
            result = _format_complex(expr)
            
            embed = math_embed(
                title="Complex Calculation",
                result=result,
                steps=[("Original", str(await parse_expression(expression)).replace('I', 'i'))]
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /complex_polar
    # -----------------------------------------------------------------------

    @cx.command(
        name="polar",
        description="Convert a complex number to polar form (modulus, argument).",
    )
    @app_commands.describe(expression="The complex number, e.g. 1+i")
    @app_commands.checks.cooldown(1, 2.0)
    async def complex_polar(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await _parse_complex(expression)
            
            r = sympy.simplify(sympy.Abs(expr))
            theta = sympy.simplify(sympy.arg(expr))
            
            r_str = str(r)
            theta_str = str(theta)
            
            result_str = f"r = {r_str}\nθ = {theta_str}"
            
            embed = math_embed(
                title="Polar Form Conversion",
                result=result_str,
                steps=[("Original", _format_complex(expr))],
                footer="Angle θ is in radians"
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /complex_rect
    # -----------------------------------------------------------------------

    @cx.command(
        name="rect",
        description="Convert polar coordinates (r, θ) to rectangular form (a + bi).",
    )
    @app_commands.describe(
        r="Modulus (distance from origin)",
        theta="Argument (angle in radians, e.g. pi/2)"
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def complex_rect(
        self,
        interaction: discord.Interaction,
        r: str,
        theta: str,
    ) -> None:
        await interaction.response.defer()
        try:
            r_expr = await parse_expression(r)
            theta_expr = await parse_expression(theta)
            
            # r * e^(i*theta)
            z = r_expr * sympy.exp(sympy.I * theta_expr)
            z_rect = sympy.expand_complex(z)
            
            result = _format_complex(z_rect)
            
            embed = math_embed(
                title="Rectangular Form Conversion",
                result=result,
                steps=[("Polar Input", f"r = {r}\nθ = {theta}")],
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /complex_conjugate
    # -----------------------------------------------------------------------

    @cx.command(
        name="conj",
        description="Return the complex conjugate of an expression.",
    )
    @app_commands.describe(expression="The complex expression")
    @app_commands.checks.cooldown(1, 2.0)
    async def complex_conjugate(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await _parse_complex(expression)
            conj = sympy.conjugate(expr)
            conj = sympy.expand_complex(conj)
            
            embed = math_embed(
                title="Complex Conjugate",
                result=_format_complex(conj),
                steps=[("Original", _format_complex(expr))]
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /complex_modulus
    # -----------------------------------------------------------------------

    @cx.command(
        name="modulus",
        description="Calculate the modulus (absolute value) |z| of a complex expression.",
    )
    @app_commands.describe(expression="The complex expression")
    @app_commands.checks.cooldown(1, 2.0)
    async def complex_modulus(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await _parse_complex(expression)
            modulus = sympy.simplify(sympy.Abs(expr))
            
            embed = math_embed(
                title="Complex Modulus",
                result=str(modulus),
                steps=[("Original", _format_complex(expr))]
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the ComplexCog into *bot*."""
    await bot.add_cog(ComplexCog(bot))
