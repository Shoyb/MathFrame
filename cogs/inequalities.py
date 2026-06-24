"""
cogs/inequalities.py — Inequality solver.

Commands
--------
/solve_ineq         expression     Solve a single inequality.
/solve_ineq_system  expressions    Solve a system of inequalities.
"""

from __future__ import annotations

import asyncio

import discord
from discord import app_commands
from discord.ext import commands
import sympy

from utils.parser    import parse_expression
from utils.formatter import math_embed, error_embed

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class InequalitiesCog(commands.Cog, name="Inequalities"):
    """Solve inequalities and systems of inequalities."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /solve_ineq
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="solve_ineq",
        description="Solve a single inequality, returning the solution set as an interval.",
    )
    @app_commands.describe(
        expression="The inequality to solve, e.g. x**2 - 3*x + 2 < 0",
        variable="Variable to solve for (leave blank to auto-detect)"
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
                # If no free variables (e.g., 3 > 2), it's just a boolean
                result = str(expr)
                embed = math_embed(
                    title="Evaluate Inequality",
                    result=result,
                    steps=[("Original", str(expr))]
                )
                await interaction.followup.send(embed=embed)
                return
                
            if variable:
                var = sympy.Symbol(variable)
            else:
                var = sorted(free_vars, key=lambda s: s.name)[0]
                
            solution = sympy.solve_univariate_inequality(expr, var, relational=False)
            
            # Format interval nicely using pretty-printing fallback
            try:
                result_str = sympy.pretty(solution, use_unicode=False)
            except Exception:
                result_str = str(solution)
                
            embed = math_embed(
                title=f"Solve Inequality for {var}",
                result=result_str,
                steps=[("Original", str(expr))]
            )
            await interaction.followup.send(embed=embed)
            
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except NotImplementedError:
            await interaction.followup.send(embed=error_embed("SymPy couldn't solve this inequality."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /solve_ineq_system
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="solve_ineq_system",
        description="Solve a system of inequalities.",
    )
    @app_commands.describe(
        expressions="Inequalities separated by semicolons, e.g. x > 0; x < 5",
        variables="Variables to solve for, comma-separated (leave blank to auto-detect)"
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
                
            # reduce_inequalities typically works on lists of inequalities
            solution = sympy.reduce_inequalities(parsed, var_syms)
            
            try:
                result_str = sympy.pretty(solution, use_unicode=False)
            except Exception:
                result_str = str(solution)
                
            steps = [
                ("System", "\n".join(str(eq) for eq in parsed)),
                ("Variables", ", ".join(str(v) for v in var_syms)),
            ]
            
            embed = math_embed(
                title="Solve System of Inequalities",
                result=result_str,
                steps=steps
            )
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
    """Load the InequalitiesCog into *bot*."""
    await bot.add_cog(InequalitiesCog(bot))
