"""
cogs/symbolic.py — Symbolic math slash commands for the math bot.

Commands
--------
/latex            expression                  Render an expression as a LaTeX PNG image.
/subs             expression  substitutions   Substitute values into an expression.
/partial_fraction expression  [variable]      Partial fraction decomposition.
/roots            expression  [variable]      Find all roots of an expression.
"""

import sympy
import discord
from discord import app_commands
from discord.ext import commands

from utils.parser    import parse_expression, _validate_raw  # noqa: PLC2701
from utils.formatter import math_embed, error_embed
from utils.renderer  import expr_to_image, result_to_image

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_substitutions(raw: str) -> dict[sympy.Symbol, sympy.Basic]:
    """
    Parse a ``"x=2, y=3"`` style substitution string into a dict mapping
    SymPy :class:`~sympy.Symbol` objects to SymPy values.

    Parameters
    ----------
    raw:
        Comma-separated ``variable=value`` pairs, e.g. ``"x=2, y=pi, z=-1/2"``.

    Returns
    -------
    dict[sympy.Symbol, sympy.Basic]

    Raises
    ------
    ValueError
        If any token is not in ``name=value`` form, if either side is blank,
        or if a value cannot be parsed by SymPy.

    Examples
    --------
    >>> _parse_substitutions("x=2, y=pi")
    {x: 2, y: pi}
    """
    if not raw.strip():
        raise ValueError(
            "Substitutions string is empty. "
            "Use format: `x=2, y=3` or `x=pi, y=1/2`."
        )

    result: dict[sympy.Symbol, sympy.Basic] = {}

    for i, token in enumerate(raw.split(","), start=1):
        token = token.strip()
        if not token:
            continue

        if "=" not in token:
            raise ValueError(
                f"Substitution #{i} `{token}` is missing `=`. "
                "Each entry must be in `variable=value` form."
            )

        lhs, _, rhs = token.partition("=")
        lhs = lhs.strip()
        rhs = rhs.strip()

        if not lhs:
            raise ValueError(f"Substitution #{i} has an empty variable name.")
        if not rhs:
            raise ValueError(f"Substitution #{i} for `{lhs}` has an empty value.")
        if not lhs.isidentifier():
            raise ValueError(
                f"`{lhs}` is not a valid variable name (substitution #{i})."
            )

        # Apply forbidden-keyword and length guard before sympify.
        _validate_raw(rhs)
        try:
            value = sympy.sympify(rhs)
        except sympy.SympifyError as exc:
            raise ValueError(
                f"Cannot parse value `{rhs}` for `{lhs}`: {exc}"
            ) from exc

        result[sympy.Symbol(lhs)] = value

    if not result:
        raise ValueError("No valid substitutions found.")

    return result


def _root_line(var: sympy.Symbol, root: sympy.Basic, mult: int) -> str:
    """
    Format a single root as a display line with a complexity indicator.

    Complex roots are prefixed with a blue circle emoji; real roots are plain.

    Parameters
    ----------
    var:
        The variable symbol (used in the label).
    root:
        The root value (SymPy expression).
    mult:
        Algebraic multiplicity of the root.
    """
    is_complex = not root.is_real
    # is_real returns None for "unknown" on symbolic roots — treat as complex
    prefix = "🔵 " if (is_complex or is_complex is None and root.has(sympy.I)) else ""
    mult_str = f"  (multiplicity: {mult})" if mult > 1 else ""
    return f"{prefix}{var} = {root}{mult_str}"


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class SymbolicCog(commands.Cog, name="Symbolic"):
    """Symbolic math commands: LaTeX rendering, substitution, partial fractions, roots."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /latex
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="latex",
        description="Render a mathematical expression as a LaTeX image.",
    )
    @app_commands.describe(
        expression="Expression to render, e.g. x^2 + 2x or \\frac{1}{x}",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def latex(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expr      = await parse_expression(expression)
            latex_str = sympy.latex(expr)
            file      = await expr_to_image(latex_str)

            embed = discord.Embed(
                title="LaTeX Render",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name="Input",
                value=f"```{expression}```",
                inline=False,
            )
            embed.add_field(
                name="LaTeX",
                value=f"```{latex_str}```",
                inline=False,
            )
            embed.set_image(url="attachment://formula.png")

            await interaction.followup.send(embed=embed, file=file)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
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
    # /subs
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="subs",
        description='Substitute values into an expression, e.g. substitutions: "x=2, y=pi".',
    )
    @app_commands.describe(
        expression="The expression to evaluate, e.g. x^2 + y",
        substitutions='Comma-separated pairs, e.g. "x=2, y=pi" or "x=1/2, y=-3"',
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def subs(
        self,
        interaction: discord.Interaction,
        expression: str,
        substitutions: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expr     = await parse_expression(expression)
            subs_map = _parse_substitutions(substitutions)

            substituted = expr.subs(subs_map)
            result      = sympy.simplify(substituted)

            # Build steps showing each substitution clearly
            subs_display = ",   ".join(
                f"{var} = {val}" for var, val in subs_map.items()
            )
            steps = [
                ("Original expression", str(expr)),
                ("Substitutions",       subs_display),
                ("After substitution",  str(substituted)),
                ("Simplified result",   str(result)),
            ]
            # Collapse steps 3 and 4 into one if simplification changed nothing
            if str(substituted) == str(result):
                steps = steps[:3]

            embed = math_embed(
                title="Substitution",
                result=str(result),
                steps=steps,
                footer=f"Substituted: {subs_display}",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
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
    # /partial_fraction
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="partial_fraction",
        description="Decompose a rational expression into partial fractions.",
    )
    @app_commands.describe(
        expression="Rational expression to decompose, e.g. 1/(x^2-1)",
        variable="Variable to decompose with respect to (default: x)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def partial_fraction(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
    ) -> None:
        await interaction.response.defer()
        try:
            expr   = await parse_expression(expression)
            var    = sympy.Symbol(variable)
            result = sympy.apart(expr, var)

            # Detect no-op: apart() returns the input unchanged when it
            # cannot be decomposed further
            already_decomposed = sympy.simplify(result - expr) == 0

            steps = [
                ("Original expression",  str(expr)),
                ("Partial fraction form", str(result)),
            ]
            footer = (
                "Expression is already fully decomposed."
                if already_decomposed
                else f"Decomposed with respect to {variable}"
            )

            embed = math_embed(
                title="Partial Fraction Decomposition",
                result=str(result),
                steps=steps,
                footer=footer,
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
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
    # /roots
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="roots",
        description="Find all roots of an expression (set equal to zero).",
    )
    @app_commands.describe(
        expression="Expression whose roots to find, e.g. x^3 - x or x^2 + 1",
        variable="Variable to solve for (default: x)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def roots(
        self,
        interaction: discord.Interaction,
        expression: str,
        variable: str = "x",
    ) -> None:
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var  = sympy.Symbol(variable)

            # sympy.roots returns {root: multiplicity} for polynomial expressions
            roots_dict = sympy.roots(expr, var)

            if not roots_dict:
                # Fall back to sympy.solve for non-polynomial or tricky cases
                solutions = sympy.solve(expr, var)
                if not solutions:
                    embed = math_embed(
                        title="Roots",
                        result="Expression has no closed-form roots.",
                        footer=f"No roots found for: {expression}",
                    )
                    await interaction.followup.send(embed=embed)
                    return
                # Wrap solve() output into the same {root: 1} format
                roots_dict = {sol: 1 for sol in solutions}

            lines       = [_root_line(var, root, mult) for root, mult in
                           sorted(roots_dict.items(), key=lambda kv: (sympy.re(kv[0]), sympy.im(kv[0])))]
            has_complex = any(
                not r.is_real or (r.is_real is None and r.has(sympy.I))
                for r in roots_dict
            )
            result_str  = "\n".join(lines)

            steps = [
                ("Expression (= 0)", str(expr)),
                ("Roots found",      result_str),
            ]

            footer_parts = [f"{len(roots_dict)} root(s) found"]
            if has_complex:
                footer_parts.append("🔵 = complex root")
            footer_parts.append(f"variable: {variable}")

            # Render result as an image too
            result_expr = sympy.Rational(0)   # placeholder; we render the original expr
            image_file  = await result_to_image(expr)

            embed = math_embed(
                title="Roots",
                result=result_str,
                steps=steps,
                footer="  |  ".join(footer_parts),
            )
            embed.set_image(url="attachment://formula.png")

            await interaction.followup.send(embed=embed, file=image_file)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
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
    """Load the SymbolicCog into *bot*."""
    await bot.add_cog(SymbolicCog(bot))
