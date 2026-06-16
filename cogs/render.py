"""
cogs/render.py — LaTeX rendering slash command for the math bot.

Commands
--------
/render   latex      Render a LaTeX expression as a PNG image.
/formula  expression Render a parsed math expression as a PNG image.

/render  accepts raw LaTeX strings (no surrounding $ needed).
/formula accepts the same natural/plain/Python/LaTeX input as all
         other cogs, parses it through SymPy, and renders the result.
"""

import discord
from discord import app_commands
from discord.ext import commands

from utils.parser    import parse_expression
from utils.renderer  import expr_to_image, result_to_image
from utils.formatter import error_embed

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class RenderCog(commands.Cog, name="Render"):
    """LaTeX and expression rendering commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /render  — raw LaTeX → PNG
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="render",
        description="Render a LaTeX expression as a PNG image.",
    )
    @app_commands.describe(
        latex=r'LaTeX string without $ delimiters, e.g.  \frac{-b \pm \sqrt{b^2-4ac}}{2a}',
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def render(self, interaction: discord.Interaction, latex: str) -> None:
        r"""
        Render *latex* to a PNG and send it as a file attachment.

        The ``$`` math-mode delimiters are added automatically — do not
        include them in the input.  Most standard LaTeX math commands work
        (fractions, integrals, summations, Greek letters, etc.), but
        complex environments like ``align`` or ``tikz`` are not supported
        since this uses matplotlib's mathtext engine rather than a full
        LaTeX installation.

        Examples
        --------
        ``\frac{1}{2}``
        ``\int_0^\infty e^{-x^2} dx``
        ``\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}``
        ``\begin{pmatrix} a & b \\ c & d \end{pmatrix}``
        """
        await interaction.response.defer()
        try:
            file = await expr_to_image(latex.strip())

            embed = discord.Embed(
                title="Rendered Formula",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name="LaTeX input",
                value=f"```latex\n{latex}\n```",
                inline=False,
            )
            embed.set_image(url="attachment://formula.png")
            embed.set_footer(text="Rendered with matplotlib mathtext")

            await interaction.followup.send(embed=embed, file=file)

        except ValueError as exc:
            await interaction.followup.send(
                embed=error_embed(
                    f"{exc}\n\n"
                    "**Tips:**\n"
                    "• Do not include surrounding `$` — they are added automatically\n"
                    r"• Use `\\` for backslash commands, e.g. `\frac{1}{2}`" "\n"
                    "• Complex environments (align, tikz) are not supported"
                )
            )

    # -----------------------------------------------------------------------
    # /formula  — parsed expression → PNG
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="formula",
        description="Parse and render any math expression as a PNG formula image.",
    )
    @app_commands.describe(
        expression='Any expression the bot understands, e.g. "x^2 + 2x + 1" or "\\frac{1}{x}"',
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def formula(self, interaction: discord.Interaction, expression: str) -> None:
        """
        Parse *expression* through SymPy (same as all math commands), convert
        to LaTeX via ``sympy.latex()``, and render it as a PNG.

        Use this when you have a result from another command and want to see
        it typeset beautifully, or when you want to visualize what the bot
        is actually computing.
        """
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            file = await result_to_image(expr)

            embed = discord.Embed(
                title="Formula",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name="Input",
                value=f"```{expression}```",
                inline=True,
            )
            embed.add_field(
                name="SymPy form",
                value=f"```{str(expr)}```",
                inline=True,
            )
            embed.set_image(url="attachment://formula.png")
            embed.set_footer(text="Rendered via sympy.latex() + matplotlib mathtext")

            await interaction.followup.send(embed=embed, file=file)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the RenderCog into *bot*."""
    await bot.add_cog(RenderCog(bot))
