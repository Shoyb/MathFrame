"""
cogs/utility.py — Utility slash commands for the math bot.

Commands
--------
/constants      Display fundamental mathematical constants.
/help_math      Paginated help listing all bot commands by cog.
/convert        Convert a value between units (length, mass, temperature).

Note: /history and /clear_history are omitted — no user data is stored.
"""

from __future__ import annotations

import asyncio
from typing import Final

import discord
import sympy
import sympy.physics.units as su
from discord import app_commands
from discord.ext import commands

from utils.formatter import error_embed, info_embed
from utils.paginator import PaginatorView, send_paginated

# ---------------------------------------------------------------------------
# Constants data
# ---------------------------------------------------------------------------

_CONSTANTS: Final[list[dict]] = [
    {
        "symbol":  "π",
        "name":    "Pi",
        "value":   sympy.pi.evalf(10),
        "desc":    "Ratio of a circle's circumference to its diameter.",
    },
    {
        "symbol":  "e",
        "name":    "Euler's Number",
        "value":   sympy.E.evalf(10),
        "desc":    "Base of the natural logarithm; lim(1 + 1/n)ⁿ as n → ∞.",
    },
    {
        "symbol":  "φ",
        "name":    "Golden Ratio",
        "value":   ((1 + sympy.sqrt(5)) / 2).evalf(10),
        "desc":    "φ = (1 + √5) / 2; ratio where a/b = (a+b)/a.",
    },
    {
        "symbol":  "√2",
        "name":    "Pythagoras' Constant",
        "value":   sympy.sqrt(2).evalf(10),
        "desc":    "Diagonal of a unit square; first known irrational number.",
    },
    {
        "symbol":  "i",
        "name":    "Imaginary Unit",
        "value":   "√(−1)",
        "desc":    "Satisfies i² = −1; foundation of complex numbers.",
    },
    {
        "symbol":  "∞",
        "name":    "Infinity",
        "value":   "∞",
        "desc":    "The concept of an unbounded quantity; not a real number.",
    },
]

# ---------------------------------------------------------------------------
# Help pages data  (cog-name → [(command, description), ...])
# ---------------------------------------------------------------------------

_HELP_PAGES: Final[list[tuple[str, list[tuple[str, str]]]]] = [
    ("Arithmetic", [
        ("/simplify",   "Simplify a mathematical expression."),
        ("/solve",      "Solve a polynomial equation step-by-step."),
        ("/expand",     "Expand (distribute) an expression."),
        ("/factor",     "Factor an expression with step-by-step working."),
    ]),
    ("Calculus", [
        ("/diff",       "Differentiate an expression (any order)."),
        ("/integrate",  "Definite or indefinite integral with steps."),
        ("/limit",      "Evaluate a limit at a point or ±∞."),
        ("/series",     "Taylor / Maclaurin series expansion."),
        ("/plot",       "Plot a function as a PNG image."),
    ]),
    ("Linear Algebra", [
        ("/matrix_det", "Determinant of a square matrix."),
        ("/matrix_inv", "Inverse of a square, non-singular matrix."),
        ("/eigenvalues","Eigenvalues with algebraic multiplicities."),
        ("/dot",        "Dot product of two vectors."),
        ("/cross",      "Cross product of two 3-D vectors."),
        ("/rref",       "Reduced row-echelon form of a matrix."),
    ]),
    ("Statistics", [
        ("/mean",       "Arithmetic mean of a data set."),
        ("/median",     "Median of a data set."),
        ("/mode",       "Mode(s) of a data set."),
        ("/stdev",      "Sample standard deviation."),
        ("/variance",   "Sample variance."),
        ("/zscore",     "Standard score (z-score) for a value."),
        ("/correlation","Pearson correlation coefficient."),
        ("/regression", "Linear regression with scatter plot."),
        ("/normal_pdf", "Plot a normal distribution PDF."),
    ]),
    ("Number Theory", [
        ("/gcd",          "GCD of a list of integers."),
        ("/lcm",          "LCM of a list of integers."),
        ("/is_prime",     "Primality test up to 10¹²."),
        ("/factorize",    "Prime factorisation up to 10¹⁵."),
        ("/primes_up_to", "All primes up to n (max 10 000)."),
        ("/modular",      "Fast modular exponentiation."),
        ("/fibonacci",    "First n Fibonacci numbers (paginated)."),
    ]),
    ("Geometry", [
        ("/circle",     "Area and circumference of a circle."),
        ("/triangle",   "Area and properties of a triangle."),
        ("/distance",   "Distance between two points."),
        ("/slope",      "Slope and line equation through two points."),
    ]),
    ("Discrete / Combinatorics", [
        ("/permutation","P(n, r) — ordered arrangements."),
        ("/combination","C(n, r) — unordered selections."),
        ("/binomial",   "Expand (a + b)ⁿ using the binomial theorem."),
    ]),
    ("Symbolic", [
        ("/latex",      "Render a LaTeX expression as an image."),
        ("/simplify_trig", "Simplify a trigonometric expression."),
    ]),
    ("Utility", [
        ("/constants",  "List fundamental mathematical constants."),
        ("/help_math",  "This paginated command reference."),
        ("/convert",    "Convert values between units."),
        ("/ping",       "Check the bot's WebSocket latency."),
    ]),
]

# ---------------------------------------------------------------------------
# Unit conversion tables
# ---------------------------------------------------------------------------

# Length: all conversion factors to metres
_LENGTH_TO_M: Final[dict[str, float]] = {
    "m":    1.0,
    "km":   1_000.0,
    "cm":   0.01,
    "mm":   0.001,
    "ft":   0.3048,
    "inch": 0.0254,
    "in":   0.0254,
    "mile": 1_609.344,
    "mi":   1_609.344,
}

# Mass: all conversion factors to kilograms
_MASS_TO_KG: Final[dict[str, float]] = {
    "kg": 1.0,
    "g":  0.001,
    "mg": 1e-6,
    "lb": 0.453_592_37,
    "oz": 0.028_349_523,
}

_TEMP_UNITS: Final[frozenset[str]] = frozenset({"c", "f", "k"})


def _normalise_unit(u: str) -> str:
    """Lower-case and strip a unit string for lookup."""
    return u.strip().lower()


def _convert_length(value: float, src: str, dst: str) -> tuple[float, str]:
    """
    Convert *value* from *src* length unit to *dst*.

    Returns
    -------
    (result, formula_str)
    """
    if src not in _LENGTH_TO_M:
        raise ValueError(
            f"Unknown length unit `{src}`. "
            f"Supported: {', '.join(_LENGTH_TO_M)}."
        )
    if dst not in _LENGTH_TO_M:
        raise ValueError(
            f"Unknown length unit `{dst}`. "
            f"Supported: {', '.join(_LENGTH_TO_M)}."
        )
    factor = _LENGTH_TO_M[src] / _LENGTH_TO_M[dst]
    result = value * factor
    formula = (
        f"{value} {src}  ×  ({_LENGTH_TO_M[src]} m/{src})  ÷  ({_LENGTH_TO_M[dst]} m/{dst})"
        f"  =  {result:.6g} {dst}"
    )
    return result, formula


def _convert_mass(value: float, src: str, dst: str) -> tuple[float, str]:
    """Convert *value* from *src* mass unit to *dst*."""
    if src not in _MASS_TO_KG:
        raise ValueError(
            f"Unknown mass unit `{src}`. "
            f"Supported: {', '.join(_MASS_TO_KG)}."
        )
    if dst not in _MASS_TO_KG:
        raise ValueError(
            f"Unknown mass unit `{dst}`. "
            f"Supported: {', '.join(_MASS_TO_KG)}."
        )
    factor = _MASS_TO_KG[src] / _MASS_TO_KG[dst]
    result = value * factor
    formula = (
        f"{value} {src}  ×  ({_MASS_TO_KG[src]} kg/{src})  ÷  ({_MASS_TO_KG[dst]} kg/{dst})"
        f"  =  {result:.6g} {dst}"
    )
    return result, formula


def _convert_temperature(value: float, src: str, dst: str) -> tuple[float, str]:
    """
    Convert *value* between Celsius (C), Fahrenheit (F), and Kelvin (K).

    Temperature is non-multiplicative so each pair gets an explicit formula.
    """
    # Normalise to Celsius first, then convert to target
    if src == "c":
        celsius = value
        formula_to_c = f"{value} °C"
    elif src == "f":
        celsius = (value - 32) * 5 / 9
        formula_to_c = f"({value} − 32) × 5/9 = {celsius:.6g} °C"
    elif src == "k":
        celsius = value - 273.15
        formula_to_c = f"{value} − 273.15 = {celsius:.6g} °C"
    else:
        raise ValueError(
            f"Unknown temperature unit `{src}`. Supported: C, F, K."
        )

    if dst == "c":
        result  = celsius
        formula_from_c = f"{celsius:.6g} °C"
    elif dst == "f":
        result  = celsius * 9 / 5 + 32
        formula_from_c = f"{celsius:.6g} × 9/5 + 32 = {result:.6g} °F"
    elif dst == "k":
        result  = celsius + 273.15
        formula_from_c = f"{celsius:.6g} + 273.15 = {result:.6g} K"
    else:
        raise ValueError(
            f"Unknown temperature unit `{dst}`. Supported: C, F, K."
        )

    if src == dst:
        formula = f"{value} {src.upper()} (no conversion needed)"
    elif src == "c" or dst == "c":
        formula = formula_from_c
    else:
        formula = f"Step 1: {formula_to_c}   →   Step 2: {formula_from_c}"

    return result, formula


def _detect_unit_category(src: str, dst: str) -> str:
    """
    Return ``"length"``, ``"mass"``, or ``"temperature"`` based on the units.

    Raises :class:`ValueError` if the units belong to different categories
    or are unrecognised.
    """
    src_is_length = src in _LENGTH_TO_M
    dst_is_length = dst in _LENGTH_TO_M
    src_is_mass   = src in _MASS_TO_KG
    dst_is_mass   = dst in _MASS_TO_KG
    src_is_temp   = src in _TEMP_UNITS
    dst_is_temp   = dst in _TEMP_UNITS

    if src_is_length and dst_is_length:
        return "length"
    if src_is_mass and dst_is_mass:
        return "mass"
    if src_is_temp and dst_is_temp:
        return "temperature"

    # Mixed or unknown
    all_known = set(_LENGTH_TO_M) | set(_MASS_TO_KG) | {u.upper() for u in _TEMP_UNITS}
    if src not in (set(_LENGTH_TO_M) | set(_MASS_TO_KG) | _TEMP_UNITS):
        raise ValueError(
            f"Unrecognised unit `{src}`. "
            "Supported length units: m, km, cm, mm, ft, inch, mile.  "
            "Mass: kg, g, mg, lb, oz.  "
            "Temperature: C, F, K."
        )
    if dst not in (set(_LENGTH_TO_M) | set(_MASS_TO_KG) | _TEMP_UNITS):
        raise ValueError(
            f"Unrecognised unit `{dst}`. "
            "Supported length units: m, km, cm, mm, ft, inch, mile.  "
            "Mass: kg, g, mg, lb, oz.  "
            "Temperature: C, F, K."
        )
    raise ValueError(
        f"Cannot convert between `{src}` and `{dst}` — they belong to different categories."
    )


# ---------------------------------------------------------------------------
# Confirmation View
# ---------------------------------------------------------------------------

class _ConfirmView(discord.ui.View):
    """
    A minimal Yes / No confirmation view.

    After either button is pressed the view disables itself and exposes
    ``self.confirmed: bool | None`` (``None`` means timed out).
    """

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__(timeout=timeout)
        self.confirmed: bool | None = None

    async def _finish(
        self,
        interaction: discord.Interaction,
        value: bool,
        label: str,
    ) -> None:
        self.confirmed = value
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(
            content=f"{'✅' if value else '❌'}  {label}",
            view=self,
        )
        self.stop()

    @discord.ui.button(label="Yes, clear it", style=discord.ButtonStyle.danger)
    async def yes(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._finish(interaction, True, "Confirmed.")

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def no(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._finish(interaction, False, "Cancelled.")


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class UtilityCog(commands.Cog, name="Utility"):
    """Utility commands: constants, help, and unit conversion."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /constants
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="constants",
        description="Display fundamental mathematical constants.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def constants(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()

        embed = discord.Embed(
            title="Mathematical Constants",
            colour=discord.Colour.gold(),
        )
        for c in _CONSTANTS:
            embed.add_field(
                name=f"{c['symbol']}  —  {c['name']}",
                value=f"**Value:** `{c['value']}`\n{c['desc']}",
                inline=False,
            )
        embed.set_footer(text="Values shown to 10 significant figures via SymPy.")
        await interaction.followup.send(embed=embed)

    # -----------------------------------------------------------------------
    # /help_math
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="help_math",
        description="Paginated reference listing all bot commands grouped by category.",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def help_math(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()

        pages: list[discord.Embed] = []
        total_pages = len(_HELP_PAGES)

        for idx, (cog_name, cmds) in enumerate(_HELP_PAGES, start=1):
            lines = "\n".join(
                f"`{cmd}`  —  {desc}" for cmd, desc in cmds
            )
            embed = discord.Embed(
                title=f"📖  Help — {cog_name}",
                description=lines,
                colour=discord.Colour.gold(),
            )
            embed.set_footer(
                text=f"Page {idx}/{total_pages}  |  Use /ping to check bot latency"
            )
            pages.append(embed)

        await send_paginated(interaction, pages)

    # -----------------------------------------------------------------------
    # /convert
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="convert",
        description="Convert a value between units (length, mass, or temperature).",
    )
    @app_commands.describe(
        value="The numeric value to convert",
        from_unit="Source unit (e.g. km, lb, C)",
        to_unit="Target unit (e.g. mile, kg, F)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def convert(
        self,
        interaction: discord.Interaction,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> None:
        await interaction.response.defer()
        try:
            src = _normalise_unit(from_unit)
            dst = _normalise_unit(to_unit)

            category = _detect_unit_category(src, dst)

            if category == "length":
                result, formula = _convert_length(value, src, dst)
                cat_label = "Length"
            elif category == "mass":
                result, formula = _convert_mass(value, src, dst)
                cat_label = "Mass"
            else:  # temperature
                result, formula = _convert_temperature(value, src, dst)
                cat_label = "Temperature"

            steps = [
                ("Input",    f"{value} {from_unit}"),
                ("Formula",  formula),
                ("Result",   f"{result:.6g} {to_unit}"),
            ]

            embed = discord.Embed(
                title=f"Unit Conversion  —  {cat_label}",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name="Result",
                value=f"```{value} {from_unit}  =  {result:.6g} {to_unit}```",
                inline=False,
            )
            embed.add_field(
                name="Steps",
                value="\n".join(f"**{i+1}. {d}**\n`{e}`" for i, (d, e) in enumerate(steps)),
                inline=False,
            )
            embed.set_footer(text=f"Category: {cat_label}")
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the UtilityCog into *bot*."""
    await bot.add_cog(UtilityCog(bot))
