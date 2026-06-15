"""
cogs/utility.py — Utility and reference slash commands for the math bot.

Commands
--------
/history          Show your recent calculation history (paginated).
/clear_history    Clear your calculation history (with confirmation).
/constants        Reference embed of common mathematical constants.
/help_math        Paginated list of every bot command, grouped by category.
/convert          Convert between units of length, mass, or temperature.
/about            Bot version, library versions, stats, and invite link.

History is stored in-memory only (see :mod:`data.history`) — nothing is
written to disk or any database, and it resets if the bot restarts.
"""

from datetime import datetime, timedelta

import matplotlib
import numpy
import scipy
import sympy
from sympy import Rational
from sympy.physics.units import (
    Quantity,
    centimeter,
    convert_to,
    foot,
    gram,
    inch,
    kilogram,
    kilometer,
    meter,
    mile,
    pound,
)
from discord import app_commands
from discord.ext import commands
import discord

from utils.formatter import math_embed, error_embed
from utils.paginator import send_paginated
from data.history import get_history, clear_history

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_HISTORY_PAGE_SIZE = 5

_BOT_VERSION = "1.0.0"

# Replace with this bot's real OAuth2 invite URL once the application is set up.
_INVITE_URL = "[TO BE FILLED]"


def _exact_and_decimal(expr: sympy.Basic, digits: int = 6) -> tuple[str, str]:
    """Render *expr* as an ``(exact, decimal)`` pair of display strings."""
    exact_str = str(expr)
    try:
        decimal_str = str(sympy.N(expr, digits))
    except Exception:
        decimal_str = "—"
    return exact_str, decimal_str


def _format_uptime(delta: timedelta) -> str:
    """
    Render a :class:`~datetime.timedelta` as a compact ``"1d 2h 3m 4s"``
    string, omitting leading zero units (e.g. ``"5m 12s"`` for under an hour).
    """
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Unit tables (for /convert)
# ---------------------------------------------------------------------------

# sympy.physics.units has no built-in "ounce"; define it relative to the
# pound (1 oz = 1/16 lb) so it can be used with convert_to() like any
# other Quantity.
_ounce = Quantity("ounce")
_ounce.set_global_relative_scale_factor(Rational(1, 16), pound)

_LENGTH_UNITS: dict[str, sympy.Basic] = {
    "m": meter,
    "km": kilometer,
    "cm": centimeter,
    "ft": foot,
    "mile": mile,
    "inch": inch,
}

_MASS_UNITS: dict[str, sympy.Basic] = {
    "kg": kilogram,
    "g": gram,
    "lb": pound,
    "oz": _ounce,
}

_TEMP_UNITS: set[str] = {"c", "f", "k"}

_TEMP_NAMES = {"c": "Celsius", "f": "Fahrenheit", "k": "Kelvin"}

# unit-key -> (category, sympy Quantity or None for temperature)
_UNIT_CATEGORIES: dict[str, tuple[str, sympy.Basic | None]] = {
    **{key: ("length", q) for key, q in _LENGTH_UNITS.items()},
    **{key: ("mass", q) for key, q in _MASS_UNITS.items()},
    **{key: ("temperature", None) for key in _TEMP_UNITS},
}

_SUPPORTED_UNITS_STR = (
    "length: m, km, cm, ft, mile, inch  |  "
    "mass: kg, g, lb, oz  |  "
    "temperature: C, F, K"
)

# Absolute-zero offset: 0°C = 273.15 K, expressed exactly.
_FREEZING_K = Rational(27315, 100)

# (from_unit, "k") -> formula(value, kelvin_result) and ("k", to_unit) -> formula(kelvin, value_result)
_TEMP_TO_KELVIN = {
    "c": lambda v, r: f"K = C + 273.15 = {v} + 273.15 = {r}",
    "f": lambda v, r: f"K = (F − 32) × 5/9 + 273.15 = ({v} − 32) × 5/9 + 273.15 = {r}",
    "k": lambda v, r: f"K = {v}",
}
_TEMP_FROM_KELVIN = {
    "c": lambda v, r: f"C = K − 273.15 = {v} − 273.15 = {r}",
    "f": lambda v, r: f"F = (K − 273.15) × 9/5 + 32 = ({v} − 273.15) × 9/5 + 32 = {r}",
    "k": lambda v, r: f"K = {v}",
}


def _temp_to_kelvin(value: sympy.Rational, unit: str) -> sympy.Rational:
    """Convert *value* (in *unit* — 'c', 'f', or 'k') to Kelvin, exactly."""
    if unit == "c":
        return value + _FREEZING_K
    if unit == "f":
        return (value - 32) * Rational(5, 9) + _FREEZING_K
    return value  # "k"


def _temp_from_kelvin(value_k: sympy.Rational, unit: str) -> sympy.Rational:
    """Convert *value_k* (Kelvin) to *unit* ('c', 'f', or 'k'), exactly."""
    if unit == "c":
        return value_k - _FREEZING_K
    if unit == "f":
        return (value_k - _FREEZING_K) * Rational(9, 5) + 32
    return value_k  # "k"


# ---------------------------------------------------------------------------
# Mathematical constants (for /constants)
# ---------------------------------------------------------------------------

_CONSTANTS: list[tuple[str, str, sympy.Basic, str]] = [
    ("π", "Pi", sympy.pi, "The ratio of a circle's circumference to its diameter."),
    ("e", "Euler's number", sympy.E, "Base of the natural logarithm; limit of (1 + 1/n)ⁿ."),
    ("φ", "Golden ratio", sympy.GoldenRatio, "(1 + √5) / 2 — appears in art, architecture, and nature."),
    ("√2", "Square root of 2", sympy.sqrt(2), "Diagonal length of a unit square; the first number proven irrational."),
    ("i", "Imaginary unit", sympy.I, "Defined by i² = −1; the foundation of the complex numbers."),
    ("∞", "Infinity", sympy.oo, "An unbounded quantity, larger than any real number."),
]


def _constant_decimal(value: sympy.Basic) -> str:
    """
    Return a 10-place decimal string for *value*, or a short note for the
    non-real constants (``i`` and ``∞``) that don't have one.
    """
    if value == sympy.I:
        return "i  (not on the real number line)"
    if value == sympy.oo:
        return "∞  (unbounded — not a finite number)"
    return str(sympy.N(value, 10))


# ---------------------------------------------------------------------------
# Confirmation view (for /clear_history)
# ---------------------------------------------------------------------------

class _ConfirmClearView(discord.ui.View):
    """
    A Yes/No confirmation prompt for ``/clear_history``.

    Only the user who triggered the original command may press a button.
    If the view times out with no response, both buttons are disabled and
    the original message is edited to say so.
    """

    def __init__(self, owner_id: int, timeout: float = 30) -> None:
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        # Set by the command after sending, so on_timeout can edit the
        # original (ephemeral) message.
        self.interaction: discord.Interaction | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "This confirmation isn't for you.", ephemeral=True
            )
            return False
        return True

    def _disable_all(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

    @discord.ui.button(label="Yes, clear it", style=discord.ButtonStyle.danger, custom_id="clear_yes")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        clear_history(self.owner_id)
        self._disable_all()
        self.stop()
        await interaction.response.edit_message(content="History cleared.", view=self)

    @discord.ui.button(label="No", style=discord.ButtonStyle.secondary, custom_id="clear_no")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self._disable_all()
        self.stop()
        await interaction.response.edit_message(content="Cancelled.", view=self)

    async def on_timeout(self) -> None:
        self._disable_all()
        if self.interaction is not None:
            try:
                await self.interaction.edit_original_response(
                    content="Confirmation timed out — history was not cleared.",
                    view=self,
                )
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class UtilityCog(commands.Cog, name="Utility"):
    """Utility commands: history, reference constants, help, and unit conversion."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /history
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="history",
        description="Show your recent calculation history.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def history(self, interaction: discord.Interaction) -> None:
        """Display the user's recent commands, 5 per page, newest first."""
        entries = get_history(interaction.user.id, limit=20)

        if not entries:
            await interaction.response.send_message("No calculations yet.", ephemeral=True)
            return

        await interaction.response.defer()

        pages: list[discord.Embed] = []
        for i in range(0, len(entries), _HISTORY_PAGE_SIZE):
            chunk = entries[i:i + _HISTORY_PAGE_SIZE]
            embed = discord.Embed(
                title="Calculation History",
                colour=discord.Colour.blurple(),
            )
            for entry in chunk:
                ts = int(entry.timestamp.timestamp())
                embed.add_field(
                    name=f"/{entry.command} `{entry.input}`",
                    value=f"→ `{entry.result}`\n<t:{ts}:R>",
                    inline=False,
                )
            pages.append(embed)

        await send_paginated(interaction, pages)

    # -----------------------------------------------------------------------
    # /clear_history
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="clear_history",
        description="Clear your calculation history.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def clear_history_cmd(self, interaction: discord.Interaction) -> None:
        """Ask for confirmation, then clear the user's history if confirmed."""
        view = _ConfirmClearView(interaction.user.id)
        await interaction.response.send_message(
            "Are you sure you want to clear your calculation history? "
            "This cannot be undone.",
            view=view,
            ephemeral=True,
        )
        view.interaction = interaction

    # -----------------------------------------------------------------------
    # /constants
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="constants",
        description="Show a reference list of common mathematical constants.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def constants(self, interaction: discord.Interaction) -> None:
        """Display π, e, φ, √2, i, and ∞ with 10-place decimal values."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="Mathematical Constants",
            colour=discord.Colour.gold(),
        )

        for symbol, name, value, description in _CONSTANTS:
            embed.add_field(
                name=f"{symbol}   —   {name}",
                value=f"`{_constant_decimal(value)}`\n{description}",
                inline=False,
            )

        await interaction.followup.send(embed=embed)

    # -----------------------------------------------------------------------
    # /help_math
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="help_math",
        description="Show all bot commands, grouped by category.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def help_math(self, interaction: discord.Interaction) -> None:
        """
        Build one page per loaded cog, each listing its slash commands and
        descriptions, and present them with the standard paginator.
        """
        await interaction.response.defer()

        pages: list[discord.Embed] = []
        for cog_name, cog in self.bot.cogs.items():
            app_cmds = cog.get_app_commands()
            if not app_cmds:
                continue

            lines = [
                f"**/{cmd.name}** — {cmd.description}"
                for cmd in sorted(app_cmds, key=lambda c: c.name)
            ]

            embed = discord.Embed(
                title=f"📘 {cog_name}",
                description="\n".join(lines),
                colour=discord.Colour.green(),
            )
            pages.append(embed)

        if not pages:
            await interaction.followup.send("No commands are currently loaded.")
            return

        await send_paginated(interaction, pages)

    # -----------------------------------------------------------------------
    # /convert
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="convert",
        description="Convert a value between units of length, mass, or temperature.",
    )
    @app_commands.describe(
        value="Numeric value to convert",
        from_unit="Unit to convert from: m, km, cm, ft, mile, inch, kg, g, lb, oz, C, F, K",
        to_unit="Unit to convert to (must be the same category as from_unit)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def convert(
        self,
        interaction: discord.Interaction,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> None:
        """Convert *value* from *from_unit* to *to_unit* (length, mass, or temperature)."""
        await interaction.response.defer()

        try:
            from_key = from_unit.strip().lower()
            to_key = to_unit.strip().lower()

            if from_key not in _UNIT_CATEGORIES:
                raise ValueError(
                    f"Unsupported unit `{from_unit}`. Supported units — {_SUPPORTED_UNITS_STR}."
                )
            if to_key not in _UNIT_CATEGORIES:
                raise ValueError(
                    f"Unsupported unit `{to_unit}`. Supported units — {_SUPPORTED_UNITS_STR}."
                )

            from_cat, from_q = _UNIT_CATEGORIES[from_key]
            to_cat, to_q = _UNIT_CATEGORIES[to_key]

            if from_cat != to_cat:
                raise ValueError(
                    f"Cannot convert `{from_unit}` ({from_cat}) to `{to_unit}` ({to_cat}) "
                    "— units must be from the same category."
                )

            val = Rational(str(value))

            if from_cat == "temperature":
                kelvin = _temp_to_kelvin(val, from_key)
                result = _temp_from_kelvin(kelvin, to_key)

                steps = []
                if from_key != "k":
                    steps.append((
                        f"Convert {_TEMP_NAMES[from_key]} to Kelvin",
                        _TEMP_TO_KELVIN[from_key](val, kelvin),
                    ))
                if to_key != "k":
                    steps.append((
                        f"Convert Kelvin to {_TEMP_NAMES[to_key]}",
                        _TEMP_FROM_KELVIN[to_key](kelvin, result),
                    ))
                if not steps:
                    steps.append(("Same unit", f"{val} K = {val} K"))

            else:
                # Conversion factor for 1 unit, computed once via
                # sympy.physics.units, then applied ourselves so the
                # formula step is easy to read.
                factor_expr = convert_to(1 * from_q, to_q)
                factor = factor_expr.as_coeff_Mul()[0] if factor_expr != 0 else sympy.Integer(0)
                result = val * factor

                steps = [
                    ("Conversion factor", f"1 {from_unit} = {factor} {to_unit}"),
                    ("Apply factor", f"{val} {from_unit} × {factor} = {result}"),
                ]

            exact_str, decimal_str = _exact_and_decimal(result)

            embed = math_embed(
                title=f"Convert {value} {from_unit} → {to_unit}",
                result=f"{exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=f"{from_cat.capitalize()} conversion",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /about
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="about",
        description="Show information about this bot.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def about(self, interaction: discord.Interaction) -> None:
        """Display version, library, stats, uptime, and invite information."""
        await interaction.response.defer()

        bot = self.bot
        user = bot.user

        embed = discord.Embed(
            title=f"About {user.name if user else 'this bot'}",
            colour=discord.Colour.green(),
        )

        if user and user.display_avatar:
            embed.set_thumbnail(url=user.display_avatar.url)

        embed.add_field(name="Version", value=_BOT_VERSION, inline=True)
        embed.add_field(name="Servers", value=str(len(bot.guilds)), inline=True)
        embed.add_field(
            name="Commands",
            value=str(len(bot.tree.get_commands())),
            inline=True,
        )

        start_time = getattr(bot, "start_time", None)
        if start_time is not None:
            uptime_str = _format_uptime(datetime.utcnow() - start_time)
        else:
            uptime_str = "Unknown"
        embed.add_field(name="Uptime", value=uptime_str, inline=True)

        embed.add_field(
            name="Library Versions",
            value=(
                f"discord.py `{discord.__version__}`\n"
                f"sympy `{sympy.__version__}`\n"
                f"numpy `{numpy.__version__}`\n"
                f"scipy `{scipy.__version__}`\n"
                f"matplotlib `{matplotlib.__version__}`"
            ),
            inline=True,
        )

        embed.add_field(
            name="Invite",
            value=f"[Add to your server]({_INVITE_URL})",
            inline=False,
        )

        embed.set_footer(text="Made with Python 🐍")

        await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the UtilityCog into *bot*."""
    await bot.add_cog(UtilityCog(bot))