"""
cogs/bot.py — Bot utility group: general utility commands + Wikipedia browser.

Merged from:  utility.py  +  wiki.py

Commands  (all under the /bot group)
--------------------------------------
/bot history      Show your recent calculation history (paginated).
/bot clear        Clear your calculation history.
/bot constants    Reference embed of common mathematical constants.
/bot help         Paginated list of every bot command, grouped by category.
/bot convert      Convert between units (length, mass, temp, time, area, volume, speed, etc).
/bot units        Convert any expression with units (compound / derived).
/bot about        Bot version, library versions, guild count, and uptime.
/bot wiki         Fetch a Wikipedia article and browse it paragraph by paragraph.
/bot wiki_search  Search Wikipedia and see a list of matching articles.
"""

from __future__ import annotations

import importlib.metadata
import re
import textwrap
from datetime import datetime, timezone, timedelta
from typing import Any

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

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
import sympy
import sympy.physics.units as spu

from utils.formatter import math_embed, error_embed
from utils.paginator import send_paginated
from data.history import get_history, clear_history, save_history   # noqa: F401


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_HISTORY_PAGE_SIZE = 5
_BOT_VERSION       = "1.0.0"
_INVITE_URL        = ""      # fill in your OAuth2 invite URL once ready


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
    Render a timedelta as ``"1d 2h 3m 4s"``, omitting leading zero units.
    """
    total = int(delta.total_seconds())
    days, r   = divmod(total, 86400)
    hours, r  = divmod(r, 3600)
    minutes, seconds = divmod(r, 60)
    parts: list[str] = []
    if days:    parts.append(f"{days}d")
    if hours   or parts: parts.append(f"{hours}h")
    if minutes or parts: parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _lib_version(name: str) -> str:
    """Return installed version of *name*, or 'unknown'."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

# ---------------------------------------------------------------------------
# Unit tables (for /convert)
# ---------------------------------------------------------------------------

# sympy.physics.units has no built-in ounce — define it relative to pound.
_ounce = Quantity("ounce")
_ounce.set_global_relative_scale_factor(Rational(1, 16), pound)

_LENGTH_UNITS: dict[str, sympy.Basic] = {
    "m":    meter,
    "km":   kilometer,
    "cm":   centimeter,
    "ft":   foot,
    "mile": mile,
    "inch": inch,
}

_MASS_UNITS: dict[str, sympy.Basic] = {
    "kg": kilogram,
    "g":  gram,
    "lb": pound,
    "oz": _ounce,
}

_TEMP_UNITS: set[str] = {"c", "f", "k"}
_TEMP_NAMES: dict[str, str] = {"c": "Celsius", "f": "Fahrenheit", "k": "Kelvin"}

_TIME_UNITS: dict[str, sympy.Basic] = {
    "s": spu.second,
    "min": spu.minute,
    "hr": spu.hour,
    "day": spu.day,
    "week": 7 * spu.day,
    "year": spu.year,
}

_AREA_UNITS: dict[str, sympy.Basic] = {
    "m2": spu.meter**2,
    "km2": spu.kilometer**2,
    "cm2": spu.centimeter**2,
    "hectare": spu.hectare,
    "acre": 43560 * spu.foot**2,
    "sq_ft": spu.foot**2,
    "sq_mile": spu.mile**2,
}

_VOLUME_UNITS: dict[str, sympy.Basic] = {
    "l": spu.liter,
    "ml": spu.milliliter,
    "m3": spu.meter**3,
    "cm3": spu.centimeter**3,
    "gal": sympy.Rational(378541, 100000) * spu.liter,
    "qt": sympy.Rational(378541, 400000) * spu.liter,
    "pt": sympy.Rational(378541, 800000) * spu.liter,
    "cup": sympy.Rational(378541, 1600000) * spu.liter,
    "fl_oz": sympy.Rational(378541, 12800000) * spu.liter,
}

_SPEED_UNITS: dict[str, sympy.Basic] = {
    "mps": spu.meter / spu.second,
    "kph": spu.kilometer / spu.hour,
    "mph": spu.mile / spu.hour,
    "knot": sympy.Rational(1852, 1000) * (spu.kilometer / spu.hour),
    "fps": spu.foot / spu.second,
}

_FORCE_UNITS: dict[str, sympy.Basic] = {
    "n": spu.newton,
    "lbf": sympy.Rational(4448222, 1000000) * spu.newton,
    "dyne": sympy.Rational(1, 100000) * spu.newton,
}

_ENERGY_UNITS: dict[str, sympy.Basic] = {
    "j": spu.joule,
    "kj": spu.kilo * spu.joule,
    "cal": sympy.Rational(4184, 1000) * spu.joule,
    "kcal": 4184 * spu.joule,
    "wh": 3600 * spu.joule,
    "kwh": 3600000 * spu.joule,
    "btu": sympy.Rational(105506, 100) * spu.joule,
    "ev": spu.electronvolt,
}

_POWER_UNITS: dict[str, sympy.Basic] = {
    "w": spu.watt,
    "kw": spu.kilo * spu.watt,
    "mw": spu.mega * spu.watt,
    "hp": sympy.Rational(7457, 10) * spu.watt,
}

# unit-key → (category, sympy Quantity | None for temperature)
_UNIT_CATEGORIES: dict[str, tuple[str, sympy.Basic | None]] = {
    **{key: ("length",      q)    for key, q in _LENGTH_UNITS.items()},
    **{key: ("mass",        q)    for key, q in _MASS_UNITS.items()},
    **{key: ("temperature", None) for key in _TEMP_UNITS},
    **{key: ("time",        q)    for key, q in _TIME_UNITS.items()},
    **{key: ("area",        q)    for key, q in _AREA_UNITS.items()},
    **{key: ("volume",      q)    for key, q in _VOLUME_UNITS.items()},
    **{key: ("speed",       q)    for key, q in _SPEED_UNITS.items()},
    **{key: ("force",       q)    for key, q in _FORCE_UNITS.items()},
    **{key: ("energy",      q)    for key, q in _ENERGY_UNITS.items()},
    **{key: ("power",       q)    for key, q in _POWER_UNITS.items()},
}

_SUPPORTED_UNITS_STR = (
    "length: m, km, cm, ft, mile, inch  |  "
    "mass: kg, g, lb, oz  |  "
    "temperature: C, F, K  |  "
    "time: s, min, hr, day, week, year  |  "
    "area: m2, km2, cm2, hectare, acre, sq_ft, sq_mile  |  "
    "volume: l, ml, m3, cm3, gal, qt, pt, cup, fl_oz  |  "
    "speed: mps, kph, mph, knot, fps  |  "
    "force: n, lbf, dyne  |  "
    "energy: j, kj, cal, kcal, wh, kwh, btu, ev  |  "
    "power: w, kw, mw, hp"
)

# ---------------------------------------------------------------------------
# Unit alias table (for /units)
# ---------------------------------------------------------------------------

# Maps lowercase user-supplied unit names to sympy.physics.units objects.
# Covers the most commonly needed SI, imperial, and derived units.
_UNIT_ALIASES: dict[str, sympy.Basic] = {
    # Length
    "m": spu.meter, "meter": spu.meter, "meters": spu.meter,
    "km": spu.kilometer, "kilometer": spu.kilometer, "kilometers": spu.kilometer,
    "cm": spu.centimeter, "centimeter": spu.centimeter,
    "mm": spu.millimeter, "millimeter": spu.millimeter,
    "ft": spu.foot, "foot": spu.foot, "feet": spu.foot,
    "inch": spu.inch, "in": spu.inch, "inches": spu.inch,
    "mile": spu.mile, "miles": spu.mile,
    "yd": spu.yard, "yard": spu.yard, "yards": spu.yard,
    # Mass
    "kg": spu.kilogram, "kilogram": spu.kilogram, "kilograms": spu.kilogram,
    "g": spu.gram, "gram": spu.gram, "grams": spu.gram,
    "mg": spu.milligram, "milligram": spu.milligram,
    "lb": spu.pound, "pound": spu.pound, "pounds": spu.pound,
    "oz": _ounce, "ounce": _ounce, "ounces": _ounce,
    # Time
    "s": spu.second, "sec": spu.second, "second": spu.second, "seconds": spu.second,
    "min": spu.minute, "minute": spu.minute, "minutes": spu.minute,
    "hr": spu.hour, "hour": spu.hour, "hours": spu.hour,
    # Area
    "m2": spu.meter**2, "km2": spu.kilometer**2, "cm2": spu.centimeter**2,
    # Volume
    "l": spu.liter, "liter": spu.liter, "litre": spu.liter,
    "ml": spu.milliliter, "milliliter": spu.milliliter,
    # Speed
    "mps": spu.meter / spu.second,
    "kph": spu.kilometer / spu.hour,
    "mph": spu.mile / spu.hour,
    # Force / Energy / Power / Pressure
    "n": spu.newton, "newton": spu.newton,
    "j": spu.joule, "joule": spu.joule,
    "kj": spu.kilo * spu.joule,
    "w": spu.watt, "watt": spu.watt,
    "kw": spu.kilo * spu.watt,
    "pa": spu.pascal, "pascal": spu.pascal,
    # Electricity
    "v": spu.volt, "volt": spu.volt,
    "a": spu.ampere, "ampere": spu.ampere,
    "ohm": spu.ohm,
    "f": spu.farad, "farad": spu.farad,
    "hz": spu.hertz, "hertz": spu.hertz,
}


def _resolve_unit(name: str) -> sympy.Basic:
    """
    Return the ``sympy.physics.units`` object for *name*.

    Raises
    ------
    ValueError
        If the name is not recognised.
    """
    key = name.strip().lower()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    # Try looking it up directly in sympy.physics.units as a last resort
    obj = getattr(spu, key, None)
    if obj is not None and isinstance(obj, sympy.Basic):
        return obj
    raise ValueError(
        f"Unit `{name}` is not recognised.  "
        "Try a common name like `m`, `kg`, `newton`, `joule`, `mps`, `kph`, etc."
    )

# 0°C = 273.15 K, kept as an exact rational throughout.
_FREEZING_K = Rational(27315, 100)


def _temp_to_kelvin(value: Rational, unit: str) -> Rational:
    """Convert *value* in *unit* ('c', 'f', 'k') to Kelvin exactly."""
    if unit == "c":
        return value + _FREEZING_K
    if unit == "f":
        return (value - 32) * Rational(5, 9) + _FREEZING_K
    return value  # already Kelvin


def _temp_from_kelvin(kelvin: Rational, unit: str) -> Rational:
    """Convert *kelvin* to *unit* ('c', 'f', 'k') exactly."""
    if unit == "c":
        return kelvin - _FREEZING_K
    if unit == "f":
        return (kelvin - _FREEZING_K) * Rational(9, 5) + 32
    return kelvin  # Kelvin → Kelvin


def _temp_steps(
    val: Rational,
    from_key: str,
    kelvin: Rational,
    to_key: str,
    result: Rational,
) -> list[tuple[str, str]]:
    """Build the step list for a temperature conversion."""
    steps: list[tuple[str, str]] = []

    # Step 1: to Kelvin (skip if already Kelvin)
    if from_key == "c":
        steps.append((
            f"Convert {_TEMP_NAMES[from_key]} → Kelvin",
            f"K = {val} + 273.15 = {sympy.N(kelvin, 8)}",
        ))
    elif from_key == "f":
        steps.append((
            f"Convert {_TEMP_NAMES[from_key]} → Kelvin",
            f"K = ({val} − 32) × 5/9 + 273.15 = {sympy.N(kelvin, 8)}",
        ))

    # Step 2: from Kelvin to target (skip if target is Kelvin)
    if to_key == "c":
        steps.append((
            f"Convert Kelvin → {_TEMP_NAMES[to_key]}",
            f"C = {sympy.N(kelvin, 8)} − 273.15 = {sympy.N(result, 8)}",
        ))
    elif to_key == "f":
        steps.append((
            f"Convert Kelvin → {_TEMP_NAMES[to_key]}",
            f"F = ({sympy.N(kelvin, 8)} − 273.15) × 9/5 + 32 = {sympy.N(result, 8)}",
        ))

    if not steps:
        steps.append(("Same unit", f"{val} K"))

    return steps

# ---------------------------------------------------------------------------
# Mathematical constants (for /constants)
# ---------------------------------------------------------------------------

_CONSTANTS: list[tuple[str, str, sympy.Basic, str]] = [
    ("π",  "Pi",             sympy.pi,          "Ratio of a circle's circumference to its diameter."),
    ("e",  "Euler's number", sympy.E,            "Base of the natural logarithm; limit of (1 + 1/n)ⁿ."),
    ("φ",  "Golden ratio",   sympy.GoldenRatio,  "(1 + √5) / 2 — appears in art, architecture, and nature."),
    ("√2", "Square root of 2", sympy.sqrt(2),   "Diagonal of a unit square; first number proven irrational."),
    ("i",  "Imaginary unit", sympy.I,            "Defined by i² = −1; foundation of complex numbers."),
    ("∞",  "Infinity",       sympy.oo,           "Unbounded quantity, larger than any real number."),
]


def _constant_decimal(value: sympy.Basic) -> str:
    """10-place decimal string, or a short note for i and ∞."""
    if value == sympy.I:
        return "i  (not a real number)"
    if value == sympy.oo:
        return "∞  (not a finite number)"
    return str(sympy.N(value, 10))

# ---------------------------------------------------------------------------
# Confirmation view (for /clear_history)
# ---------------------------------------------------------------------------

class _ConfirmClearView(discord.ui.View):
    """
    Ephemeral Yes / No prompt for ``/clear_history``.

    Only the invoking user can press either button.  On timeout the buttons
    are disabled and the message is edited to say the action was cancelled.
    """

    def __init__(self, owner_id: int, timeout: float = 30.0) -> None:
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        self.message: discord.InteractionMessage | None = None  # set after send

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
    async def confirm(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        cleared = clear_history(self.owner_id)   # sync — in-memory, no await needed
        self._disable_all()
        self.stop()
        await interaction.response.edit_message(
            content=f"History cleared ({cleared} entr{'y' if cleared == 1 else 'ies'} removed).",
            view=self,
        )

    @discord.ui.button(label="No", style=discord.ButtonStyle.secondary, custom_id="clear_no")
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self._disable_all()
        self.stop()
        await interaction.response.edit_message(content="Cancelled.", view=self)

    async def on_timeout(self) -> None:
        self._disable_all()
        if self.message is not None:
            try:
                await self.message.edit(
                    content="Confirmation timed out — history was not cleared.",
                    view=self,
                )
            except discord.HTTPException:
                pass

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE          = "https://en.wikipedia.org"
_REST_BASE     = f"{_BASE}/api/rest_v1/page"
_API_BASE      = f"{_BASE}/w/api.php"
_TIMEOUT       = aiohttp.ClientTimeout(total=10)

# Characters per embed description — Discord hard limit is 4096
_PAGE_CHARS    = 1800
# Max search results shown in /wiki_search
_SEARCH_LIMIT  = 5
# Thumbnail size hint for Wikipedia image URLs
_THUMB_WIDTH   = 480
# Colour used on all wiki embeds
_COLOUR        = discord.Colour.from_rgb(255, 255, 255)   # Wikipedia white

# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """
    Strip MediaWiki artefacts from plain-text article content.

    Removes:
    * Reference markers like ``[1]``, ``[note 2]``
    * Leftover ``\\n`` runs (collapse to single blank line)
    * Leading / trailing whitespace per line
    """
    text = re.sub(r"\[\d+\]",          "",   text)   # numeric refs
    text = re.sub(r"\[[a-z]+ \d+\]",   "",   text)   # named refs
    text = re.sub(r"\[note \d+\]",     "",   text)
    text = re.sub(r" {2,}",            " ",  text)   # collapse spaces
    text = re.sub(r"\n{3,}",           "\n\n", text) # max one blank line
    return text.strip()


def _thumbnail_url(summary: dict[str, Any]) -> str | None:
    """Extract the best available thumbnail URL from a summary response."""
    original = summary.get("originalimage") or {}
    thumb    = summary.get("thumbnail") or {}
    url      = original.get("source") or thumb.get("source")
    if url and "width=" in url:
        # Request a sensible width so we don't embed a 4000 px image
        url = re.sub(r"/\d+px-", f"/{_THUMB_WIDTH}px-", url)
    return url


async def _fetch_summary(session: aiohttp.ClientSession, title: str) -> dict[str, Any]:
    """
    Fetch the Wikipedia summary for *title*.

    Returns the parsed JSON dict on success.

    Raises
    ------
    ValueError
        If the page does not exist (404) or the request fails.
    """
    url = f"{_REST_BASE}/summary/{aiohttp.helpers.quote(title, safe='')}"
    async with session.get(url, timeout=_TIMEOUT) as resp:
        if resp.status == 404:
            raise ValueError(
                f"No Wikipedia article found for **{title}**. "
                "Try `/wiki_search` to find the right title."
            )
        if resp.status != 200:
            raise ValueError(
                f"Wikipedia returned HTTP {resp.status}. Please try again."
            )
        return await resp.json()


async def _fetch_sections(
    session: aiohttp.ClientSession,
    title: str,
) -> list[dict[str, Any]]:
    """
    Fetch all sections of *title* via the mobile-sections API.

    Returns a list of section dicts, each with at least ``"title"``
    and ``"text"`` keys (plain text, already stripped of most markup).

    Falls back to an empty list if the endpoint fails — the summary
    paragraph is always shown even when sections can't be fetched.
    """
    url = f"{_REST_BASE}/mobile-sections/{aiohttp.helpers.quote(title, safe='')}"
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []

    sections: list[dict[str, Any]] = []

    # Lead section lives at data["lead"]["sections"][0]
    lead_sections = (data.get("lead") or {}).get("sections") or []
    for s in lead_sections:
        text = _clean(_strip_html(s.get("text") or ""))
        if text:
            sections.append({"title": "Introduction", "text": text})

    # Remaining sections live under data["remaining"]["sections"]
    remaining = (data.get("remaining") or {}).get("sections") or []
    for s in remaining:
        text = _clean(_strip_html(s.get("text") or ""))
        if text:
            heading = s.get("line") or s.get("title") or "Section"
            heading = _clean(_strip_html(heading))
            sections.append({"title": heading, "text": text})

    return sections


async def _search_wikipedia(
    session: aiohttp.ClientSession,
    query: str,
    limit: int = _SEARCH_LIMIT,
) -> list[dict[str, str]]:
    """
    Search Wikipedia for *query* and return up to *limit* results.

    Each result is a dict with ``"title"`` and ``"snippet"`` keys.
    """
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  str(limit),
        "format":   "json",
        "utf8":     "1",
    }
    async with session.get(_API_BASE, params=params, timeout=_TIMEOUT) as resp:
        if resp.status != 200:
            raise ValueError(
                f"Wikipedia search returned HTTP {resp.status}. Please try again."
            )
        data   = await resp.json()
        hits   = (data.get("query") or {}).get("search") or []
        return [{"title": h["title"], "snippet": _clean(_strip_html(h.get("snippet", "")))}
                for h in hits]


def _strip_html(text: str) -> str:
    """Remove all HTML tags from *text*, replacing ``<br>`` with newlines."""
    text = re.sub(r"<br\s*/?>",  "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>",   "",   text)
    text = re.sub(r"&amp;",      "&",  text)
    text = re.sub(r"&lt;",       "<",  text)
    text = re.sub(r"&gt;",       ">",  text)
    text = re.sub(r"&nbsp;",     " ",  text)
    text = re.sub(r"&#\d+;",     "",   text)
    return text

# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------

def _article_pages(
    title: str,
    wiki_url: str,
    sections: list[dict[str, Any]],
    thumbnail: str | None,
) -> list[discord.Embed]:
    """
    Convert a list of article sections into a list of Discord embeds,
    splitting long sections across multiple pages so no embed exceeds
    Discord's description limit.

    Parameters
    ----------
    title:
        Article title used in every embed's title field.
    wiki_url:
        Link to the full article (shown in the first embed only).
    sections:
        List of ``{"title": str, "text": str}`` dicts.
    thumbnail:
        Optional image URL; set on the first embed only.

    Returns
    -------
    list[discord.Embed]
        At least one embed; ready to pass to :func:`~utils.paginator.send_paginated`.
    """
    pages: list[discord.Embed] = []

    for sec_idx, section in enumerate(sections):
        heading = section["title"]
        text    = section["text"]

        # Split section text into chunks that fit the embed description limit
        chunks = textwrap.wrap(
            text,
            width=_PAGE_CHARS,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        if not chunks:
            continue

        for chunk_idx, chunk in enumerate(chunks):
            # Section heading only on the first chunk of each section
            display_heading = heading if chunk_idx == 0 else f"{heading} (cont.)"

            embed = discord.Embed(
                title=f"Wikipedia — {title}",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name=display_heading,
                value=chunk,
                inline=False,
            )

            # First page gets the article link and thumbnail
            if not pages:
                embed.url = wiki_url
                if thumbnail:
                    embed.set_thumbnail(url=thumbnail)

            pages.append(embed)

    if not pages:
        # Shouldn't happen, but guard against completely empty articles
        embed = discord.Embed(
            title=f"Wikipedia — {title}",
            description="*(article has no readable content)*",
            colour=discord.Colour.blurple(),
            url=wiki_url,
        )
        pages.append(embed)

    return pages


def _search_result_embed(query: str, results: list[dict[str, str]]) -> discord.Embed:
    """Build a single embed listing search results for *query*."""
    embed = discord.Embed(
        title=f"Wikipedia Search: \"{query}\"",
        description=(
            "Here are the closest matches. "
            "Use `/wiki <title>` with the exact title to open an article."
        ),
        colour=discord.Colour.gold(),
    )
    for i, result in enumerate(results, start=1):
        snippet = result["snippet"]
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        embed.add_field(
            name=f"{i}. {result['title']}",
            value=snippet or "*(no snippet available)*",
            inline=False,
        )
    embed.set_footer(text=f"{len(results)} result(s) found  |  Powered by Wikipedia")
    return embed

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class BotCog(commands.Cog, name="Bot"):
    """Bot utility commands: history, constants, help, unit conversion, and Wikipedia."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._session: aiohttp.ClientSession | None = None

    utility_grp = app_commands.Group(name="bot", description="Bot utility and reference commands.")


    # -----------------------------------------------------------------------
    # /history
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="history",
        description="Show your recent calculation history.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def history(self, interaction: discord.Interaction) -> None:
        """Display the user's last 20 calculations, 5 per page, newest first."""
        entries = get_history(interaction.user.id, limit=20)

        if not entries:
            await interaction.response.send_message(
                "No calculations yet.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        pages: list[discord.Embed] = []
        for i in range(0, len(entries), _HISTORY_PAGE_SIZE):
            chunk = entries[i:i + _HISTORY_PAGE_SIZE]
            embed = discord.Embed(title="Calculation History", colour=discord.Colour.blurple())
            for entry in chunk:
                ts = int(entry.created_at.timestamp())
                embed.add_field(
                    name=f"/{entry.command}  `{entry.input}`",
                    value=f"→ `{entry.result}`\n<t:{ts}:R>",
                    inline=False,
                )
            pages.append(embed)

        await send_paginated(interaction, pages)

    # -----------------------------------------------------------------------
    # /clear_history
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="clear",
        description="Clear your calculation history.",
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def clear_history_cmd(self, interaction: discord.Interaction) -> None:
        """Ask for confirmation, then clear the user's history if confirmed."""
        view = _ConfirmClearView(interaction.user.id)
        await interaction.response.send_message(
            "Are you sure you want to clear your calculation history? "
            "This **cannot** be undone.",
            view=view,
            ephemeral=True,
        )
        # Store the message reference so on_timeout can edit it.
        view.message = await interaction.original_response()

    # -----------------------------------------------------------------------
    # /constants
    # -----------------------------------------------------------------------

    @utility_grp.command(
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
                name=f"{symbol}  —  {name}",
                value=f"`{_constant_decimal(value)}`\n{description}",
                inline=False,
            )

        await interaction.followup.send(embed=embed)

    # -----------------------------------------------------------------------
    # /help_math
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="help",
        description="Show all bot commands, grouped by category.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def help_math(self, interaction: discord.Interaction) -> None:
        """One page per loaded cog, listing its slash commands and descriptions."""
        await interaction.response.defer()

        pages: list[discord.Embed] = []
        for cog_name, cog in sorted(self.bot.cogs.items()):
            app_cmds = cog.get_app_commands()
            if not app_cmds:
                continue

            lines: list[str] = []
            for cmd in sorted(app_cmds, key=lambda c: c.name):
                if isinstance(cmd, app_commands.Group):
                    # Expand subcommands: show  /group sub  — description
                    for subcmd in sorted(cmd.commands, key=lambda c: c.name):
                        lines.append(f"**/{cmd.name} {subcmd.name}** — {subcmd.description}")
                else:
                    lines.append(f"**/{cmd.name}** — {cmd.description}")

            if not lines:
                continue

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

    @utility_grp.command(
        name="convert",
        description="Convert a value between units (length, mass, temp, time, area, volume, speed, force, energy, power).",
    )
    @app_commands.describe(
        value     = "Numeric value to convert",
        from_unit = "Unit to convert from (e.g. m, kg, C, s, m2, l, mph, n, j, w)",
        to_unit   = "Unit to convert to (must be the same category as from_unit)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def convert(
        self,
        interaction: discord.Interaction,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> None:
        """Convert *value* from *from_unit* to *to_unit*."""
        await interaction.response.defer()

        try:
            from_key = from_unit.strip().lower()
            to_key   = to_unit.strip().lower()

            if from_key not in _UNIT_CATEGORIES:
                raise ValueError(
                    f"Unsupported unit `{from_unit}`.\n"
                    f"Supported: {_SUPPORTED_UNITS_STR}."
                )
            if to_key not in _UNIT_CATEGORIES:
                raise ValueError(
                    f"Unsupported unit `{to_unit}`.\n"
                    f"Supported: {_SUPPORTED_UNITS_STR}."
                )

            from_cat, from_q = _UNIT_CATEGORIES[from_key]
            to_cat,   to_q   = _UNIT_CATEGORIES[to_key]

            if from_cat != to_cat:
                raise ValueError(
                    f"Cannot convert `{from_unit}` ({from_cat}) to "
                    f"`{to_unit}` ({to_cat}) — units must be the same category."
                )

            val = Rational(str(value))   # exact rational, avoids float drift

            if from_cat == "temperature":
                kelvin = _temp_to_kelvin(val, from_key)
                result = _temp_from_kelvin(kelvin, to_key)
                steps  = _temp_steps(val, from_key, kelvin, to_key, result)
            else:
                # Derive the conversion factor via SymPy units, then divide
                # out the target unit to get a plain dimensionless number.
                factor_expr = convert_to(from_q, to_q)
                factor = sympy.nsimplify(
                    factor_expr / to_q, rational=True
                )
                result = val * factor
                steps  = [
                    ("Conversion factor", f"1 {from_unit} = {factor} {to_unit}"),
                    ("Apply factor",      f"{val} × {factor} = {sympy.N(result, 8)} {to_unit}"),
                ]

            exact_str, decimal_str = _exact_and_decimal(result)
            result_display = (
                f"{exact_str}  ≈  {decimal_str} {to_unit.upper()}"
                if exact_str != decimal_str
                else f"{exact_str} {to_unit.upper()}"
            )

            embed = math_embed(
                title=f"Convert  {value} {from_unit.upper()} → {to_unit.upper()}",
                result=result_display,
                steps=steps,
                footer=f"{from_cat.capitalize()} conversion",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /units
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="units",
        description="Convert any expression with units, including compound and derived units.",
    )
    @app_commands.describe(
        value='Numeric value to convert, e.g. "9.8" or "1/3"',
        from_unit='Source unit expression, e.g. "m/s^2", "km/h", "kg*m^2"',
        to_unit='Target unit expression, e.g. "ft/s^2", "mph", "joule"',
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def units(
        self,
        interaction: discord.Interaction,
        value: str,
        from_unit: str,
        to_unit: str,
    ) -> None:
        """
        Convert *value from_unit* to *to_unit* using ``sympy.physics.units``.

        Handles compound and derived units (e.g. ``m/s^2``, ``kg*m/s^2``,
        ``kWh``).  The existing ``/convert`` command is faster and simpler for
        everyday single-unit conversions.
        """
        await interaction.response.defer()

        try:
            # ---- Parse the numeric value --------------------------------
            try:
                val = sympy.Rational(str(value).strip())
            except Exception:
                raise ValueError(
                    f"`{value}` is not a valid number. "
                    "Use a decimal (e.g. `9.8`) or fraction (e.g. `1/3`)."
                )

            # ---- Parse unit expressions ---------------------------------
            # We build a SymPy expression from tokens split on * / ^
            # Each alphabetic token is resolved through _resolve_unit().
            def _parse_unit_expr(raw: str) -> sympy.Basic:
                import re as _re
                raw = raw.strip()
                # Replace ^ with ** for Python eval-style parsing
                raw = raw.replace("^", "**")
                # Tokenise: split on * and / while keeping the delimiters
                tokens = _re.split(r"([*/])", raw)
                result: sympy.Basic | None = None
                op = "*"
                for tok in tokens:
                    tok = tok.strip()
                    if tok in ("*", "/"):
                        op = tok
                        continue
                    if not tok:
                        continue
                    # Check for exponentiation: name**n
                    exp_match = _re.match(r"^([A-Za-z]\w*)(\*\*[\-\d]+)$", tok)
                    if exp_match:
                        base_name = exp_match.group(1)
                        exponent  = int(exp_match.group(2).replace("**", ""))
                        unit_obj  = _resolve_unit(base_name) ** exponent
                    elif tok.lstrip("-").isdigit():
                        unit_obj = sympy.Integer(int(tok))
                    else:
                        unit_obj = _resolve_unit(tok)

                    if result is None:
                        result = unit_obj
                    elif op == "*":
                        result = result * unit_obj
                    else:
                        result = result / unit_obj
                if result is None:
                    raise ValueError(f"Could not parse unit expression `{raw}`.")
                return result

            src_unit = _parse_unit_expr(from_unit)
            tgt_unit = _parse_unit_expr(to_unit)

            # ---- Perform conversion -------------------------------------
            source_quantity = val * src_unit
            converted = spu.convert_to(source_quantity, tgt_unit)

            # Strip the target unit to get a dimensionless numeric result.
            # nsimplify with rational=True preserves exact fractions.
            numeric = sympy.nsimplify(converted / tgt_unit, rational=True)
            numeric_simplified = sympy.simplify(numeric)

            exact_str   = str(numeric_simplified)
            decimal_str = str(sympy.N(numeric_simplified, 8))

            result_display = (
                f"{exact_str}  ≈  {decimal_str}  {to_unit}"
                if exact_str != decimal_str
                else f"{exact_str}  {to_unit}"
            )

            steps = [
                ("Input",       f"{value}  {from_unit}"),
                ("Target unit", to_unit),
                ("Converted",   result_display),
            ]
            embed = math_embed(
                title=f"Unit Conversion  {from_unit} → {to_unit}",
                result=result_display,
                steps=steps,
                footer="Powered by sympy.physics.units  |  Use /convert for simple everyday conversions",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(
                    f"Conversion failed: {exc}\n"
                    "Make sure the units are compatible (e.g. you can't convert meters to kilograms)."
                )
            )

    # -----------------------------------------------------------------------
    # /about
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="about",
        description="Show information about this bot.",
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def about(self, interaction: discord.Interaction) -> None:
        """Display version, library versions, guild count, and uptime."""
        await interaction.response.defer()

        user = self.bot.user
        embed = discord.Embed(
            title=f"About {user.name if user else 'MathFrame'}",
            colour=discord.Colour.green(),
        )

        if user and user.display_avatar:
            embed.set_thumbnail(url=user.display_avatar.url)

        embed.add_field(name="Version",  value=_BOT_VERSION,                   inline=True)
        embed.add_field(name="Servers",  value=str(len(self.bot.guilds)),       inline=True)
        embed.add_field(name="Commands", value=str(len(self.bot.tree.get_commands())), inline=True)

        start_time: datetime | None = getattr(self.bot, "start_time", None)
        uptime_str = _format_uptime(datetime.now(tz=timezone.utc) - start_time) if start_time else "Unknown"
        embed.add_field(name="Uptime", value=uptime_str, inline=True)

        embed.add_field(
            name="Libraries",
            value=(
                f"discord.py `{discord.__version__}`\n"
                f"sympy `{_lib_version('sympy')}`\n"
                f"numpy `{_lib_version('numpy')}`\n"
                f"scipy `{_lib_version('scipy')}`\n"
                f"matplotlib `{_lib_version('matplotlib')}`"
            ),
            inline=True,
        )

        if _INVITE_URL:
            embed.add_field(name="Invite", value=f"[Add to your server]({_INVITE_URL})", inline=False)

        embed.set_footer(text="Made with Python 🐍")
        await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
    async def cog_load(self) -> None:
        """Create the shared HTTP session when the cog is loaded."""
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "MathBot/1.0 (Discord bot; educational use)"}
        )

    async def cog_unload(self) -> None:
        """Close the HTTP session cleanly when the cog is unloaded."""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "MathBot/1.0 (Discord bot; educational use)"}
            )
        return self._session

    # -----------------------------------------------------------------------
    # /wiki
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="wiki",
        description="Fetch a Wikipedia article and browse it paragraph by paragraph.",
    )
    @app_commands.describe(
        topic="Article title or topic to look up, e.g. 'Pythagorean theorem'",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def wiki(
        self,
        interaction: discord.Interaction,
        topic: str,
    ) -> None:
        await interaction.response.defer()
        try:
            summary   = await _fetch_summary(self.session, topic)
            title     = summary.get("title", topic)
            wiki_url  = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
            thumbnail = _thumbnail_url(summary)

            # Try to get full sectioned content; fall back to summary extract
            sections = await _fetch_sections(self.session, title)

            if not sections:
                # Graceful fallback: use the summary extract as a single section
                extract = _clean(summary.get("extract") or "*(no content)*")
                sections = [{"title": "Summary", "text": extract}]

            pages = _article_pages(title, wiki_url, sections, thumbnail)
            await send_paginated(interaction, pages)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except aiohttp.ClientError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Network error fetching Wikipedia: {exc}")
            )

    # -----------------------------------------------------------------------
    # /wiki_search
    # -----------------------------------------------------------------------

    @utility_grp.command(
        name="wiki_search",
        description="Search Wikipedia and see a list of matching articles.",
    )
    @app_commands.describe(
        topic="Search query, e.g. 'Fourier transform' or 'prime numbers'",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def wiki_search(
        self,
        interaction: discord.Interaction,
        topic: str,
    ) -> None:
        await interaction.response.defer()
        try:
            results = await _search_wikipedia(self.session, topic)

            if not results:
                await interaction.followup.send(
                    embed=error_embed(
                        f"No Wikipedia articles found for **{topic}**. "
                        "Try different keywords."
                    )
                )
                return

            embed = _search_result_embed(topic, results)
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except aiohttp.ClientError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Network error searching Wikipedia: {exc}")
            )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the BotCog into *bot*."""
    await bot.add_cog(BotCog(bot))