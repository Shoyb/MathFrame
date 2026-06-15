"""
utils/formatter.py — Discord embed builders for the math bot.

All cogs should use these helpers instead of constructing embeds by hand
so that the bot's visual style stays consistent across every command.

Functions
---------
math_embed   — Success embed with a result and optional step-by-step field.
error_embed  — Uniform error embed shown when a command fails.
info_embed   — Neutral informational embed (help text, constants, etc.).
"""

from datetime import datetime, timezone
import re

import discord

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STEPS_FIELD_LIMIT = 1024   # Discord hard limit for field values


def _format_steps(steps: list[tuple[str, str]]) -> str:
    """
    Render a list of ``(description, expression)`` tuples into a numbered
    list that fits inside a single Discord embed field.

    Each step is formatted as::

        **1. description**
        `expression`

    If the fully-rendered string would exceed ``_STEPS_FIELD_LIMIT``
    characters the function stops adding steps and appends a truncation
    notice so the embed remains valid.

    Parameters
    ----------
    steps:
        Ordered list of ``(description, expression)`` pairs, e.g.
        ``[("Move constant", "x² - 4 = 0"), ("Factor", "(x-2)(x+2) = 0")]``.

    Returns
    -------
    str
        Ready-to-embed string, guaranteed to be ≤ 1 024 characters.
    """
    truncation_notice = "\n*…steps truncated (field limit reached)*"
    # Reserve space for the notice so we can always append it if needed.
    budget = _STEPS_FIELD_LIMIT - len(truncation_notice)

    lines: list[str] = []
    accumulated = 0
    truncated = False

    for i, (description, expression) in enumerate(steps, start=1):
        chunk = f"**{i}. {description}**\n`{expression}`"
        # Two-newline separator between steps (not added before the first).
        separator = "\n\n" if lines else ""
        needed = len(separator) + len(chunk)

        if accumulated + needed > budget:
            truncated = True
            break

        lines.append(separator + chunk)
        accumulated += needed

    body = "".join(lines)
    return body + truncation_notice if truncated else body


def _now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)

# ---------------------------------------------------------------------------
# Public embed builders
# ---------------------------------------------------------------------------

def to_readable_text(expr) -> str:
    """
    Convert a SymPy expression or result string to a human-readable format.
    
    - Replaces Python exponentiation '**' with math caret '^'.
    - Removes unnecessary multiplication signs '*' (e.g., 2*x -> 2x, x*y -> xy).
    """
    s = str(expr)
    
    # 1. Handle exponents
    s = s.replace("**", "^")
    
    # 2. Handle implicit multiplication using regex with lookarounds to support 
    # overlapping cases like a*b*c.
    # Matches '*' if:
    #   Preceded by: a digit, a letter, or a closing parenthesis
    #   Followed by: a letter or an opening parenthesis
    # This avoids matching 2*3 (which stays 2*3) while catching 2*x -> 2x.
    s = re.sub(r'(?<=[a-zA-Z\d\)])\*(?=[a-zA-Z\(])', '', s)
    
    return s

def math_embed(
    title: str,
    result: str,
    steps: list[tuple[str, str]] | None = None,
    footer: str = "",
) -> discord.Embed:
    """
    Build a success embed displaying a math result with an optional
    step-by-step breakdown.

    Parameters
    ----------
    title:
        Embed title, typically the command name (e.g. ``"Simplify"``).
    result:
        The computed answer; displayed inside a code block so monospace
        fonts render it cleanly.
    steps:
        Optional ordered list of ``(description, expression)`` tuples that
        walk the user through the computation.  Each tuple becomes one
        numbered entry in the **Steps** field.
    footer:
        Optional footer text (e.g. a hint or attribution string).

    Returns
    -------
    discord.Embed

    Example
    -------
    ::

        embed = math_embed(
            "Solve for x",
            "x = 2, x = -2",
            steps=[
                ("Set equation to zero", "x² - 4 = 0"),
                ("Factor",               "(x - 2)(x + 2) = 0"),
                ("Apply zero-product",   "x = 2  or  x = -2"),
            ],
        )
    """
    embed = discord.Embed(
        title=title,
        colour=discord.Colour.blurple(),
        timestamp=_now(),
    )

    embed.add_field(
        name="Result",
        value=f"```{result}```",
        inline=False,
    )

    if steps:
        embed.add_field(
            name="Steps",
            value=_format_steps(steps),
            inline=False,
        )

    if footer:
        embed.set_footer(text=footer)

    return embed


def error_embed(message: str) -> discord.Embed:
    """
    Build a uniform error embed.

    Parameters
    ----------
    message:
        User-friendly description of what went wrong.  This comes directly
        from the ``ValueError`` messages raised in ``utils/parser.py`` or
        the cog's own validation logic.

    Returns
    -------
    discord.Embed

    Example
    -------
    ::

        await interaction.followup.send(embed=error_embed(str(e)))
    """
    return discord.Embed(
        title="❌ Error",
        description=message,
        colour=discord.Colour.red(),
        timestamp=_now(),
    )


def info_embed(title: str, description: str) -> discord.Embed:
    """
    Build a neutral informational embed.

    Intended for help text, mathematical constants listings, bot
    information, and similar non-result content.

    Parameters
    ----------
    title:
        Short header for the embed.
    description:
        Body text; supports Discord markdown.

    Returns
    -------
    discord.Embed
    """
    return discord.Embed(
        title=title,
        description=description,
        colour=discord.Colour.gold(),
        timestamp=_now(),
    )
