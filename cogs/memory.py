"""
cogs/memory.py — Personal variable memory workspace.

Commands
--------
/mem set   <name> <value>   — store a number, expression, or matrix
/mem get   <name>           — display a stored value
/mem list                   — show all stored variables
/mem del   <name>           — delete one variable
/mem clear                  — delete everything (with confirmation)
/mem eval  <expression>     — evaluate an expression with $name substitution

Using stored values in other commands
--------------------------------------
Any command that accepts an expression supports the $name token syntax once
the cog passes the expression through ``memory.resolve()``.  The calls that
already do this are:

    /simplify  /expand  /factor  /solve
    /diff      /integrate

Example:
    /mem set  k  0.5
    /diff     $k * x^2          →  treats k as 0.5
    /solve    x^2 + 2*x + 3     →  x is still a free symbol — no conflict

Adding $-support to any other command takes one line before parse_expression::

    expression = memory.resolve(interaction.guild_id or 0,
                                interaction.user.id, expression)
"""

from __future__ import annotations

import asyncio

import discord
import sympy
from discord import app_commands
from discord.ext import commands

from data.memory import (
    MAX_ENTRIES,
    MemEntry,
    MemType,
    MemoryStore,
    SYMPY_BUILTIN_NAMES,
    memory,
    parse_matrix,
    validate_name,
)
from utils.formatter import error_embed, math_embed
from utils.parser import parse_expression

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gid(interaction: discord.Interaction) -> int:
    """Return guild_id, or 0 for DMs."""
    return interaction.guild_id or 0


async def _parse_value(raw: str) -> MemEntry:
    """
    Auto-detect the type of *raw* and return a :class:`MemEntry`.

    Detection order:
    1. Starts with ``[``  → try matrix parsing.
    2. Parse as a SymPy expression via ``parse_expression``.
       - No free symbols → NUMBER
       - Has free symbols → EXPRESSION
    """
    raw = raw.strip()

    # --- Matrix -----------------------------------------------------------
    if raw.startswith("["):
        try:
            mat = parse_matrix(raw)
            rows, cols = mat.shape
            return MemEntry(
                name="",
                mem_type=MemType.MATRIX,
                value=mat,
                raw=raw,
            )
        except ValueError:
            pass  # fall through and try as an expression

    # --- Scalar / Expression ---------------------------------------------
    try:
        expr = await parse_expression(raw)
    except Exception as exc:
        raise ValueError(f"Could not parse value: {exc}") from exc

    mem_type = MemType.NUMBER if not expr.free_symbols else MemType.EXPRESSION
    return MemEntry(name="", mem_type=mem_type, value=expr, raw=raw)


def _mem_embed(
    title: str,
    entry: MemEntry,
    *,
    colour: discord.Colour = discord.Colour.blurple(),
    extra_fields: list[tuple[str, str, bool]] | None = None,
    footer: str | None = None,
) -> discord.Embed:
    """Build a consistent embed for memory display."""
    embed = discord.Embed(title=title, colour=colour)

    # Value display — pretty for matrices, code-block for scalars
    if entry.mem_type == MemType.MATRIX:
        rows, cols = entry.value.shape
        embed.add_field(
            name=f"{entry.emoji} Value  ({rows}×{cols} matrix)",
            value=f"```\n{entry.display_str()}\n```",
            inline=False,
        )
    else:
        embed.add_field(
            name=f"{entry.emoji} Value  ({entry.type_label})",
            value=f"```\n{entry.display_str()}\n```",
            inline=False,
        )

    embed.add_field(
        name="Original input",
        value=f"`{entry.raw}`",
        inline=False,
    )

    if extra_fields:
        for name, value, inline in extra_fields:
            embed.add_field(name=name, value=value, inline=inline)

    if footer:
        embed.set_footer(text=footer)

    return embed


# ---------------------------------------------------------------------------
# Confirmation view for /mem clear
# ---------------------------------------------------------------------------

class _ConfirmClearView(discord.ui.View):
    def __init__(self, owner_id: int) -> None:
        super().__init__(timeout=30.0)
        self.owner_id   = owner_id
        self.confirmed  = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "This confirmation isn't for you.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Yes, clear everything", style=discord.ButtonStyle.danger)
    async def confirm(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        self.confirmed = True
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        self.stop()
        await interaction.response.defer()

    async def on_timeout(self) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class MemoryCog(commands.Cog, name="Memory"):
    """Personal variable memory — store numbers, expressions, and matrices."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # The slash-command group — all sub-commands registered below
    mem = app_commands.Group(
        name="mem",
        description="Personal variable memory — store and recall values across commands.",
    )

    # -----------------------------------------------------------------------
    # /mem set
    # -----------------------------------------------------------------------

    @mem.command(
        name="set",
        description="Store a number, expression, or matrix under a name.",
    )
    @app_commands.describe(
        name=(
            "Variable name (letters, digits, underscores — e.g. k, alpha, M2). "
            "Overwrites if the name already exists."
        ),
        value=(
            "Value to store.  "
            "Number: 3.14 | pi/2 | sqrt(3)   "
            "Expression: x^2 + 2*x   "
            "Matrix: [[1,2],[3,4]]"
        ),
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def mem_set(
        self,
        interaction: discord.Interaction,
        name: str,
        value: str,
    ) -> None:
        """Store *value* in memory under *name*."""
        await interaction.response.defer(ephemeral=True)

        # Validate name first so we give a fast error before heavy parsing
        try:
            validate_name(name)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return

        try:
            entry = await _parse_value(value)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return

        try:
            memory.set(_gid(interaction), interaction.user.id, name, entry)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return

        count = memory.count(_gid(interaction), interaction.user.id)
        footer_parts = [f"Reference as ${name} in expressions", f"{count}/{MAX_ENTRIES} slots used"]

        warn = ""
        if name in SYMPY_BUILTIN_NAMES:
            warn = (
                f"\n⚠️  `{name}` shadows a SymPy built-in.  "
                f"Use `${name}` (with `$`) when you want your stored value; "
                f"bare `{name}` in expressions still refers to the built-in."
            )

        embed = _mem_embed(
            title=f"📥 Stored — {name}",
            entry=entry,
            colour=discord.Colour.green(),
            footer="  |  ".join(footer_parts),
        )
        if warn:
            embed.add_field(name="⚠️ Name collision", value=warn, inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)

    # -----------------------------------------------------------------------
    # /mem get
    # -----------------------------------------------------------------------

    @mem.command(
        name="get",
        description="Display a stored variable.",
    )
    @app_commands.describe(name="Name of the stored variable to show.")
    @app_commands.checks.cooldown(1, 1.0)
    async def mem_get(
        self,
        interaction: discord.Interaction,
        name: str,
    ) -> None:
        """Retrieve and display the value stored under *name*."""
        await interaction.response.defer(ephemeral=True)

        entry = memory.get(_gid(interaction), interaction.user.id, name)
        if entry is None:
            await interaction.followup.send(
                embed=error_embed(
                    f"`{name}` is not in your memory.  "
                    "Use `/mem list` to see what's stored."
                ),
                ephemeral=True,
            )
            return

        footer = f"Reference as ${name} in expressions" if entry.mem_type != MemType.MATRIX else f"View only — matrices cannot be inlined with $"

        embed = _mem_embed(
            title=f"📦 {name}",
            entry=entry,
            footer=footer,
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    # -----------------------------------------------------------------------
    # /mem list
    # -----------------------------------------------------------------------

    @mem.command(
        name="list",
        description="Show all variables stored in your memory.",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def mem_list(
        self,
        interaction: discord.Interaction,
    ) -> None:
        """List every stored variable with a one-line preview."""
        await interaction.response.defer(ephemeral=True)

        entries = memory.list_all(_gid(interaction), interaction.user.id)

        if not entries:
            embed = discord.Embed(
                title="📭 Memory — empty",
                description=(
                    "Nothing stored yet.\n"
                    "Use `/mem set <name> <value>` to store a value."
                ),
                colour=discord.Colour.light_grey(),
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        # Build a compact table
        lines: list[str] = []
        for entry_name, entry in sorted(entries.items()):
            emoji   = entry.emoji
            preview = entry.short_display()
            if entry.mem_type == MemType.MATRIX:
                rows, cols = entry.value.shape
                preview = f"[{rows}×{cols} matrix]"
            lines.append(f"{emoji}  {entry_name:<16}  {preview}")

        table = "\n".join(lines)

        embed = discord.Embed(
            title=f"🗂️ Memory  ({len(entries)}/{MAX_ENTRIES} slots)",
            description=f"```\n{table}\n```",
            colour=discord.Colour.blurple(),
        )
        embed.set_footer(text="Use $name in any supported expression to inline a value.")
        await interaction.followup.send(embed=embed, ephemeral=True)

    # -----------------------------------------------------------------------
    # /mem del
    # -----------------------------------------------------------------------

    @mem.command(
        name="del",
        description="Delete a stored variable from memory.",
    )
    @app_commands.describe(name="Name of the variable to delete.")
    @app_commands.checks.cooldown(1, 1.0)
    async def mem_del(
        self,
        interaction: discord.Interaction,
        name: str,
    ) -> None:
        """Delete the variable named *name*."""
        await interaction.response.defer(ephemeral=True)

        deleted = memory.delete(_gid(interaction), interaction.user.id, name)
        if not deleted:
            await interaction.followup.send(
                embed=error_embed(
                    f"`{name}` is not in your memory — nothing was deleted.\n"
                    "Use `/mem list` to see what's stored."
                ),
                ephemeral=True,
            )
            return

        count = memory.count(_gid(interaction), interaction.user.id)
        embed = discord.Embed(
            title=f"🗑️ Deleted — {name}",
            description=f"`{name}` has been removed from memory.",
            colour=discord.Colour.orange(),
        )
        embed.set_footer(text=f"{count}/{MAX_ENTRIES} slots remaining")
        await interaction.followup.send(embed=embed, ephemeral=True)

    # -----------------------------------------------------------------------
    # /mem clear
    # -----------------------------------------------------------------------

    @mem.command(
        name="clear",
        description="Delete all variables from your memory (asks for confirmation).",
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def mem_clear(
        self,
        interaction: discord.Interaction,
    ) -> None:
        """Clear the entire memory namespace after user confirmation."""
        count = memory.count(_gid(interaction), interaction.user.id)

        if count == 0:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="📭 Memory already empty",
                    description="There's nothing to clear.",
                    colour=discord.Colour.light_grey(),
                ),
                ephemeral=True,
            )
            return

        view = _ConfirmClearView(owner_id=interaction.user.id)

        await interaction.response.send_message(
            embed=discord.Embed(
                title="⚠️ Clear all memory?",
                description=(
                    f"This will permanently delete all **{count}** stored "
                    f"variable{'s' if count != 1 else ''}.  This cannot be undone."
                ),
                colour=discord.Colour.yellow(),
            ),
            view=view,
            ephemeral=True,
        )

        await view.wait()

        if view.confirmed:
            deleted = memory.clear(_gid(interaction), interaction.user.id)
            embed = discord.Embed(
                title="🗑️ Memory cleared",
                description=f"Deleted {deleted} variable{'s' if deleted != 1 else ''}.",
                colour=discord.Colour.red(),
            )
        else:
            embed = discord.Embed(
                title="↩️ Cancelled",
                description="Memory was not changed.",
                colour=discord.Colour.light_grey(),
            )

        await interaction.edit_original_response(embed=embed, view=None)

    # -----------------------------------------------------------------------
    # /mem eval
    # -----------------------------------------------------------------------

    @mem.command(
        name="eval",
        description="Evaluate an expression — use $name to reference stored values.",
    )
    @app_commands.describe(
        expression=(
            "Expression to evaluate.  "
            "Use $name to substitute stored values.  "
            "Example: $k * x^2 + $offset"
        ),
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def mem_eval(
        self,
        interaction: discord.Interaction,
        expression: str,
    ) -> None:
        """Resolve $-references and simplify the expression."""
        await interaction.response.defer()

        try:
            # Resolve $name tokens
            resolved = memory.resolve(_gid(interaction), interaction.user.id, expression)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
            return

        try:
            expr   = await parse_expression(resolved)
            result = sympy.simplify(expr)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
            return
        except sympy.SympifyError:
            await interaction.followup.send(
                embed=error_embed("Could not parse the expression after substitution.")
            )
            return

        # Format result
        result_str = str(result)
        result_str = result_str.replace("**", "^")

        footer = None
        if memory.has_refs(expression):
            # Show what the expression looked like after substitution
            resolved_clean = resolved.replace("**", "^")
            footer = f"After substitution: {resolved_clean}"

        embed = math_embed(
            title="Memory Eval",
            result=result_str,
            footer=footer,
        )
        await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MemoryCog(bot))