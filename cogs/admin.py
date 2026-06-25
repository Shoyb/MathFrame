"""
cogs/admin.py — Server admin commands for MathFrame.

Commands
--------
/admin enable  [command] [channel]   Allow a command in a channel (or server-wide).
/admin disable [command] [channel]   Block a command in a channel (or server-wide).
/admin reset   [command] [channel]   Remove an existing permission rule.
/admin status                        Show all active rules for this server.

All commands require the ``Manage Guild`` permission.  Admins can target
specific channels and commands, or leave either blank to apply the rule
server-wide / to all commands respectively.
"""

from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from data.permissions import (
    set_permission,
    clear_permission,
    get_guild_status,
)

# ---------------------------------------------------------------------------
# Group definition
# ---------------------------------------------------------------------------

class AdminCog(commands.Cog, name="Admin"):
    """Server administration commands for managing command permissions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------
    # Shared permission check
    # ------------------------------------------------------------------

    @staticmethod
    def _require_manage_guild(interaction: discord.Interaction) -> bool:
        """Return True only if the caller has Manage Guild permission."""
        if interaction.guild is None:
            return False
        member = interaction.user
        if isinstance(member, discord.Member):
            return member.guild_permissions.manage_guild
        return False

    # ------------------------------------------------------------------
    # /admin group
    # ------------------------------------------------------------------

    admin_group = app_commands.Group(
        name="admin",
        description="Manage per-channel and server-wide command permissions.",
        guild_only=True,
    )

    # ------------------------------------------------------------------
    # /admin enable
    # ------------------------------------------------------------------

    @admin_group.command(
        name="enable",
        description="Allow a command in a specific channel or server-wide.",
    )
    @app_commands.describe(
        command="Command name to enable (without /). Leave blank to enable all commands.",
        channel="Channel to target. Leave blank to apply server-wide.",
    )
    async def admin_enable(
        self,
        interaction: discord.Interaction,
        command: str = "",
        channel: discord.TextChannel | None = None,
    ) -> None:
        """Enable *command* in *channel* (or everywhere if channel is omitted)."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        cmd_name    = command.strip().lstrip("/") or None
        channel_id  = channel.id if channel else None
        guild_id    = interaction.guild_id

        set_permission(guild_id, channel_id, cmd_name, enabled=True)

        cmd_label     = f"`/{cmd_name}`" if cmd_name else "**all commands**"
        channel_label = channel.mention if channel else "**all channels**"
        await interaction.response.send_message(
            f"✅ Enabled {cmd_label} in {channel_label}.",
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # /admin disable
    # ------------------------------------------------------------------

    @admin_group.command(
        name="disable",
        description="Block a command in a specific channel or server-wide.",
    )
    @app_commands.describe(
        command="Command name to disable (without /). Leave blank to disable all commands.",
        channel="Channel to target. Leave blank to apply server-wide.",
    )
    async def admin_disable(
        self,
        interaction: discord.Interaction,
        command: str = "",
        channel: discord.TextChannel | None = None,
    ) -> None:
        """Disable *command* in *channel* (or everywhere if channel is omitted)."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        cmd_name    = command.strip().lstrip("/") or None
        channel_id  = channel.id if channel else None
        guild_id    = interaction.guild_id

        set_permission(guild_id, channel_id, cmd_name, enabled=False)

        cmd_label     = f"`/{cmd_name}`" if cmd_name else "**all commands**"
        channel_label = channel.mention if channel else "**all channels**"
        await interaction.response.send_message(
            f"🚫 Disabled {cmd_label} in {channel_label}.",
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # /admin reset
    # ------------------------------------------------------------------

    @admin_group.command(
        name="reset",
        description="Remove an existing permission rule (restore default behaviour).",
    )
    @app_commands.describe(
        command="Command name to reset (without /). Leave blank to reset the all-commands rule.",
        channel="Channel to reset for. Leave blank to reset the server-wide rule.",
    )
    async def admin_reset(
        self,
        interaction: discord.Interaction,
        command: str = "",
        channel: discord.TextChannel | None = None,
    ) -> None:
        """Remove the permission rule for *command* in *channel*."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        cmd_name   = command.strip().lstrip("/") or None
        channel_id = channel.id if channel else None
        guild_id   = interaction.guild_id

        removed = clear_permission(guild_id, channel_id, cmd_name)

        cmd_label     = f"`/{cmd_name}`" if cmd_name else "all-commands rule"
        channel_label = channel.mention if channel else "server-wide"

        if removed:
            await interaction.response.send_message(
                f"🔄 Reset the {cmd_label} rule for {channel_label}.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"ℹ️ No rule found for {cmd_label} / {channel_label} — nothing to reset.",
                ephemeral=True,
            )

    # ------------------------------------------------------------------
    # /admin status
    # ------------------------------------------------------------------

    @admin_group.command(
        name="status",
        description="List all active command permission rules for this server.",
    )
    async def admin_status(
        self,
        interaction: discord.Interaction,
    ) -> None:
        """Show all permission rules configured for this guild."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        rows = get_guild_status(interaction.guild_id)

        if not rows:
            await interaction.response.send_message(
                "ℹ️ No permission rules are configured for this server. "
                "All commands are currently **allowed everywhere** (default).",
                ephemeral=True,
            )
            return

        lines: list[str] = []
        for row in rows:
            icon = "✅" if row["enabled"] else "🚫"
            lines.append(f"{icon} **{row['command']}** — {row['channel']}")

        embed = discord.Embed(
            title="📋 Command Permission Rules",
            description="\n".join(lines),
            colour=discord.Colour.blurple(),
        )
        embed.set_footer(text=f"{len(rows)} rule(s) active  |  Use /admin enable/disable to change")
        await interaction.response.send_message(embed=embed, ephemeral=True)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the AdminCog into *bot*."""
    cog = AdminCog(bot)
    await bot.add_cog(cog)
