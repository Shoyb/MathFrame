"""
cogs/admin.py — Server admin commands for MathFrame.

Commands
--------
/admin enable        [command] [channel]   Allow a command in a channel (or server-wide).
/admin disable       [command] [channel]   Block a command in a channel (or server-wide).
/admin reset         [command] [channel]   Remove an existing permission rule.
/admin status                              Show all active rules for this server.
/admin panic                               EMERGENCY: instantly disable every command, everywhere.
/admin unpanic                             Restore whatever rules were in place before /admin panic.
/admin list_commands [group]               Browse every registered command (or one group's subcommands).
/admin diagnostics                         Uptime, CPU/memory/disk usage, latency, guild/command counts.
/admin logs          [lines]               Tail the bot's log file (bot owner only — spans all servers).

All commands require the ``Manage Guild`` permission. Admins can target
specific channels and commands, or leave either blank to apply the rule
server-wide / to all commands respectively.

Command-name matching (important)
-----------------------------------
``command`` is matched against a slash command's *qualified* name — e.g.
``quiz practice``, ``prob sample``, ``rand sample`` — not just its last
word. Several subcommands across different groups share a short name
(``sample`` exists under both ``/rand`` and ``/prob``; ``solve``,
``convert``, ``table``, and ``clear`` also collide), so matching on the
short name alone would make disabling one silently disable the other.
The ``command`` parameter has autocomplete — start typing and Discord
will suggest real, fully-qualified command names, so there's no need to
remember the exact spelling.

Scoping note: /admin diagnostics vs. /admin logs
----------------------------------------------------
Most ``/admin`` commands only need the ``Manage Guild`` permission
because they only affect (and only reveal) *this* server's own data.
``/admin diagnostics`` (uptime, CPU/memory/disk, latency) follows that
same rule — it's general bot-health info, comparable to what a public
"bot stats" command on many Discord bots already shows.

``/admin logs`` is different: the bot process is shared across every
server it's in, so its log file can contain guild IDs, user IDs, and
error details from *other* servers, not just this one. Showing that to
any server's ``Manage Guild`` admin would leak other servers' activity.
It's gated to the bot's owner (``bot.is_owner()``) instead.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

from data.permissions import (
    set_permission,
    clear_permission,
    get_guild_status,
    panic_lock,
    panic_unlock,
)
from utils.bot_diagnostics import (
    get_process_stats,
    get_data_dir_size_mb,
    tail_log,
    format_uptime,
)

# ---------------------------------------------------------------------------
# Autocomplete helpers
# ---------------------------------------------------------------------------


def _all_qualified_command_names(interaction: discord.Interaction) -> list[str]:
    """
    Every real, invokable (leaf) slash command's qualified name, e.g.
    ``["ping", "quiz practice", "prob sample", "rand sample", ...]``.

    ``CommandTree.walk_commands()`` yields ``Group`` objects too (e.g. a
    bare ``"quiz"`` entry for the group itself) — those are filtered out
    since a bare group is never itself the thing checked against the
    permission system at runtime; only leaf commands are.
    """
    tree = interaction.client.tree
    return sorted(
        {
            cmd.qualified_name
            for cmd in tree.walk_commands()
            if isinstance(cmd, app_commands.Command)
        }
    )


async def _command_name_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Suggest real, fully-qualified command names as the admin types."""
    names = _all_qualified_command_names(interaction)
    typed = current.strip().lstrip("/").lower()

    choices: list[app_commands.Choice[str]] = []
    if not typed:
        choices.append(app_commands.Choice(name="(all commands)", value=""))

    matches = [n for n in names if typed in n.lower()] if typed else names
    choices.extend(app_commands.Choice(name=n, value=n) for n in matches[: 25 - len(choices)])
    return choices[:25]


async def _group_name_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Suggest top-level command group names for /admin list_commands."""
    names = _all_qualified_command_names(interaction)
    groups = sorted({n.split(" ", 1)[0] for n in names})
    typed = current.strip().lstrip("/").lower()
    matches = [g for g in groups if typed in g.lower()] if typed else groups
    return [app_commands.Choice(name=g, value=g) for g in matches[:25]]


# ---------------------------------------------------------------------------
# Cog
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
        command="Command to enable — start typing for suggestions. Leave blank for all commands.",
        channel="Channel to target. Leave blank to apply server-wide.",
    )
    @app_commands.autocomplete(command=_command_name_autocomplete)
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

        await set_permission(guild_id, channel_id, cmd_name, enabled=True)

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
        command="Command to disable — start typing for suggestions. Leave blank for all commands.",
        channel="Channel to target. Leave blank to apply server-wide.",
    )
    @app_commands.autocomplete(command=_command_name_autocomplete)
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

        await set_permission(guild_id, channel_id, cmd_name, enabled=False)

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
        command="Command to reset — start typing for suggestions. Leave blank to reset the all-commands rule.",
        channel="Channel to reset for. Leave blank to reset the server-wide rule.",
    )
    @app_commands.autocomplete(command=_command_name_autocomplete)
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

        removed = await clear_permission(guild_id, channel_id, cmd_name)

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

        rows = await get_guild_status(interaction.guild_id)

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

    # ------------------------------------------------------------------
    # /admin panic / /admin unpanic
    # ------------------------------------------------------------------

    @admin_group.command(
        name="panic",
        description="EMERGENCY: instantly disable every command in every channel (e.g. during a raid/spam incident).",
    )
    async def admin_panic(self, interaction: discord.Interaction) -> None:
        """Lock the whole server down immediately; previous rules are saved for /admin unpanic."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        await panic_lock(interaction.guild_id)
        await interaction.response.send_message(
            "🚨 **Panic lockdown activated.** Every command is now disabled in every channel.\n"
            "Run `/admin unpanic` to restore whatever rules were in place before this.",
        )

    @admin_group.command(
        name="unpanic",
        description="Restore command permissions to how they were before the last /admin panic.",
    )
    async def admin_unpanic(self, interaction: discord.Interaction) -> None:
        """Undo the most recent /admin panic, restoring the prior ruleset exactly."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        restored = await panic_unlock(interaction.guild_id)
        if restored:
            message = "✅ Panic lockdown lifted — previous permission rules have been restored."
        else:
            message = (
                "✅ Panic lockdown lifted. There was no prior ruleset on record, so all "
                "commands are now allowed everywhere (default)."
            )
        await interaction.response.send_message(message)

    # ------------------------------------------------------------------
    # /admin list_commands
    # ------------------------------------------------------------------

    @admin_group.command(
        name="list_commands",
        description="Browse every registered slash command, or one group's subcommands.",
    )
    @app_commands.describe(
        group="Top-level command group to list, e.g. 'quiz'. Leave blank to see all groups.",
    )
    @app_commands.autocomplete(group=_group_name_autocomplete)
    async def admin_list_commands(
        self,
        interaction: discord.Interaction,
        group: str = "",
    ) -> None:
        """List command groups, or every subcommand under one group, by qualified name."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        names = _all_qualified_command_names(interaction)
        group_key = group.strip().lstrip("/").lower()

        if not group_key:
            counts: dict[str, int] = {}
            for n in names:
                top = n.split(" ", 1)[0]
                counts[top] = counts.get(top, 0) + 1
            lines = [f"**/{g}** — {c} command(s)" for g, c in sorted(counts.items())]
            embed = discord.Embed(
                title="📋 Registered Command Groups",
                description="\n".join(lines) or "No commands registered.",
                colour=discord.Colour.blurple(),
            )
            embed.set_footer(
                text=f"{len(names)} commands total  |  /admin list_commands group:<name> for details"
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        matched = [n for n in names if n == group_key or n.startswith(group_key + " ")]
        if not matched:
            await interaction.response.send_message(
                f"ℹ️ No commands found under `/{group_key}`. Try `/admin list_commands` with no "
                "argument to see available groups.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f"📋 Commands under /{group_key}",
            description="\n".join(f"`/{n}`" for n in matched),
            colour=discord.Colour.blurple(),
        )
        embed.set_footer(text=f"{len(matched)} command(s)  |  use the full name with /admin enable/disable")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------
    # /admin diagnostics
    # ------------------------------------------------------------------

    @admin_group.command(
        name="diagnostics",
        description="Bot health: uptime, CPU/memory/disk usage, latency, guild and command counts.",
    )
    async def admin_diagnostics(self, interaction: discord.Interaction) -> None:
        """Show process- and system-level resource usage alongside general bot health stats."""
        if not self._require_manage_guild(interaction):
            await interaction.response.send_message(
                "❌ You need the **Manage Guild** permission to use this command.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)

        stats = get_process_stats()
        data_mb = get_data_dir_size_mb()

        start_time: datetime | None = getattr(self.bot, "start_time", None)
        uptime_str = (
            format_uptime((datetime.now(tz=timezone.utc) - start_time).total_seconds())
            if start_time
            else "Unknown"
        )

        embed = discord.Embed(title="🩺 Bot Diagnostics", colour=discord.Colour.blurple())

        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(name="Latency", value=f"{round(self.bot.latency * 1000)} ms", inline=True)
        embed.add_field(name="Servers", value=str(len(self.bot.guilds)), inline=True)

        embed.add_field(
            name="Process (this bot)",
            value=(
                f"CPU: {stats['process_cpu_percent']:.1f}%\n"
                f"Memory: {stats['process_memory_mb']:.1f} MB ({stats['process_memory_percent']:.1f}%)\n"
                f"Threads: {stats['thread_count']}"
            ),
            inline=True,
        )
        embed.add_field(
            name="System (host machine)",
            value=(
                f"CPU: {stats['system_cpu_percent']:.1f}%\n"
                f"Memory: {stats['system_memory_used_gb']:.1f} / {stats['system_memory_total_gb']:.1f} GB "
                f"({stats['system_memory_percent']:.1f}%)\n"
                f"Disk: {stats['disk_used_gb']:.1f} / {stats['disk_total_gb']:.1f} GB ({stats['disk_percent']:.1f}%)"
            ),
            inline=True,
        )
        embed.add_field(
            name="Data storage",
            value=f"{data_mb:.2f} MB  (JSON stores + log file, in data/)",
            inline=True,
        )

        embed.set_footer(text="Process = just this bot  |  System = the whole host machine it runs on")
        await interaction.followup.send(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------
    # /admin logs
    # ------------------------------------------------------------------

    @admin_group.command(
        name="logs",
        description="Tail the bot's log file (bot owner only — logs span every server the bot is in).",
    )
    @app_commands.describe(lines="How many recent log lines to show (1-500, default 50).")
    async def admin_logs(
        self,
        interaction: discord.Interaction,
        lines: app_commands.Range[int, 1, 500] = 50,
    ) -> None:
        """Send the last *lines* lines of the bot's rotating log file as a text attachment."""
        if not await self.bot.is_owner(interaction.user):
            await interaction.response.send_message(
                "❌ This command is restricted to the bot owner — the log file can contain "
                "activity from every server the bot is in, not just this one.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)

        tail = tail_log(lines)
        if not tail:
            await interaction.followup.send(
                "ℹ️ No log entries found yet (the log file may not have been created yet, "
                "or the bot was just restarted).",
                ephemeral=True,
            )
            return

        content = "\n".join(tail)
        buf = io.BytesIO(content.encode("utf-8"))
        file = discord.File(buf, filename="bot_recent.log")
        await interaction.followup.send(
            content=f"🧾 Last {len(tail)} log line(s):",
            file=file,
            ephemeral=True,
        )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the AdminCog into *bot*."""
    cog = AdminCog(bot)
    await bot.add_cog(cog)
