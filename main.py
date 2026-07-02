"""
main.py — Bot entry point.

Start the bot with:
    python main.py
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

import config
from data.permissions import is_command_allowed
from utils.bot_diagnostics import LOG_DIR, LOG_FILE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
#
# Two handlers: the existing console stream (unchanged — still what you see
# when running `python main.py` directly) plus a rotating file so there's
# somewhere for /admin logs to actually read from. Capped at 5 MB x 3 backup
# files (~15 MB max on disk) so a busy bot can't quietly fill the disk.
# LOG_FILE is defined once in utils/bot_diagnostics.py and imported here AND
# by cogs/admin.py, so the writer and the reader can never point at
# different paths.

os.makedirs(LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_log_formatter)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cogs to load (order matters: earlier cogs can depend on shared utilities)
# ---------------------------------------------------------------------------

COGS: list[str] = [
    "cogs.admin",        # /admin  — already a group (no changes needed)
    "cogs.algebra",      # /alg    — merged: arithmetic + equations + inequalities
    "cogs.calculus",     # /calc
    "cogs.transforms",   # /tf
    "cogs.linear_algebra",  # /mat
    "cogs.statistics",   # /stat
    "cogs.number_theory",   # /nt
    "cogs.geometry",     # /geo
    "cogs.discrete",     # /dis
    "cogs.symbolic",     # /sym
    "cogs.complex",      # /cx
    "cogs.base_n",       # /base
    "cogs.random_tools", # /rand   — Phase 1 of Random/Probability/Quiz plan
    "cogs.probability",  # /prob   — Phase 3 of Random/Probability/Quiz plan
    "cogs.quiz",         # /quiz   — Phase 4 of Random/Probability/Quiz plan (solo practice)
    "cogs.memory",       # /mem   — already a group (no changes needed)
    "cogs.bot",          # /bot   — merged: utility + wiki
    "cogs.render",       # /render
    "cogs.plot_engine",  # /plot
    "cogs.csv_tools",   # /csv   — CSV upload + stats + interactive plots
]

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True   # required to read message content in prefix commands

bot = commands.Bot(command_prefix=config.PREFIX, intents=intents)


async def _setup_hook() -> None:
    """
    Runs once, before login completes and before the first on_ready.
    Opens the shared SQLite connection (and runs the one-time JSON
    migration) before any cog can be loaded, since several cogs
    (quiz, admin) touch the database as soon as they're used.
    """
    from data.db import init_db
    await init_db()
    log.info("Database initialized.")


bot.setup_hook = _setup_hook

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready() -> None:
    """Called once the bot has connected and is ready to receive events."""
    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
    print(f"\n[OK] Logged in as {bot.user}  |  id: {bot.user.id}")
    print("-" * 48)

    # Record start time once (on_ready can fire again after a reconnect,
    # but uptime should be measured from the first connection).
    if not hasattr(bot, "start_time"):
        bot.start_time = datetime.now(tz=timezone.utc)

    # Load cogs ---------------------------------------------------------------
    loaded: list[str] = []
    failed: list[str] = []

    for cog in COGS:
        try:
            await bot.load_extension(cog)
            loaded.append(cog)
            log.info("Loaded cog: %s", cog)
        except commands.ExtensionNotFound:
            failed.append(cog)
            log.warning("Cog not found (skipped): %s", cog)
        except commands.ExtensionAlreadyLoaded:
            loaded.append(cog)
            log.warning("Cog already loaded: %s", cog)
        except Exception as exc:                          # noqa: BLE001
            failed.append(cog)
            log.exception("Failed to load cog %s: %s", cog, exc)

    print(f"  Cogs loaded : {len(loaded)}/{len(COGS)}")
    if failed:
        print(f"  [WARN] Skipped  : {', '.join(failed)}")

    # Sync slash commands globally — guarded so reconnects don't re-fire.
    # Discord rate-limits global syncs heavily; one sync per process lifetime
    # is the correct pattern.
    if not getattr(bot, "_commands_synced", False):
        bot._commands_synced = True
        try:
            synced = await bot.tree.sync()
            log.info("Synced %d slash command(s) globally.", len(synced))
            print(f"  Slash cmds  : {len(synced)} synced globally")
        except discord.HTTPException as exc:
            log.error("Failed to sync slash commands: %s", exc)
            print(f"  [WARN] Slash command sync failed: {exc}")

    print("-" * 48 + "\n")

# ---------------------------------------------------------------------------
# Built-in slash commands
# ---------------------------------------------------------------------------

@bot.tree.command(name="ping", description="Check the bot's latency")
async def ping(interaction: discord.Interaction) -> None:
    """Reply with Pong and the current WebSocket latency."""
    latency_ms = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"Pong! `{latency_ms} ms`"
    )

# ---------------------------------------------------------------------------
# Global before_invoke — permission enforcement
# ---------------------------------------------------------------------------

async def _permission_check(interaction: discord.Interaction) -> bool:
    """
    Global pre-invoke check for all slash commands.

    Consults ``data.permissions.is_command_allowed()`` using the guild ID,
    channel ID, and command name. If the rule denies the command, an
    ephemeral error message is sent and ``False`` is returned so discord.py
    cancels the invocation before the cog handler runs.

    DMs (no guild) are always allowed; the admin system is guild-only.

    Uses ``interaction.command.qualified_name`` (e.g. ``"quiz practice"``,
    ``"prob sample"``) rather than ``interaction.command.name`` (just
    ``"practice"``, just ``"sample"``). Several subcommands across
    different groups share a leaf name (``sample`` exists under both
    ``/rand`` and ``/prob``; ``solve``, ``convert``, ``table``, and
    ``clear`` also collide) — checking only the leaf name would make
    ``/admin disable`` on one of those silently disable the other,
    unrelated command too.
    """
    if interaction.guild_id is None:
        return True  # DMs always allowed

    command_name = interaction.command.qualified_name if interaction.command else None
    if command_name is None:
        return True

    channel_id = interaction.channel_id or 0

    if not await is_command_allowed(interaction.guild_id, channel_id, command_name):
        await interaction.response.send_message(
            f"🚫 `/{command_name}` is disabled in this channel. "
            "Ask a server admin to enable it with `/admin enable`.",
            ephemeral=True,
        )
        return False

    return True


# NOTE: this MUST be a direct assignment, not a ``@bot.tree.interaction_check``
# decorator. ``CommandTree.interaction_check`` is documented as an
# overridable method (meant for subclassing), not a registration hook —
# decorating with it just calls the bound method once at import time with
# ``_permission_check`` passed in as the ``interaction`` argument, produces
# an unawaited coroutine, and leaves the tree's default ``interaction_check``
# (which unconditionally returns ``True``) as what actually runs during
# dispatch. That silently made every ``/admin enable`` / ``/admin disable``
# rule a no-op — the permission system was never actually being consulted.
# Direct assignment onto the tree instance is the correct, documented way
# to install a check function that isn't part of a CommandTree subclass.
bot.tree.interaction_check = _permission_check

# ---------------------------------------------------------------------------
# Global app_commands error handler
# ---------------------------------------------------------------------------

async def _ephemeral_reply(interaction: discord.Interaction, message: str) -> None:
    """
    Send an ephemeral reply regardless of whether the interaction has
    already been responded to (e.g. the cog called ``defer()`` first).
    """
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:
    """
    Catch-all handler for slash command errors.

    Handled cases
    -------------
    CommandOnCooldown
        Tells the user how many seconds remain before they can retry.
    MissingPermissions
        Tells the user they lack the required permissions.
    BotMissingPermissions
        Tells the user the bot itself lacks required permissions.
    Everything else
        Shows a generic "something went wrong" message and logs the
        full traceback so it can be investigated.
    """
    if isinstance(error, app_commands.CommandOnCooldown):
        await _ephemeral_reply(
            interaction,
            f"[!] Slow down! This command is on cooldown. "
            f"Try again in **{error.retry_after:.1f}s**.",
        )
        return

    if isinstance(error, app_commands.MissingPermissions):
        await _ephemeral_reply(
            interaction,
            "[X] You don't have permission to use this command.",
        )
        return

    if isinstance(error, app_commands.BotMissingPermissions):
        await _ephemeral_reply(
            interaction,
            "[X] I'm missing permissions to do that.",
        )
        return

    # Unknown / unexpected error - log full traceback -------------------------
    log.exception(
        "Unhandled error in command /%s: %s",
        interaction.command.qualified_name if interaction.command else "unknown",
        error,
    )
    await _ephemeral_reply(
        interaction,
        "[ERROR] Something went wrong while running that command. Please try again.",
    )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        raise RuntimeError(
            "DISCORD_TOKEN is not set. "
            "Copy .env.example → .env and fill in your token."
        )
    bot.run(config.DISCORD_TOKEN, log_handler=None)  # logging already configured above