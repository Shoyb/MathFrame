"""
main.py — Bot entry point.

Start the bot with:
    python main.py
"""

import logging
from datetime import datetime

import discord
from discord import app_commands
from discord.ext import commands

import config
from data.permissions import is_command_allowed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cogs to load (order matters: earlier cogs can depend on shared utilities)
# ---------------------------------------------------------------------------

COGS: list[str] = [
    "cogs.admin",
    "cogs.arithmetic",
    "cogs.calculus",
    "cogs.transforms",
    "cogs.linear_algebra",
    "cogs.statistics",
    "cogs.number_theory",
    "cogs.geometry",
    "cogs.discrete",
    "cogs.symbolic",
    "cogs.equations",
    "cogs.inequalities",
    "cogs.complex",
    "cogs.base_n",
    "cogs.utility",
    "cogs.render",
    "cogs.plot_engine",
    "cogs.wiki"
]

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True   # required to read message content in prefix commands

bot = commands.Bot(command_prefix=config.PREFIX, intents=intents)

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
        bot.start_time = datetime.utcnow()

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

    # Sync slash commands globally --------------------------------------------
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

@bot.tree.interaction_check
async def _permission_check(interaction: discord.Interaction) -> bool:
    """
    Global pre-invoke check for all slash commands.

    Consults ``data.permissions.is_command_allowed()`` using the guild ID,
    channel ID, and command name.  If the rule denies the command, an
    ephemeral error message is sent and ``False`` is returned so discord.py
    cancels the invocation before the cog handler runs.

    DMs (no guild) are always allowed; the admin system is guild-only.
    """
    if interaction.guild_id is None:
        return True  # DMs always allowed

    command_name = interaction.command.name if interaction.command else None
    if command_name is None:
        return True

    channel_id = interaction.channel_id or 0

    if not is_command_allowed(interaction.guild_id, channel_id, command_name):
        await interaction.response.send_message(
            f"🚫 `/{command_name}` is disabled in this channel. "
            "Ask a server admin to enable it with `/admin enable`.",
            ephemeral=True,
        )
        return False

    return True

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
        interaction.command.name if interaction.command else "unknown",
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