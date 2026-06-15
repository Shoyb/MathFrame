from discord.ext import commands
import discord

class LinearAlgebraCog(commands.Cog, name="Linear Algebra"):
    """Placeholder cog for matrix operations and vector math."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

async def setup(bot: commands.Bot) -> None:
    """Load the LinearAlgebraCog into *bot*."""
    await bot.add_cog(LinearAlgebraCog(bot))