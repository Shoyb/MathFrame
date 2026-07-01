"""
cogs/random_tools.py — Pure randomness utilities (Phase 1 of the
Random/Probability/Quiz plan; see RANDOM_PROBABILITY_QUIZ_PLAN.md).

Everything in this cog answers "give me one random thing" — a number, a
choice, a roll, a generated matrix/polynomial. It deliberately does NOT
answer "what's the likelihood/distribution of X" — that's the Phase 3
probability cog (`/prob`). See the plan doc's "Random vs Probability"
table for the exact dividing line used to decide where a new command idea
belongs.

Commands
--------
/rand int       min max                          Random integer in [min, max].
/rand float     min max [decimals]                Random float, rounded for display.
/rand choice    options                           Pick one from a comma-separated list.
/rand shuffle   items                             Return a comma-separated list reordered.
/rand sample    items k                           Sample k items without replacement.
/rand dice      notation                          Roll dice, e.g. "2d6+3".
/rand coin      [bias]                            Single weighted coin flip.
/rand matrix    rows cols [min] [max] [integer]   Random matrix.
/rand vector    dim [min] [max]                   Random vector.
/rand poly      degree [min_coeff] [max_coeff] [var]  Random polynomial expression.
/rand prime     min max                           Random prime in range.
/rand token     [length]                          Secure random token (uses `secrets`).
/rand seed      [value]                            Set/clear your reproducible session seed.

Several of these (matrix, vector, poly, prime) exist specifically to feed
other parts of the bot — pasting their output straight into `/mat`, `/alg`,
or `/calc` commands — and will later double as the question generators for
the Phase 4/5 quiz cog.
"""

from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

import sympy

from utils.formatter import math_embed, error_embed
from utils.rng import get_user_rng, secure_token
from utils.dice import parse_dice
from data.rng_seed import seed_store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_list(raw: str, label: str = "items") -> list[str]:
    """
    Split a comma-separated string into a cleaned, non-empty list of items.

    Raises
    ------
    ValueError
        If *raw* contains no usable items after stripping.
    """
    items = [s.strip() for s in raw.split(",")]
    items = [s for s in items if s]
    if not items:
        raise ValueError(f"No usable {label} found — give at least one, comma-separated.")
    return items


def _format_matrix(mat: sympy.Matrix) -> str:
    """Render *mat* as a plain-text grid, one row per line."""
    rows = []
    for i in range(mat.rows):
        entries = "   ".join(str(mat[i, j]) for j in range(mat.cols))
        rows.append(f"[ {entries} ]")
    return "\n".join(rows)


def _format_vector(vec: list) -> str:
    return "[ " + "   ".join(str(v) for v in vec) + " ]"


def _build_poly_string(
    degree: int,
    min_coeff: int,
    max_coeff: int,
    var: str,
    rng,
) -> str:
    """
    Build a random polynomial as a SymPy-parseable string of the given
    *degree* (the leading coefficient is guaranteed nonzero so the result
    is genuinely that degree, not a lower one masquerading as it).

    Uses ``**`` for exponentiation — this codebase's parser does not
    enable ``convert_xor``, so ``^`` would be misread as bitwise XOR.
    """
    if min_coeff > max_coeff:
        raise ValueError("`min_coeff` must be ≤ `max_coeff`.")
    if min_coeff == max_coeff == 0:
        raise ValueError("Coefficient range can't be exactly [0, 0] — every term would vanish.")

    # Leading coefficient: resample until nonzero (range is guaranteed to
    # contain at least one nonzero value by the check above).
    leading = 0
    while leading == 0:
        leading = rng.randint(min_coeff, max_coeff)

    terms = [_format_term(leading, degree, var)]
    for power in range(degree - 1, -1, -1):
        coeff = rng.randint(min_coeff, max_coeff)
        if coeff == 0:
            continue
        terms.append(_format_term(coeff, power, var))

    return " + ".join(terms).replace("+ -", "- ")


def _format_term(coeff: int, power: int, var: str) -> str:
    if power == 0:
        return str(coeff)
    var_part = var if power == 1 else f"{var}**{power}"
    if coeff == 1:
        return var_part
    if coeff == -1:
        return f"-{var_part}"
    return f"{coeff}*{var_part}"


_MAX_DICE_TO_LIST = 50  # above this, show only the total (avoid a spammy embed)

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class RandomToolsCog(commands.Cog, name="Random"):
    """Randomness utilities — dice, shuffles, generators."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    rand = app_commands.Group(name="rand", description="Randomness utilities.")

    # -----------------------------------------------------------------------
    # /rand int
    # -----------------------------------------------------------------------

    @rand.command(name="int", description="Random integer in [min, max].")
    @app_commands.describe(min="Lower bound (inclusive)", max="Upper bound (inclusive)")
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_int(self, interaction: discord.Interaction, min: int, max: int) -> None:
        await interaction.response.defer()
        try:
            if min > max:
                raise ValueError("`min` must be ≤ `max`.")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            result = rng.randint(min, max)
            embed = math_embed(
                title="Random Integer",
                result=str(result),
                footer=f"Range: [{min}, {max}]",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand float
    # -----------------------------------------------------------------------

    @rand.command(name="float", description="Random float in [min, max].")
    @app_commands.describe(
        min="Lower bound", max="Upper bound", decimals="Decimal places to display (0-10, default 2)"
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_float(
        self,
        interaction: discord.Interaction,
        min: float,
        max: float,
        decimals: app_commands.Range[int, 0, 10] = 2,
    ) -> None:
        await interaction.response.defer()
        try:
            if min > max:
                raise ValueError("`min` must be ≤ `max`.")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            result = rng.uniform(min, max)
            embed = math_embed(
                title="Random Float",
                result=f"{result:.{decimals}f}",
                footer=f"Range: [{min}, {max}]",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand choice
    # -----------------------------------------------------------------------

    @rand.command(name="choice", description="Pick one item from a comma-separated list.")
    @app_commands.describe(options='Comma-separated options, e.g. "pizza, sushi, tacos"')
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_choice(self, interaction: discord.Interaction, options: str) -> None:
        await interaction.response.defer()
        try:
            items = _parse_list(options, "options")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            result = rng.choice(items)
            embed = math_embed(
                title="Random Choice",
                result=result,
                footer=f"Chosen from {len(items)} option(s)",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand shuffle
    # -----------------------------------------------------------------------

    @rand.command(name="shuffle", description="Shuffle a comma-separated list.")
    @app_commands.describe(items='Comma-separated items, e.g. "a, b, c, d"')
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_shuffle(self, interaction: discord.Interaction, items: str) -> None:
        await interaction.response.defer()
        try:
            parsed = _parse_list(items, "items")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            shuffled = parsed.copy()
            rng.shuffle(shuffled)
            embed = math_embed(
                title="Shuffled",
                result=", ".join(shuffled),
                footer=f"{len(shuffled)} item(s)",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand sample
    # -----------------------------------------------------------------------

    @rand.command(name="sample", description="Sample k items from a list, without replacement.")
    @app_commands.describe(
        items='Comma-separated items, e.g. "a, b, c, d, e"', k="How many to sample"
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_sample(self, interaction: discord.Interaction, items: str, k: int) -> None:
        await interaction.response.defer()
        try:
            parsed = _parse_list(items, "items")
            if not (1 <= k <= len(parsed)):
                raise ValueError(
                    f"`k` must be between 1 and {len(parsed)} (the number of items given)."
                )
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            result = rng.sample(parsed, k)
            embed = math_embed(
                title="Random Sample",
                result=", ".join(result),
                footer=f"{k} of {len(parsed)} item(s), without replacement",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand dice
    # -----------------------------------------------------------------------

    @rand.command(name="dice", description='Roll dice, e.g. "2d6+3" or "d20".')
    @app_commands.describe(notation='Dice notation, e.g. "2d6", "d20", "3d6+2"')
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_dice(self, interaction: discord.Interaction, notation: str) -> None:
        await interaction.response.defer()
        try:
            spec = parse_dice(notation)
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            rolls = [rng.randint(1, spec.sides) for _ in range(spec.count)]
            total = sum(rolls) + spec.modifier

            if spec.count <= _MAX_DICE_TO_LIST:
                modifier_str = (
                    f"  {'+' if spec.modifier >= 0 else '-'}  {abs(spec.modifier)}"
                    if spec.modifier
                    else ""
                )
                breakdown = f"[{', '.join(str(r) for r in rolls)}]{modifier_str}"
                footer = f"{notation}  |  range [{spec.min_total}, {spec.max_total}]  |  rolls: {breakdown}"
            else:
                footer = f"{notation}  |  range [{spec.min_total}, {spec.max_total}]  |  ({spec.count} dice — individual rolls omitted)"

            embed = math_embed(title="Dice Roll", result=str(total), footer=footer)
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand coin
    # -----------------------------------------------------------------------

    @rand.command(name="coin", description="Flip a coin (optionally weighted).")
    @app_commands.describe(bias="Probability of heads, 0-1 (default 0.5, fair coin)")
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_coin(
        self, interaction: discord.Interaction, bias: app_commands.Range[float, 0.0, 1.0] = 0.5
    ) -> None:
        await interaction.response.defer()
        rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
        result = "Heads" if rng.random() < bias else "Tails"
        embed = math_embed(
            title="Coin Flip",
            result=result,
            footer=f"P(heads) = {bias:g}" if bias != 0.5 else "Fair coin",
        )
        await interaction.followup.send(embed=embed)

    # -----------------------------------------------------------------------
    # /rand matrix
    # -----------------------------------------------------------------------

    @rand.command(name="matrix", description="Generate a random matrix.")
    @app_commands.describe(
        rows="Number of rows (1-10)",
        cols="Number of columns (1-10)",
        min="Minimum entry value (default -9)",
        max="Maximum entry value (default 9)",
        integer="Integer entries (default True); False gives 2-decimal floats",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def rand_matrix(
        self,
        interaction: discord.Interaction,
        rows: app_commands.Range[int, 1, 10],
        cols: app_commands.Range[int, 1, 10],
        min: int = -9,
        max: int = 9,
        integer: bool = True,
    ) -> None:
        await interaction.response.defer()
        try:
            if min > max:
                raise ValueError("`min` must be ≤ `max`.")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)

            if integer:
                entries = [rng.randint(min, max) for _ in range(rows * cols)]
            else:
                entries = [round(rng.uniform(min, max), 2) for _ in range(rows * cols)]

            mat = sympy.Matrix(rows, cols, entries)
            embed = math_embed(
                title="Random Matrix",
                result=_format_matrix(mat),
                footer=f"{rows}×{cols}  |  entries in [{min}, {max}]"
                + ("" if integer else "  |  float"),
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand vector
    # -----------------------------------------------------------------------

    @rand.command(name="vector", description="Generate a random vector.")
    @app_commands.describe(
        dim="Dimension (1-20)", min="Minimum entry value (default -9)", max="Maximum entry value (default 9)"
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_vector(
        self,
        interaction: discord.Interaction,
        dim: app_commands.Range[int, 1, 20],
        min: int = -9,
        max: int = 9,
    ) -> None:
        await interaction.response.defer()
        try:
            if min > max:
                raise ValueError("`min` must be ≤ `max`.")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            entries = [rng.randint(min, max) for _ in range(dim)]
            embed = math_embed(
                title="Random Vector",
                result=_format_vector(entries),
                footer=f"dim={dim}  |  entries in [{min}, {max}]",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand poly
    # -----------------------------------------------------------------------

    @rand.command(name="poly", description="Generate a random polynomial expression.")
    @app_commands.describe(
        degree="Degree of the polynomial (0-10)",
        min_coeff="Minimum coefficient (default -9)",
        max_coeff="Maximum coefficient (default 9)",
        var="Variable name (default x)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_poly(
        self,
        interaction: discord.Interaction,
        degree: app_commands.Range[int, 0, 10],
        min_coeff: int = -9,
        max_coeff: int = 9,
        var: str = "x",
    ) -> None:
        await interaction.response.defer()
        try:
            if not var.isidentifier():
                raise ValueError(f"`{var}` isn't a valid variable name.")
            rng = get_user_rng(interaction.guild_id or 0, interaction.user.id)
            poly_str = _build_poly_string(degree, min_coeff, max_coeff, var, rng)
            embed = math_embed(
                title="Random Polynomial",
                result=poly_str,
                footer=f"degree {degree}  |  coefficients in [{min_coeff}, {max_coeff}]  |  "
                "paste this straight into /alg or /calc commands",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand prime
    # -----------------------------------------------------------------------

    @rand.command(name="prime", description="Random prime number in [min, max].")
    @app_commands.describe(min="Lower bound (inclusive)", max="Upper bound (exclusive, per SymPy's randprime)")
    @app_commands.checks.cooldown(1, 3.0)
    async def rand_prime(self, interaction: discord.Interaction, min: int, max: int) -> None:
        await interaction.response.defer()
        try:
            if min < 2:
                raise ValueError("`min` must be at least 2 (2 is the smallest prime).")
            if min >= max:
                raise ValueError("`min` must be < `max`.")
            if max > 10**9:
                raise ValueError("`max` is capped at 10^9 to keep this fast.")
            result = sympy.randprime(min, max)
            embed = math_embed(
                title="Random Prime",
                result=str(result),
                footer=f"Range: [{min}, {max})",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rand token
    # -----------------------------------------------------------------------

    @rand.command(name="token", description="Generate a secure random token.")
    @app_commands.describe(length="Approximate token length in characters (4-128, default 16)")
    @app_commands.checks.cooldown(1, 3.0)
    async def rand_token(
        self, interaction: discord.Interaction, length: app_commands.Range[int, 4, 128] = 16
    ) -> None:
        # Ephemeral — a generated token/password shouldn't sit in plain
        # channel history.
        await interaction.response.defer(ephemeral=True)
        # secrets.token_urlsafe's `nbytes` param isn't a 1:1 character
        # count (base64 expands ~1.3x), so scale down to land close to
        # the requested display length.
        token = secure_token(max(3, round(length * 0.75)))
        embed = math_embed(
            title="Secure Token",
            result=token,
            footer="Generated with `secrets` (cryptographically secure)  |  only visible to you",
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    # -----------------------------------------------------------------------
    # /rand seed
    # -----------------------------------------------------------------------

    @rand.command(
        name="seed",
        description="Set or clear your reproducible randomness seed for this session.",
    )
    @app_commands.describe(value="Seed to set; omit to clear your current seed")
    @app_commands.checks.cooldown(1, 2.0)
    async def rand_seed(
        self, interaction: discord.Interaction, value: int | None = None
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        guild_id = interaction.guild_id or 0
        user_id = interaction.user.id

        if value is None:
            cleared = seed_store.clear(guild_id, user_id)
            msg = "Seed cleared — back to fresh randomness." if cleared else "You didn't have a seed set."
            embed = math_embed(title="Random Seed", result=msg)
        else:
            seed_store.set(guild_id, user_id, value)
            embed = math_embed(
                title="Random Seed",
                result=f"Seed set to {value}",
                footer="All /rand commands you run this session will now be reproducible. "
                "Run /rand seed with no value to clear it.",
            )
        await interaction.followup.send(embed=embed, ephemeral=True)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    """Load the RandomToolsCog into *bot*."""
    await bot.add_cog(RandomToolsCog(bot))
