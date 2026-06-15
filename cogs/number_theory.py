"""
cogs/number_theory.py — Number theory slash commands for the math bot.

Commands
--------
/gcd            numbers             GCD of a list of integers.
/lcm            numbers             LCM of a list of integers.
/is_prime       n                   Primality test with factorisation fallback.
/factorize      n                   Prime factorisation in superscript notation.
/primes_up_to   n                   All primes up to n (capped, paginated).
/modular        base  exp  mod      Modular exponentiation.
/fibonacci      n                   First n Fibonacci numbers (paginated if large).
"""

import math
from typing import Final

import discord
import sympy
from discord import app_commands
from discord.ext import commands

from utils.formatter import error_embed, math_embed
from utils.paginator import send_paginated

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRIME_CAP:       Final[int] = 10 ** 12
_FACTORIZE_CAP:   Final[int] = 10 ** 15
_PRIMES_UPTO_CAP: Final[int] = 10_000
_PRIMES_SHOW_MAX: Final[int] = 50
_FIB_CAP:         Final[int] = 200
_FIB_PAGE_SIZE:   Final[int] = 20

# Unicode superscript digits 0-9
_SUPERSCRIPTS: dict[str, str] = str.maketrans(
    "0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _superscript(n: int) -> str:
    """Convert an integer to its Unicode superscript string, e.g. 3 → '³'."""
    return str(n).translate(_SUPERSCRIPTS)


def _parse_integers(s: str, param_name: str = "numbers") -> list[int]:
    """
    Parse a comma-separated string of integers.

    Parameters
    ----------
    s:
        Raw user input, e.g. ``"12, 18, 24"``.
    param_name:
        Label used in error messages.

    Returns
    -------
    list[int]

    Raises
    ------
    ValueError
        If the string is empty, any token is not a valid integer (including
        floats), or fewer than 2 values are provided.
    """
    if not s.strip():
        raise ValueError(f"`{param_name}` is empty. Provide comma-separated integers.")

    results: list[int] = []
    for i, token in enumerate(s.split(",")):
        token = token.strip()
        if not token:
            raise ValueError(
                f"Empty value at position {i + 1} in `{param_name}`. "
                "Check for trailing or double commas."
            )
        # Reject floats explicitly so "1.5" doesn't silently truncate to 1
        if "." in token:
            raise ValueError(
                f"`{token}` (position {i + 1}) is a decimal — only integers are accepted."
            )
        try:
            results.append(int(token))
        except ValueError:
            raise ValueError(
                f"Cannot convert `{token}` (position {i + 1}) to an integer."
            )

    if not results:
        raise ValueError(f"No integers found in `{param_name}`.")
    if len(results) < 2:
        raise ValueError(f"At least 2 integers are required (got {len(results)}).")
    return results


def _validate_int_arg(value: int, name: str, lo: int | None = None, hi: int | None = None) -> None:
    """
    Raise :class:`ValueError` if *value* is outside [*lo*, *hi*].

    Parameters
    ----------
    value:
        The integer to validate.
    name:
        Parameter name shown in the error message.
    lo, hi:
        Optional inclusive bounds.
    """
    if lo is not None and value < lo:
        raise ValueError(f"`{name}` must be ≥ {lo:,} (got {value:,}).")
    if hi is not None and value > hi:
        raise ValueError(f"`{name}` must be ≤ {hi:,} (got {value:,}).")


def _list_gcd(nums: list[int]) -> int:
    """Return the GCD of all integers in *nums* using iterative reduction."""
    result = nums[0]
    for n in nums[1:]:
        result = math.gcd(result, n)
    return result


def _list_lcm(nums: list[int]) -> int:
    """Return the LCM of all integers in *nums* using iterative reduction."""
    def _lcm2(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    result = nums[0]
    for n in nums[1:]:
        result = _lcm2(result, n)
    return result


def _format_factorisation(factors: dict[int, int]) -> str:
    """
    Format a ``{prime: exponent}`` dict as ``2³ × 3² × 5¹``.

    Exponents of 1 are shown explicitly for consistency.
    """
    parts = [
        f"{p}{_superscript(e)}"
        for p, e in sorted(factors.items())
    ]
    return " × ".join(parts)


def _fibonacci_list(n: int) -> list[int]:
    """Return the first *n* Fibonacci numbers (1-indexed: F(1)=1, F(2)=1, …)."""
    if n == 1:
        return [1]
    fibs = [1, 1]
    for _ in range(n - 2):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def _paginate_list(
    items: list[str],
    page_size: int,
    title: str,
    footer_prefix: str = "",
) -> list[discord.Embed]:
    """
    Split *items* into pages of *page_size* and build one embed per page.

    Parameters
    ----------
    items:
        Pre-formatted string items to display.
    page_size:
        Maximum items per embed.
    title:
        Embed title (shared across all pages).
    footer_prefix:
        Optional text prepended to the page-number footer.

    Returns
    -------
    list[discord.Embed]
    """
    pages: list[discord.Embed] = []
    chunks = [items[i : i + page_size] for i in range(0, len(items), page_size)]
    total_pages = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        embed = discord.Embed(
            title=title,
            description=", ".join(chunk),
            colour=discord.Colour.blurple(),
        )
        footer = f"{footer_prefix}  |  " if footer_prefix else ""
        footer += f"Page {idx}/{total_pages}"
        embed.set_footer(text=footer)
        pages.append(embed)

    return pages


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class NumberTheoryCog(commands.Cog, name="Number Theory"):
    """GCD, LCM, primality, factorisation, modular arithmetic, and Fibonacci."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /gcd
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="gcd",
        description="Compute the Greatest Common Divisor of a list of integers.",
    )
    @app_commands.describe(numbers='Comma-separated integers, e.g. "12, 18, 24"')
    @app_commands.checks.cooldown(1, 2.0)
    async def gcd(self, interaction: discord.Interaction, numbers: str) -> None:
        await interaction.response.defer()
        try:
            nums   = _parse_integers(numbers, "numbers")
            result = _list_gcd(nums)
            embed  = math_embed(
                title="Greatest Common Divisor",
                result=str(result),
                steps=[
                    ("Inputs",   ", ".join(str(n) for n in nums)),
                    ("GCD",      f"gcd({', '.join(str(n) for n in nums)}) = {result}"),
                ],
                footer=f"{len(nums)} integers  |  iterative gcd reduction",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /lcm
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="lcm",
        description="Compute the Least Common Multiple of a list of integers.",
    )
    @app_commands.describe(numbers='Comma-separated integers, e.g. "4, 6, 10"')
    @app_commands.checks.cooldown(1, 2.0)
    async def lcm(self, interaction: discord.Interaction, numbers: str) -> None:
        await interaction.response.defer()
        try:
            nums   = _parse_integers(numbers, "numbers")
            result = _list_lcm(nums)
            embed  = math_embed(
                title="Least Common Multiple",
                result=str(result),
                steps=[
                    ("Inputs",  ", ".join(str(n) for n in nums)),
                    ("Formula", "lcm(a, b) = |a × b| / gcd(a, b), extended iteratively"),
                    ("LCM",     f"lcm({', '.join(str(n) for n in nums)}) = {result}"),
                ],
                footer=f"{len(nums)} integers  |  iterative lcm reduction",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /is_prime
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="is_prime",
        description=f"Test whether an integer is prime (n ≤ 10¹²).",
    )
    @app_commands.describe(n="The integer to test")
    @app_commands.checks.cooldown(1, 2.0)
    async def is_prime(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=2, hi=_PRIME_CAP)
            prime = sympy.isprime(n)

            if prime:
                result_str = f"✅  Yes, {n:,} is prime."
                steps = [
                    ("Input",  str(n)),
                    ("Result", f"{n:,} has no divisors other than 1 and itself → prime"),
                ]
                footer = "Tested with SymPy's deterministic Miller-Rabin algorithm"
            else:
                factors = sympy.factorint(n)
                factors_str = _format_factorisation(factors)
                result_str = f"❌  No, {n:,} is not prime."
                steps = [
                    ("Input",           str(n)),
                    ("Factorisation",   factors_str),
                    ("Conclusion",      f"{n:,} = {factors_str}  →  composite"),
                ]
                footer = "Prime factorisation shown above"

            embed = math_embed(
                title="Primality Test",
                result=result_str,
                steps=steps,
                footer=footer,
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /factorize
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="factorize",
        description=f"Prime-factorise an integer (n ≤ 10¹⁵).",
    )
    @app_commands.describe(n="The integer to factorise")
    @app_commands.checks.cooldown(1, 2.0)
    async def factorize(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=2, hi=_FACTORIZE_CAP)
            factors = sympy.factorint(n)
            fmt     = _format_factorisation(factors)
            num_factors = sum(factors.values())  # total prime factors with multiplicity

            embed = math_embed(
                title="Prime Factorisation",
                result=fmt,
                steps=[
                    ("Input",          f"{n:,}"),
                    ("Factor pairs",   "  ×  ".join(f"{p}^{e}" for p, e in sorted(factors.items()))),
                    ("Superscript form", fmt),
                ],
                footer=(
                    f"{'Prime' if len(factors) == 1 and list(factors.values())[0] == 1 else 'Composite'}  |  "
                    f"{len(factors)} distinct prime factor(s)  |  "
                    f"{num_factors} total prime factor(s) with multiplicity"
                ),
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /primes_up_to
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="primes_up_to",
        description=f"List all primes up to n (n ≤ {_PRIMES_UPTO_CAP:,}).",
    )
    @app_commands.describe(n=f"Upper bound (inclusive, max {_PRIMES_UPTO_CAP:,})")
    @app_commands.checks.cooldown(1, 2.0)
    async def primes_up_to(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=2, hi=_PRIMES_UPTO_CAP)
            primes = list(sympy.primerange(2, n + 1))
            total  = len(primes)

            if total == 0:
                await interaction.followup.send(
                    embed=error_embed(f"No primes found up to {n:,}.")
                )
                return

            prime_strs = [str(p) for p in primes]
            footer_prefix = f"{total:,} prime(s) up to {n:,}"

            pages = _paginate_list(
                items=prime_strs,
                page_size=_PRIMES_SHOW_MAX,
                title=f"Primes up to {n:,}",
                footer_prefix=footer_prefix,
            )
            await send_paginated(interaction, pages)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /modular
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="modular",
        description="Compute base^exp mod m using fast modular exponentiation.",
    )
    @app_commands.describe(
        base="The base integer",
        exp="The exponent (must be ≥ 0)",
        mod="The modulus (must be ≥ 2)",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def modular(
        self,
        interaction: discord.Interaction,
        base: int,
        exp: int,
        mod: int,
    ) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(exp, "exp", lo=0)
            _validate_int_arg(mod, "mod", lo=2)

            result = pow(base, exp, mod)

            steps = [
                ("Formula",      "result = base^exp  mod  m"),
                ("Substitute",   f"result = {base}^{exp}  mod  {mod}"),
                ("Result",       f"{base}^{exp}  mod  {mod}  =  {result}"),
            ]
            embed = math_embed(
                title="Modular Exponentiation",
                result=str(result),
                steps=steps,
                footer="Computed with Python's built-in pow(base, exp, mod) — O(log exp) time",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /fibonacci
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="fibonacci",
        description=f"Generate the first n Fibonacci numbers (n ≤ {_FIB_CAP}).",
    )
    @app_commands.describe(n=f"How many terms to generate (max {_FIB_CAP})")
    @app_commands.checks.cooldown(1, 2.0)
    async def fibonacci(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=1, hi=_FIB_CAP)
            fibs      = _fibonacci_list(n)
            fib_strs  = [str(f) for f in fibs]
            last      = fibs[-1]

            if n <= _FIB_PAGE_SIZE:
                # Single embed — no pagination needed
                embed = math_embed(
                    title=f"Fibonacci Sequence  (n = {n})",
                    result=", ".join(fib_strs),
                    footer=f"F({n}) = {last:,}  |  iteratively generated",
                )
                await interaction.followup.send(embed=embed)
            else:
                # Paginate: _FIB_PAGE_SIZE terms per page
                pages = _paginate_list(
                    items=fib_strs,
                    page_size=_FIB_PAGE_SIZE,
                    title=f"Fibonacci Sequence  (n = {n})",
                    footer_prefix=f"F({n}) = {last:,}",
                )
                await send_paginated(interaction, pages)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the NumberTheoryCog into *bot*."""
    await bot.add_cog(NumberTheoryCog(bot))