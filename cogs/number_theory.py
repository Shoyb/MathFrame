"""
cogs/number_theory.py — Number theory slash commands for the math bot.

Commands
--------
/gcd              numbers             GCD of a list of integers.
/lcm              numbers             LCM of a list of integers.
/is_prime         n                   Primality test with factorisation fallback.
/factorize        n                   Prime factorisation in superscript notation.
/primes_up_to     n                   All primes up to n (capped, paginated).
/modular          base  exp  mod      Modular exponentiation.
/fibonacci        n                   First n Fibonacci numbers (paginated if large).
/totient          n                   Euler's totient φ(n).
/divisors         n                   All divisors of n.
/is_perfect       n                   Check if n is a perfect number.
/mobius           n                   Möbius function μ(n).
/chinese_remainder remainders moduli  Chinese Remainder Theorem.
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
_TOTIENT_CAP:     Final[int] = 10 ** 9
_DIVISORS_CAP:    Final[int] = 10 ** 12
_DIVISORS_PAGE:   Final[int] = 30   # paginate divisor lists longer than this

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


def _parse_single_integer(s: str, param_name: str = "n") -> int:
    """
    Parse a single integer from a user-supplied string.

    Raises
    ------
    ValueError
        If the string is not a valid integer.
    """
    s = s.strip()
    if not s:
        raise ValueError(f"`{param_name}` is empty.")
    if "." in s:
        raise ValueError(f"`{s}` is a decimal — only integers are accepted.")
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"Cannot convert `{s}` to an integer.")


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
    """GCD, LCM, primality, factorisation, modular arithmetic, Fibonacci, and number-theoretic functions."""

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

    # -----------------------------------------------------------------------
    # /totient  (T1-6)
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="totient",
        description="Compute Euler's totient φ(n) — the count of integers up to n coprime to n.",
    )
    @app_commands.describe(n=f"Positive integer (max {_TOTIENT_CAP:,})")
    @app_commands.checks.cooldown(1, 2.0)
    async def totient(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=1, hi=_TOTIENT_CAP)
            result = sympy.totient(n)
            embed = math_embed(
                title="Euler's Totient Function",
                result=f"φ({n:,}) = {result:,}",
                steps=[
                    ("Definition", "φ(n) = count of integers k ≤ n with gcd(k, n) = 1"),
                    ("Input",      f"n = {n:,}"),
                    ("Result",     f"φ({n:,}) = {result:,}"),
                ],
                footer=f"Computed using Euler's product formula over prime factors of {n:,}",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /divisors  (T1-6)
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="divisors",
        description="List all positive divisors of n.",
    )
    @app_commands.describe(n=f"Positive integer (max {_DIVISORS_CAP:,})")
    @app_commands.checks.cooldown(1, 2.0)
    async def divisors(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=1, hi=_DIVISORS_CAP)
            divs = sympy.divisors(n)
            total = len(divs)
            div_strs = [str(d) for d in divs]
            footer_prefix = f"{total} divisor(s) of {n:,}"

            if total <= _DIVISORS_PAGE:
                embed = math_embed(
                    title=f"Divisors of {n:,}",
                    result=",  ".join(div_strs),
                    footer=footer_prefix,
                )
                await interaction.followup.send(embed=embed)
            else:
                pages = _paginate_list(
                    items=div_strs,
                    page_size=_DIVISORS_PAGE,
                    title=f"Divisors of {n:,}",
                    footer_prefix=footer_prefix,
                )
                await send_paginated(interaction, pages)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /is_perfect  (T1-6)
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="is_perfect",
        description="Check whether n is a perfect number (equals the sum of its proper divisors).",
    )
    @app_commands.describe(n=f"Positive integer (max {_DIVISORS_CAP:,})")
    @app_commands.checks.cooldown(1, 2.0)
    async def is_perfect(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=1, hi=_DIVISORS_CAP)
            divs = sympy.divisors(n)
            proper_sum = sum(divs[:-1])   # all divisors except n itself
            perfect = (n > 1) and (proper_sum == n)

            if perfect:
                result_str = f"✅  Yes, {n:,} is a perfect number."
                footer = f"Sum of proper divisors = {proper_sum:,} = {n:,}"
            else:
                result_str = f"❌  No, {n:,} is not a perfect number."
                footer = f"Sum of proper divisors = {proper_sum:,} ≠ {n:,}"

            embed = math_embed(
                title="Perfect Number Test",
                result=result_str,
                steps=[
                    ("Definition",           "A perfect number equals the sum of its proper divisors."),
                    ("Proper divisors of n",  ", ".join(str(d) for d in divs[:-1]) if len(divs) <= 20
                                             else f"{len(divs)-1} divisors (too many to list)"),
                    ("Sum of proper divisors", f"{proper_sum:,}"),
                    ("Verdict",              result_str),
                ],
                footer=footer,
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /mobius  (T1-6)
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="mobius",
        description="Compute the Möbius function μ(n): returns −1, 0, or 1.",
    )
    @app_commands.describe(n=f"Positive integer (max {_DIVISORS_CAP:,})")
    @app_commands.checks.cooldown(1, 2.0)
    async def mobius(self, interaction: discord.Interaction, n: int) -> None:
        await interaction.response.defer()
        try:
            _validate_int_arg(n, "n", lo=1, hi=_DIVISORS_CAP)
            mu = sympy.mobius(n)

            if mu == 0:
                label = "0  (n has a squared prime factor)"
            elif mu == 1:
                label = "1  (n = 1 or n has an even number of distinct prime factors)"
            else:
                label = "−1  (n has an odd number of distinct prime factors)"

            factors = sympy.factorint(n)

            embed = math_embed(
                title="Möbius Function μ(n)",
                result=f"μ({n:,}) = {label}",
                steps=[
                    ("Factorisation",   _format_factorisation(factors)),
                    ("μ(n) value",      label),
                ],
                footer="μ(n) is 0 if n has a squared prime factor; else (−1)^k where k = number of distinct primes",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /chinese_remainder  (T1-6)
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="chinese_remainder",
        description="Solve a system of congruences using the Chinese Remainder Theorem.",
    )
    @app_commands.describe(
        remainders="Comma-separated remainders, e.g. '2, 3, 1'",
        moduli="Comma-separated moduli (must be pairwise coprime), e.g. '3, 5, 7'",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def chinese_remainder(
        self,
        interaction: discord.Interaction,
        remainders: str,
        moduli: str,
    ) -> None:
        """
        Solve the system  x ≡ rᵢ (mod mᵢ)  for all i.

        Uses :func:`sympy.ntheory.modular.crt` which requires the moduli to
        be pairwise coprime.  Returns the unique solution modulo lcm(mᵢ).
        """
        await interaction.response.defer()
        try:
            from sympy.ntheory.modular import crt

            rs = _parse_integers(remainders, "remainders")
            ms = _parse_integers(moduli, "moduli")

            if len(rs) != len(ms):
                raise ValueError(
                    f"Number of remainders ({len(rs)}) must match number of moduli ({len(ms)})."
                )
            if any(m <= 0 for m in ms):
                raise ValueError("All moduli must be positive integers.")

            result, lcm_val = crt(ms, rs, symmetric=False)

            if result is None:
                raise ValueError(
                    "No solution exists. "
                    "Check that the moduli are pairwise coprime and remainders are valid."
                )

            congruences = "  ∧  ".join(
                f"x ≡ {r} (mod {m})" for r, m in zip(rs, ms)
            )
            embed = math_embed(
                title="Chinese Remainder Theorem",
                result=f"x ≡ {result} (mod {lcm_val})",
                steps=[
                    ("System",    congruences),
                    ("Solution",  f"x ≡ {result} (mod {lcm_val})"),
                    ("LCM of moduli", f"{lcm_val:,}"),
                ],
                footer=f"Remainders: {rs}  |  Moduli: {ms}  |  Solution is unique mod {lcm_val:,}",
            )
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An error occurred: {exc}"))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the NumberTheoryCog into *bot*."""
    await bot.add_cog(NumberTheoryCog(bot))