"""
utils/probability_math.py — Core probability computations, kept separate
from the Discord command layer so every function here can be (and has
been) unit-verified against known results / brute force / manual
combinatorics before being wired into ``cogs/probability.py``.

Every function is pure (no Discord objects, no I/O) and safe to call from
an executor thread.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from utils.dice import DiceSpec

# ---------------------------------------------------------------------------
# Dice-sum distribution (exact, via polynomial convolution)
# ---------------------------------------------------------------------------


def dice_sum_distribution(spec: DiceSpec) -> dict[int, int]:
    """
    Return the EXACT outcome-count distribution for rolling *spec*'s dice
    (ignoring the modifier — that's just a shift, applied by the caller).

    Computed via repeated polynomial convolution of a single die's
    distribution (each face equally likely) — this is exact combinatorics,
    not simulation. Uses Python's arbitrary-precision ints (object dtype)
    so large dice pools (e.g. 100d6) don't silently overflow into floats.

    Returns
    -------
    dict[int, int]
        ``{sum_value: number_of_ways_to_roll_it}``. Values sum to
        ``spec.sides ** spec.count``.
    """
    single = np.zeros(spec.sides + 1, dtype=object)
    single[1:] = 1  # faces 1..sides, each with weight 1

    dist = np.array([1], dtype=object)
    for _ in range(spec.count):
        dist = np.convolve(dist, single)

    # index i of `dist` holds the count for unmodified sum i (before shift)
    return {i: int(c) for i, c in enumerate(dist) if c != 0}


def dice_sum_probability(spec: DiceSpec, target: int) -> tuple[int, int]:
    """
    Exact ``P(sum == target)`` for dice *spec* (modifier included), returned
    as ``(ways, total_outcomes)`` so the caller can show both the reduced
    probability and, if useful, the raw combinatorics.
    """
    dist = dice_sum_distribution(spec)
    unshifted_target = target - spec.modifier
    ways = dist.get(unshifted_target, 0)
    total = spec.sides**spec.count
    return ways, total


# ---------------------------------------------------------------------------
# Bayes' theorem
# ---------------------------------------------------------------------------


def bayes_theorem(prior: float, sensitivity: float, false_positive_rate: float) -> float:
    """
    Compute ``P(A|B)`` given:

    - *prior*: ``P(A)``
    - *sensitivity*: ``P(B|A)``
    - *false_positive_rate*: ``P(B|¬A)``

    via ``P(A|B) = P(B|A)P(A) / [P(B|A)P(A) + P(B|¬A)P(¬A)]``.

    Raises
    ------
    ValueError
        If any input is outside ``[0, 1]``, or if the evidence ``B`` has
        zero probability under both hypotheses (division by zero).
    """
    for name, val in (
        ("prior", prior),
        ("sensitivity", sensitivity),
        ("false_positive_rate", false_positive_rate),
    ):
        if not (0 <= val <= 1):
            raise ValueError(f"`{name}` must be between 0 and 1 (got {val}).")

    p_b = sensitivity * prior + false_positive_rate * (1 - prior)
    if p_b == 0:
        raise ValueError(
            "P(B) = 0 — the evidence has zero probability under either "
            "hypothesis, so P(A|B) is undefined."
        )
    return (sensitivity * prior) / p_b


# ---------------------------------------------------------------------------
# Hypergeometric (card draws / urn draws)
# ---------------------------------------------------------------------------


def hypergeometric_pmf(population: int, successes_in_pop: int, draws: int, target_successes: int) -> float:
    """
    Exact ``P(exactly target_successes successes in draws)`` drawing without
    replacement from a population of size *population* containing
    *successes_in_pop* "success" items.

    Thin wrapper around ``scipy.stats.hypergeom.pmf``, verified against
    manual ``comb(K,k)*comb(N-K,n-k)/comb(N,n)`` combinatorics.
    """
    if not (0 <= successes_in_pop <= population):
        raise ValueError("`successes_in_pop` must be between 0 and `population`.")
    if not (0 <= draws <= population):
        raise ValueError("`draws` must be between 0 and `population`.")
    if not (0 <= target_successes <= draws):
        raise ValueError("`target_successes` must be between 0 and `draws`.")
    return float(scipy_stats.hypergeom.pmf(target_successes, population, successes_in_pop, draws))


def urn_distribution(red: int, blue: int, draws: int) -> dict[int, float]:
    """
    Exact distribution of "number of red drawn" when drawing *draws* balls
    without replacement from an urn of *red* red and *blue* blue balls.

    Returns ``{r: P(exactly r red drawn)}`` for every feasible ``r``.
    """
    total = red + blue
    if draws > total:
        raise ValueError(f"Can't draw {draws} balls from an urn of only {total}.")
    return {
        r: float(scipy_stats.hypergeom.pmf(r, total, red, draws))
        for r in range(draws + 1)
    }


def urn_monte_carlo(red: int, blue: int, draws: int, trials: int, rng: random.Random) -> dict[int, float]:
    """
    Empirical "number of red drawn" distribution over *trials* repeated
    draws-without-replacement trials — a simulated cross-check against
    :func:`urn_distribution`'s exact math, shown side by side for
    trust-building in the ``/prob urn`` command.
    """
    pool = ["R"] * red + ["B"] * blue
    counts: dict[int, int] = {}
    for _ in range(trials):
        sample = rng.sample(pool, draws)
        r = sample.count("R")
        counts[r] = counts.get(r, 0) + 1
    return {r: c / trials for r, c in counts.items()}


# ---------------------------------------------------------------------------
# Birthday paradox
# ---------------------------------------------------------------------------


def birthday_probability(n_people: int, days: int = 365) -> float:
    """
    Exact probability that at least two of *n_people* share a birthday
    (uniform over *days* equally likely days, no leap-year adjustment).
    """
    if n_people < 0:
        raise ValueError("`n_people` must be non-negative.")
    if n_people <= 1:
        return 0.0
    if n_people > days:
        return 1.0
    p_no_match = 1.0
    for i in range(n_people):
        p_no_match *= (days - i) / days
    return 1 - p_no_match


def birthday_monte_carlo(n_people: int, trials: int, rng: random.Random, days: int = 365) -> float:
    """Empirical birthday-match probability over *trials* simulated rooms of *n_people*."""
    if n_people <= 1:
        return 0.0
    matches = 0
    for _ in range(trials):
        birthdays = [rng.randrange(days) for _ in range(n_people)]
        if len(set(birthdays)) < n_people:
            matches += 1
    return matches / trials


# ---------------------------------------------------------------------------
# Monte Carlo demonstrations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloPiResult:
    estimate: float
    trials: int
    inside: int


def monte_carlo_pi(trials: int, rng: random.Random) -> MonteCarloPiResult:
    """Estimate π by sampling random points in [-1,1]² and checking the unit circle."""
    inside = 0
    for _ in range(trials):
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if x * x + y * y <= 1:
            inside += 1
    return MonteCarloPiResult(estimate=4 * inside / trials, trials=trials, inside=inside)


def buffon_pi(needle_length: float, line_spacing: float, trials: int, rng: random.Random) -> float | None:
    """
    Estimate π via Buffon's needle: drop a needle of *needle_length* onto a
    floor ruled with parallel lines *line_spacing* apart and count crossings.

    Uses the classic short-needle formula, which requires
    ``needle_length <= line_spacing``.

    Returns
    -------
    float | None
        The π estimate, or ``None`` if zero needles crossed a line in
        *trials* attempts (division by zero — caller should ask for more
        trials rather than treat this as an error).
    """
    if needle_length <= 0 or line_spacing <= 0:
        raise ValueError("`needle_length` and `line_spacing` must be positive.")
    if needle_length > line_spacing:
        raise ValueError(
            "This uses the classic short-needle formula, which requires "
            "`needle_length` ≤ `line_spacing`."
        )

    crosses = 0
    for _ in range(trials):
        center = rng.uniform(0, line_spacing / 2)
        angle = rng.uniform(0, math.pi / 2)
        if center <= (needle_length / 2) * math.sin(angle):
            crosses += 1

    if crosses == 0:
        return None
    return (2 * needle_length * trials) / (line_spacing * crosses)


# ---------------------------------------------------------------------------
# Distribution sampling ("set generator")
# ---------------------------------------------------------------------------

_SUPPORTED_DISTRIBUTIONS = ("normal", "binomial", "poisson", "uniform", "exponential")

DIST_PARAM_NAMES: dict[str, list[str]] = {
    "normal": ["mean", "stdev"],
    "binomial": ["n", "p"],
    "poisson": ["lam"],
    "uniform": ["low", "high"],
    "exponential": ["rate"],
}


def sample_distribution(
    dist: str, n: int, params: dict[str, float], seed: int | None = None
) -> np.ndarray:
    """
    Draw *n* samples from the named distribution.

    Parameters
    ----------
    dist:
        One of ``"normal"``, ``"binomial"``, ``"poisson"``, ``"uniform"``,
        ``"exponential"``.
    params:
        Distribution parameters — see :data:`DIST_PARAM_NAMES` for the
        required keys per distribution.
    seed:
        Optional seed for reproducibility.

    Raises
    ------
    ValueError
        For an unknown distribution name or invalid parameter values
        (e.g. negative stdev).
    """
    if dist not in _SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution `{dist}`. Supported: {', '.join(_SUPPORTED_DISTRIBUTIONS)}."
        )

    rng = np.random.default_rng(seed)

    if dist == "normal":
        if params["stdev"] <= 0:
            raise ValueError("`stdev` must be positive.")
        return rng.normal(params["mean"], params["stdev"], n)

    if dist == "binomial":
        if params["n"] < 1 or params["n"] != int(params["n"]):
            raise ValueError("`n` must be a positive integer.")
        if not (0 <= params["p"] <= 1):
            raise ValueError("`p` must be between 0 and 1.")
        return rng.binomial(int(params["n"]), params["p"], n)

    if dist == "poisson":
        if params["lam"] <= 0:
            raise ValueError("`lam` must be positive.")
        return rng.poisson(params["lam"], n)

    if dist == "uniform":
        if params["low"] >= params["high"]:
            raise ValueError("`low` must be < `high`.")
        return rng.uniform(params["low"], params["high"], n)

    if dist == "exponential":
        if params["rate"] <= 0:
            raise ValueError("`rate` must be positive.")
        return rng.exponential(1 / params["rate"], n)

    raise AssertionError("unreachable")  # _SUPPORTED_DISTRIBUTIONS guards this above
