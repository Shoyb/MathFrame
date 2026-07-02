"""
cogs/probability.py — Probability & likelihood slash commands (Phase 3 of
the Random/Probability/Quiz plan; see RANDOM_PROBABILITY_QUIZ_PLAN.md).

Everything in this cog answers "what's the likelihood/distribution of X" —
as opposed to ``cogs/random_tools.py``'s ``/rand`` group, which answers
"give me one random outcome." See the plan doc's "Random vs Probability"
table for the exact dividing line.

All the actual math lives in ``utils/probability_math.py`` and has been
unit-verified there (against brute force, manual combinatorics, and known
textbook results) before being wired into the Discord commands below —
this file is intentionally thin: parameter validation, formatting, and
plotting only.

Commands
--------
/prob sample        distribution n [params...]        Draw n samples, show histogram + full stats summary.
/prob dice_sum       notation target                    Exact P(dice sum == target).
/prob bayes          prior sensitivity fpr               Bayes' theorem: P(A|B).
/prob conditional    2x2 contingency table counts        P(A|B), P(B|A), independence check.
/prob card_draw      population successes draws target  Hypergeometric: P(exactly target successes).
/prob urn            red blue draws [trials]             Urn draw: exact distribution + Monte Carlo cross-check.
/prob birthday       n_people [days] [simulate]          Birthday paradox: exact (+ optional simulated) probability.
/prob set_sample     items k [replacement] [trials]      Repeated set sampling: empirical outcome-frequency distribution.
/prob monte_carlo_pi trials                              Estimate π via random points in a circle.
/prob buffon         needle_length line_spacing trials   Estimate π via Buffon's needle.
"""

from __future__ import annotations

import asyncio
import io
import random

import discord
import matplotlib
import numpy as np
from discord import app_commands
from discord.ext import commands

from utils.formatter import error_embed, math_embed
from utils.dice import parse_dice
from utils.stats_summary import compute_summary
from utils.probability_math import (
    dice_sum_distribution,
    dice_sum_probability,
    bayes_theorem,
    conditional_from_counts,
    hypergeometric_pmf,
    urn_distribution,
    urn_monte_carlo,
    birthday_probability,
    birthday_monte_carlo,
    monte_carlo_pi,
    buffon_pi,
    sample_distribution,
    set_sample_distribution,
    DIST_PARAM_NAMES,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Plotting helpers (run inside an executor — the calling command defers first)
# ---------------------------------------------------------------------------


def _histogram_bytes(samples: np.ndarray, title: str, xlabel: str) -> io.BytesIO:
    """Render a histogram of *samples* to a PNG and return it as BytesIO."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bins = min(50, max(10, int(np.sqrt(len(samples)))))
    ax.hist(samples, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.mean(samples)), color="crimson", linewidth=2, linestyle="--", label="Mean")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=12, pad=6)
    ax.grid(True, alpha=0.3)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _dice_dist_bytes(dist: dict[int, int], modifier: int, target: int | None) -> io.BytesIO:
    """Render a dice-sum distribution as a bar chart, highlighting *target* if given."""
    sums = sorted(dist.keys())
    shifted_sums = [s + modifier for s in sums]
    total = sum(dist.values())
    probs = [dist[s] / total for s in sums]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [
        "crimson" if target is not None and s == target else "steelblue"
        for s in shifted_sums
    ]
    ax.bar(shifted_sums, probs, color=colors, edgecolor="white")
    ax.set_xlabel("Sum", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title("Dice Sum Distribution", fontsize=12, pad=6)
    ax.grid(True, alpha=0.3, axis="y")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _urn_dist_bytes(exact: dict[int, float], mc: dict[int, float] | None, draws: int) -> io.BytesIO:
    """Render exact urn distribution as bars, with Monte Carlo overlay points if given."""
    rs = sorted(exact.keys())
    exact_probs = [exact[r] for r in rs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(rs, exact_probs, color="steelblue", edgecolor="white", alpha=0.8, label="Exact")
    if mc is not None:
        mc_probs = [mc.get(r, 0) for r in rs]
        ax.scatter(rs, mc_probs, color="crimson", zorder=3, s=50, label="Monte Carlo")

    ax.set_xlabel("Red balls drawn", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(f"Urn Draw Distribution ({draws} draws)", fontsize=12, pad=6)
    ax.set_xticks(rs)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _monte_carlo_pi_bytes(trials: int, rng: random.Random) -> tuple[io.BytesIO, float]:
    """Scatter-plot the Monte Carlo π estimation points (inside/outside the unit circle)."""
    xs = [rng.uniform(-1, 1) for _ in range(min(trials, 5000))]
    # Reuse the same rng stream position conceptually isn't required here —
    # this is a fresh illustrative sample of up to 5000 points for the plot,
    # separate from the trials used for the numeric estimate.
    ys = [rng.uniform(-1, 1) for _ in range(len(xs))]
    inside_mask = [x * x + y * y <= 1 for x, y in zip(xs, ys)]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    inside_x = [x for x, m in zip(xs, inside_mask) if m]
    inside_y = [y for y, m in zip(ys, inside_mask) if m]
    outside_x = [x for x, m in zip(xs, inside_mask) if not m]
    outside_y = [y for y, m in zip(ys, inside_mask) if not m]

    ax.scatter(inside_x, inside_y, s=4, color="steelblue", label="Inside circle")
    ax.scatter(outside_x, outside_y, s=4, color="lightgray", label="Outside circle")
    circle = plt.Circle((0, 0), 1, fill=False, color="crimson", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Monte Carlo π estimation (illustrating {len(xs)} of {trials} points)", fontsize=11, pad=6)
    ax.legend(loc="upper right", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _set_sample_bytes(counts: dict[tuple, int], trials: int, top_n: int = 15) -> io.BytesIO:
    """Render the top *top_n* most frequent outcomes from a set_sample run as a bar chart."""
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [", ".join(map(str, outcome)) for outcome, _ in ranked]
    freqs = [c / trials for _, c in ranked]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels) + 1)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = range(len(labels))
    ax.barh(list(y_pos), freqs, color="steelblue", edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # highest frequency at the top
    ax.set_xlabel("Empirical probability", fontsize=10)
    ax.set_title(f"Top {len(labels)} outcomes over {trials} trials", fontsize=12, pad=6)
    ax.grid(True, alpha=0.3, axis="x")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class ProbabilityCog(commands.Cog, name="Probability"):
    """Probability, likelihood, and distribution commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    prob = app_commands.Group(name="prob", description="Probability and likelihood tools.")

    # -----------------------------------------------------------------------
    # /prob sample
    # -----------------------------------------------------------------------

    @prob.command(
        name="sample",
        description="Draw samples from a distribution — histogram + full stats summary.",
    )
    @app_commands.describe(
        distribution="Which distribution to sample from",
        n="Number of samples to draw (10-100000)",
        params='Comma-separated parameters — see the distribution choice for the order',
    )
    @app_commands.choices(distribution=[
        app_commands.Choice(name="Normal      — params: mean, stdev", value="normal"),
        app_commands.Choice(name="Binomial    — params: n, p", value="binomial"),
        app_commands.Choice(name="Poisson     — params: lam", value="poisson"),
        app_commands.Choice(name="Uniform     — params: low, high", value="uniform"),
        app_commands.Choice(name="Exponential — params: rate", value="exponential"),
    ])
    @app_commands.checks.cooldown(1, 4.0)
    async def sample(
        self,
        interaction: discord.Interaction,
        distribution: str,
        n: app_commands.Range[int, 10, 100000],
        params: str,
    ) -> None:
        await interaction.response.defer()
        try:
            expected = DIST_PARAM_NAMES[distribution]
            tokens = [t.strip() for t in params.split(",")]
            if len(tokens) != len(expected):
                raise ValueError(
                    f"`{distribution}` expects {len(expected)} parameter(s) "
                    f"({', '.join(expected)}), got {len(tokens)}."
                )
            parsed_params = {}
            for name, tok in zip(expected, tokens):
                try:
                    parsed_params[name] = float(tok)
                except ValueError:
                    raise ValueError(f"Parameter `{name}` must be a number (got `{tok}`).")

            loop = asyncio.get_running_loop()

            def _do_sample():
                return sample_distribution(distribution, n, parsed_params)

            samples = await loop.run_in_executor(None, _do_sample)
            summary = compute_summary(list(samples))

            def _do_plot():
                return _histogram_bytes(
                    samples,
                    title=f"{distribution.title()} distribution — n={n}",
                    xlabel="Value",
                )

            buf = await loop.run_in_executor(None, _do_plot)
            file = discord.File(buf, filename="sample.png")

            result = (
                f"Mean = {summary.mean:.6g}   Median = {summary.median:.6g}\n"
                f"Sample stdev = {summary.sample_stdev:.6g}   Variance = {summary.sample_variance:.6g}\n"
                f"Min / Max = {summary.minimum:.6g} / {summary.maximum:.6g}\n"
                f"Q1 / Q3 = {summary.q1:.6g} / {summary.q3:.6g}"
            )
            embed = math_embed(
                title=f"Sample: {distribution.title()}({params})",
                result=result,
                footer=f"n = {n}  |  see full breakdown via /stat summary on the raw values",
            )
            embed.set_image(url="attachment://sample.png")
            await interaction.followup.send(embed=embed, file=file)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob dice_sum
    # -----------------------------------------------------------------------

    @prob.command(name="dice_sum", description="Exact probability of a dice-roll sum.")
    @app_commands.describe(
        notation='Dice notation, e.g. "2d6" or "3d6+2"', target="The sum to compute P(sum = target) for"
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def dice_sum(self, interaction: discord.Interaction, notation: str, target: int) -> None:
        await interaction.response.defer()
        try:
            spec = parse_dice(notation)
            loop = asyncio.get_running_loop()

            def _do_compute():
                dist = dice_sum_distribution(spec)
                ways, total = dice_sum_probability(spec, target)
                return dist, ways, total

            dist, ways, total = await loop.run_in_executor(None, _do_compute)

            prob_val = ways / total
            result = (
                f"P(sum = {target}) = {ways} / {total} = {prob_val:.6g}\n"
                f"Possible range: [{spec.min_total}, {spec.max_total}]"
            )
            embed = math_embed(
                title=f"Dice Sum Probability — {notation}",
                result=result,
                footer=f"{spec.count} die/dice, {spec.sides} sides each"
                + (f", modifier {spec.modifier:+d}" if spec.modifier else ""),
            )

            def _do_plot():
                return _dice_dist_bytes(dist, spec.modifier, target)

            buf = await loop.run_in_executor(None, _do_plot)
            embed.set_image(url="attachment://dist.png")
            file = discord.File(buf, filename="dist.png")

            await interaction.followup.send(embed=embed, file=file)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob bayes
    # -----------------------------------------------------------------------

    @prob.command(name="bayes", description="Bayes' theorem: compute P(A|B).")
    @app_commands.describe(
        prior="P(A) — prior probability of the hypothesis",
        sensitivity="P(B|A) — probability of the evidence given the hypothesis is true",
        false_positive_rate="P(B|¬A) — probability of the evidence given the hypothesis is false",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def bayes(
        self,
        interaction: discord.Interaction,
        prior: app_commands.Range[float, 0.0, 1.0],
        sensitivity: app_commands.Range[float, 0.0, 1.0],
        false_positive_rate: app_commands.Range[float, 0.0, 1.0],
    ) -> None:
        await interaction.response.defer()
        try:
            posterior = bayes_theorem(prior, sensitivity, false_positive_rate)
            result = (
                f"P(A|B) = {posterior:.6g}\n\n"
                f"Given:\n"
                f"P(A) = {prior:g}\n"
                f"P(B|A) = {sensitivity:g}\n"
                f"P(B|¬A) = {false_positive_rate:g}"
            )
            embed = math_embed(
                title="Bayes' Theorem",
                result=result,
                footer="P(A|B) = P(B|A)P(A) / [P(B|A)P(A) + P(B|¬A)P(¬A)]",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob conditional
    # -----------------------------------------------------------------------

    @prob.command(
        name="conditional",
        description="Conditional probability from a 2x2 contingency table: P(A|B), P(B|A), independence check.",
    )
    @app_commands.describe(
        a_and_b="Count of outcomes where both A and B occurred",
        a_and_not_b="Count of outcomes where A occurred but B did not",
        not_a_and_b="Count of outcomes where B occurred but A did not",
        not_a_and_not_b="Count of outcomes where neither A nor B occurred",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def conditional(
        self,
        interaction: discord.Interaction,
        a_and_b: app_commands.Range[int, 0, None],
        a_and_not_b: app_commands.Range[int, 0, None],
        not_a_and_b: app_commands.Range[int, 0, None],
        not_a_and_not_b: app_commands.Range[int, 0, None],
    ) -> None:
        await interaction.response.defer()
        try:
            r = conditional_from_counts(a_and_b, a_and_not_b, not_a_and_b, not_a_and_not_b)
            total = a_and_b + a_and_not_b + not_a_and_b + not_a_and_not_b
            result = (
                f"P(A) = {r.p_a:.6g}   P(B) = {r.p_b:.6g}   P(A ∩ B) = {r.p_a_and_b:.6g}\n\n"
                f"P(A|B) = {r.p_a_given_b:.6g}\n"
                f"P(B|A) = {r.p_b_given_a:.6g}\n\n"
                f"A and B are {'independent' if r.independent else 'NOT independent'} "
                f"(P(A∩B) {'==' if r.independent else '≠'} P(A)·P(B))"
            )
            embed = math_embed(
                title="Conditional Probability",
                result=result,
                footer=(
                    f"Table (n={total}): A∩B={a_and_b}, A∩¬B={a_and_not_b}, "
                    f"¬A∩B={not_a_and_b}, ¬A∩¬B={not_a_and_not_b}"
                ),
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob card_draw
    # -----------------------------------------------------------------------

    @prob.command(
        name="card_draw",
        description="Hypergeometric: probability of exactly k successes in a hand.",
    )
    @app_commands.describe(
        population="Total population size (e.g. 52 for a standard deck)",
        successes="Number of 'success' items in the population (e.g. 4 for aces)",
        draws="Hand/sample size drawn without replacement",
        target="Exact number of successes to compute P(X = target) for",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def card_draw(
        self,
        interaction: discord.Interaction,
        population: app_commands.Range[int, 1, 100000],
        successes: int,
        draws: int,
        target: int,
    ) -> None:
        await interaction.response.defer()
        try:
            p = hypergeometric_pmf(population, successes, draws, target)
            result = f"P(exactly {target} success(es)) = {p:.6g}"
            embed = math_embed(
                title="Hypergeometric Probability",
                result=result,
                footer=f"population={population}, successes_in_pop={successes}, draws={draws}",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob urn
    # -----------------------------------------------------------------------

    @prob.command(
        name="urn",
        description="Urn draw: exact distribution of red balls drawn, with Monte Carlo cross-check.",
    )
    @app_commands.describe(
        red="Number of red balls in the urn",
        blue="Number of blue balls in the urn",
        draws="Number of balls drawn without replacement",
        trials="Monte Carlo trials for the cross-check plot (0 to skip, default 5000)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def urn(
        self,
        interaction: discord.Interaction,
        red: app_commands.Range[int, 1, 10000],
        blue: app_commands.Range[int, 1, 10000],
        draws: int,
        trials: app_commands.Range[int, 0, 50000] = 5000,
    ) -> None:
        await interaction.response.defer()
        try:
            loop = asyncio.get_running_loop()

            def _do_compute():
                exact = urn_distribution(red, blue, draws)
                mc = None
                if trials > 0:
                    mc = urn_monte_carlo(red, blue, draws, trials, random.Random())
                return exact, mc

            exact, mc = await loop.run_in_executor(None, _do_compute)

            lines = [f"P(r red) for r=0..{draws}:"]
            for r in sorted(exact.keys()):
                lines.append(f"  r={r}: {exact[r]:.4g}")
            result = "\n".join(lines)

            embed = math_embed(
                title="Urn Draw Distribution",
                result=result,
                footer=f"urn: {red} red, {blue} blue  |  {draws} drawn without replacement"
                + (f"  |  Monte Carlo: {trials} trials" if trials > 0 else ""),
            )

            def _do_plot():
                return _urn_dist_bytes(exact, mc, draws)

            buf = await loop.run_in_executor(None, _do_plot)
            embed.set_image(url="attachment://urn.png")
            file = discord.File(buf, filename="urn.png")

            await interaction.followup.send(embed=embed, file=file)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob birthday
    # -----------------------------------------------------------------------

    @prob.command(name="birthday", description="Birthday paradox: P(at least two share a birthday).")
    @app_commands.describe(
        n_people="Number of people in the room",
        days="Days in the year to assume (default 365)",
        simulate="Also run a Monte Carlo simulation to cross-check (default False)",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def birthday(
        self,
        interaction: discord.Interaction,
        n_people: app_commands.Range[int, 0, 100000],
        days: app_commands.Range[int, 2, 100000] = 365,
        simulate: bool = False,
    ) -> None:
        await interaction.response.defer()
        try:
            exact = birthday_probability(n_people, days)
            result = f"P(shared birthday) = {exact:.6g}"

            if simulate:
                loop = asyncio.get_running_loop()

                def _do_sim():
                    return birthday_monte_carlo(n_people, 20000, random.Random(), days)

                mc = await loop.run_in_executor(None, _do_sim)
                result += f"\nMonte Carlo (20000 trials) = {mc:.6g}"

            embed = math_embed(
                title="Birthday Paradox",
                result=result,
                footer=f"n_people={n_people}, days={days}",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob set_sample
    # -----------------------------------------------------------------------

    @prob.command(
        name="set_sample",
        description="Repeated sampling from a set — empirical outcome-frequency distribution over many trials.",
    )
    @app_commands.describe(
        items="Comma-separated set of items, e.g. 'red,blue,green'",
        k="Number of items drawn per trial",
        replacement="Sample with replacement each trial? (default False)",
        trials="Number of repeated trials (default 1000)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def set_sample(
        self,
        interaction: discord.Interaction,
        items: str,
        k: app_commands.Range[int, 1, 20],
        replacement: bool = False,
        trials: app_commands.Range[int, 1, 100000] = 1000,
    ) -> None:
        await interaction.response.defer()
        try:
            item_list = [t.strip() for t in items.split(",") if t.strip()]
            if not item_list:
                raise ValueError("`items` must contain at least one non-empty, comma-separated value.")

            loop = asyncio.get_running_loop()

            def _do_sample():
                return set_sample_distribution(item_list, k, replacement, trials, random.Random())

            counts = await loop.run_in_executor(None, _do_sample)

            def _do_plot():
                return _set_sample_bytes(counts, trials)

            buf = await loop.run_in_executor(None, _do_plot)
            file = discord.File(buf, filename="set_sample.png")

            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            top = ranked[:5]
            lines = [f"{', '.join(outcome)} — {c / trials:.4f} ({c}/{trials})" for outcome, c in top]
            result = f"{len(counts)} distinct outcome(s) observed. Top 5:\n" + "\n".join(lines)
            if len(result) > 1000:
                result = result[:1000] + "…"

            items_preview = ", ".join(item_list)
            if len(items_preview) > 200:
                items_preview = items_preview[:200] + f"… (+{len(item_list)} items total)"

            embed = math_embed(
                title=f"Set Sample — draw {k} of {len(item_list)} items",
                result=result,
                footer=f"items: {items_preview}  |  trials={trials}  |  replacement={replacement}",
            )
            embed.set_image(url="attachment://set_sample.png")
            await interaction.followup.send(embed=embed, file=file)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /prob monte_carlo_pi
    # -----------------------------------------------------------------------

    @prob.command(name="monte_carlo_pi", description="Estimate π via random points in a circle.")
    @app_commands.describe(trials="Number of random points to sample (100-1000000)")
    @app_commands.checks.cooldown(1, 4.0)
    async def monte_carlo_pi_cmd(
        self, interaction: discord.Interaction, trials: app_commands.Range[int, 100, 1000000]
    ) -> None:
        await interaction.response.defer()
        loop = asyncio.get_running_loop()

        def _do_estimate():
            return monte_carlo_pi(trials, random.Random())

        result_obj = await loop.run_in_executor(None, _do_estimate)

        def _do_plot():
            return _monte_carlo_pi_bytes(trials, random.Random())

        buf = await loop.run_in_executor(None, _do_plot)

        import math as _math
        error = abs(result_obj.estimate - _math.pi)
        result = (
            f"π ≈ {result_obj.estimate:.6f}\n"
            f"Actual π = {_math.pi:.6f}\n"
            f"Error = {error:.6f}"
        )
        embed = math_embed(
            title="Monte Carlo π Estimation",
            result=result,
            footer=f"{result_obj.inside} / {result_obj.trials} points landed inside the circle",
        )
        embed.set_image(url="attachment://mc_pi.png")
        file = discord.File(buf, filename="mc_pi.png")
        await interaction.followup.send(embed=embed, file=file)

    # -----------------------------------------------------------------------
    # /prob buffon
    # -----------------------------------------------------------------------

    @prob.command(name="buffon", description="Estimate π via Buffon's needle.")
    @app_commands.describe(
        needle_length="Needle length (must be ≤ line_spacing)",
        line_spacing="Distance between parallel lines",
        trials="Number of needle drops (100-1000000)",
    )
    @app_commands.checks.cooldown(1, 4.0)
    async def buffon_cmd(
        self,
        interaction: discord.Interaction,
        needle_length: float,
        line_spacing: float,
        trials: app_commands.Range[int, 100, 1000000],
    ) -> None:
        await interaction.response.defer()
        try:
            loop = asyncio.get_running_loop()

            def _do_estimate():
                return buffon_pi(needle_length, line_spacing, trials, random.Random())

            estimate = await loop.run_in_executor(None, _do_estimate)

            if estimate is None:
                raise ValueError(
                    "No needle crossed a line in that many trials — try more trials "
                    "or a longer needle relative to the line spacing."
                )

            import math as _math
            error = abs(estimate - _math.pi)
            result = f"π ≈ {estimate:.6f}\nActual π = {_math.pi:.6f}\nError = {error:.6f}"
            embed = math_embed(
                title="Buffon's Needle π Estimation",
                result=result,
                footer=f"needle={needle_length:g}, spacing={line_spacing:g}, trials={trials}",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    """Load the ProbabilityCog into *bot*."""
    await bot.add_cog(ProbabilityCog(bot))
