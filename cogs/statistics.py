"""
cogs/statistics.py — Descriptive statistics and probability slash commands.

Commands
--------
/mean          data                        Arithmetic mean.
/median        data                        Median value.
/mode          data                        Most common value(s).
/stdev         data                        Sample standard deviation.
/variance      data                        Sample variance.
/zscore        value  mean  stdev          Standard score (z-score).
/correlation   data_x  data_y             Pearson correlation coefficient.
/regression    data_x  data_y             Linear regression with scatter plot.
/normal_pdf    mean  stdev                Normal distribution PDF plot.
/normal_cdf    mean stdev upper [lower]   Normal distribution CDF plot.
/inv_normal    probability mean stdev     Inverse normal.
/binomial_cdf  n p x                      Binomial CDF plot.
/poisson_cdf   lam x                      Poisson CDF plot.

Data input
----------
Comma-separated numeric strings: ``"1, 2, 3, 4, 5"``
"""

import asyncio
import io
import statistics as _stats

import discord
import matplotlib
import numpy as np
from discord import app_commands
from discord.ext import commands
from scipy import stats as scipy_stats

from utils.formatter import error_embed, math_embed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def parse_numbers(s: str) -> list[float]:
    """
    Parse a comma-separated string of numbers into a list of floats.

    Parameters
    ----------
    s:
        Raw user input, e.g. ``"1, 2, 3.5, -4"``.

    Returns
    -------
    list[float]

    Raises
    ------
    ValueError
        If the string is empty, results in an empty list, or contains any
        non-numeric token.
    """
    if not s.strip():
        raise ValueError("Data string is empty. Provide comma-separated numbers.")

    results: list[float] = []
    for i, token in enumerate(s.split(",")):
        token = token.strip()
        if not token:
            raise ValueError(
                f"Empty value at position {i + 1}. "
                "Check for trailing commas or double commas."
            )
        try:
            results.append(float(token))
        except ValueError:
            raise ValueError(
                f"Cannot convert `{token}` (position {i + 1}) to a number."
            )

    if not results:
        raise ValueError("No numeric data found.")
    return results


def _correlation_label(r: float) -> str:
    """Return a human-readable strength label for a Pearson *r* value."""
    abs_r = abs(r)
    direction = "positive" if r >= 0 else "negative"
    if abs_r > 0.8:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"
    return f"{strength} {direction}"


def _regression_plot_bytes(
    xs: list[float],
    ys: list[float],
    slope: float,
    intercept: float,
    r_squared: float,
) -> io.BytesIO:
    """
    Render a scatter plot with the fitted regression line overlaid.

    This runs synchronously and is intended to be called inside an executor
    or where blocking is acceptable (the calling command already deferred).

    Returns
    -------
    io.BytesIO
        PNG bytes seeked to position 0.
    """
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 300)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(xs_arr, ys_arr, s=35, zorder=3, label="Data points")
    ax.plot(
        x_line, y_line,
        color="crimson", linewidth=2,
        label=f"y = {slope:.4g}x + {intercept:.4g}  (R²={r_squared:.4f})",
    )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title("Linear Regression", fontsize=13)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _normal_pdf_bytes(mean: float, stdev: float) -> io.BytesIO:
    """
    Render the PDF curve for N(*mean*, *stdev*²) over [mean − 4σ, mean + 4σ].

    Returns
    -------
    io.BytesIO
        PNG bytes seeked to position 0.
    """
    x = np.linspace(mean - 4 * stdev, mean + 4 * stdev, 800)
    y = scipy_stats.norm.pdf(x, loc=mean, scale=stdev)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x, y, linewidth=2, color="steelblue")
    ax.fill_between(x, y, alpha=0.15, color="steelblue")

    # Mark ±1σ, ±2σ, ±3σ reference lines
    for k, alpha in ((1, 0.7), (2, 0.45), (3, 0.25)):
        for sign in (-1, 1):
            ax.axvline(
                mean + sign * k * stdev,
                color="gray", linewidth=0.8, linestyle="--", alpha=alpha,
            )

    ax.axvline(mean, color="dimgray", linewidth=1, linestyle="-", label=f"μ = {mean}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Normal Distribution  N(μ={mean}, σ={stdev})", fontsize=13)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _normal_cdf_bytes(mean: float, stdev: float, lower: float, upper: float) -> io.BytesIO:
    x = np.linspace(mean - 4 * stdev, mean + 4 * stdev, 800)
    y = scipy_stats.norm.pdf(x, loc=mean, scale=stdev)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x, y, linewidth=2, color="steelblue")
    
    x_fill = np.linspace(max(mean - 4 * stdev, lower), min(mean + 4 * stdev, upper), 200)
    y_fill = scipy_stats.norm.pdf(x_fill, loc=mean, scale=stdev)
    ax.fill_between(x_fill, y_fill, alpha=0.5, color="steelblue", label=f"P({lower} ≤ X ≤ {upper})")

    ax.axvline(mean, color="dimgray", linewidth=1, linestyle="-", label=f"μ = {mean}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Normal Distribution  N(μ={mean}, σ={stdev})", fontsize=13)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def _inv_normal_bytes(mean: float, stdev: float, prob: float, x_val: float) -> io.BytesIO:
    x = np.linspace(mean - 4 * stdev, mean + 4 * stdev, 800)
    y = scipy_stats.norm.pdf(x, loc=mean, scale=stdev)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x, y, linewidth=2, color="steelblue")
    
    x_fill = np.linspace(mean - 4 * stdev, x_val, 200)
    y_fill = scipy_stats.norm.pdf(x_fill, loc=mean, scale=stdev)
    ax.fill_between(x_fill, y_fill, alpha=0.5, color="mediumseagreen", label=f"Area = {prob}")

    ax.axvline(x_val, color="crimson", linewidth=1.5, linestyle="--", label=f"x = {x_val:.4f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Inverse Normal  N(μ={mean}, σ={stdev})", fontsize=13)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def _binomial_cdf_bytes(n: int, p: float, x_val: int) -> io.BytesIO:
    x = np.arange(0, n + 1)
    y = scipy_stats.binom.pmf(x, n, p)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["steelblue" if val <= x_val else "lightgray" for val in x]
    ax.bar(x, y, color=colors, edgecolor="black", alpha=0.7)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(f"Binomial Distribution  B(n={n}, p={p})", fontsize=13)
    ax.set_xlabel("Number of Successes (k)", fontsize=11)
    ax.set_ylabel("Probability P(X=k)", fontsize=11)
    
    import matplotlib.patches as mpatches
    shaded_patch = mpatches.Patch(color='steelblue', alpha=0.7, label=f'P(X ≤ {x_val})')
    ax.legend(handles=[shaded_patch], fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def _poisson_cdf_bytes(lam: float, x_val: int) -> io.BytesIO:
    upper_bound = max(15, int(lam + 4 * np.sqrt(lam)))
    x = np.arange(0, upper_bound + 1)
    y = scipy_stats.poisson.pmf(x, lam)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["steelblue" if val <= x_val else "lightgray" for val in x]
    ax.bar(x, y, color=colors, edgecolor="black", alpha=0.7)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(f"Poisson Distribution  Po(λ={lam})", fontsize=13)
    ax.set_xlabel("Number of Events (k)", fontsize=11)
    ax.set_ylabel("Probability P(X=k)", fontsize=11)
    
    import matplotlib.patches as mpatches
    shaded_patch = mpatches.Patch(color='steelblue', alpha=0.7, label=f'P(X ≤ {x_val})')
    ax.legend(handles=[shaded_patch], fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class StatisticsCog(commands.Cog, name="Statistics"):
    """Descriptive statistics and probability distribution commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /mean
    # -----------------------------------------------------------------------

    @app_commands.command(name="mean", description="Compute the arithmetic mean of a data set.")
    @app_commands.describe(data='Comma-separated numbers, e.g. "1, 2, 3, 4, 5"')
    @app_commands.checks.cooldown(1, 2.0)
    async def mean(self, interaction: discord.Interaction, data: str) -> None:
        await interaction.response.defer()
        try:
            nums   = parse_numbers(data)
            result = _stats.mean(nums)
            embed  = math_embed(
                title="Mean",
                result=str(result),
                footer=f"n = {len(nums)}  |  arithmetic mean",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /median
    # -----------------------------------------------------------------------

    @app_commands.command(name="median", description="Compute the median of a data set.")
    @app_commands.describe(data='Comma-separated numbers, e.g. "1, 2, 3, 4, 5"')
    @app_commands.checks.cooldown(1, 2.0)
    async def median(self, interaction: discord.Interaction, data: str) -> None:
        await interaction.response.defer()
        try:
            nums   = parse_numbers(data)
            result = _stats.median(nums)
            embed  = math_embed(
                title="Median",
                result=str(result),
                footer=f"n = {len(nums)}  |  middle value of sorted data",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /mode
    # -----------------------------------------------------------------------

    @app_commands.command(name="mode", description="Find the mode (most common value) of a data set.")
    @app_commands.describe(data='Comma-separated numbers, e.g. "1, 2, 2, 3"')
    @app_commands.checks.cooldown(1, 2.0)
    async def mode(self, interaction: discord.Interaction, data: str) -> None:
        await interaction.response.defer()
        try:
            nums = parse_numbers(data)
            modes = _stats.multimode(nums)
            result_str = ", ".join(str(m) for m in modes)
            footer_note = "all modes shown" if len(modes) > 1 else "unique mode"

            embed = math_embed(
                title="Mode",
                result=result_str,
                footer=f"n = {len(nums)}  |  {footer_note}",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /stdev
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="stdev",
        description="Compute the sample standard deviation of a data set.",
    )
    @app_commands.describe(data='Comma-separated numbers, e.g. "2, 4, 4, 4, 5, 5, 7, 9"')
    @app_commands.checks.cooldown(1, 2.0)
    async def stdev(self, interaction: discord.Interaction, data: str) -> None:
        await interaction.response.defer()
        try:
            nums = parse_numbers(data)
            if len(nums) < 2:
                raise ValueError(
                    "Sample standard deviation requires at least 2 data points."
                )
            result = _stats.stdev(nums)
            embed  = math_embed(
                title="Standard Deviation",
                result=str(result),
                footer=(
                    f"n = {len(nums)}  |  sample std dev (÷ n−1 Bessel correction)  |  "
                    "for population std dev use /variance or divide by n instead"
                ),
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /variance
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="variance",
        description="Compute the sample variance of a data set.",
    )
    @app_commands.describe(data='Comma-separated numbers, e.g. "2, 4, 4, 4, 5, 5, 7, 9"')
    @app_commands.checks.cooldown(1, 2.0)
    async def variance(self, interaction: discord.Interaction, data: str) -> None:
        await interaction.response.defer()
        try:
            nums = parse_numbers(data)
            if len(nums) < 2:
                raise ValueError("Sample variance requires at least 2 data points.")
            result = _stats.variance(nums)
            embed  = math_embed(
                title="Variance",
                result=str(result),
                footer=(
                    f"n = {len(nums)}  |  sample variance (÷ n−1)  |  "
                    "population variance divides by n"
                ),
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /zscore
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="zscore",
        description="Compute the z-score (standard score) for a value.",
    )
    @app_commands.describe(
        value="The data point to standardise",
        mean="Population or sample mean",
        stdev="Population or sample standard deviation",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def zscore(
        self,
        interaction: discord.Interaction,
        value: float,
        mean: float,
        stdev: float,
    ) -> None:
        await interaction.response.defer()
        try:
            if stdev == 0:
                raise ValueError(
                    "Standard deviation is 0 — z-score is undefined when all values are identical."
                )
            z = (value - mean) / stdev

            steps = [
                ("Formula",      "z = (x − μ) / σ"),
                ("Substitute",   f"z = ({value} − {mean}) / {stdev}"),
                ("Numerator",    f"{value} − {mean} = {value - mean}"),
                ("Result",       f"z = {value - mean} / {stdev} = {z:.6g}"),
            ]
            embed = math_embed(
                title="Z-Score",
                result=f"{z:.6g}",
                steps=steps,
                footer=f"x = {value}  |  μ = {mean}  |  σ = {stdev}",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /correlation
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="correlation",
        description="Compute the Pearson correlation coefficient between two data series.",
    )
    @app_commands.describe(
        data_x='First series as comma-separated numbers, e.g. "1,2,3,4"',
        data_y='Second series as comma-separated numbers, e.g. "2,4,5,4"',
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def correlation(
        self,
        interaction: discord.Interaction,
        data_x: str,
        data_y: str,
    ) -> None:
        await interaction.response.defer()
        try:
            xs = parse_numbers(data_x)
            ys = parse_numbers(data_y)
            if len(xs) != len(ys):
                raise ValueError(
                    f"data_x and data_y must have the same length "
                    f"(got {len(xs)} and {len(ys)})."
                )
            if len(xs) < 2:
                raise ValueError("Correlation requires at least 2 data points.")

            r = float(np.corrcoef(xs, ys)[0, 1])
            label = _correlation_label(r)

            steps = [
                ("Pearson r formula",  "r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / √[Σ(xᵢ−x̄)²·Σ(yᵢ−ȳ)²]"),
                ("Computed r",         f"r = {r:.6f}"),
                ("Interpretation",     f"|r| = {abs(r):.6f}  →  {label} correlation"),
            ]
            embed = math_embed(
                title="Pearson Correlation",
                result=f"r = {r:.6f}",
                steps=steps,
                footer=f"n = {len(xs)}  |  {label} correlation  |  r ∈ [−1, 1]",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /regression
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="regression",
        description="Fit a linear regression to two data series and plot the result.",
    )
    @app_commands.describe(
        data_x='Independent variable as comma-separated numbers',
        data_y='Dependent variable as comma-separated numbers',
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def regression(
        self,
        interaction: discord.Interaction,
        data_x: str,
        data_y: str,
    ) -> None:
        await interaction.response.defer()
        try:
            xs = parse_numbers(data_x)
            ys = parse_numbers(data_y)
            if len(xs) != len(ys):
                raise ValueError(
                    f"data_x and data_y must have the same length "
                    f"(got {len(xs)} and {len(ys)})."
                )
            if len(xs) < 2:
                raise ValueError("Regression requires at least 2 data points.")

            slope, intercept = np.polyfit(xs, ys, 1)
            r               = float(np.corrcoef(xs, ys)[0, 1])
            r_squared       = r ** 2
            label           = _correlation_label(r)

            sign    = "+" if intercept >= 0 else "−"
            abs_int = abs(intercept)
            eq_str  = f"y = {slope:.6g}x  {sign}  {abs_int:.6g}"

            steps = [
                ("Model",           "y = mx + b  (ordinary least squares)"),
                ("Slope  m",        f"m = {slope:.6g}"),
                ("Intercept  b",    f"b = {intercept:.6g}"),
                ("Equation",        eq_str),
                ("R² (fit quality)", f"R² = {r_squared:.6f}  ({label} fit)"),
            ]

            embed = math_embed(
                title="Linear Regression",
                result=eq_str,
                steps=steps,
                footer=f"n = {len(xs)}  |  R² = {r_squared:.4f}  |  {label} fit",
            )
            embed.set_image(url="attachment://plot.png")

            loop = asyncio.get_running_loop()
            buf  = await loop.run_in_executor(
                None, _regression_plot_bytes, xs, ys, slope, intercept, r_squared
            )
            file = discord.File(buf, filename="plot.png")

            await interaction.followup.send(embed=embed, file=file)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /normal_pdf
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="normal_pdf",
        description="Plot the probability density function of a normal distribution.",
    )
    @app_commands.describe(
        mean="Mean (μ) of the distribution",
        stdev="Standard deviation (σ) — must be > 0",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def normal_pdf(
        self,
        interaction: discord.Interaction,
        mean: float,
        stdev: float,
    ) -> None:
        await interaction.response.defer()
        try:
            if stdev <= 0:
                raise ValueError(
                    f"Standard deviation must be positive (got σ = {stdev})."
                )

            peak = scipy_stats.norm.pdf(mean, loc=mean, scale=stdev)

            embed = discord.Embed(
                title=f"Normal Distribution  N(μ={mean}, σ={stdev})",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name="Parameters",
                value=f"μ = {mean}   |   σ = {stdev}   |   σ² = {stdev**2:.6g}",
                inline=False,
            )
            embed.add_field(
                name="Key values",
                value=(
                    f"Peak PDF  = {peak:.6g}\n"
                    f"68% of data within  [μ−σ, μ+σ]  =  [{mean-stdev:.6g}, {mean+stdev:.6g}]\n"
                    f"95% of data within  [μ−2σ, μ+2σ] =  [{mean-2*stdev:.6g}, {mean+2*stdev:.6g}]\n"
                    f"99.7% of data within [μ−3σ, μ+3σ] = [{mean-3*stdev:.6g}, {mean+3*stdev:.6g}]"
                ),
                inline=False,
            )
            embed.set_footer(text="Dashed lines mark ±1σ, ±2σ, ±3σ from the mean.")
            embed.set_image(url="attachment://normal_pdf.png")

            loop = asyncio.get_running_loop()
            buf  = await loop.run_in_executor(None, _normal_pdf_bytes, mean, stdev)
            file = discord.File(buf, filename="normal_pdf.png")

            await interaction.followup.send(embed=embed, file=file)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /normal_cdf
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="normal_cdf",
        description="Compute the cumulative probability P(lower ≤ X ≤ upper) for a normal distribution.",
    )
    @app_commands.describe(
        mean="Mean (μ) of the distribution",
        stdev="Standard deviation (σ) — must be > 0",
        lower="Lower bound (default: -oo)",
        upper="Upper bound",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def normal_cdf(
        self,
        interaction: discord.Interaction,
        mean: float,
        stdev: float,
        upper: float,
        lower: float = -np.inf,
    ) -> None:
        await interaction.response.defer()
        try:
            if stdev <= 0:
                raise ValueError("Standard deviation must be positive.")
                
            prob = scipy_stats.norm.cdf(upper, loc=mean, scale=stdev) - scipy_stats.norm.cdf(lower, loc=mean, scale=stdev)
            lower_str = "-∞" if lower == -np.inf else str(lower)
            
            embed = math_embed(
                title=f"Normal CDF  N(μ={mean}, σ={stdev})",
                result=f"P({lower_str} ≤ X ≤ {upper}) = {prob:.6g}",
            )
            embed.set_image(url="attachment://normal_cdf.png")

            loop = asyncio.get_running_loop()
            buf  = await loop.run_in_executor(None, _normal_cdf_bytes, mean, stdev, lower, upper)
            file = discord.File(buf, filename="normal_cdf.png")

            await interaction.followup.send(embed=embed, file=file)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /inv_normal
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="inv_normal",
        description="Inverse normal: find x given P(X ≤ x) = probability.",
    )
    @app_commands.describe(
        probability="Cumulative probability P(X ≤ x) between 0 and 1",
        mean="Mean (μ) of the distribution",
        stdev="Standard deviation (σ) — must be > 0",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def inv_normal(
        self,
        interaction: discord.Interaction,
        probability: float,
        mean: float,
        stdev: float,
    ) -> None:
        await interaction.response.defer()
        try:
            if stdev <= 0:
                raise ValueError("Standard deviation must be positive.")
            if not 0 < probability < 1:
                raise ValueError("Probability must be strictly between 0 and 1.")
                
            x_val = scipy_stats.norm.ppf(probability, loc=mean, scale=stdev)
            
            embed = math_embed(
                title=f"Inverse Normal  N(μ={mean}, σ={stdev})",
                result=f"x = {x_val:.6g}",
                footer=f"P(X ≤ {x_val:.6g}) = {probability}",
            )
            embed.set_image(url="attachment://inv_normal.png")

            loop = asyncio.get_running_loop()
            buf  = await loop.run_in_executor(None, _inv_normal_bytes, mean, stdev, probability, x_val)
            file = discord.File(buf, filename="inv_normal.png")

            await interaction.followup.send(embed=embed, file=file)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /binomial_cdf
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="binomial_cdf",
        description="Compute P(X ≤ x) for a Binomial distribution.",
    )
    @app_commands.describe(
        n="Number of trials",
        p="Probability of success on each trial (0 to 1)",
        x="Number of successes to compute cumulative probability for",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def binomial_cdf(
        self,
        interaction: discord.Interaction,
        n: int,
        p: float,
        x: int,
    ) -> None:
        await interaction.response.defer()
        try:
            if n <= 0:
                raise ValueError("Number of trials (n) must be positive.")
            if not 0 <= p <= 1:
                raise ValueError("Probability (p) must be between 0 and 1.")
            if x < 0 or x > n:
                raise ValueError(f"Successes (x) must be between 0 and {n}.")
                
            prob = scipy_stats.binom.cdf(x, n, p)
            pmf_val = scipy_stats.binom.pmf(x, n, p)
            
            embed = math_embed(
                title=f"Binomial CDF  B(n={n}, p={p})",
                result=f"P(X ≤ {x}) = {prob:.6g}",
                footer=f"P(X = {x}) = {pmf_val:.6g}",
            )
            embed.set_image(url="attachment://binomial_cdf.png")

            loop = asyncio.get_running_loop()
            buf  = await loop.run_in_executor(None, _binomial_cdf_bytes, n, p, x)
            file = discord.File(buf, filename="binomial_cdf.png")

            await interaction.followup.send(embed=embed, file=file)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /poisson_cdf
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="poisson_cdf",
        description="Compute P(X ≤ x) for a Poisson distribution.",
    )
    @app_commands.describe(
        lam="Average rate of success (λ) — must be > 0",
        x="Number of occurrences to compute cumulative probability for",
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def poisson_cdf(
        self,
        interaction: discord.Interaction,
        lam: float,
        x: int,
    ) -> None:
        await interaction.response.defer()
        try:
            if lam <= 0:
                raise ValueError("Average rate (λ) must be positive.")
            if x < 0:
                raise ValueError("Occurrences (x) cannot be negative.")
                
            prob = scipy_stats.poisson.cdf(x, lam)
            pmf_val = scipy_stats.poisson.pmf(x, lam)
            
            embed = math_embed(
                title=f"Poisson CDF  Po(λ={lam})",
                result=f"P(X ≤ {x}) = {prob:.6g}",
                footer=f"P(X = {x}) = {pmf_val:.6g}",
            )
            embed.set_image(url="attachment://poisson_cdf.png")

            loop = asyncio.get_running_loop()
            buf  = await loop.run_in_executor(None, _poisson_cdf_bytes, lam, x)
            file = discord.File(buf, filename="poisson_cdf.png")

            await interaction.followup.send(embed=embed, file=file)
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the StatisticsCog into *bot*."""
    await bot.add_cog(StatisticsCog(bot))