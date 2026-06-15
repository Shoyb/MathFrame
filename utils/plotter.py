"""
utils/plotter.py — Matplotlib plot generator for the math bot.

Produces PNG plot images and returns them as ``discord.File`` objects
ready to attach to any interaction reply.

Both public functions are async and safe to call concurrently: the
blocking matplotlib work runs inside a thread-pool executor so it never
stalls the event loop.

Usage
-----
::

    from utils.plotter import plot_function, plot_points

    # Plot a SymPy expression over a range:
    import sympy
    x = sympy.Symbol("x")
    expr = sympy.parse_expr("sin(x) / x")
    file = await plot_function(expr, x, x_min=-10, x_max=10, title="sinc(x)")
    await interaction.followup.send(file=file)

    # Scatter plot for stats / regression data:
    file = await plot_points(xs, ys, title="Regression", xlabel="x", ylabel="y")
    await interaction.followup.send(file=file)
"""

import asyncio
import io
import warnings
from concurrent.futures import ThreadPoolExecutor

import discord
import matplotlib
import numpy as np
import sympy

matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)

# ---------------------------------------------------------------------------
# Module-level constants and executor
# ---------------------------------------------------------------------------

_PLOT_POINTS = 800          # number of x-samples for function plots
_Y_CLIP = 1e6               # clip y values to [-_Y_CLIP, _Y_CLIP] before plotting

# Dedicated executor so plotting never competes with the renderer's pool.
_plot_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plotter")

# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

def _apply_axes_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """
    Apply the bot's standard axes style to *ax* in-place.

    Adds a subtle grid, zero-crossing reference lines, labels, and a title.

    Parameters
    ----------
    ax:
        The matplotlib :class:`~matplotlib.axes.Axes` to style.
    title:
        Plot title string.
    xlabel:
        Label for the x-axis.
    ylabel:
        Label for the y-axis.
    """
    # Zero-crossing reference lines
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def _save_fig_to_bytes(fig: plt.Figure) -> io.BytesIO:
    """
    Save *fig* to a PNG :class:`io.BytesIO` buffer and close the figure.

    Returns
    -------
    io.BytesIO
        Seeked to position 0, ready for reading or passing to
        :class:`discord.File`.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=150,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)   # release figure memory immediately
    buf.seek(0)
    return buf

# ---------------------------------------------------------------------------
# Blocking implementations (run inside the executor)
# ---------------------------------------------------------------------------

def _plot_function_blocking(
    expr: sympy.Expr,
    var: sympy.Symbol,
    x_min: float,
    x_max: float,
    title: str,
) -> io.BytesIO:
    """
    Evaluate *expr* over [*x_min*, *x_max*] and render a line plot.

    Numpy warnings for ``inf`` / ``nan`` values are suppressed; the
    resulting y-array is clipped to ``[-_Y_CLIP, _Y_CLIP]`` so that
    singularities (e.g. ``tan(x)`` near π/2) don't collapse the scale.

    Parameters
    ----------
    expr:
        A SymPy expression in *var*.
    var:
        The independent variable symbol.
    x_min, x_max:
        Domain endpoints.
    title:
        Plot title; falls back to ``str(expr)`` if empty.

    Returns
    -------
    io.BytesIO
        PNG bytes seeked to 0.
    """
    f = sympy.lambdify(var, expr, modules=["numpy"])

    xs = np.linspace(x_min, x_max, _PLOT_POINTS)

    # Suppress numpy's RuntimeWarning for division-by-zero / overflow so the
    # bot doesn't spam logs; we handle the bad values via clipping instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ys = np.asarray(f(xs), dtype=float)

    # Replace non-finite values with NaN so matplotlib draws gaps rather than
    # wild spikes, then clip finite values to a sensible range.
    ys = np.where(np.isfinite(ys), np.clip(ys, -_Y_CLIP, _Y_CLIP), np.nan)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(xs, ys, linewidth=2)

    _apply_axes_style(
        ax,
        title=title or str(expr),
        xlabel=str(var),
        ylabel="f({})".format(str(var)),
    )

    fig.tight_layout()
    return _save_fig_to_bytes(fig)


def _plot_points_blocking(
    xs: list,
    ys: list,
    title: str,
    xlabel: str,
    ylabel: str,
) -> io.BytesIO:
    """
    Render a scatter plot from parallel *xs* / *ys* sequences.

    Parameters
    ----------
    xs, ys:
        Equal-length sequences of numeric values.
    title:
        Plot title.
    xlabel, ylabel:
        Axis labels.

    Returns
    -------
    io.BytesIO
        PNG bytes seeked to 0.
    """
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(xs_arr, ys_arr, s=30, zorder=3)

    _apply_axes_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    fig.tight_layout()
    return _save_fig_to_bytes(fig)

# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def plot_function(
    expr: sympy.Expr,
    var: sympy.Symbol,
    x_min: float = -10,
    x_max: float = 10,
    title: str = "",
) -> discord.File:
    """
    Plot a SymPy expression as a line graph and return a :class:`discord.File`.

    The expression is evaluated at 800 equally-spaced points across
    [*x_min*, *x_max*].  Infinite or NaN values are replaced with gaps
    in the line rather than raising an error, so functions with
    singularities (e.g. ``1/x``, ``tan(x)``) plot cleanly.

    Parameters
    ----------
    expr:
        A SymPy expression to plot, e.g. ``sympy.parse_expr("sin(x)/x")``.
    var:
        The free symbol to use as the x-axis variable.
    x_min:
        Left boundary of the plot domain (default ``-10``).
    x_max:
        Right boundary of the plot domain (default ``10``).
    title:
        Optional plot title.  If omitted, ``str(expr)`` is used.

    Returns
    -------
    discord.File
        A file named ``plot.png``.

    Raises
    ------
    ValueError
        If lambdification or plotting fails for any reason.

    Example
    -------
    ::

        x = sympy.Symbol("x")
        file = await plot_function(sympy.sin(x) / x, x, title="sinc(x)")
        await interaction.followup.send(file=file)
    """
    if x_min >= x_max:
        raise ValueError(
            f"x_min ({x_min}) must be strictly less than x_max ({x_max})."
        )

    loop = asyncio.get_event_loop()
    try:
        buf = await loop.run_in_executor(
            _plot_executor,
            _plot_function_blocking,
            expr, var, x_min, x_max, title,
        )
    except Exception as exc:
        raise ValueError(f"Could not generate plot: {exc}") from exc

    return discord.File(buf, filename="plot.png")


async def plot_points(
    xs: list,
    ys: list,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
) -> discord.File:
    """
    Render a scatter plot from paired data lists and return a
    :class:`discord.File`.

    Intended for use by statistics and regression cogs where the data
    points are already computed (e.g. residual plots, fitted curves).

    Parameters
    ----------
    xs:
        Sequence of numeric x-values.
    ys:
        Sequence of numeric y-values; must be the same length as *xs*.
    title:
        Optional plot title.
    xlabel:
        Label for the x-axis (default ``"x"``).
    ylabel:
        Label for the y-axis (default ``"y"``).

    Returns
    -------
    discord.File
        A file named ``plot.png``.

    Raises
    ------
    ValueError
        If *xs* and *ys* have different lengths, or if plotting fails.

    Example
    -------
    ::

        file = await plot_points(
            xs=[1, 2, 3, 4, 5],
            ys=[2.1, 3.9, 6.2, 8.0, 9.8],
            title="Linear Regression",
            xlabel="x",
            ylabel="y",
        )
        await interaction.followup.send(file=file)
    """
    if len(xs) != len(ys):
        raise ValueError(
            f"xs and ys must have the same length "
            f"(got {len(xs)} and {len(ys)})."
        )
    if not xs:
        raise ValueError("Cannot plot an empty data set.")

    loop = asyncio.get_event_loop()
    try:
        buf = await loop.run_in_executor(
            _plot_executor,
            _plot_points_blocking,
            xs, ys, title, xlabel, ylabel,
        )
    except Exception as exc:
        raise ValueError(f"Could not generate scatter plot: {exc}") from exc

    return discord.File(buf, filename="plot.png")
