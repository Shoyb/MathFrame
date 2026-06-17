"""
utils/plotter.py — Matplotlib plot generator for the math bot.

Produces PNG plot images and returns them as ``discord.File`` objects
ready to attach to any interaction reply.

All public functions are async and safe to call concurrently: blocking
matplotlib work runs inside a thread-pool executor so it never stalls
the event loop.  Each blocking call uses ``matplotlib.rc_context`` so
that style overrides are fully isolated — concurrent renders never
clobber each other's rcParams.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Public API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2-D single-variable
    plot_function       — line plot of f(x)
    plot_points         — scatter plot of (xs, ys) data

2-D multi-variable
    plot_contour        — filled contour map of f(x, y)
    plot_vector_field   — 2-D quiver / streamplot of a vector field
    plot_parametric_2d  — parametric curve (x(t), y(t))

3-D
    plot_surface        — 3-D surface of f(x, y)
    plot_wireframe      — 3-D wireframe of f(x, y)
    plot_parametric_3d  — 3-D parametric curve (x(t), y(t), z(t))
    plot_scatter_3d     — 3-D scatter of (xs, ys, zs) data

Multi-panel
    plot_multi          — arbitrary grid of sub-plots from a list of
                          PlotSpec descriptors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adding a new plot type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Write a ``_plot_<name>_blocking(...)`` function that accepts plain
   Python / NumPy / SymPy arguments, builds a matplotlib Figure, and
   returns ``_save_fig_to_bytes(fig)``.  Accept a ``style: StyleOptions``
   keyword argument and pass it to ``_apply_line_style`` / ``_apply_axes_style``
   as appropriate.

2. Write the matching ``async def plot_<name>(...)`` public coroutine
   that validates inputs, calls ``loop.run_in_executor(_plot_executor,
   _plot_<name>_blocking, ...)`` inside ``_run_blocking``, and returns a
   ``discord.File``.

3. If the type can appear inside ``plot_multi``, add a branch for it in
   ``_render_spec_onto_axes`` and add the ``kind`` string to the docstring
   of ``PlotSpec``.

4. Export the new function in ``__all__`` at the bottom of this file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
::

    from utils.plotter import (
        plot_function, plot_points,
        plot_contour, plot_vector_field, plot_parametric_2d,
        plot_surface, plot_wireframe, plot_parametric_3d, plot_scatter_3d,
        plot_multi, PlotSpec, StyleOptions,
    )

    import sympy
    x = sympy.Symbol("x")

    # Line plot with custom style
    style = StyleOptions(color="#e74c3c", line_width=2.5)
    file  = await plot_function(sympy.sin(x) / x, x, title="sinc(x)", style=style)

    # 3-D surface
    x, y = sympy.symbols("x y")
    file  = await plot_surface(sympy.sin(x) * sympy.cos(y), x, y, title="sin·cos")

    # Multi-panel (2×2 grid)
    specs = [
        PlotSpec("function",  expr=sympy.sin(x), var=x, title="sin(x)"),
        PlotSpec("function",  expr=sympy.cos(x), var=x, title="cos(x)"),
        PlotSpec("contour",   expr=x**2 + y**2,  x_var=x, y_var=y, title="Paraboloid"),
        PlotSpec("points",    xs=[1,2,3], ys=[1,4,9], title="y=x²"),
    ]
    file = await plot_multi(specs, ncols=2, title="Gallery")

    await interaction.followup.send(file=file)
"""

from __future__ import annotations

import asyncio
import io
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import discord
import matplotlib
import numpy as np
import sympy

matplotlib.use("Agg")  # headless — must be before pyplot import

import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib import cm                              # noqa: E402
from matplotlib.collections import LineCollection      # noqa: E402
from mpl_toolkits.mplot3d import Axes3D               # noqa: F401 (registers projection)

# ---------------------------------------------------------------------------
# Module-level constants  (read-only after import — never mutate at runtime)
# ---------------------------------------------------------------------------

PLOT_POINTS  = 800    # x-samples for 1-D line plots
GRID_POINTS  = 120    # grid resolution for 2-D / 3-D surface plots
Y_CLIP       = 1e6    # clip |y| beyond this for 1-D plots
Z_CLIP       = 1e6    # clip |z| beyond this for surface plots
PARAM_POINTS = 1000   # t-samples for parametric curves

# Worker pool — 4 threads is enough; matplotlib is CPU-bound not I/O-bound.
_plot_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="plotter")

# ---------------------------------------------------------------------------
# StyleOptions — centralised style bag passed into every blocking function
# ---------------------------------------------------------------------------

@dataclass
class StyleOptions:
    """
    All visual style knobs that can be applied to a plot.

    Every public plotting function accepts an optional ``style`` keyword
    argument of this type.  Defaults reproduce the original bot appearance.
    Fields that don't apply to a particular plot type are silently ignored.

    Attributes
    ----------
    color : str
        Line / scatter colour (matplotlib colour string or hex).
    line_width : float
        Stroke width for lines and wireframes.
    line_style : str
        Matplotlib line-style string: ``"-"``, ``"--"``, ``":"`, ``"-."``.
    marker : str | None
        Marker style.  ``None`` means no markers.
    marker_size : float
        Marker diameter in points.
    colormap : str
        Named matplotlib colormap used for surfaces, contours, vector fields,
        parametric curves, and 3-D scatter.
    alpha : float
        Opacity 0–1 for surfaces.
    show_grid : bool
        Whether to draw a background grid.
    dpi : int
        Output resolution for the saved PNG.
    fig_width : float
        Figure width in inches.
    fig_height : float
        Figure height in inches.

    Notes
    -----
    Add new style fields here as the bot grows — every blocking function
    already receives the whole ``StyleOptions`` object, so new fields are
    available everywhere without changing any call sites.
    """

    color:      str   = "#1f77b4"
    line_width: float = 2.0
    line_style: str   = "-"
    marker:     Optional[str] = None
    marker_size: float = 6.0
    colormap:   str   = "viridis"
    theme:      str   = "default"
    alpha:      float = 0.9
    show_grid:  bool  = True
    dpi:        int   = 150
    fig_width:  float = 8.0
    fig_height: float = 4.5

    def rc_overrides(self) -> Dict:
        """
        Return a dict suitable for ``matplotlib.rc_context(rc=...)``.
        """
        overrides = {
            "lines.linewidth":  self.line_width,
            "lines.linestyle":  self.line_style,
            "lines.color":      self.color,
            "lines.markersize": self.marker_size,
            "lines.marker":     self.marker or "None",
            "scatter.marker":   self.marker or "o",
            "axes.grid":        self.show_grid,
            "grid.alpha":       0.3,
            "figure.dpi":       self.dpi,
        }
        
        if self.theme == "dark":
            overrides.update({
                "axes.facecolor": "#2c2f33",
                "figure.facecolor": "#2c2f33",
                "axes.edgecolor": "#ffffff",
                "axes.labelcolor": "#ffffff",
                "text.color": "#ffffff",
                "xtick.color": "#ffffff",
                "ytick.color": "#ffffff",
                "grid.color": "#ffffff",
            })
        elif self.theme == "cyberpunk":
            overrides.update({
                "axes.facecolor": "#212946",
                "figure.facecolor": "#212946",
                "axes.edgecolor": "#08F7FE",
                "axes.labelcolor": "#08F7FE",
                "text.color": "#08F7FE",
                "xtick.color": "#08F7FE",
                "ytick.color": "#08F7FE",
                "grid.color": "#08F7FE",
                "lines.linewidth": 3.0,
            })
        elif self.theme == "academic":
            overrides.update({
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "axes.facecolor": "#ffffff",
                "figure.facecolor": "#ffffff",
                "axes.grid": True,
                "grid.color": "#cccccc",
                "grid.linestyle": "--",
            })
        elif self.theme == "seaborn":
            overrides.update({
                "axes.facecolor": "#EAEAF2",
                "figure.facecolor": "white",
                "axes.edgecolor": "white",
                "grid.color": "white",
                "grid.linestyle": "-",
                "axes.axisbelow": True,
            })
            
        return overrides


# Default style instance — used when callers don't supply one.
_DEFAULT_STYLE = StyleOptions()

# ---------------------------------------------------------------------------
# PlotSpec — descriptor for multi-panel plots
# ---------------------------------------------------------------------------

@dataclass
class PlotSpec:
    """
    Descriptor for one sub-plot panel inside :func:`plot_multi`.

    Parameters
    ----------
    kind : str
        One of ``"function"``, ``"points"``, ``"contour"``,
        ``"vector_field"``, ``"parametric_2d"``.

        .. note::
            3-D kinds (``"surface"``, ``"wireframe"``, ``"parametric_3d"``,
            ``"scatter_3d"``) cannot be mixed into a ``plot_multi`` grid
            because matplotlib's subplot grid does not support mixed 2-D/3-D
            projections.  Use the dedicated 3-D async functions for those.

    title : str
        Sub-plot title.
    style : StyleOptions | None
        Per-panel style override; ``None`` inherits the grid's default style.

    2-D single-variable (kind="function")
        expr, var, x_min, x_max

    Scatter (kind="points")
        xs, ys, xlabel, ylabel

    2-D multivariable (kind="contour")
        expr, x_var, y_var, x_range, y_range

    Vector field (kind="vector_field")
        u_expr, v_expr, x_var, y_var, x_range, y_range,
        stream (bool) — use streamplot instead of quiver

    Parametric 2-D (kind="parametric_2d")
        x_expr, y_expr, t_var, t_min, t_max, xlabel, ylabel
    """

    kind: str

    # shared
    title: str = ""
    style: Optional[StyleOptions] = None   # None → use caller's default

    # 1-D function
    expr: Optional[sympy.Expr] = None
    var:  Optional[sympy.Symbol] = None
    x_min: float = -10.0
    x_max: float  = 10.0

    # scatter / points
    xs: Optional[list] = None
    ys: Optional[list] = None
    zs: Optional[list] = None
    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"

    # 2-D / 3-D multivariable
    x_var: Optional[sympy.Symbol] = None
    y_var: Optional[sympy.Symbol] = None
    x_range: Tuple[float, float] = (-5.0, 5.0)
    y_range: Tuple[float, float] = (-5.0, 5.0)

    # vector field
    u_expr: Optional[sympy.Expr] = None
    v_expr: Optional[sympy.Expr] = None
    stream: bool = False

    # parametric
    x_expr: Optional[sympy.Expr] = None
    y_expr: Optional[sympy.Expr] = None
    z_expr: Optional[sympy.Expr] = None
    t_var:  Optional[sympy.Symbol] = None
    t_min:  float = 0.0
    t_max:  float = 2 * float(sympy.pi)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_axes_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    show_grid: bool = True,
) -> None:
    """Apply the bot's standard 2-D axes style."""
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(show_grid, alpha=0.3)
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)


def _apply_line_style(ax: plt.Axes, style: StyleOptions) -> None:
    """
    Retroactively apply line style to the most-recently-added Line2D on *ax*.

    Call this right after ``ax.plot(...)`` to honour per-call style options
    without relying solely on rcParams (which are set via rc_context anyway,
    but an explicit override makes the intent crystal-clear).
    """
    if not ax.lines:
        return
    line = ax.lines[-1]
    line.set_color(style.color)
    line.set_linewidth(style.line_width)
    line.set_linestyle(style.line_style)
    if style.marker:
        line.set_marker(style.marker)
        line.set_markersize(style.marker_size)


def _save_fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    """Save *fig* to a PNG BytesIO buffer and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _white_fig(style: StyleOptions, **kw) -> Tuple[plt.Figure, plt.Axes]:
    """Create a white-background figure + axes pair, sized from *style*."""
    kw.setdefault("figsize", (style.fig_width, style.fig_height))
    fig, ax = plt.subplots(**kw)
    # Facecolor is handled by rcParams
    return fig, ax


def _make_3d_axes(style: StyleOptions) -> Tuple[plt.Figure, "Axes3D"]:
    """Create a white-background figure with a 3-D projection axes."""
    fig = plt.figure(figsize=(style.fig_width, style.fig_height))
    # Facecolor is handled by rcParams
    ax  = fig.add_subplot(111, projection="3d")
    # Facecolor is handled by rcParams
    return fig, ax


def _lambdify1(expr: sympy.Expr, var: sympy.Symbol) -> Callable:
    return sympy.lambdify(var, expr, modules=["numpy"])


def _lambdify2(expr: sympy.Expr,
               xv: sympy.Symbol,
               yv: sympy.Symbol) -> Callable:
    return sympy.lambdify((xv, yv), expr, modules=["numpy"])


def _eval1(f: Callable, xs: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ys = np.asarray(f(xs), dtype=float)
    return np.where(np.isfinite(ys), np.clip(ys, -Y_CLIP, Y_CLIP), np.nan)


def _eval2(f: Callable, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Z = np.asarray(f(X, Y), dtype=float)
    return np.where(np.isfinite(Z), np.clip(Z, -Z_CLIP, Z_CLIP), np.nan)


def _meshgrid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n: int = GRID_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    return np.meshgrid(xs, ys)


async def _run_blocking(fn: Callable, *args) -> io.BytesIO:
    """
    Run a blocking plot function in the thread-pool executor.

    All public async functions funnel through here so error handling
    and executor dispatch are in one place.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_plot_executor, fn, *args)
    except Exception as exc:
        raise ValueError(f"Plot failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Blocking implementations
# ---------------------------------------------------------------------------
# Convention:
#   • Every function receives *style* as its last positional argument.
#   • Every function wraps its body in ``with matplotlib.rc_context(rc=style.rc_overrides()):``
#     so that concurrent calls never share global state.
# ---------------------------------------------------------------------------

# ── 2-D single-variable ────────────────────────────────────────────────────

def _plot_function_blocking(
    expr: sympy.Expr,
    var:  sympy.Symbol,
    x_min: float,
    x_max: float,
    title: str,
    style: StyleOptions,
    additional_exprs: list = None,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        f  = _lambdify1(expr, var)
        xs = np.linspace(x_min, x_max, PLOT_POINTS)
        ys = _eval1(f, xs)

        fig, ax = _white_fig(style)
        ax.plot(xs, ys, label=str(expr))
        _apply_line_style(ax, style)
        
        if additional_exprs:
            for extra in additional_exprs:
                f_extra = _lambdify1(extra, var)
                ys_extra = _eval1(f_extra, xs)
                ax.plot(xs, ys_extra, label=str(extra))
            ax.legend(loc="upper right")
            
        _apply_axes_style(ax, title or str(expr), str(var), f"f({var})", style.show_grid)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_points_blocking(
    xs: list,
    ys: list,
    title: str,
    xlabel: str,
    ylabel: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        fig, ax = _white_fig(style)
        # Facecolor is handled by rcParams
        ax.scatter(
            np.asarray(xs, float),
            np.asarray(ys, float),
            s=style.marker_size ** 2,
            color=style.color,
            zorder=3,
        )
        _apply_axes_style(ax, title, xlabel, ylabel, style.show_grid)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


# ── 2-D multivariable ──────────────────────────────────────────────────────

def _plot_contour_blocking(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title:   str,
    levels:  int,
    style:   StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range)
        Z    = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        # Facecolor is handled by rcParams

        cf = ax.contourf(X, Y, Z, levels=levels, cmap=style.colormap, alpha=0.85)
        ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.85, label=f"f({x_var},{y_var})")

        ax.set_title(title or str(expr), fontsize=12, pad=6)
        ax.set_xlabel(str(x_var), fontsize=10)
        ax.set_ylabel(str(y_var), fontsize=10)
        ax.grid(style.show_grid, alpha=0.2)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_vector_field_blocking(
    u_expr:  sympy.Expr,
    v_expr:  sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title:   str,
    stream:  bool,
    density: float,
    style:   StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n  = 24 if not stream else GRID_POINTS
        xs = np.linspace(x_range[0], x_range[1], n)
        ys = np.linspace(y_range[0], y_range[1], n)
        X, Y = np.meshgrid(xs, ys)

        fu = _lambdify2(u_expr, x_var, y_var)
        fv = _lambdify2(v_expr, x_var, y_var)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            U = np.asarray(fu(X, Y), dtype=float)
            V = np.asarray(fv(X, Y), dtype=float)

        mag      = np.hypot(U, V)
        mag_safe = np.where(mag == 0, 1, mag)

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        # Facecolor is handled by rcParams

        if stream:
            sp = ax.streamplot(
                xs, ys, U, V,
                color=mag, cmap=style.colormap,
                density=density, linewidth=style.line_width,
                arrowsize=1.2,
            )
            fig.colorbar(sp.lines, ax=ax, shrink=0.85, label="magnitude")
        else:
            q = ax.quiver(
                X, Y, U / mag_safe, V / mag_safe, mag,
                cmap=style.colormap, scale=25, width=0.003,
            )
            fig.colorbar(q, ax=ax, shrink=0.85, label="magnitude")

        ax.set_title(title or f"({u_expr}, {v_expr})", fontsize=12, pad=6)
        ax.set_xlabel(str(x_var), fontsize=10)
        ax.set_ylabel(str(y_var), fontsize=10)
        ax.grid(style.show_grid, alpha=0.2)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_parametric_2d_blocking(
    x_expr: sympy.Expr,
    y_expr: sympy.Expr,
    t_var:  sympy.Symbol,
    t_min:  float,
    t_max:  float,
    title:  str,
    xlabel: str,
    ylabel: str,
    style:  StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        fx = _lambdify1(x_expr, t_var)
        fy = _lambdify1(y_expr, t_var)
        ts = np.linspace(t_min, t_max, PARAM_POINTS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xs = np.asarray(fx(ts), dtype=float)
            ys = np.asarray(fy(ts), dtype=float)

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        # Facecolor is handled by rcParams

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        lc     = LineCollection(segs, cmap=style.colormap, linewidth=style.line_width)
        lc.set_array(ts[:-1])
        ax.add_collection(lc)
        ax.autoscale()
        fig.colorbar(lc, ax=ax, shrink=0.85, label=str(t_var))

        ax.set_title(title or f"({x_expr}, {y_expr})", fontsize=12, pad=6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(style.show_grid, alpha=0.3)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


# ── 3-D ────────────────────────────────────────────────────────────────────

def _plot_surface_blocking(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title:   str,
    style:   StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range)
        Z    = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _make_3d_axes(style)
        surf = ax.plot_surface(
            X, Y, Z,
            cmap=style.colormap, alpha=style.alpha,
            linewidth=0, antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=f"f({x_var},{y_var})")

        ax.set_title(title or str(expr), fontsize=12)
        ax.set_xlabel(str(x_var), fontsize=9)
        ax.set_ylabel(str(y_var), fontsize=9)
        ax.set_zlabel(f"f({x_var},{y_var})", fontsize=9)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_wireframe_blocking(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title:   str,
    rstride: int,
    cstride: int,
    style:   StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n    = min(GRID_POINTS, 60)
        X, Y = _meshgrid(x_range, y_range, n)
        Z    = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _make_3d_axes(style)
        ax.plot_wireframe(
            X, Y, Z,
            color=style.color, linewidth=style.line_width,
            rstride=rstride, cstride=cstride,
        )

        ax.set_title(title or str(expr), fontsize=12)
        ax.set_xlabel(str(x_var), fontsize=9)
        ax.set_ylabel(str(y_var), fontsize=9)
        ax.set_zlabel(f"f({x_var},{y_var})", fontsize=9)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_parametric_3d_blocking(
    x_expr: sympy.Expr,
    y_expr: sympy.Expr,
    z_expr: sympy.Expr,
    t_var:  sympy.Symbol,
    t_min:  float,
    t_max:  float,
    title:  str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    style:  StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        ts = np.linspace(t_min, t_max, PARAM_POINTS)
        fx = _lambdify1(x_expr, t_var)
        fy = _lambdify1(y_expr, t_var)
        fz = _lambdify1(z_expr, t_var)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xs = np.asarray(fx(ts), dtype=float)
            ys = np.asarray(fy(ts), dtype=float)
            zs = np.asarray(fz(ts), dtype=float)

        norm   = plt.Normalize(ts.min(), ts.max())
        colors = plt.get_cmap(style.colormap)(norm(ts))

        fig, ax = _make_3d_axes(style)
        for i in range(len(ts) - 1):
            ax.plot(
                xs[i:i+2], ys[i:i+2], zs[i:i+2],
                color=colors[i], linewidth=style.line_width,
            )

        sm = cm.ScalarMappable(cmap=plt.get_cmap(style.colormap), norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.1, label=str(t_var))

        ax.set_title(title or f"({x_expr}, {y_expr}, {z_expr})", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_zlabel(zlabel, fontsize=9)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_scatter_3d_blocking(
    xs:      list,
    ys:      list,
    zs:      list,
    title:   str,
    xlabel:  str,
    ylabel:  str,
    zlabel:  str,
    style:   StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        xa = np.asarray(xs, float)
        ya = np.asarray(ys, float)
        za = np.asarray(zs, float)

        fig, ax = _make_3d_axes(style)
        sc = ax.scatter(
            xa, ya, za,
            c=za, cmap=style.colormap,
            s=style.marker_size ** 2,
            depthshade=True,
        )
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1, label=zlabel)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_zlabel(zlabel, fontsize=9)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


# ── Multi-panel ─────────────────────────────────────────────────────────────

def _render_spec_onto_axes(
    spec:          PlotSpec,
    ax:            plt.Axes,
    default_style: StyleOptions,
) -> None:
    """
    Draw a single :class:`PlotSpec` onto *ax*.

    The spec's own ``style`` overrides ``default_style`` when provided.
    Add new ``kind`` branches here when adding new 2-D plot types that
    should be composable inside ``plot_multi``.
    """
    style = spec.style or default_style
    kind  = spec.kind.lower()

    if kind == "function":
        f  = _lambdify1(spec.expr, spec.var)
        xs = np.linspace(spec.x_min, spec.x_max, PLOT_POINTS)
        ys = _eval1(f, xs)
        ax.plot(xs, ys, color=style.color, linewidth=style.line_width,
                linestyle=style.line_style)
        _apply_axes_style(ax, spec.title or str(spec.expr),
                          str(spec.var), f"f({spec.var})", style.show_grid)

    elif kind == "points":
        ax.scatter(
            np.asarray(spec.xs, float),
            np.asarray(spec.ys, float),
            s=style.marker_size ** 2,
            color=style.color,
            zorder=3,
        )
        _apply_axes_style(ax, spec.title, spec.xlabel, spec.ylabel, style.show_grid)

    elif kind == "contour":
        X, Y = _meshgrid(spec.x_range, spec.y_range, n=60)
        Z    = _eval2(_lambdify2(spec.expr, spec.x_var, spec.y_var), X, Y)
        cf   = ax.contourf(X, Y, Z, levels=12, cmap=style.colormap, alpha=0.85)
        ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.3, alpha=0.4)
        ax.figure.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(spec.title or str(spec.expr), fontsize=11)
        ax.set_xlabel(str(spec.x_var), fontsize=9)
        ax.set_ylabel(str(spec.y_var), fontsize=9)
        ax.grid(style.show_grid, alpha=0.2)

    elif kind == "vector_field":
        n  = 18
        xs = np.linspace(spec.x_range[0], spec.x_range[1], n)
        ys = np.linspace(spec.y_range[0], spec.y_range[1], n)
        X, Y = np.meshgrid(xs, ys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            U = np.asarray(_lambdify2(spec.u_expr, spec.x_var, spec.y_var)(X, Y), float)
            V = np.asarray(_lambdify2(spec.v_expr, spec.x_var, spec.y_var)(X, Y), float)
        mag  = np.hypot(U, V)
        safe = np.where(mag == 0, 1, mag)
        ax.quiver(X, Y, U / safe, V / safe, mag,
                  cmap=style.colormap, scale=22, width=0.004)
        ax.set_title(spec.title or f"({spec.u_expr},{spec.v_expr})", fontsize=11)
        ax.set_xlabel(str(spec.x_var), fontsize=9)
        ax.set_ylabel(str(spec.y_var), fontsize=9)
        ax.grid(style.show_grid, alpha=0.2)

    elif kind == "parametric_2d":
        ts = np.linspace(spec.t_min, spec.t_max, PARAM_POINTS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xs = np.asarray(_lambdify1(spec.x_expr, spec.t_var)(ts), float)
            ys = np.asarray(_lambdify1(spec.y_expr, spec.t_var)(ts), float)
        ax.plot(xs, ys, color=style.color, linewidth=style.line_width,
                linestyle=style.line_style)
        ax.set_title(spec.title or f"({spec.x_expr},{spec.y_expr})", fontsize=11)
        ax.set_xlabel(spec.xlabel, fontsize=9)
        ax.set_ylabel(spec.ylabel, fontsize=9)
        ax.grid(style.show_grid, alpha=0.3)

    else:
        # Unknown kind — leave axes blank with a visible error label.
        # This makes it easy to spot missing branches during development.
        ax.text(0.5, 0.5, f"Unknown kind:\n'{spec.kind}'",
                ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set_title(spec.title)


def _plot_multi_blocking(
    specs:      List[PlotSpec],
    ncols:      int,
    title:      str,
    fig_width:  float,
    row_height: float,
    style:      StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n     = len(specs)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, row_height * nrows),
            squeeze=False,
        )
        # Facecolor is handled by rcParams

        flat = axes.flatten()
        for i, spec in enumerate(specs):
            flat[i].set_facecolor("white")
            _render_spec_onto_axes(spec, flat[i], style)

        for j in range(n, len(flat)):
            flat[j].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=14, y=1.01)

        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def plot_function(
    expr:  sympy.Expr,
    var:   sympy.Symbol,
    x_min: float = -10.0,
    x_max: float = 10.0,
    title: str   = "",
    style: StyleOptions = _DEFAULT_STYLE,
    additional_exprs: list = None,
) -> discord.File:
    """
    Plot a SymPy expression as a line graph.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to plot, e.g. ``sympy.sin(x) / x``.
    var : sympy.Symbol
        Free variable used as the x-axis.
    x_min, x_max : float
        Domain boundaries (default ``-10``, ``10``).
    title : str
        Plot title; defaults to ``str(expr)``.
    style : StyleOptions
        Visual style options.

    Returns
    -------
    discord.File
        PNG named ``plot.png``.
    """
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max}).")
    buf = await _run_blocking(
        _plot_function_blocking, expr, var, x_min, x_max, title, style, additional_exprs,
    )
    return discord.File(buf, filename="plot.png")


async def plot_points(
    xs:     list,
    ys:     list,
    title:  str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    style:  StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    Scatter plot from paired numeric lists.

    Parameters
    ----------
    xs, ys : list
        Equal-length sequences of numeric values.
    title, xlabel, ylabel : str
        Optional labels.
    style : StyleOptions
        Visual style options.

    Returns
    -------
    discord.File
        PNG named ``plot.png``.
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys lengths differ ({len(xs)} vs {len(ys)}).")
    if not xs:
        raise ValueError("Cannot plot an empty data set.")
    buf = await _run_blocking(
        _plot_points_blocking, xs, ys, title, xlabel, ylabel, style,
    )
    return discord.File(buf, filename="plot.png")


async def plot_contour(
    expr:     sympy.Expr,
    x_var:    sympy.Symbol,
    y_var:    sympy.Symbol,
    x_range:  Tuple[float, float] = (-5.0, 5.0),
    y_range:  Tuple[float, float] = (-5.0, 5.0),
    title:    str = "",
    levels:   int = 20,
    style:    StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    Filled contour map of a two-variable SymPy expression f(x, y).

    Parameters
    ----------
    expr : sympy.Expr
        Expression in *x_var* and *y_var*.
    x_var, y_var : sympy.Symbol
        The two free variables.
    x_range, y_range : (float, float)
        Axis domain bounds (default ``(-5, 5)`` each).
    title : str
        Plot title.
    levels : int
        Number of contour levels (default ``20``).
    style : StyleOptions
        Visual style options (``colormap`` is used here).

    Returns
    -------
    discord.File
        PNG named ``contour.png``.
    """
    buf = await _run_blocking(
        _plot_contour_blocking,
        expr, x_var, y_var, x_range, y_range, title, levels, style,
    )
    return discord.File(buf, filename="contour.png")


async def plot_vector_field(
    u_expr:  sympy.Expr,
    v_expr:  sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    title:   str   = "",
    stream:  bool  = False,
    density: float = 1.2,
    style:   StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    2-D vector field plot F(x, y) = (u, v).

    Parameters
    ----------
    u_expr, v_expr : sympy.Expr
        x- and y-components of the field.
    x_var, y_var : sympy.Symbol
        Free variables.
    x_range, y_range : (float, float)
        Domain bounds.
    title : str
        Plot title.
    stream : bool
        Use streamplot instead of quiver.
    density : float
        Stream density when ``stream=True``.
    style : StyleOptions
        Visual style options.

    Returns
    -------
    discord.File
        PNG named ``vector_field.png``.
    """
    buf = await _run_blocking(
        _plot_vector_field_blocking,
        u_expr, v_expr, x_var, y_var,
        x_range, y_range, title, stream, density, style,
    )
    return discord.File(buf, filename="vector_field.png")


async def plot_parametric_2d(
    x_expr: sympy.Expr,
    y_expr: sympy.Expr,
    t_var:  sympy.Symbol,
    t_min:  float = 0.0,
    t_max:  float = 2 * float(sympy.pi),
    title:  str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    style:  StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    2-D parametric curve (x(t), y(t)).

    Parameters
    ----------
    x_expr, y_expr : sympy.Expr
        Coordinate expressions in *t_var*.
    t_var : sympy.Symbol
        The parameter symbol.
    t_min, t_max : float
        Parameter range.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    style : StyleOptions
        Visual style options.

    Returns
    -------
    discord.File
        PNG named ``parametric_2d.png``.
    """
    if t_min >= t_max:
        raise ValueError(f"t_min ({t_min}) must be < t_max ({t_max}).")
    buf = await _run_blocking(
        _plot_parametric_2d_blocking,
        x_expr, y_expr, t_var, t_min, t_max, title, xlabel, ylabel, style,
    )
    return discord.File(buf, filename="parametric_2d.png")


async def plot_surface(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    title:   str = "",
    style:   StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    3-D surface plot of f(x, y).

    Parameters
    ----------
    expr : sympy.Expr
        Expression in *x_var* and *y_var*.
    x_var, y_var : sympy.Symbol
        Free variables.
    x_range, y_range : (float, float)
        Domain bounds.
    title : str
        Plot title.
    style : StyleOptions
        Visual style options (``colormap`` and ``alpha`` are used here).

    Returns
    -------
    discord.File
        PNG named ``surface.png``.
    """
    buf = await _run_blocking(
        _plot_surface_blocking,
        expr, x_var, y_var, x_range, y_range, title, style,
    )
    return discord.File(buf, filename="surface.png")


async def plot_wireframe(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    title:   str = "",
    rstride: int = 3,
    cstride: int = 3,
    style:   StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    3-D wireframe plot of f(x, y).

    Parameters
    ----------
    expr : sympy.Expr
        Expression in *x_var* and *y_var*.
    x_var, y_var : sympy.Symbol
        Free variables.
    x_range, y_range : (float, float)
        Domain bounds.
    title : str
        Plot title.
    rstride, cstride : int
        Row/column decimation for wireframe density.
    style : StyleOptions
        Visual style options (``color`` is used for wire colour).

    Returns
    -------
    discord.File
        PNG named ``wireframe.png``.
    """
    buf = await _run_blocking(
        _plot_wireframe_blocking,
        expr, x_var, y_var, x_range, y_range, title, rstride, cstride, style,
    )
    return discord.File(buf, filename="wireframe.png")


async def plot_parametric_3d(
    x_expr: sympy.Expr,
    y_expr: sympy.Expr,
    z_expr: sympy.Expr,
    t_var:  sympy.Symbol,
    t_min:  float = 0.0,
    t_max:  float = 2 * float(sympy.pi),
    title:  str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    style:  StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    3-D parametric curve (x(t), y(t), z(t)).

    Parameters
    ----------
    x_expr, y_expr, z_expr : sympy.Expr
        Coordinate expressions in *t_var*.
    t_var : sympy.Symbol
        Parameter symbol.
    t_min, t_max : float
        Parameter range.
    title : str
        Plot title.
    xlabel, ylabel, zlabel : str
        Axis labels.
    style : StyleOptions
        Visual style options.

    Returns
    -------
    discord.File
        PNG named ``parametric_3d.png``.
    """
    if t_min >= t_max:
        raise ValueError(f"t_min ({t_min}) must be < t_max ({t_max}).")
    buf = await _run_blocking(
        _plot_parametric_3d_blocking,
        x_expr, y_expr, z_expr, t_var,
        t_min, t_max, title, xlabel, ylabel, zlabel, style,
    )
    return discord.File(buf, filename="parametric_3d.png")


async def plot_scatter_3d(
    xs:      list,
    ys:      list,
    zs:      list,
    title:   str = "",
    xlabel:  str = "x",
    ylabel:  str = "y",
    zlabel:  str = "z",
    style:   StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    3-D scatter plot from three parallel numeric lists.

    Parameters
    ----------
    xs, ys, zs : list
        Equal-length sequences of numeric values.
    title : str
        Plot title.
    xlabel, ylabel, zlabel : str
        Axis labels.
    style : StyleOptions
        Visual style options (``colormap`` used for z-colouring).

    Returns
    -------
    discord.File
        PNG named ``scatter_3d.png``.
    """
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("xs, ys, and zs must all have the same length.")
    if not xs:
        raise ValueError("Cannot plot an empty data set.")
    buf = await _run_blocking(
        _plot_scatter_3d_blocking,
        xs, ys, zs, title, xlabel, ylabel, zlabel, style,
    )
    return discord.File(buf, filename="scatter_3d.png")


async def plot_multi(
    specs:      Sequence[PlotSpec],
    ncols:      int   = 2,
    title:      str   = "",
    fig_width:  float = 14.0,
    row_height: float = 5.0,
    style:      StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    """
    Compose multiple plots into a single grid image.

    Parameters
    ----------
    specs : list[PlotSpec]
        One :class:`PlotSpec` per sub-plot panel.
    ncols : int
        Number of columns in the grid (default ``2``).
    title : str
        Overall figure title.
    fig_width : float
        Total figure width in inches (default ``14``).
    row_height : float
        Height per row in inches (default ``5``).
    style : StyleOptions
        Default style applied to any spec that does not supply its own.

    Returns
    -------
    discord.File
        PNG named ``multi_plot.png``.
    """
    if not specs:
        raise ValueError("specs must contain at least one PlotSpec.")
    if ncols < 1:
        raise ValueError("ncols must be ≥ 1.")
    buf = await _run_blocking(
        _plot_multi_blocking,
        list(specs), ncols, title, fig_width, row_height, style,
    )
    return discord.File(buf, filename="multi_plot.png")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


from matplotlib.animation import FuncAnimation, PillowWriter

def _plot_animation_function_blocking(
    expr: sympy.Expr,
    var: sympy.Symbol,
    anim_var: sympy.Symbol,
    x_min: float,
    x_max: float,
    title: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        fig, ax = _white_fig(style)
        xs = np.linspace(x_min, x_max, PLOT_POINTS)
        line, = ax.plot([], [])
        _apply_line_style(ax, style)
        _apply_axes_style(ax, title or str(expr), str(var), f"f({var})", style.show_grid)

        frames = 30
        anim_min, anim_max = 0, 10
        a_vals = np.linspace(anim_min, anim_max, frames)

        # we need to pre-compute y limits or autoscale on the fly
        f = _lambdify2(expr, var, anim_var)
        
        all_ys = []
        for a in a_vals:
            ys = _eval1(lambda x: f(x, a), xs)
            all_ys.append(ys)
        
        flat_ys = np.concatenate(all_ys)
        valid = flat_ys[np.isfinite(flat_ys)]
        if len(valid) > 0:
            ax.set_ylim(valid.min() - 0.5, valid.max() + 0.5)
        ax.set_xlim(x_min, x_max)

        def init():
            line.set_data([], [])
            return line,

        def update(frame_idx):
            line.set_data(xs, all_ys[frame_idx])
            ax.set_title(f"{title or str(expr)} ({anim_var}={a_vals[frame_idx]:.2f})")
            return line,

        ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_name = tmp.name

        try:
            writer = PillowWriter(fps=15)
            ani.save(tmp_name, writer=writer)
            with open(tmp_name, "rb") as f:
                buf = io.BytesIO(f.read())
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

        plt.close(fig)
        buf.seek(0)
        return buf

async def plot_animation_function(cfg) -> discord.File:
    import sympy
    x = sympy.Symbol("x")
    from cogs.plot_engine import _clean_sympy_expr, _sympy_expr
    expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, sympy.Symbol(cfg.anim_param or "a"))
    anim_var = sympy.Symbol(cfg.anim_param or "a")
    
    buf = await _run_blocking(
        _plot_animation_function_blocking,
        expr, x, anim_var, cfg.x_min, cfg.x_max, cfg.title, cfg.to_style()
    )
    return discord.File(buf, filename="anim.gif")


__all__ = [
    # Data classes
    "StyleOptions",
    "PlotSpec",
    # 2-D single-variable
    "plot_function",
    "plot_points",
    # 2-D multivariable
    "plot_contour",
    "plot_vector_field",
    "plot_parametric_2d",
    # 3-D
    "plot_surface",
    "plot_wireframe",
    "plot_parametric_3d",
    "plot_scatter_3d",
    # Multi-panel
    "plot_multi",
]
