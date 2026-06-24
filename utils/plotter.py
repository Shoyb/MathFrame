"""
utils/plotter.py — Matplotlib plot generator for the math bot.

Produces PNG plot images and returns them as ``discord.File`` objects
ready to attach to any interaction reply.

All public functions are async and safe to call concurrently: blocking
matplotlib work runs inside a thread-pool executor so it never stalls
the event loop.  Each blocking call uses ``matplotlib.rc_context`` so
that style overrides are fully isolated — concurrent renders never
clobber each other's rcParams.
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
from utils.expr_utils import _clean_sympy_expr, _sympy_expr

# ---------------------------------------------------------------------------
# Module-level constants  (read-only after import — never mutate at runtime)
# ---------------------------------------------------------------------------

PLOT_POINTS  = 800    # x-samples for 1-D line plots
GRID_POINTS  = 120    # grid resolution for 2-D / 3-D surface plots
Y_CLIP       = 1e6    # hard clip for values outside smart auto-range
Z_CLIP       = 1e6    # clip |z| beyond this for surface plots
PARAM_POINTS = 1000   # t-samples for parametric curves

ANIM_FRAMES      = 30
ANIM_PARAM_MIN   = 0.0
ANIM_PARAM_MAX   = 10.0
ANIM_GRID_POINTS = 70
ANIM_PARAM_POINTS = 400

_plot_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="plotter")

# ---------------------------------------------------------------------------
# StyleOptions
# ---------------------------------------------------------------------------

@dataclass
class StyleOptions:
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

    x_log: bool = False
    y_log: bool = False

    fill_below: bool = False
    fill_color: str  = ""

    x_lim: Optional[Tuple[float, float]] = None
    y_lim: Optional[Tuple[float, float]] = None

    def rc_overrides(self) -> Dict:
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


_DEFAULT_STYLE = StyleOptions()

# ---------------------------------------------------------------------------
# PlotSpec
# ---------------------------------------------------------------------------

@dataclass
class PlotSpec:
    """
    Descriptor for one sub-plot panel inside :func:`plot_multi`.

    kind : str
        One of ``"function"``, ``"points"``, ``"contour"``,
        ``"vector_field"``, ``"parametric_2d"``, ``"polar"``,
        ``"implicit"``, ``"histogram"``, ``"errorbar"``, or ``"heatmap"``.
    """

    kind: str

    title: str = ""
    style: Optional[StyleOptions] = None

    expr: Optional[sympy.Expr] = None
    var:  Optional[sympy.Symbol] = None
    x_min: float = -10.0
    x_max: float  = 10.0

    xs: Optional[list] = None
    ys: Optional[list] = None
    zs: Optional[list] = None
    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"

    x_var: Optional[sympy.Symbol] = None
    y_var: Optional[sympy.Symbol] = None
    x_range: Tuple[float, float] = (-5.0, 5.0)
    y_range: Tuple[float, float] = (-5.0, 5.0)

    u_expr: Optional[sympy.Expr] = None
    v_expr: Optional[sympy.Expr] = None
    stream: bool = False

    x_expr: Optional[sympy.Expr] = None
    y_expr: Optional[sympy.Expr] = None
    z_expr: Optional[sympy.Expr] = None
    t_var:  Optional[sympy.Symbol] = None
    t_min:  float = 0.0
    t_max:  float = 2 * float(sympy.pi)

    additional_exprs: List[sympy.Expr] = field(default_factory=list)
    implicit_rhs: float = 0.0
    inequality_op: str = "<="
    hist_bins: int = 20
    box_violin: str = "box"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_math_label(s: str) -> str:
    if not s:
        return s
    try:
        parsed = sympy.sympify(s)
        latex = sympy.latex(parsed)
        if "\\begin" in latex or "\\end" in latex:
            return str(parsed)
        return f"${latex}$"
    except Exception:
        return s


def _apply_axes_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    show_grid: bool = True,
    style: Optional[StyleOptions] = None,
) -> None:
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(show_grid, alpha=0.3)
    ax.set_title(_to_math_label(title), fontsize=12, pad=6)
    ax.set_xlabel(_to_math_label(xlabel), fontsize=10)
    ax.set_ylabel(_to_math_label(ylabel), fontsize=10)

    if style is not None:
        if style.x_log:
            ax.set_xscale("log")
        if style.y_log:
            ax.set_yscale("log")
        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)


def _apply_line_style(ax: plt.Axes, style: StyleOptions) -> None:
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _white_fig(style: StyleOptions, **kw) -> Tuple[plt.Figure, plt.Axes]:
    kw.setdefault("figsize", (style.fig_width, style.fig_height))
    fig, ax = plt.subplots(**kw)
    return fig, ax


def _make_3d_axes(style: StyleOptions) -> Tuple[plt.Figure, "Axes3D"]:
    fig = plt.figure(figsize=(style.fig_width, style.fig_height))
    ax  = fig.add_subplot(111, projection="3d")
    return fig, ax


def _lambdify1(expr: sympy.Expr, var: sympy.Symbol) -> Callable:
    return sympy.lambdify(var, expr, modules=["numpy"])


def _lambdify2(expr: sympy.Expr,
               xv: sympy.Symbol,
               yv: sympy.Symbol) -> Callable:
    return sympy.lambdify((xv, yv), expr, modules=["numpy"])


def _lambdify3(expr: sympy.Expr,
               av: sympy.Symbol,
               bv: sympy.Symbol,
               cv: sympy.Symbol) -> Callable:
    return sympy.lambdify((av, bv, cv), expr, modules=["numpy"])


def _eval1(f: Callable, xs: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ys = np.asarray(f(xs), dtype=float)
    if ys.shape == ():
        ys = np.full_like(xs, float(ys), dtype=float)
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


def _smart_ylim(
    ys: np.ndarray,
    style: StyleOptions,
    pad_frac: float = 0.08,
) -> Optional[Tuple[float, float]]:
    if style.y_lim is not None or style.y_log:
        return None

    finite = ys[np.isfinite(ys)]
    if len(finite) == 0:
        return None

    lo = float(np.nanpercentile(finite, 2))
    hi = float(np.nanpercentile(finite, 98))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _as_float(value: sympy.Basic) -> Optional[float]:
    try:
        numeric = float(sympy.N(value))
    except Exception:
        return None
    if np.isfinite(numeric):
        return numeric
    return None


def _finite_points_in_window(
    values: Sequence[sympy.Basic],
    lo: float,
    hi: float,
) -> List[float]:
    points: List[float] = []
    for value in values:
        numeric = _as_float(value)
        if numeric is not None and lo <= numeric <= hi:
            points.append(numeric)
    return points


def _interesting_x_points(
    expr: sympy.Expr,
    var: sympy.Symbol,
    lo: float,
    hi: float,
) -> List[float]:
    points: List[float] = []
    for candidate in (expr, sympy.diff(expr, var)):
        try:
            solved = sympy.solve(candidate, var)
        except Exception:
            solved = []
        points.extend(_finite_points_in_window(solved[:16], lo, hi))

    try:
        singular = sympy.singularities(expr, var)
    except Exception:
        singular = []
    if singular is not sympy.S.EmptySet:
        try:
            points.extend(_finite_points_in_window(list(singular)[:16], lo, hi))
        except Exception:
            pass

    return sorted(set(round(p, 12) for p in points))


def _auto_domain(
    expr: sympy.Expr,
    var: sympy.Symbol,
    x_min: float,
    x_max: float,
    style: StyleOptions,
) -> Tuple[float, float]:
    if style.x_log or style.x_lim is not None:
        return x_min, x_max
    if abs(x_min + 10.0) > 1e-9 or abs(x_max - 10.0) > 1e-9:
        return x_min, x_max

    points = _interesting_x_points(expr, var, -50.0, 50.0)
    if not points:
        return x_min, x_max

    lo, hi = min(points), max(points)
    span = max(hi - lo, 4.0)
    pad = max(span * 0.35, 1.0)
    return lo - pad, hi + pad


def _singularities_in_range(
    expr: sympy.Expr,
    var: sympy.Symbol,
    x_min: float,
    x_max: float,
) -> List[float]:
    try:
        singular = sympy.singularities(expr, var)
    except Exception:
        return []
    if singular is sympy.S.EmptySet:
        return []
    try:
        return _finite_points_in_window(list(singular)[:32], x_min, x_max)
    except Exception:
        return []


def _insert_function_gaps(
    xs: np.ndarray,
    ys: np.ndarray,
    singularities: Sequence[float],
) -> np.ndarray:
    gapped = np.array(ys, copy=True)
    if xs.size < 2:
        return gapped

    step = float(np.nanmedian(np.diff(xs)))
    gap_width = max(abs(step) * 2.5, 1e-12)
    for point in singularities:
        gapped[np.abs(xs - point) <= gap_width] = np.nan

    finite = np.isfinite(gapped)
    diffs = np.abs(np.diff(gapped))
    finite_pairs = finite[:-1] & finite[1:]
    finite_values = np.abs(gapped[finite])
    scale = float(np.nanpercentile(finite_values, 95)) if finite_values.size else 1.0
    jump_threshold = max(scale * 8.0, 100.0)
    jumps = np.where(finite_pairs & (diffs > jump_threshold))[0]
    gapped[jumps + 1] = np.nan
    return gapped


async def _run_blocking(fn: Callable, *args) -> io.BytesIO:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_plot_executor, fn, *args)
    except Exception as exc:
        raise ValueError(f"Plot failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Blocking implementations
# ---------------------------------------------------------------------------

def _plot_function_blocking(
    expr: sympy.Expr,
    var:  sympy.Symbol,
    x_min: float,
    x_max: float,
    title: str,
    style: StyleOptions,
    additional_exprs: list = None,
    n_1d: int = PLOT_POINTS,
) -> io.BytesIO:
    if style.x_log and x_min <= 0:
        raise ValueError(
            f"Log x-axis requires x_min > 0, but x_min={x_min}. "
            "Change the domain or disable log scale."
        )

    with matplotlib.rc_context(rc=style.rc_overrides()):
        x_min, x_max = _auto_domain(expr, var, x_min, x_max, style)
        if style.x_log:
            xs = np.logspace(np.log10(x_min), np.log10(x_max), n_1d)
        else:
            xs = np.linspace(x_min, x_max, n_1d)

        f  = _lambdify1(expr, var)
        singularities = _singularities_in_range(expr, var, x_min, x_max)
        ys = _insert_function_gaps(xs, _eval1(f, xs), singularities)

        fig, ax = _white_fig(style)
        ax.plot(xs, ys, label=str(expr))
        _apply_line_style(ax, style)

        if style.fill_below:
            fill_c = style.fill_color if style.fill_color else style.color
            ax.fill_between(xs, ys, 0, alpha=0.25, color=fill_c)

        if additional_exprs:
            for extra in additional_exprs:
                f_extra = _lambdify1(extra, var)
                extra_singularities = _singularities_in_range(extra, var, x_min, x_max)
                ys_extra = _insert_function_gaps(
                    xs, _eval1(f_extra, xs), extra_singularities,
                )
                ax.plot(xs, ys_extra, label=str(extra))
                if style.fill_below:
                    fill_c = style.fill_color if style.fill_color else style.color
                    ax.fill_between(xs, ys_extra, 0, alpha=0.15, color=fill_c)
            ax.legend(loc="upper right")

        _apply_axes_style(
            ax, title or str(expr), str(var), f"f({var})",
            style.show_grid, style=style,
        )

        all_ys = ys if additional_exprs is None else np.concatenate(
            [ys] + [
                _insert_function_gaps(
                    xs,
                    _eval1(_lambdify1(e, var), xs),
                    _singularities_in_range(e, var, x_min, x_max),
                )
                for e in (additional_exprs or [])
            ]
        )
        ylim = _smart_ylim(all_ys, style)
        if ylim is not None:
            ax.set_ylim(*ylim)

        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_riemann_blocking(
    expr: sympy.Expr,
    var: sympy.Symbol,
    a: float,
    b: float,
    n: int,
    method: str,
    title: str,
    style: StyleOptions,
    n_1d: int = PLOT_POINTS,
) -> io.BytesIO:
    if a == b:
        raise ValueError("Riemann bounds must be different.")

    method = method.lower().strip()
    if method not in ("left", "right", "midpoint"):
        raise ValueError("Riemann method must be left, right, or midpoint.")

    n = max(1, min(500, int(n)))
    lo, hi = (a, b) if a < b else (b, a)

    with matplotlib.rc_context(rc=style.rc_overrides()):
        xs = np.linspace(lo, hi, n_1d)
        f = _lambdify1(expr, var)
        singularities = _singularities_in_range(expr, var, lo, hi)
        ys = _insert_function_gaps(xs, _eval1(f, xs), singularities)

        edges = np.linspace(a, b, n + 1)
        width = (b - a) / n
        if method == "left":
            sample_xs = edges[:-1]
        elif method == "right":
            sample_xs = edges[1:]
        else:
            sample_xs = (edges[:-1] + edges[1:]) / 2
        heights = _eval1(f, sample_xs)

        fig, ax = _white_fig(style)
        rect_color = style.fill_color or style.color
        ax.bar(
            edges[:-1],
            heights,
            width=width,
            align="edge",
            alpha=0.25,
            color=rect_color,
            edgecolor=style.color,
            linewidth=0.8,
            label=f"{method} rectangles (n={n})",
        )
        ax.plot(xs, ys, color=style.color, linewidth=style.line_width,
                linestyle=style.line_style, label=str(expr), zorder=3)

        finite_heights = heights[np.isfinite(heights)]
        estimate = float(np.sum(finite_heights) * width) if finite_heights.size == n else np.nan
        if np.isfinite(estimate):
            label = f"estimate = {estimate:.6g}"
            ax.text(
                0.02, 0.96, label,
                ha="left", va="top", transform=ax.transAxes,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "alpha": 0.7, "facecolor": "white"},
            )

        _apply_axes_style(
            ax,
            title or f"Riemann sum: {expr}",
            str(var),
            f"f({var})",
            style.show_grid,
            style=style,
        )
        ylim = _smart_ylim(np.concatenate([ys, heights]), style)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="best")
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
        ax.scatter(
            np.asarray(xs, float),
            np.asarray(ys, float),
            s=style.marker_size ** 2,
            color=style.color,
            zorder=3,
        )
        _apply_axes_style(ax, title, xlabel, ylabel, style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_contour_blocking(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title:   str,
    levels:  int,
    style:   StyleOptions,
    n_2d:    int = GRID_POINTS,
) -> io.BytesIO:
    if style.x_log and x_range[0] <= 0:
        raise ValueError("Log x-axis requires x_range[0] > 0.")
    if style.y_log and y_range[0] <= 0:
        raise ValueError("Log y-axis requires y_range[0] > 0.")

    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=n_2d)
        Z    = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))

        cf = ax.contourf(X, Y, Z, levels=levels, cmap=style.colormap, alpha=0.85)
        ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.85, label=f"f({x_var},{y_var})")

        ax.set_title(_to_math_label(title or str(expr)), fontsize=12, pad=6)
        ax.set_xlabel(_to_math_label(str(x_var)), fontsize=10)
        ax.set_ylabel(_to_math_label(str(y_var)), fontsize=10)
        ax.grid(style.show_grid, alpha=0.2)

        if style.x_log:
            ax.set_xscale("log")
        if style.y_log:
            ax.set_yscale("log")
        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)

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
    n_2d:    int = GRID_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n  = 24 if not stream else n_2d
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

        ax.set_title(_to_math_label(title or f"({u_expr}, {v_expr})"), fontsize=12, pad=6)
        ax.set_xlabel(_to_math_label(str(x_var)), fontsize=10)
        ax.set_ylabel(_to_math_label(str(y_var)), fontsize=10)
        ax.grid(style.show_grid, alpha=0.2)

        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)

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
    n_1d:   int = PARAM_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        fx = _lambdify1(x_expr, t_var)
        fy = _lambdify1(y_expr, t_var)
        ts = np.linspace(t_min, t_max, n_1d)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xs = np.asarray(fx(ts), dtype=float)
            ys = np.asarray(fy(ts), dtype=float)

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        lc     = LineCollection(segs, cmap=style.colormap, linewidth=style.line_width)
        lc.set_array(ts[:-1])
        ax.add_collection(lc)
        ax.autoscale()
        fig.colorbar(lc, ax=ax, shrink=0.85, label=str(t_var))

        ax.set_title(_to_math_label(title or f"({x_expr}, {y_expr})"), fontsize=12, pad=6)
        ax.set_xlabel(_to_math_label(xlabel), fontsize=10)
        ax.set_ylabel(_to_math_label(ylabel), fontsize=10)
        ax.grid(style.show_grid, alpha=0.3)

        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)

        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_surface_blocking(
    expr:    sympy.Expr,
    x_var:   sympy.Symbol,
    y_var:   sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title:   str,
    style:   StyleOptions,
    n_2d:    int = GRID_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=n_2d)
        Z    = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _make_3d_axes(style)
        surf = ax.plot_surface(
            X, Y, Z,
            cmap=style.colormap, alpha=style.alpha,
            linewidth=0, antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=f"f({x_var},{y_var})")

        ax.set_title(_to_math_label(title or str(expr)), fontsize=12)
        ax.set_xlabel(_to_math_label(str(x_var)), fontsize=9)
        ax.set_ylabel(_to_math_label(str(y_var)), fontsize=9)
        ax.set_zlabel(_to_math_label(f"f({x_var},{y_var})"), fontsize=9)
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
    n_2d:    int = GRID_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n    = min(n_2d, 60)
        X, Y = _meshgrid(x_range, y_range, n)
        Z    = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _make_3d_axes(style)
        ax.plot_wireframe(
            X, Y, Z,
            color=style.color, linewidth=style.line_width,
            rstride=rstride, cstride=cstride,
        )

        ax.set_title(_to_math_label(title or str(expr)), fontsize=12)
        ax.set_xlabel(_to_math_label(str(x_var)), fontsize=9)
        ax.set_ylabel(_to_math_label(str(y_var)), fontsize=9)
        ax.set_zlabel(_to_math_label(f"f({x_var},{y_var})"), fontsize=9)
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
    n_1d:   int = PARAM_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        ts = np.linspace(t_min, t_max, n_1d)
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

        ax.set_title(_to_math_label(title or f"({x_expr}, {y_expr}, {z_expr})"), fontsize=11)
        ax.set_xlabel(_to_math_label(xlabel), fontsize=9)
        ax.set_ylabel(_to_math_label(ylabel), fontsize=9)
        ax.set_zlabel(_to_math_label(zlabel), fontsize=9)
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

        ax.set_title(_to_math_label(title), fontsize=12)
        ax.set_xlabel(_to_math_label(xlabel), fontsize=9)
        ax.set_ylabel(_to_math_label(ylabel), fontsize=9)
        ax.set_zlabel(_to_math_label(zlabel), fontsize=9)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_polar_blocking(
    expr:            sympy.Expr,
    theta_var:       sympy.Symbol,
    theta_min:       float,
    theta_max:       float,
    title:           str,
    style:           StyleOptions,
    additional_exprs: list = None,
    n_1d:            int   = PARAM_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        thetas = np.linspace(theta_min, theta_max, n_1d)

        f_main = _lambdify1(expr, theta_var)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_main = np.asarray(f_main(thetas), dtype=float)
        r_main = np.where(np.isfinite(r_main), r_main, np.nan)

        fig = plt.figure(figsize=(style.fig_width, style.fig_height))
        ax  = fig.add_subplot(111, projection="polar")

        has_extra = bool(additional_exprs)

        ax.plot(
            thetas, r_main,
            color=style.color,
            linewidth=style.line_width,
            linestyle=style.line_style,
            label=_to_math_label(str(expr)) if has_extra else None,
        )

        if has_extra:
            cmap_fn  = plt.get_cmap(style.colormap)
            n_extra  = len(additional_exprs)
            offsets  = np.linspace(0.15, 0.85, n_extra) if n_extra > 1 else [0.5]

            for extra_expr, offset in zip(additional_exprs, offsets):
                f_extra = _lambdify1(extra_expr, theta_var)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    r_extra = np.asarray(f_extra(thetas), dtype=float)
                r_extra = np.where(np.isfinite(r_extra), r_extra, np.nan)
                ax.plot(
                    thetas, r_extra,
                    color=cmap_fn(offset),
                    linewidth=style.line_width,
                    linestyle=style.line_style,
                    label=_to_math_label(str(extra_expr)),
                )

            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.3, 1.1),
                fontsize=8,
                framealpha=0.7,
            )

        ax.grid(style.show_grid, alpha=0.3)

        ax.set_thetagrids(
            np.degrees(np.linspace(0, 2 * np.pi, 9)[:-1]),
            labels=["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4"],
            fontsize=7,
        )

        title_str = title or str(expr)
        ax.set_title(_to_math_label(title_str), fontsize=12, pad=14)

        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


# ── Multi-panel ─────────────────────────────────────────────────────────────

def _plot_implicit_blocking(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    rhs: float,
    title: str,
    style: StyleOptions,
    n_2d: int = GRID_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=n_2d)
        Z = _eval2(_lambdify2(expr, x_var, y_var), X, Y) - rhs

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        cs = ax.contour(X, Y, Z, levels=[0.0], colors=[style.color],
                        linewidths=[style.line_width])
        if not cs.allsegs or not cs.allsegs[0]:
            ax.text(0.5, 0.5, "No visible zero contour", ha="center",
                    va="center", transform=ax.transAxes)

        _apply_axes_style(ax, title or f"{expr} = {rhs:g}", str(x_var),
                          str(y_var), style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_inequality_blocking(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    op: str,
    rhs: float,
    title: str,
    style: StyleOptions,
    n_2d: int = GRID_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=n_2d)
        Z = _eval2(_lambdify2(expr, x_var, y_var), X, Y)
        if op == "<":
            mask = Z < rhs
        elif op == "<=":
            mask = Z <= rhs
        elif op == ">":
            mask = Z > rhs
        elif op == ">=":
            mask = Z >= rhs
        else:
            raise ValueError("inequality_op must be one of <, <=, >, >=.")

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5],
                    colors=[style.fill_color or style.color], alpha=0.25)
        ax.contour(X, Y, Z - rhs, levels=[0.0], colors=[style.color],
                   linewidths=[style.line_width])
        _apply_axes_style(ax, title or f"{expr} {op} {rhs:g}", str(x_var),
                          str(y_var), style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_histogram_blocking(
    values: list,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        data = np.asarray(values, float)
        fig, ax = _white_fig(style)
        ax.hist(data, bins=bins, color=style.color, alpha=style.alpha,
                edgecolor="black", linewidth=0.6)
        _apply_axes_style(ax, title or "Histogram", xlabel, ylabel,
                          style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_errorbar_blocking(
    xs: list,
    ys: list,
    yerr: list,
    title: str,
    xlabel: str,
    ylabel: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        fig, ax = _white_fig(style)
        ax.errorbar(np.asarray(xs, float), np.asarray(ys, float),
                    yerr=np.asarray(yerr, float), fmt=style.marker or "o",
                    color=style.color, capsize=4, linewidth=style.line_width)
        _apply_axes_style(ax, title or "Error bar plot", xlabel, ylabel,
                          style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_heatmap_blocking(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    style: StyleOptions,
    n_2d: int = GRID_POINTS,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=n_2d)
        Z = _eval2(_lambdify2(expr, x_var, y_var), X, Y)

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        im = ax.imshow(Z, origin="lower", cmap=style.colormap,
                       extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                       aspect="auto", alpha=style.alpha)
        fig.colorbar(im, ax=ax, shrink=0.85, label=f"f({x_var},{y_var})")
        _apply_axes_style(ax, title or str(expr), str(x_var), str(y_var),
                          style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _plot_boxplot_blocking(
    groups: list,
    mode: str,
    title: str,
    ylabel: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        clean_groups = [np.asarray(g, float) for g in groups if len(g) > 0]
        fig, ax = _white_fig(style)
        if mode == "violin":
            parts = ax.violinplot(clean_groups, showmeans=True, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(style.color)
                body.set_alpha(style.alpha)
        else:
            ax.boxplot(clean_groups, patch_artist=True,
                       boxprops={"facecolor": style.color, "alpha": style.alpha},
                       medianprops={"color": "black"})
        _apply_axes_style(ax, title or "Box plot", "group", ylabel,
                          style.show_grid, style=style)
        fig.tight_layout()
        return _save_fig_to_bytes(fig, style.dpi)


def _render_spec_onto_axes(
    spec:          PlotSpec,
    ax:            plt.Axes,
    default_style: StyleOptions,
) -> None:
    style = spec.style or default_style
    kind  = spec.kind.lower()

    if kind == "function":
        f  = _lambdify1(spec.expr, spec.var)
        xs = np.linspace(spec.x_min, spec.x_max, PLOT_POINTS)
        ys = _eval1(f, xs)
        ax.plot(xs, ys, color=style.color, linewidth=style.line_width,
                linestyle=style.line_style)

        if style.fill_below:
            fill_c = style.fill_color if style.fill_color else style.color
            ax.fill_between(xs, ys, 0, alpha=0.25, color=fill_c)

        _apply_axes_style(ax, spec.title or str(spec.expr),
                          str(spec.var), f"f({spec.var})", style.show_grid,
                          style=style)

        ylim = _smart_ylim(ys, style)
        if ylim is not None:
            ax.set_ylim(*ylim)

    elif kind == "points":
        ax.scatter(
            np.asarray(spec.xs, float),
            np.asarray(spec.ys, float),
            s=style.marker_size ** 2,
            color=style.color,
            zorder=3,
        )
        _apply_axes_style(ax, spec.title, spec.xlabel, spec.ylabel, style.show_grid,
                          style=style)

    elif kind == "contour":
        X, Y = _meshgrid(spec.x_range, spec.y_range, n=60)
        Z    = _eval2(_lambdify2(spec.expr, spec.x_var, spec.y_var), X, Y)
        cf   = ax.contourf(X, Y, Z, levels=12, cmap=style.colormap, alpha=0.85)
        ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.3, alpha=0.4)
        ax.figure.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(_to_math_label(spec.title or str(spec.expr)), fontsize=11)
        ax.set_xlabel(_to_math_label(str(spec.x_var)), fontsize=9)
        ax.set_ylabel(_to_math_label(str(spec.y_var)), fontsize=9)
        ax.grid(style.show_grid, alpha=0.2)
        if style.x_log:
            ax.set_xscale("log")
        if style.y_log:
            ax.set_yscale("log")
        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)

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
        ax.set_title(_to_math_label(spec.title or f"({spec.u_expr},{spec.v_expr})"), fontsize=11)
        ax.set_xlabel(_to_math_label(str(spec.x_var)), fontsize=9)
        ax.set_ylabel(_to_math_label(str(spec.y_var)), fontsize=9)
        ax.grid(style.show_grid, alpha=0.2)
        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)

    elif kind == "parametric_2d":
        ts = np.linspace(spec.t_min, spec.t_max, PARAM_POINTS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xs = np.asarray(_lambdify1(spec.x_expr, spec.t_var)(ts), float)
            ys = np.asarray(_lambdify1(spec.y_expr, spec.t_var)(ts), float)
        ax.plot(xs, ys, color=style.color, linewidth=style.line_width,
                linestyle=style.line_style)
        ax.set_title(_to_math_label(spec.title or f"({spec.x_expr},{spec.y_expr})"), fontsize=11)
        ax.set_xlabel(_to_math_label(spec.xlabel), fontsize=9)
        ax.set_ylabel(_to_math_label(spec.ylabel), fontsize=9)
        ax.grid(style.show_grid, alpha=0.3)
        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)

    elif kind == "polar":
        ss = ax.get_subplotspec()
        ax.set_visible(False)
        polar_ax = ax.get_figure().add_subplot(ss, projection="polar")

        thetas = np.linspace(spec.t_min, spec.t_max, PARAM_POINTS)
        f_main = _lambdify1(spec.expr, spec.var)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_main = np.asarray(f_main(thetas), dtype=float)
        r_main = np.where(np.isfinite(r_main), r_main, np.nan)

        has_extra = bool(spec.additional_exprs)
        polar_ax.plot(
            thetas, r_main,
            color=style.color,
            linewidth=style.line_width,
            linestyle=style.line_style,
            label=_to_math_label(str(spec.expr)) if has_extra else None,
        )

        if has_extra:
            cmap_fn = plt.get_cmap(style.colormap)
            n_extra = len(spec.additional_exprs)
            offsets = np.linspace(0.15, 0.85, n_extra) if n_extra > 1 else [0.5]
            for extra_expr, offset in zip(spec.additional_exprs, offsets):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    r_extra = np.asarray(
                        _lambdify1(extra_expr, spec.var)(thetas), dtype=float
                    )
                r_extra = np.where(np.isfinite(r_extra), r_extra, np.nan)
                polar_ax.plot(
                    thetas, r_extra,
                    color=cmap_fn(offset),
                    linewidth=style.line_width,
                    label=_to_math_label(str(extra_expr)),
                )
            polar_ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                            fontsize=7, framealpha=0.7)

        polar_ax.grid(style.show_grid, alpha=0.3)
        polar_ax.set_title(
            _to_math_label(spec.title or str(spec.expr)), fontsize=11, pad=14
        )

    else:
        ax.text(0.5, 0.5, f"Unknown kind:\n'{spec.kind}'",
                ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set_title(_to_math_label(spec.title))


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

        flat = axes.flatten()
        for i, spec in enumerate(specs):
            flat[i].set_facecolor("white")
            _render_spec_onto_axes(spec, flat[i], style)

        for j in range(n, len(flat)):
            flat[j].set_visible(False)

        if title:
            fig.suptitle(_to_math_label(title), fontsize=14, y=1.01)

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
    resolution_1d: int = PLOT_POINTS,
) -> discord.File:
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max}).")
    buf = await _run_blocking(
        _plot_function_blocking,
        expr, var, x_min, x_max, title, style, additional_exprs, resolution_1d,
    )
    return discord.File(buf, filename="plot.png")


async def plot_riemann(
    expr: sympy.Expr,
    var: sympy.Symbol,
    a: float,
    b: float,
    n: int = 8,
    method: str = "left",
    title: str = "",
    style: StyleOptions = _DEFAULT_STYLE,
    resolution_1d: int = PLOT_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_riemann_blocking,
        expr, var, a, b, n, method, title, style, resolution_1d,
    )
    return discord.File(buf, filename="riemann.png")


async def plot_points(
    xs:     list,
    ys:     list,
    title:  str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    style:  StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
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
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_contour_blocking,
        expr, x_var, y_var, x_range, y_range, title, levels, style, resolution_2d,
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
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_vector_field_blocking,
        u_expr, v_expr, x_var, y_var,
        x_range, y_range, title, stream, density, style, resolution_2d,
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
    resolution_1d: int = PARAM_POINTS,
) -> discord.File:
    if t_min >= t_max:
        raise ValueError(f"t_min ({t_min}) must be < t_max ({t_max}).")
    buf = await _run_blocking(
        _plot_parametric_2d_blocking,
        x_expr, y_expr, t_var, t_min, t_max, title, xlabel, ylabel, style,
        resolution_1d,
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
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_surface_blocking,
        expr, x_var, y_var, x_range, y_range, title, style, resolution_2d,
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
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_wireframe_blocking,
        expr, x_var, y_var, x_range, y_range, title, rstride, cstride, style,
        resolution_2d,
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
    resolution_1d: int = PARAM_POINTS,
) -> discord.File:
    if t_min >= t_max:
        raise ValueError(f"t_min ({t_min}) must be < t_max ({t_max}).")
    buf = await _run_blocking(
        _plot_parametric_3d_blocking,
        x_expr, y_expr, z_expr, t_var,
        t_min, t_max, title, xlabel, ylabel, zlabel, style, resolution_1d,
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
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("xs, ys, and zs must all have the same length.")
    if not xs:
        raise ValueError("Cannot plot an empty data set.")
    buf = await _run_blocking(
        _plot_scatter_3d_blocking,
        xs, ys, zs, title, xlabel, ylabel, zlabel, style,
    )
    return discord.File(buf, filename="scatter_3d.png")


async def plot_polar(
    expr:            sympy.Expr,
    theta_var:       sympy.Symbol,
    theta_min:       float = 0.0,
    theta_max:       float = 2 * float(sympy.pi),
    title:           str   = "",
    style:           StyleOptions = _DEFAULT_STYLE,
    additional_exprs: list  = None,
    resolution_1d:   int   = PARAM_POINTS,
) -> discord.File:
    if theta_min >= theta_max:
        raise ValueError(
            f"theta_min ({theta_min}) must be < theta_max ({theta_max})."
        )
    buf = await _run_blocking(
        _plot_polar_blocking,
        expr, theta_var, theta_min, theta_max, title, style,
        additional_exprs, resolution_1d,
    )
    return discord.File(buf, filename="polar.png")


async def plot_implicit(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    rhs: float = 0.0,
    title: str = "",
    style: StyleOptions = _DEFAULT_STYLE,
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_implicit_blocking,
        expr, x_var, y_var, x_range, y_range, rhs, title, style, resolution_2d,
    )
    return discord.File(buf, filename="implicit.png")


async def plot_inequality(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    op: str = "<=",
    rhs: float = 0.0,
    title: str = "",
    style: StyleOptions = _DEFAULT_STYLE,
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_inequality_blocking,
        expr, x_var, y_var, x_range, y_range, op, rhs, title, style,
        resolution_2d,
    )
    return discord.File(buf, filename="inequality.png")


async def plot_histogram(
    values: list,
    bins: int = 20,
    title: str = "",
    xlabel: str = "value",
    ylabel: str = "count",
    style: StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    if not values:
        raise ValueError("Cannot plot an empty histogram data set.")
    bins = max(1, min(500, int(bins)))
    buf = await _run_blocking(
        _plot_histogram_blocking, values, bins, title, xlabel, ylabel, style,
    )
    return discord.File(buf, filename="histogram.png")


async def plot_errorbar(
    xs: list,
    ys: list,
    yerr: list,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    style: StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    if not (len(xs) == len(ys) == len(yerr)):
        raise ValueError("xs, ys, and error values must have the same length.")
    if not xs:
        raise ValueError("Cannot plot an empty errorbar data set.")
    buf = await _run_blocking(
        _plot_errorbar_blocking, xs, ys, yerr, title, xlabel, ylabel, style,
    )
    return discord.File(buf, filename="errorbar.png")


async def plot_heatmap(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    title: str = "",
    style: StyleOptions = _DEFAULT_STYLE,
    resolution_2d: int = GRID_POINTS,
) -> discord.File:
    buf = await _run_blocking(
        _plot_heatmap_blocking,
        expr, x_var, y_var, x_range, y_range, title, style, resolution_2d,
    )
    return discord.File(buf, filename="heatmap.png")


async def plot_boxplot(
    groups: list,
    mode: str = "box",
    title: str = "",
    ylabel: str = "value",
    style: StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
    clean_groups = [g for g in groups if len(g) > 0]
    if not clean_groups:
        raise ValueError("Cannot plot an empty boxplot data set.")
    mode = "violin" if mode == "violin" else "box"
    buf = await _run_blocking(
        _plot_boxplot_blocking, clean_groups, mode, title, ylabel, style,
    )
    return discord.File(buf, filename=f"{mode}plot.png")


async def plot_multi(
    specs:      Sequence[PlotSpec],
    ncols:      int   = 2,
    title:      str   = "",
    fig_width:  float = 14.0,
    row_height: float = 5.0,
    style:      StyleOptions = _DEFAULT_STYLE,
) -> discord.File:
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
# Animation helpers
# ---------------------------------------------------------------------------

from matplotlib.animation import FuncAnimation, PillowWriter


def _save_animation_to_gif(fig: plt.Figure, ani: FuncAnimation, fps: int = 15) -> io.BytesIO:
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        writer = PillowWriter(fps=fps)
        ani.save(tmp_name, writer=writer)
        with open(tmp_name, "rb") as fh:
            buf = io.BytesIO(fh.read())
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    plt.close(fig)
    buf.seek(0)
    return buf


def _anim_param_values() -> np.ndarray:
    return np.linspace(ANIM_PARAM_MIN, ANIM_PARAM_MAX, ANIM_FRAMES)


def _plot_animation_function_blocking(
    exprs: list,
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

        lines = [ax.plot([], [], label=str(expr))[0] for expr in exprs]
        funcs = [_lambdify2(expr, var, anim_var) for expr in exprs]

        if len(exprs) == 1:
            _apply_line_style(ax, style)
        else:
            ax.legend(loc="upper right")

        combined_title = title or ", ".join(str(e) for e in exprs)
        title_math = _to_math_label(combined_title)
        _apply_axes_style(ax, combined_title, str(var), f"f({var})", style.show_grid,
                          style=style)

        a_vals = _anim_param_values()

        all_ys = [
            [_eval1(lambda x, f=f, a=a: f(x, a), xs) for a in a_vals]
            for f in funcs
        ]

        flat_ys = np.concatenate([np.concatenate(per_func) for per_func in all_ys])
        valid = flat_ys[np.isfinite(flat_ys)]
        if len(valid) > 0:
            ax.set_ylim(valid.min() - 0.5, valid.max() + 0.5)
        ax.set_xlim(x_min, x_max)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame_idx):
            for line, per_func in zip(lines, all_ys):
                line.set_data(xs, per_func[frame_idx])
            ax.set_title(f"{title_math} ({anim_var}={a_vals[frame_idx]:.2f})")
            return lines

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, init_func=init, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_contour_blocking(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    anim_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    levels: int,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=ANIM_GRID_POINTS)
        f = _lambdify3(expr, x_var, y_var, anim_var)

        a_vals = _anim_param_values()
        all_Z = [_eval2(lambda X, Y, a=a: f(X, Y, a), X, Y) for a in a_vals]

        flat = np.concatenate([Z.ravel() for Z in all_Z])
        valid = flat[np.isfinite(flat)]
        vmin, vmax = (float(valid.min()), float(valid.max())) if len(valid) else (-1.0, 1.0)
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        combined_title = title or str(expr)
        title_math = _to_math_label(combined_title)
        xlabel_math = _to_math_label(str(x_var))
        ylabel_math = _to_math_label(str(y_var))

        def draw(idx: int):
            ax.clear()
            cf = ax.contourf(X, Y, all_Z[idx], levels=levels, cmap=style.colormap,
                             vmin=vmin, vmax=vmax)
            ax.set_title(f"{title_math} ({anim_var}={a_vals[idx]:.2f})", fontsize=12, pad=6)
            ax.set_xlabel(xlabel_math, fontsize=10)
            ax.set_ylabel(ylabel_math, fontsize=10)
            ax.grid(style.show_grid, alpha=0.2)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            return cf

        first_cf = draw(0)
        fig.colorbar(first_cf, ax=ax, shrink=0.85, label=f"f({x_var},{y_var})")

        def update(frame_idx: int):
            draw(frame_idx)
            return []

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_vector_field_blocking(
    u_expr: sympy.Expr,
    v_expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    anim_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    stream: bool,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n = 20 if not stream else min(ANIM_GRID_POINTS, 50)
        xs = np.linspace(x_range[0], x_range[1], n)
        ys = np.linspace(y_range[0], y_range[1], n)
        X, Y = np.meshgrid(xs, ys)

        fu = _lambdify3(u_expr, x_var, y_var, anim_var)
        fv = _lambdify3(v_expr, x_var, y_var, anim_var)

        a_vals = _anim_param_values()
        all_U, all_V, all_mag = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for a in a_vals:
                U = np.asarray(fu(X, Y, a), dtype=float)
                V = np.asarray(fv(X, Y, a), dtype=float)
                U = np.where(np.isfinite(U), U, 0.0)
                V = np.where(np.isfinite(V), V, 0.0)
                all_U.append(U)
                all_V.append(V)
                all_mag.append(np.hypot(U, V))

        flat_mag = np.concatenate([m.ravel() for m in all_mag])
        finite_mag = flat_mag[np.isfinite(flat_mag)]
        vmax = float(finite_mag.max()) if len(finite_mag) else 1.0
        vmax = vmax or 1.0

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        combined_title = title or f"({u_expr}, {v_expr})"
        title_math = _to_math_label(combined_title)
        xlabel_math = _to_math_label(str(x_var))
        ylabel_math = _to_math_label(str(y_var))

        def draw(idx: int):
            ax.clear()
            U, V, mag = all_U[idx], all_V[idx], all_mag[idx]
            if stream:
                ax.streamplot(
                    xs, ys, U, V,
                    color=mag, cmap=style.colormap,
                    density=1.2, linewidth=style.line_width, arrowsize=1.2,
                )
            else:
                mag_safe = np.where(mag == 0, 1, mag)
                ax.quiver(
                    X, Y, U / mag_safe, V / mag_safe, mag,
                    cmap=style.colormap, scale=25, width=0.003, clim=(0, vmax),
                )
            ax.set_title(f"{title_math} ({anim_var}={a_vals[idx]:.2f})", fontsize=12, pad=6)
            ax.set_xlabel(xlabel_math, fontsize=10)
            ax.set_ylabel(ylabel_math, fontsize=10)
            ax.grid(style.show_grid, alpha=0.2)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)

        draw(0)

        def update(frame_idx: int):
            draw(frame_idx)
            return []

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_parametric_2d_blocking(
    x_expr: sympy.Expr,
    y_expr: sympy.Expr,
    t_var: sympy.Symbol,
    anim_var: sympy.Symbol,
    t_min: float,
    t_max: float,
    title: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        ts = np.linspace(t_min, t_max, PARAM_POINTS)
        fx = _lambdify2(x_expr, t_var, anim_var)
        fy = _lambdify2(y_expr, t_var, anim_var)

        a_vals = _anim_param_values()
        all_xs, all_ys = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for a in a_vals:
                all_xs.append(np.asarray(fx(ts, a), dtype=float))
                all_ys.append(np.asarray(fy(ts, a), dtype=float))

        flat_x = np.concatenate(all_xs)
        flat_y = np.concatenate(all_ys)
        vx = flat_x[np.isfinite(flat_x)]
        vy = flat_y[np.isfinite(flat_y)]
        xmin, xmax = (float(vx.min()), float(vx.max())) if len(vx) else (-1.0, 1.0)
        ymin, ymax = (float(vy.min()), float(vy.max())) if len(vy) else (-1.0, 1.0)
        pad_x = (xmax - xmin) * 0.1 or 1.0
        pad_y = (ymax - ymin) * 0.1 or 1.0

        fig, ax = _white_fig(style, figsize=(style.fig_width, style.fig_height))
        line, = ax.plot([], [])
        _apply_line_style(ax, style)
        combined_title = title or f"({x_expr}, {y_expr})"
        title_math = _to_math_label(combined_title)
        _apply_axes_style(ax, combined_title, "x", "y", style.show_grid, style=style)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        def init():
            line.set_data([], [])
            return line,

        def update(frame_idx):
            line.set_data(all_xs[frame_idx], all_ys[frame_idx])
            ax.set_title(f"{title_math} ({anim_var}={a_vals[frame_idx]:.2f})", fontsize=12, pad=6)
            return line,

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, init_func=init, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_surface_blocking(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    anim_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        X, Y = _meshgrid(x_range, y_range, n=ANIM_GRID_POINTS)
        f = _lambdify3(expr, x_var, y_var, anim_var)

        a_vals = _anim_param_values()
        all_Z = [_eval2(lambda X, Y, a=a: f(X, Y, a), X, Y) for a in a_vals]

        flat = np.concatenate([Z.ravel() for Z in all_Z])
        valid = flat[np.isfinite(flat)]
        zmin, zmax = (float(valid.min()), float(valid.max())) if len(valid) else (-1.0, 1.0)
        if zmin == zmax:
            zmin, zmax = zmin - 1.0, zmax + 1.0

        fig, ax = _make_3d_axes(style)
        combined_title = title or str(expr)
        title_math = _to_math_label(combined_title)
        xlabel_math = _to_math_label(str(x_var))
        ylabel_math = _to_math_label(str(y_var))
        zlabel_math = _to_math_label(f"f({x_var},{y_var})")

        def draw(idx: int):
            ax.clear()
            ax.plot_surface(X, Y, all_Z[idx], cmap=style.colormap, alpha=style.alpha,
                            linewidth=0, antialiased=True)
            ax.set_zlim(zmin, zmax)
            ax.set_title(f"{title_math} ({anim_var}={a_vals[idx]:.2f})", fontsize=12)
            ax.set_xlabel(xlabel_math, fontsize=9)
            ax.set_ylabel(ylabel_math, fontsize=9)
            ax.set_zlabel(zlabel_math, fontsize=9)

        draw(0)

        def update(frame_idx: int):
            draw(frame_idx)
            return []

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_wireframe_blocking(
    expr: sympy.Expr,
    x_var: sympy.Symbol,
    y_var: sympy.Symbol,
    anim_var: sympy.Symbol,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n = min(ANIM_GRID_POINTS, 50)
        X, Y = _meshgrid(x_range, y_range, n)
        f = _lambdify3(expr, x_var, y_var, anim_var)

        a_vals = _anim_param_values()
        all_Z = [_eval2(lambda X, Y, a=a: f(X, Y, a), X, Y) for a in a_vals]

        flat = np.concatenate([Z.ravel() for Z in all_Z])
        valid = flat[np.isfinite(flat)]
        zmin, zmax = (float(valid.min()), float(valid.max())) if len(valid) else (-1.0, 1.0)
        if zmin == zmax:
            zmin, zmax = zmin - 1.0, zmax + 1.0

        fig, ax = _make_3d_axes(style)
        combined_title = title or str(expr)
        title_math = _to_math_label(combined_title)
        xlabel_math = _to_math_label(str(x_var))
        ylabel_math = _to_math_label(str(y_var))
        zlabel_math = _to_math_label(f"f({x_var},{y_var})")

        def draw(idx: int):
            ax.clear()
            ax.plot_wireframe(X, Y, all_Z[idx], color=style.color,
                              linewidth=style.line_width)
            ax.set_zlim(zmin, zmax)
            ax.set_title(f"{title_math} ({anim_var}={a_vals[idx]:.2f})", fontsize=12)
            ax.set_xlabel(xlabel_math, fontsize=9)
            ax.set_ylabel(ylabel_math, fontsize=9)
            ax.set_zlabel(zlabel_math, fontsize=9)

        draw(0)

        def update(frame_idx: int):
            draw(frame_idx)
            return []

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_polar_blocking(
    expr,
    theta_var,
    anim_var,
    theta_min,
    theta_max,
    title,
    style,
):
    with matplotlib.rc_context(rc=style.rc_overrides()):
        thetas = np.linspace(theta_min, theta_max, ANIM_PARAM_POINTS)
        f = _lambdify2(expr, theta_var, anim_var)
        a_vals = _anim_param_values()

        all_r = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for a in a_vals:
                r = np.asarray(f(thetas, a), dtype=float)
                all_r.append(np.where(np.isfinite(r), r, np.nan))

        fig = plt.figure(figsize=(style.fig_width, style.fig_height))
        ax = fig.add_subplot(111, projection="polar")
        (line,) = ax.plot([], [], color=style.color, linewidth=style.line_width,
                          linestyle=style.line_style)
        title_obj = ax.set_title("", fontsize=12, pad=14)
        title_str = title or str(expr)
        ax.grid(style.show_grid, alpha=0.3)

        def draw(idx):
            line.set_data(thetas, all_r[idx])
            finite = all_r[idx][np.isfinite(all_r[idx])]
            rmax = float(np.nanmax(np.abs(finite))) * 1.1 if finite.size else 1.0
            ax.set_rmax(rmax or 1.0)
            title_obj.set_text(
                f"{_to_math_label(title_str)} ({anim_var}={a_vals[idx]:.2f})"
            )

        draw(0)

        def update(frame_idx):
            draw(frame_idx)
            return [line]

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_implicit_blocking(
    expr,
    x_var,
    y_var,
    anim_var,
    x_range,
    y_range,
    rhs,
    title,
    style,
):
    with matplotlib.rc_context(rc=style.rc_overrides()):
        n = min(ANIM_GRID_POINTS, 80)
        X, Y = _meshgrid(x_range, y_range, n)
        f = _lambdify3(expr, x_var, y_var, anim_var)
        a_vals = _anim_param_values()

        fig, ax = _white_fig(style)
        title_str = title or f"{expr} = {rhs:g}"
        title_obj = ax.set_title("", fontsize=12, pad=6)
        ax.set_xlabel(_to_math_label(str(x_var)), fontsize=10)
        ax.set_ylabel(_to_math_label(str(y_var)), fontsize=10)
        ax.grid(style.show_grid, alpha=0.3)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        def draw(idx):
            ax.collections.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                Z = np.asarray(f(X, Y, a_vals[idx]), dtype=float) - rhs
            try:
                ax.contour(X, Y, Z, levels=[0.0], colors=[style.color],
                           linewidths=[style.line_width])
            except Exception:
                pass
            title_obj.set_text(
                f"{_to_math_label(title_str)} ({anim_var}={a_vals[idx]:.2f})"
            )

        draw(0)

        def update(frame_idx):
            draw(frame_idx)
            return []

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


def _plot_animation_parametric_3d_blocking(
    x_expr: sympy.Expr,
    y_expr: sympy.Expr,
    z_expr: sympy.Expr,
    t_var: sympy.Symbol,
    anim_var: sympy.Symbol,
    t_min: float,
    t_max: float,
    title: str,
    style: StyleOptions,
) -> io.BytesIO:
    with matplotlib.rc_context(rc=style.rc_overrides()):
        ts = np.linspace(t_min, t_max, ANIM_PARAM_POINTS)
        fx = _lambdify2(x_expr, t_var, anim_var)
        fy = _lambdify2(y_expr, t_var, anim_var)
        fz = _lambdify2(z_expr, t_var, anim_var)

        a_vals = _anim_param_values()
        all_xs, all_ys, all_zs = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for a in a_vals:
                all_xs.append(np.asarray(fx(ts, a), dtype=float))
                all_ys.append(np.asarray(fy(ts, a), dtype=float))
                all_zs.append(np.asarray(fz(ts, a), dtype=float))

        def _bounds(arrs):
            flat = np.concatenate(arrs)
            v = flat[np.isfinite(flat)]
            if not len(v):
                return -1.0, 1.0
            lo, hi = float(v.min()), float(v.max())
            pad = (hi - lo) * 0.1 or 1.0
            return lo - pad, hi + pad

        xlim = _bounds(all_xs)
        ylim = _bounds(all_ys)
        zlim = _bounds(all_zs)

        fig, ax = _make_3d_axes(style)
        combined_title = title or f"({x_expr}, {y_expr}, {z_expr})"

        def draw(idx: int):
            ax.clear()
            ax.plot(all_xs[idx], all_ys[idx], all_zs[idx],
                    color=style.color, linewidth=style.line_width)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_title(f"{combined_title} ({anim_var}={a_vals[idx]:.2f})", fontsize=12)
            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("y", fontsize=9)
            ax.set_zlabel("z", fontsize=9)

        draw(0)

        def update(frame_idx: int):
            draw(frame_idx)
            return []

        ani = FuncAnimation(fig, update, frames=ANIM_FRAMES, blit=False)
        return _save_animation_to_gif(fig, ani)


async def plot_animation(cfg) -> discord.File:
    anim_var = sympy.Symbol(cfg.anim_param or "a")
    style = cfg.to_style()
    pt = cfg.plot_type

    if pt == "function":
        x = sympy.Symbol("x")
        exprs = [_sympy_expr(_clean_sympy_expr(cfg.expr_main), x, anim_var)]
        for e in cfg.additional_exprs:
            try:
                exprs.append(_sympy_expr(_clean_sympy_expr(e), x, anim_var))
            except Exception:
                pass
        buf = await _run_blocking(
            _plot_animation_function_blocking,
            exprs, x, anim_var, cfg.x_min, cfg.x_max, cfg.title, style,
        )

    elif pt in ("contour", "surface", "wireframe"):
        x, y = sympy.Symbol("x"), sympy.Symbol("y")
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y, anim_var)
        x_range, y_range = (cfg.x_min, cfg.x_max), (cfg.y_min, cfg.y_max)
        if pt == "contour":
            buf = await _run_blocking(
                _plot_animation_contour_blocking,
                expr, x, y, anim_var, x_range, y_range, cfg.title, cfg.levels, style,
            )
        elif pt == "surface":
            buf = await _run_blocking(
                _plot_animation_surface_blocking,
                expr, x, y, anim_var, x_range, y_range, cfg.title, style,
            )
        else:
            buf = await _run_blocking(
                _plot_animation_wireframe_blocking,
                expr, x, y, anim_var, x_range, y_range, cfg.title, style,
            )

    elif pt == "vector-field":
        x, y = sympy.Symbol("x"), sympy.Symbol("y")
        u = _sympy_expr(_clean_sympy_expr(cfg.expr_u), x, y, anim_var)
        v = _sympy_expr(_clean_sympy_expr(cfg.expr_v), x, y, anim_var)
        buf = await _run_blocking(
            _plot_animation_vector_field_blocking,
            u, v, x, y, anim_var, (cfg.x_min, cfg.x_max), (cfg.y_min, cfg.y_max),
            cfg.title, cfg.stream, style,
        )

    elif pt in ("parametric-2d", "parametric-3d"):
        t = sympy.Symbol("t")
        xe = _sympy_expr(_clean_sympy_expr(cfg.expr_x), t, anim_var)
        ye = _sympy_expr(_clean_sympy_expr(cfg.expr_y), t, anim_var)
        if pt == "parametric-2d":
            buf = await _run_blocking(
                _plot_animation_parametric_2d_blocking,
                xe, ye, t, anim_var, cfg.t_min, cfg.t_max, cfg.title, style,
            )
        else:
            ze = _sympy_expr(_clean_sympy_expr(cfg.expr_z), t, anim_var)
            buf = await _run_blocking(
                _plot_animation_parametric_3d_blocking,
                xe, ye, ze, t, anim_var, cfg.t_min, cfg.t_max, cfg.title, style,
            )

    elif pt == "polar":
        theta_sym = sympy.Symbol(cfg.theta_symbol or "theta")
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), theta_sym, anim_var)
        buf = await _run_blocking(
            _plot_animation_polar_blocking,
            expr, theta_sym, anim_var, cfg.t_min, cfg.t_max, cfg.title, style,
        )

    elif pt == "implicit":
        x, y = sympy.Symbol("x"), sympy.Symbol("y")
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y, anim_var)
        buf = await _run_blocking(
            _plot_animation_implicit_blocking,
            expr, x, y, anim_var,
            (cfg.x_min, cfg.x_max), (cfg.y_min, cfg.y_max),
            cfg.implicit_rhs, cfg.title, style,
        )

    elif pt in ("scatter", "scatter-3d"):
        raise ValueError(
            "Animation isn't supported for scatter plots — there's no expression "
            "to vary a parameter over, just raw data points. Try function, "
            "contour, surface, wireframe, vector-field, or parametric instead."
        )

    else:
        raise ValueError(f"Animation isn't supported for plot type `{pt}` yet.")

    return discord.File(buf, filename="anim.gif")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "StyleOptions",
    "PlotSpec",
    "plot_function",
    "plot_riemann",
    "plot_points",
    "plot_polar",
    "plot_implicit",
    "plot_inequality",
    "plot_histogram",
    "plot_errorbar",
    "plot_heatmap",
    "plot_boxplot",
    "plot_contour",
    "plot_vector_field",
    "plot_parametric_2d",
    "plot_surface",
    "plot_wireframe",
    "plot_parametric_3d",
    "plot_scatter_3d",
    "plot_multi",
    "plot_animation",
]