"""
cogs/plot_engine.py — Interactive plot-builder for the math bot.

Provides three slash commands:

  /plot        Open the full interactive plot builder (ephemeral control panel).
  /quickplot   Instantly render a single function expression as a PNG.
  /multiplot   Plot up to 4 functions side-by-side in one image.

Note: /plot from cogs/calculus.py has been removed.  All plotting lives here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/plot  —  Interactive builder user flow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. /plot
   └── Control Panel embed appears (ephemeral)
        │  [Select] Plot type
        │  [Button] Expressions   → Modal: expressions + domain
        │  [Button] Style         → Modal: color / style options
        │  [Button] Axes & Labels → Modal: title, axis labels, ranges
        │  [Button] Advanced      → Modal: resolution, alpha, colormap, figsize
        │  [Button] Colormap      → Sub-view: colormap picker
        │  [Button] Stream ON/OFF → Toggle streamplot for vector fields
        │  [Button] Preview       → Refresh embed with current state
        │  [Button] Render        → Generate PNG and post publicly
        └── [Button] Reset        → Restore defaults (keep plot type)

All session state lives in a ``PlotConfig`` dataclass attached to each
``PlotEngineView`` instance.  Nothing is shared between sessions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adding a new plot type to the builder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Add the type string to ``PLOT_TYPES``.
2. Add a hint line to ``_type_hint()``.
3. Add a branch in ``_config_embed()`` for the expression summary fields.
4. Add a pre-fill branch in ``ExpressionModal.__init__()`` and a store
   branch in ``ExpressionModal.on_submit()``.
5. Add a dispatch branch in ``_render()`` that calls the appropriate
   function from ``utils.plotter``.
   Pass a ``StyleOptions`` built from the current ``PlotConfig`` — that
   is the only place ``StyleOptions`` is constructed in this file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  discord.py ≥ 2.3   (app_commands, ui.Modal, ui.Select, ui.Button)
  sympy
  utils.plotter      (plot functions + StyleOptions + PlotSpec)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import discord
import sympy
from discord import app_commands, ui
from discord.ext import commands

from utils.plotter import (
    PlotSpec,
    StyleOptions,
    plot_contour,
    plot_function,
    plot_parametric_2d,
    plot_parametric_3d,
    plot_points,
    plot_scatter_3d,
    plot_surface,
    plot_vector_field,
    plot_wireframe,
    plot_multi,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PLOT_TYPES = [
    "function",
    "contour",
    "vector-field",
    "parametric-2d",
    "surface",
    "wireframe",
    "parametric-3d",
    "scatter",
    "scatter-3d",
]

# Colormaps shown in the picker.  Extend freely — unknown names are accepted
# at render time and validated by matplotlib.
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdBu", "seismic",
    "Blues", "Greens", "Oranges", "Reds", "Purples",
    "rainbow", "jet", "turbo",
    "gray", "bone", "pink",
]

LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]
_LS_MAP     = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}

MARKERS = ["none", ".", "o", "s", "^", "D", "*", "+", "x"]

EMBED_COLOR = discord.Color.from_rgb(88, 101, 242)

# ─────────────────────────────────────────────────────────────────────────────
# PlotConfig — single source of truth for a builder session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlotConfig:
    """
    All customisation state for one /plot session.

    When you add a new plot type, add the expression fields it needs here
    with sensible defaults so the builder always has something to pre-fill.
    """

    # ── plot type ──────────────────────────────────────────────────────────
    plot_type: str = "function"

    # ── expressions (raw strings, parsed on render) ────────────────────────
    expr_main:  str = "sin(x)"
    expr_u:     str = "-y"
    expr_v:     str = "x"
    expr_x:     str = "cos(t)"
    expr_y:     str = "sin(t)"
    expr_z:     str = "t"
    scatter_xs: str = "1,2,3,4,5"
    scatter_ys: str = "1,4,9,16,25"
    scatter_zs: str = "0,1,0,1,0"

    # ── domain ────────────────────────────────────────────────────────────
    x_min: float = -10.0
    x_max: float  = 10.0
    y_min: float  = -10.0
    y_max: float  = 10.0
    t_min: float  = 0.0
    t_max: float  = 6.2832   # 2π

    # ── axes & labels ─────────────────────────────────────────────────────
    title:       str  = ""
    xlabel:      str  = "x"
    ylabel:      str  = "y"
    zlabel:      str  = "z"
    show_grid:   bool = True

    # ── line / scatter style ──────────────────────────────────────────────
    line_color:  str   = "#1f77b4"
    line_style:  str   = "solid"
    line_width:  float = 2.0
    marker:      str   = "none"
    marker_size: float = 6.0

    # ── surface / contour / vector style ──────────────────────────────────
    colormap: str   = "viridis"
    alpha:    float = 0.9
    levels:   int   = 20
    stream:   bool  = False

    # ── resolution & figure ───────────────────────────────────────────────
    resolution: int   = 120
    fig_width:  float = 8.0
    fig_height: float = 5.0
    dpi:        int   = 150

    # ── internal ──────────────────────────────────────────────────────────
    last_error: str = ""

    def to_style(self) -> StyleOptions:
        """
        Build a :class:`~utils.plotter.StyleOptions` from the current config.

        This is the single conversion point between PlotConfig and the
        plotter API — nothing else in this file should construct StyleOptions.
        """
        return StyleOptions(
            color       = self.line_color,
            line_width  = self.line_width,
            line_style  = _LS_MAP.get(self.line_style, "-"),
            marker      = None if self.marker == "none" else self.marker,
            marker_size = self.marker_size,
            colormap    = self.colormap,
            alpha       = self.alpha,
            show_grid   = self.show_grid,
            dpi         = self.dpi,
            fig_width   = self.fig_width,
            fig_height  = self.fig_height,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_float(s: str, default: float) -> float:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return default


def _parse_int(s: str, default: int) -> int:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return default


def _parse_bool(s: str, default: bool) -> bool:
    return s.strip().lower() in ("yes", "true", "1", "on") if s else default


def _parse_floatlist(s: str) -> list[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def _sympy_expr(s: str, *syms: sympy.Symbol) -> sympy.Expr:
    """Parse *s* as a SymPy expression; raises ValueError on failure."""
    try:
        local = {str(sym): sym for sym in syms}
        return sympy.sympify(s, locals=local)
    except Exception as exc:
        raise ValueError(f"Cannot parse expression `{s}`: {exc}") from exc


def _type_hint(pt: str) -> str:
    """One-line description shown in the plot-type select menu."""
    return {
        "function":      "f(x) line plot",
        "contour":       "filled contour map of f(x,y)",
        "vector-field":  "quiver / streamplot of (u,v)",
        "parametric-2d": "curve (x(t), y(t))",
        "surface":       "3-D shaded surface of f(x,y)",
        "wireframe":     "3-D wireframe of f(x,y)",
        "parametric-3d": "3-D curve (x,y,z)(t)",
        "scatter":       "scatter plot of (xs, ys)",
        "scatter-3d":    "3-D scatter of (xs, ys, zs)",
    }.get(pt, "")


def _config_embed(cfg: PlotConfig) -> discord.Embed:
    """Build the control-panel embed showing the current session state."""
    embed = discord.Embed(
        title="Plot Engine",
        description="Configure your plot using the controls below, then click **Render**.",
        color=EMBED_COLOR,
    )
    embed.add_field(name="Type",     value=f"`{cfg.plot_type}`",    inline=True)
    embed.add_field(name="Title",    value=cfg.title or "*(auto)*", inline=True)
    embed.add_field(name="Colormap", value=f"`{cfg.colormap}`",     inline=True)

    # ── type-specific expression summary ──────────────────────────────────
    # Add a new elif block here when adding a new plot type.
    if cfg.plot_type == "function":
        embed.add_field(name="f(x)",   value=f"`{cfg.expr_main}`",              inline=False)
        embed.add_field(name="Domain", value=f"x ∈ [{cfg.x_min}, {cfg.x_max}]", inline=True)

    elif cfg.plot_type in ("contour", "surface", "wireframe"):
        embed.add_field(name="f(x,y)",  value=f"`{cfg.expr_main}`",               inline=False)
        embed.add_field(name="x range", value=f"[{cfg.x_min}, {cfg.x_max}]",      inline=True)
        embed.add_field(name="y range", value=f"[{cfg.y_min}, {cfg.y_max}]",      inline=True)

    elif cfg.plot_type == "vector-field":
        embed.add_field(name="u(x,y)", value=f"`{cfg.expr_u}`",                    inline=True)
        embed.add_field(name="v(x,y)", value=f"`{cfg.expr_v}`",                    inline=True)
        embed.add_field(name="Stream", value="yes" if cfg.stream else "no",        inline=True)

    elif cfg.plot_type == "parametric-2d":
        embed.add_field(name="x(t)",    value=f"`{cfg.expr_x}`",                   inline=True)
        embed.add_field(name="y(t)",    value=f"`{cfg.expr_y}`",                   inline=True)
        embed.add_field(name="t range", value=f"[{cfg.t_min}, {cfg.t_max}]",       inline=True)

    elif cfg.plot_type == "parametric-3d":
        embed.add_field(name="x(t)",    value=f"`{cfg.expr_x}`",                   inline=True)
        embed.add_field(name="y(t)",    value=f"`{cfg.expr_y}`",                   inline=True)
        embed.add_field(name="z(t)",    value=f"`{cfg.expr_z}`",                   inline=True)
        embed.add_field(name="t range", value=f"[{cfg.t_min}, {cfg.t_max}]",       inline=True)

    elif cfg.plot_type in ("scatter", "scatter-3d"):
        embed.add_field(name="xs", value=f"`{cfg.scatter_xs[:60]}`",               inline=False)
        embed.add_field(name="ys", value=f"`{cfg.scatter_ys[:60]}`",               inline=False)
        if cfg.plot_type == "scatter-3d":
            embed.add_field(name="zs", value=f"`{cfg.scatter_zs[:60]}`",           inline=False)

    embed.add_field(
        name="Style",
        value=(f"color `{cfg.line_color}` · "
               f"line `{cfg.line_style}` · "
               f"lw `{cfg.line_width}` · "
               f"marker `{cfg.marker}`"),
        inline=False,
    )
    embed.add_field(
        name="Figure",
        value=(f"{cfg.fig_width}×{cfg.fig_height} in | {cfg.dpi} dpi | "
               f"res {cfg.resolution} | alpha {cfg.alpha}"),
        inline=False,
    )

    if cfg.last_error:
        embed.add_field(
            name="⚠ Last error",
            value=f"```{cfg.last_error[:900]}```",
            inline=False,
        )
        embed.color = discord.Color.red()

    embed.set_footer(
        text="Plot Engine · expressions use SymPy syntax  (sin, cos, exp, sqrt, pi …)"
    )
    return embed


# ─────────────────────────────────────────────────────────────────────────────
# Modals
# ─────────────────────────────────────────────────────────────────────────────

class ExpressionModal(ui.Modal, title="Expressions & Domain"):
    """Collects function expressions and domain/range values."""

    expr_a   = ui.TextInput(label="Expression A  (see hint in embed)",
                             placeholder="e.g. sin(x)/x   or   x**2 + y**2",
                             required=False, max_length=300)
    expr_b   = ui.TextInput(label="Expression B  (u for vector-field, y(t) for parametric)",
                             placeholder="e.g. cos(t)  or  -y",
                             required=False, max_length=300)
    expr_c   = ui.TextInput(label="Expression C  (v for vector-field, z(t) for 3-D)",
                             placeholder="e.g. x  or  t/(2*pi)",
                             required=False, max_length=300)
    domain_x = ui.TextInput(label="x range  OR  t range  (min, max)",
                             placeholder="-10, 10",
                             required=False, max_length=40)
    domain_y = ui.TextInput(label="y range  (ignored for 1-D / parametric)",
                             placeholder="-10, 10",
                             required=False, max_length=40)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view

        pt = cfg.plot_type
        # Pre-fill A/B/C based on plot type.
        # Add a new branch here when adding a new plot type.
        if pt == "function":
            self.expr_a.default = cfg.expr_main
        elif pt in ("contour", "surface", "wireframe"):
            self.expr_a.default = cfg.expr_main
        elif pt == "vector-field":
            self.expr_a.default = cfg.expr_u
            self.expr_b.default = cfg.expr_v
        elif pt == "parametric-2d":
            self.expr_a.default = cfg.expr_x
            self.expr_b.default = cfg.expr_y
        elif pt == "parametric-3d":
            self.expr_a.default = cfg.expr_x
            self.expr_b.default = cfg.expr_y
            self.expr_c.default = cfg.expr_z
        elif pt in ("scatter", "scatter-3d"):
            self.expr_a.default = cfg.scatter_xs
            self.expr_b.default = cfg.scatter_ys
            if pt == "scatter-3d":
                self.expr_c.default = cfg.scatter_zs

        self.domain_x.default = f"{cfg.x_min}, {cfg.x_max}"
        self.domain_y.default = f"{cfg.y_min}, {cfg.y_max}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        pt  = cfg.plot_type

        a = self.expr_a.value.strip()
        b = self.expr_b.value.strip()
        c = self.expr_c.value.strip()

        # Store expressions by type.
        # Add a new branch here when adding a new plot type.
        if pt == "function":
            if a: cfg.expr_main = a
        elif pt in ("contour", "surface", "wireframe"):
            if a: cfg.expr_main = a
        elif pt == "vector-field":
            if a: cfg.expr_u = a
            if b: cfg.expr_v = b
        elif pt == "parametric-2d":
            if a: cfg.expr_x = a
            if b: cfg.expr_y = b
        elif pt == "parametric-3d":
            if a: cfg.expr_x = a
            if b: cfg.expr_y = b
            if c: cfg.expr_z = c
        elif pt in ("scatter", "scatter-3d"):
            if a: cfg.scatter_xs = a
            if b: cfg.scatter_ys = b
            if c: cfg.scatter_zs = c

        def _range(raw: str, lo: str, hi: str) -> None:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) == 2:
                setattr(cfg, lo, _parse_float(parts[0], getattr(cfg, lo)))
                setattr(cfg, hi, _parse_float(parts[1], getattr(cfg, hi)))

        dx = self.domain_x.value
        dy = self.domain_y.value
        if pt in ("parametric-2d", "parametric-3d"):
            if dx: _range(dx, "t_min", "t_max")
        else:
            if dx: _range(dx, "x_min", "x_max")
            if dy: _range(dy, "y_min", "y_max")

        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class StyleModal(ui.Modal, title="Line & Marker Style"):
    color      = ui.TextInput(label="Line / scatter colour  (hex or name)",
                               placeholder="#1f77b4  or  red  or  steelblue",
                               required=False, max_length=40)
    width      = ui.TextInput(label="Line width  (float, e.g. 2.0)",
                               placeholder="2.0", required=False, max_length=10)
    style      = ui.TextInput(label="Line style  (solid | dashed | dotted | dashdot)",
                               placeholder="solid", required=False, max_length=10)
    marker     = ui.TextInput(label="Marker  (none | . | o | s | ^ | D | * | + | x)",
                               placeholder="none", required=False, max_length=5)
    markersize = ui.TextInput(label="Marker size  (float, e.g. 6.0)",
                               placeholder="6.0", required=False, max_length=10)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.color.default      = cfg.line_color
        self.width.default      = str(cfg.line_width)
        self.style.default      = cfg.line_style
        self.marker.default     = cfg.marker
        self.markersize.default = str(cfg.marker_size)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        if self.color.value:
            cfg.line_color  = self.color.value.strip()
        if self.width.value:
            cfg.line_width  = _parse_float(self.width.value, cfg.line_width)
        if self.style.value and self.style.value.strip() in LINE_STYLES:
            cfg.line_style  = self.style.value.strip()
        if self.marker.value and self.marker.value.strip() in MARKERS:
            cfg.marker      = self.marker.value.strip()
        if self.markersize.value:
            cfg.marker_size = _parse_float(self.markersize.value, cfg.marker_size)
        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class AxesModal(ui.Modal, title="Axes, Labels & Grid"):
    plot_title = ui.TextInput(label="Plot title  (blank = auto)",
                               placeholder="My Beautiful Plot",
                               required=False, max_length=120)
    xl   = ui.TextInput(label="x-axis label", placeholder="x",
                        required=False, max_length=60)
    yl   = ui.TextInput(label="y-axis label", placeholder="y",
                        required=False, max_length=60)
    zl   = ui.TextInput(label="z-axis label  (3-D only)", placeholder="z",
                        required=False, max_length=60)
    grid = ui.TextInput(label="Show grid?  (yes / no)",
                        placeholder="yes", required=False, max_length=5)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.plot_title.default = cfg.title
        self.xl.default         = cfg.xlabel
        self.yl.default         = cfg.ylabel
        self.zl.default         = cfg.zlabel
        self.grid.default       = "yes" if cfg.show_grid else "no"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        cfg.title     = self.plot_title.value.strip()
        if self.xl.value:   cfg.xlabel    = self.xl.value.strip()
        if self.yl.value:   cfg.ylabel    = self.yl.value.strip()
        if self.zl.value:   cfg.zlabel    = self.zl.value.strip()
        cfg.show_grid = _parse_bool(self.grid.value, cfg.show_grid)
        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class AdvancedModal(ui.Modal, title="Advanced Options"):
    cmap       = ui.TextInput(label="Colormap",
                               placeholder="viridis", required=False, max_length=30)
    alpha_val  = ui.TextInput(label="Opacity / alpha  (0.0 – 1.0)",
                               placeholder="0.9", required=False, max_length=6)
    levels_val = ui.TextInput(label="Contour levels  (integer, contour only)",
                               placeholder="20", required=False, max_length=5)
    res        = ui.TextInput(label="Grid resolution  (integer, 40–400)",
                               placeholder="120", required=False, max_length=5)
    figsize    = ui.TextInput(label="Figure size  (width, height) in inches",
                               placeholder="8, 5", required=False, max_length=15)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.cmap.default       = cfg.colormap
        self.alpha_val.default  = str(cfg.alpha)
        self.levels_val.default = str(cfg.levels)
        self.res.default        = str(cfg.resolution)
        self.figsize.default    = f"{cfg.fig_width}, {cfg.fig_height}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        if self.cmap.value:
            cfg.colormap = self.cmap.value.strip()   # validated by matplotlib at render
        if self.alpha_val.value:
            cfg.alpha    = max(0.0, min(1.0, _parse_float(self.alpha_val.value, cfg.alpha)))
        if self.levels_val.value:
            cfg.levels   = max(2, min(100, _parse_int(self.levels_val.value, cfg.levels)))
        if self.res.value:
            cfg.resolution = max(20, min(500, _parse_int(self.res.value, cfg.resolution)))
        if self.figsize.value:
            parts = self.figsize.value.split(",")
            if len(parts) == 2:
                cfg.fig_width  = max(2.0, min(24.0, _parse_float(parts[0], cfg.fig_width)))
                cfg.fig_height = max(2.0, min(24.0, _parse_float(parts[1], cfg.fig_height)))
        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


# ─────────────────────────────────────────────────────────────────────────────
# Render  —  PlotConfig → discord.File
# ─────────────────────────────────────────────────────────────────────────────

async def _render(cfg: PlotConfig) -> discord.File:
    """
    Translate a :class:`PlotConfig` into a :class:`discord.File`.

    Style options are gathered once via ``cfg.to_style()`` and forwarded to
    the plotter.  No rcParams are touched here — that is entirely the
    plotter's responsibility (it uses ``rc_context`` internally).

    To add a new plot type, add a branch below and call the matching
    ``plot_*`` function from ``utils.plotter``.
    """
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    t = sympy.Symbol("t")

    style = cfg.to_style()
    pt    = cfg.plot_type

    if pt == "function":
        expr = _sympy_expr(cfg.expr_main, x)
        return await plot_function(
            expr, x,
            x_min=cfg.x_min, x_max=cfg.x_max,
            title=cfg.title or str(expr),
            style=style,
        )

    elif pt == "contour":
        expr = _sympy_expr(cfg.expr_main, x, y)
        return await plot_contour(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            levels=cfg.levels,
            style=style,
        )

    elif pt == "vector-field":
        u = _sympy_expr(cfg.expr_u, x, y)
        v = _sympy_expr(cfg.expr_v, x, y)
        return await plot_vector_field(
            u, v, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            stream=cfg.stream,
            style=style,
        )

    elif pt == "parametric-2d":
        xe = _sympy_expr(cfg.expr_x, t)
        ye = _sympy_expr(cfg.expr_y, t)
        return await plot_parametric_2d(
            xe, ye, t,
            t_min=cfg.t_min, t_max=cfg.t_max,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            style=style,
        )

    elif pt == "surface":
        expr = _sympy_expr(cfg.expr_main, x, y)
        return await plot_surface(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            style=style,
        )

    elif pt == "wireframe":
        expr = _sympy_expr(cfg.expr_main, x, y)
        return await plot_wireframe(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            style=style,
        )

    elif pt == "parametric-3d":
        xe = _sympy_expr(cfg.expr_x, t)
        ye = _sympy_expr(cfg.expr_y, t)
        ze = _sympy_expr(cfg.expr_z, t)
        return await plot_parametric_3d(
            xe, ye, ze, t,
            t_min=cfg.t_min, t_max=cfg.t_max,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            zlabel=cfg.zlabel,
            style=style,
        )

    elif pt == "scatter":
        xs_list = _parse_floatlist(cfg.scatter_xs)
        ys_list = _parse_floatlist(cfg.scatter_ys)
        return await plot_points(
            xs_list, ys_list,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            style=style,
        )

    elif pt == "scatter-3d":
        xs_list = _parse_floatlist(cfg.scatter_xs)
        ys_list = _parse_floatlist(cfg.scatter_ys)
        zs_list = _parse_floatlist(cfg.scatter_zs)
        return await plot_scatter_3d(
            xs_list, ys_list, zs_list,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            zlabel=cfg.zlabel,
            style=style,
        )

    else:
        raise ValueError(f"Unknown plot type: `{pt}`")


# ─────────────────────────────────────────────────────────────────────────────
# PlotEngineView — the persistent control panel
# ─────────────────────────────────────────────────────────────────────────────

class PlotEngineView(ui.View):
    """
    Persistent ephemeral Discord UI for configuring and rendering a plot.
    Times out after 10 minutes of inactivity.
    """

    def __init__(self, cfg: PlotConfig) -> None:
        super().__init__(timeout=600)
        self.cfg = cfg
        self._add_type_select()

    # ── Plot type selector ────────────────────────────────────────────────

    def _add_type_select(self) -> None:
        options = [
            discord.SelectOption(
                label=pt,
                value=pt,
                description=_type_hint(pt),
                default=(pt == self.cfg.plot_type),
            )
            for pt in PLOT_TYPES
        ]
        select = ui.Select(placeholder="Choose plot type…", options=options, row=0)
        select.callback = self._on_type_select
        self.add_item(select)

    async def _on_type_select(self, interaction: discord.Interaction) -> None:
        self.cfg.plot_type  = interaction.data["values"][0]
        self.cfg.last_error = ""
        self.clear_items()
        self._add_type_select()
        self._add_buttons()
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    # ── Buttons ───────────────────────────────────────────────────────────

    def _add_buttons(self) -> None:
        """Re-add all action buttons.  Called after every select rebuild."""

        def _btn(label, style, row, cb):
            b = ui.Button(label=label, style=style, row=row)
            b.callback = cb
            self.add_item(b)

        # Row 1 — expression / domain / style / axes
        _btn("Expressions",  discord.ButtonStyle.primary,   1, self._on_expr)
        _btn("Style",        discord.ButtonStyle.primary,   1, self._on_style)
        _btn("Axes & Labels",discord.ButtonStyle.primary,   1, self._on_axes)

        # Row 2 — advanced / colormap / stream toggle
        _btn("Advanced",     discord.ButtonStyle.secondary, 2, self._on_advanced)
        _btn("Colormap",     discord.ButtonStyle.secondary, 2, self._on_cmap)

        stream_label = "Stream: " + ("ON" if self.cfg.stream else "OFF")
        stream_style = (discord.ButtonStyle.success
                        if self.cfg.stream else discord.ButtonStyle.secondary)
        _btn(stream_label, stream_style, 2, self._on_stream)

        # Row 3 — preview / render / reset
        _btn("Preview config", discord.ButtonStyle.secondary, 3, self._on_preview)
        _btn("Render plot",    discord.ButtonStyle.success,   3, self._on_render)
        _btn("Reset",          discord.ButtonStyle.danger,    3, self._on_reset)

    # ── Button callbacks ──────────────────────────────────────────────────

    async def _on_expr(self,     interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(ExpressionModal(self.cfg, self))

    async def _on_style(self,    interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(StyleModal(self.cfg, self))

    async def _on_axes(self,     interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AxesModal(self.cfg, self))

    async def _on_advanced(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AdvancedModal(self.cfg, self))

    async def _on_stream(self, interaction: discord.Interaction) -> None:
        self.cfg.stream = not self.cfg.stream
        self.clear_items()
        self._add_type_select()
        self._add_buttons()
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def _on_cmap(self, interaction: discord.Interaction) -> None:
        view = ColormapPickerView(self.cfg, parent=self)
        await interaction.response.send_message(
            embed=discord.Embed(
                title="Choose a Colormap",
                description=(
                    "Select one of the presets below.\n"
                    "For a custom name, use **Advanced** → colormap field."
                ),
                color=EMBED_COLOR,
            ),
            view=view,
            ephemeral=True,
        )

    async def _on_preview(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def _on_render(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            file = await _render(self.cfg)
            self.cfg.last_error = ""
        except Exception as exc:
            self.cfg.last_error = str(exc)
            await interaction.followup.edit_message(
                message_id=interaction.message.id,
                embed=_config_embed(self.cfg),
                view=self,
            )
            await interaction.followup.send(f"⚠ Render failed: {exc}", ephemeral=True)
            return

        await interaction.followup.edit_message(
            message_id=interaction.message.id,
            embed=_config_embed(self.cfg),
            view=self,
        )

        embed_out = discord.Embed(
            title=self.cfg.title or f"{self.cfg.plot_type} plot",
            color=EMBED_COLOR,
        )
        embed_out.set_image(url="attachment://plot.png")
        embed_out.set_footer(
            text=(f"type={self.cfg.plot_type} | cmap={self.cfg.colormap} | "
                  f"alpha={self.cfg.alpha} | res={self.cfg.resolution}")
        )
        await interaction.followup.send(embed=embed_out, file=file)

    async def _on_reset(self, interaction: discord.Interaction) -> None:
        pt = self.cfg.plot_type
        self.cfg.__init__()
        self.cfg.plot_type = pt
        self.clear_items()
        self._add_type_select()
        self._add_buttons()
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True


# ─────────────────────────────────────────────────────────────────────────────
# Colormap picker — secondary ephemeral view
# ─────────────────────────────────────────────────────────────────────────────

class ColormapPickerView(ui.View):
    """Paginated colormap selector that writes back to the parent view."""

    PAGE_SIZE = 25

    def __init__(self, cfg: PlotConfig, parent: PlotEngineView) -> None:
        super().__init__(timeout=120)
        self._cfg    = cfg
        self._parent = parent
        self._page   = 0
        self._build()

    def _build(self) -> None:
        self.clear_items()
        start = self._page * self.PAGE_SIZE
        chunk = COLORMAPS[start: start + self.PAGE_SIZE]
        options = [
            discord.SelectOption(label=c, value=c, default=(c == self._cfg.colormap))
            for c in chunk
        ]
        sel = ui.Select(placeholder="Pick a colormap…", options=options)
        sel.callback = self._on_pick
        self.add_item(sel)

        if start > 0:
            btn = ui.Button(label="< Prev", style=discord.ButtonStyle.secondary, row=1)
            btn.callback = self._on_prev
            self.add_item(btn)
        if start + self.PAGE_SIZE < len(COLORMAPS):
            btn = ui.Button(label="Next >", style=discord.ButtonStyle.secondary, row=1)
            btn.callback = self._on_next
            self.add_item(btn)

    async def _on_pick(self, interaction: discord.Interaction) -> None:
        self._cfg.colormap = interaction.data["values"][0]
        self._parent.clear_items()
        self._parent._add_type_select()
        self._parent._add_buttons()
        await interaction.response.edit_message(
            content=f"Colormap set to `{self._cfg.colormap}`.", view=None
        )

    async def _on_prev(self, interaction: discord.Interaction) -> None:
        self._page -= 1
        self._build()
        await interaction.response.edit_message(view=self)

    async def _on_next(self, interaction: discord.Interaction) -> None:
        self._page += 1
        self._build()
        await interaction.response.edit_message(view=self)


# ─────────────────────────────────────────────────────────────────────────────
# Cog
# ─────────────────────────────────────────────────────────────────────────────

class PlotEngine(commands.Cog):
    """
    Interactive plot builder and quick-plot commands.

    Commands
    --------
    /plot        Full interactive builder (ephemeral control panel).
    /quickplot   Instantly render a function — no builder UI.
    /multiplot   Up to 4 functions side-by-side in one image.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ── /plot ──────────────────────────────────────────────────────────────

    @app_commands.command(
        name="plot",
        description="Open the interactive plot builder to create a customised plot.",
    )
    @app_commands.describe(
        plot_type="Starting plot type (can be changed inside the builder).",
        quick_expr="Quick expression to pre-fill (e.g. sin(x)*exp(-x/4)).",
    )
    @app_commands.choices(plot_type=[
        app_commands.Choice(name=pt, value=pt) for pt in PLOT_TYPES
    ])
    async def plot(
        self,
        interaction: discord.Interaction,
        plot_type:  Optional[str] = "function",
        quick_expr: Optional[str] = None,
    ) -> None:
        cfg = PlotConfig(plot_type=plot_type or "function")
        if quick_expr:
            cfg.expr_main = quick_expr
            cfg.expr_x    = quick_expr

        view = PlotEngineView(cfg)
        view._add_buttons()

        await interaction.response.send_message(
            embed=_config_embed(cfg),
            view=view,
            ephemeral=True,
        )

    # ── /quickplot ─────────────────────────────────────────────────────────

    @app_commands.command(
        name="quickplot",
        description="Plot a function expression instantly with no builder UI.",
    )
    @app_commands.describe(
        expression="SymPy expression in x, e.g. sin(x)/x or x**3 - 2*x",
        x_min="Left boundary (default -10).",
        x_max="Right boundary (default 10).",
        title="Optional plot title.",
        color="Line colour (hex or name, default #1f77b4).",
        line_style="Line style: solid | dashed | dotted | dashdot.",
    )
    async def quickplot(
        self,
        interaction: discord.Interaction,
        expression: str,
        x_min: float = -10.0,
        x_max: float  = 10.0,
        title: str    = "",
        color: str    = "#1f77b4",
        line_style: app_commands.Choice[str] = None,
    ) -> None:
        await interaction.response.defer(thinking=True)

        cfg            = PlotConfig()
        cfg.plot_type  = "function"
        cfg.expr_main  = expression
        cfg.x_min      = x_min
        cfg.x_max      = x_max
        cfg.title      = title
        cfg.line_color = color
        if line_style:
            cfg.line_style = line_style.value

        try:
            file = await _render(cfg)
        except Exception as exc:
            await interaction.followup.send(f"⚠ Could not render: {exc}", ephemeral=True)
            return

        embed = discord.Embed(title=title or expression, color=EMBED_COLOR)
        embed.set_image(url="attachment://plot.png")
        embed.set_footer(text=f"x ∈ [{x_min}, {x_max}] | color={color}")
        await interaction.followup.send(embed=embed, file=file)

    # ── /multiplot ─────────────────────────────────────────────────────────

    @app_commands.command(
        name="multiplot",
        description="Plot up to 4 functions side-by-side in one image.",
    )
    @app_commands.describe(
        f1="First expression in x.",
        f2="Second expression in x (optional).",
        f3="Third expression in x (optional).",
        f4="Fourth expression in x (optional).",
        x_min="Domain left boundary (default -10).",
        x_max="Domain right boundary (default 10).",
        ncols="Columns in grid (1 or 2, default 2).",
        title="Overall figure title.",
    )
    async def multiplot(
        self,
        interaction: discord.Interaction,
        f1: str,
        f2: Optional[str] = None,
        f3: Optional[str] = None,
        f4: Optional[str] = None,
        x_min: float = -10.0,
        x_max: float  = 10.0,
        ncols: int    = 2,
        title: str    = "",
    ) -> None:
        await interaction.response.defer(thinking=True)

        x     = sympy.Symbol("x")
        exprs = [e for e in (f1, f2, f3, f4) if e]
        specs: List[PlotSpec] = []

        try:
            for raw in exprs:
                expr = _sympy_expr(raw, x)
                specs.append(PlotSpec(
                    kind="function",
                    expr=expr,
                    var=x,
                    x_min=x_min,
                    x_max=x_max,
                    title=str(expr),
                ))
        except ValueError as exc:
            await interaction.followup.send(f"⚠ Expression error: {exc}", ephemeral=True)
            return

        try:
            file = await plot_multi(specs, ncols=min(max(ncols, 1), 2), title=title)
        except Exception as exc:
            await interaction.followup.send(f"⚠ Render error: {exc}", ephemeral=True)
            return

        embed = discord.Embed(title=title or "Multi-function plot", color=EMBED_COLOR)
        embed.set_image(url="attachment://multi_plot.png")
        embed.set_footer(text="  |  ".join(exprs))
        await interaction.followup.send(embed=embed, file=file)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PlotEngine(bot))
