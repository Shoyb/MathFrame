"""
cogs/plot_engine.py — Interactive plot-builder for the math bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import base64
import json
import zlib
import re

import discord
import sympy
from discord import app_commands, ui
from discord.ext import commands

from utils.plotter import (
    PlotSpec,
    StyleOptions,
    plot_contour,
    plot_boxplot,
    plot_errorbar,
    plot_function,
    plot_heatmap,
    plot_histogram,
    plot_implicit,
    plot_inequality,
    plot_parametric_2d,
    plot_parametric_3d,
    plot_points,
    plot_polar,
    plot_riemann,
    plot_scatter_3d,
    plot_surface,
    plot_vector_field,
    plot_wireframe,
    plot_multi,
)
from utils.expr_utils import (
    _clean_sympy_expr as _shared_clean_sympy_expr,
    _sympy_expr as _shared_sympy_expr,
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
    "polar",
    "implicit",
    "inequality",
    "riemann",
    "histogram",
    "errorbar",
    "heatmap",
    "boxplot",
]

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
    plot_type: str = "function"

    expr_main:  str = "sin(x)"
    expr_u:     str = "-y"
    expr_v:     str = "x"
    expr_x:     str = "cos(t)"
    expr_y:     str = "sin(t)"
    expr_z:     str = "t"
    scatter_xs: str = "1,2,3,4,5"
    scatter_ys: str = "1,4,9,16,25"
    scatter_zs: str = "0,1,0,1,0"
    additional_exprs: List[str] = field(default_factory=list)

    x_min: float = -10.0
    x_max: float  = 10.0
    y_min: float  = -10.0
    y_max: float  = 10.0
    t_min: float  = 0.0
    t_max: float  = 6.2832   # 2π

    title:       str  = ""
    xlabel:      str  = "x"
    ylabel:      str  = "y"
    zlabel:      str  = "z"
    show_grid:   bool = True

    x_log: bool = False
    y_log: bool = False

    line_color:  str   = "#1f77b4"
    line_style:  str   = "solid"
    line_width:  float = 2.0
    marker:      str   = "none"
    marker_size: float = 6.0

    fill_below: bool = False
    fill_color: str  = ""

    colormap: str   = "viridis"
    theme:    str   = "default"
    alpha:    float = 0.9
    levels:   int   = 20
    stream:   bool  = False
    anim_param: str = ""

    theta_symbol: str = "theta"
    implicit_rhs: float = 0.0
    inequality_op: str = "<="
    hist_bins: int = 20
    box_violin: str = "box"
    riemann_n: int = 8
    riemann_method: str = "left"

    x_lim_min: Optional[float] = None
    x_lim_max: Optional[float] = None
    y_lim_min: Optional[float] = None
    y_lim_max: Optional[float] = None

    resolution_1d: int = 800
    resolution_2d: int = 120

    fig_width:  float = 8.0
    fig_height: float = 5.0
    dpi:        int   = 150

    last_error: str = ""

    def to_style(self) -> StyleOptions:
        x_lim = (
            (self.x_lim_min, self.x_lim_max)
            if self.x_lim_min is not None and self.x_lim_max is not None
            else None
        )
        y_lim = (
            (self.y_lim_min, self.y_lim_max)
            if self.y_lim_min is not None and self.y_lim_max is not None
            else None
        )

        return StyleOptions(
            color       = self.line_color,
            line_width  = self.line_width,
            line_style  = _LS_MAP.get(self.line_style, "-"),
            marker      = None if self.marker == "none" else self.marker,
            marker_size = self.marker_size,
            colormap    = self.colormap,
            theme       = self.theme,
            alpha       = self.alpha,
            show_grid   = self.show_grid,
            dpi         = self.dpi,
            fig_width   = self.fig_width,
            fig_height  = self.fig_height,
            x_log       = self.x_log,
            y_log       = self.y_log,
            fill_below  = self.fill_below,
            fill_color  = self.fill_color,
            x_lim       = x_lim,
            y_lim       = y_lim,
        )

    def export_config(self) -> str:
        skip = {"last_error"}
        data = {k: v for k, v in self.__dict__.items() if k not in skip}
        js = json.dumps(data)
        compressed = zlib.compress(js.encode("utf-8"))
        return base64.urlsafe_b64encode(compressed).decode("ascii")

    @classmethod
    def import_config(cls, data: str) -> "PlotConfig":
        try:
            compressed = base64.urlsafe_b64decode(data.encode("ascii"))
            js = zlib.decompress(compressed).decode("utf-8")
            state = json.loads(js)

            if "resolution" in state and "resolution_2d" not in state:
                state["resolution_2d"] = state.pop("resolution")
                state.setdefault("resolution_1d", 800)
            elif "resolution" in state:
                state.pop("resolution")

            cfg = cls()
            for k, v in state.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        except Exception as exc:
            raise ValueError(f"Invalid config string: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_ASSIGNMENT_PREFIX_RE = re.compile(r'^\s*[A-Za-z_]\w*\s*(?:\([^)]*\))?\s*=(?!=)\s*(.+)$')


def _clean_sympy_expr(s: str) -> str:
    return _shared_clean_sympy_expr(s)


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
    result = []
    for v in s.split(","):
        v = v.strip()
        if not v:
            continue
        try:
            result.append(float(v))
        except ValueError:
            raise ValueError(
                f"Could not parse `{v}` as a number. "
                "Data fields expect comma-separated numbers, e.g. `1, 2.5, -3`."
            )
    return result


def _parse_optional_float(s: str) -> Optional[float]:
    s = s.strip().lower()
    if not s or s in ("none", "auto", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _sympy_expr(s: str, *syms: sympy.Symbol) -> sympy.Expr:
    return _shared_sympy_expr(s, *syms)


def _type_hint(pt: str) -> str:
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
        "polar":         "polar curve r(θ)",
        "implicit":      "implicit curve f(x,y)=k",
        "inequality":    "shaded region f(x,y) <= k",
        "riemann":       "Riemann rectangles under f(x)",
        "histogram":     "distribution of raw data",
        "errorbar":      "points with vertical errors",
        "heatmap":       "imshow heatmap of f(x,y)",
        "boxplot":       "box or violin grouped data",
    }.get(pt, "")


def _log_label(cfg: PlotConfig) -> str:
    axes = []
    if cfg.x_log:
        axes.append("x")
    if cfg.y_log:
        axes.append("y")
    return ", ".join(axes) if axes else "none"


def _lim_label(cfg: PlotConfig) -> str:
    parts = []
    if cfg.x_lim_min is not None or cfg.x_lim_max is not None:
        lo = cfg.x_lim_min if cfg.x_lim_min is not None else "auto"
        hi = cfg.x_lim_max if cfg.x_lim_max is not None else "auto"
        parts.append(f"x [{lo}, {hi}]")
    if cfg.y_lim_min is not None or cfg.y_lim_max is not None:
        lo = cfg.y_lim_min if cfg.y_lim_min is not None else "auto"
        hi = cfg.y_lim_max if cfg.y_lim_max is not None else "auto"
        parts.append(f"y [{lo}, {hi}]")
    return " | ".join(parts) if parts else "auto"


def _config_embed(cfg: PlotConfig) -> discord.Embed:
    embed = discord.Embed(
        title="Plot Engine",
        description="Configure your plot using the controls below, then click **Render**.",
        color=EMBED_COLOR,
    )
    embed.add_field(name="Type",     value=f"`{cfg.plot_type}`",    inline=True)
    embed.add_field(name="Title",    value=cfg.title or "*(auto)*", inline=True)
    embed.add_field(name="Theme",    value=f"`{cfg.theme}`",        inline=True)

    if cfg.plot_type == "function":
        embed.add_field(name="f(x)",   value=f"`{cfg.expr_main}`",              inline=False)
        if cfg.additional_exprs:
            embed.add_field(name="Extra f(x)", value=", ".join(f"`{e}`" for e in cfg.additional_exprs), inline=False)
        embed.add_field(name="Domain", value=f"x ∈ [{cfg.x_min}, {cfg.x_max}]", inline=True)

    elif cfg.plot_type in ("contour", "surface", "wireframe", "heatmap"):
        embed.add_field(name="f(x,y)",  value=f"`{cfg.expr_main}`",               inline=False)
        embed.add_field(name="x range", value=f"[{cfg.x_min}, {cfg.x_max}]",      inline=True)
        embed.add_field(name="y range", value=f"[{cfg.y_min}, {cfg.y_max}]",      inline=True)

    elif cfg.plot_type == "implicit":
        embed.add_field(name="Equation", value=f"`{cfg.expr_main} = {cfg.implicit_rhs}`", inline=False)
        embed.add_field(name="x range", value=f"[{cfg.x_min}, {cfg.x_max}]", inline=True)
        embed.add_field(name="y range", value=f"[{cfg.y_min}, {cfg.y_max}]", inline=True)

    elif cfg.plot_type == "inequality":
        embed.add_field(name="Inequality", value=f"`{cfg.expr_main} {cfg.inequality_op} {cfg.implicit_rhs}`", inline=False)
        embed.add_field(name="x range", value=f"[{cfg.x_min}, {cfg.x_max}]", inline=True)
        embed.add_field(name="y range", value=f"[{cfg.y_min}, {cfg.y_max}]", inline=True)

    elif cfg.plot_type == "riemann":
        embed.add_field(name="f(x)", value=f"`{cfg.expr_main}`", inline=False)
        embed.add_field(name="Bounds", value=f"[{cfg.x_min}, {cfg.x_max}]", inline=True)
        embed.add_field(name="Rectangles", value=f"`{cfg.riemann_n}`", inline=True)
        embed.add_field(name="Method", value=f"`{cfg.riemann_method}`", inline=True)

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

    elif cfg.plot_type in ("scatter", "scatter-3d", "errorbar"):
        embed.add_field(name="xs", value=f"`{cfg.scatter_xs[:60]}`",               inline=False)
        embed.add_field(name="ys", value=f"`{cfg.scatter_ys[:60]}`",               inline=False)
        if cfg.plot_type == "scatter-3d":
            embed.add_field(name="zs", value=f"`{cfg.scatter_zs[:60]}`",           inline=False)
        elif cfg.plot_type == "errorbar":
            embed.add_field(name="y errors", value=f"`{cfg.expr_z[:60]}`",          inline=False)

    elif cfg.plot_type == "histogram":
        embed.add_field(name="data", value=f"`{cfg.scatter_xs[:80]}`", inline=False)
        embed.add_field(name="bins", value=f"`{cfg.hist_bins}`", inline=True)

    elif cfg.plot_type == "boxplot":
        embed.add_field(name="groups", value=f"`{cfg.scatter_xs[:80]}`", inline=False)
        embed.add_field(name="mode", value=f"`{cfg.box_violin}`", inline=True)

    elif cfg.plot_type == "polar":
        embed.add_field(name="r(θ)",     value=f"`{cfg.expr_main}`",                inline=False)
        if cfg.additional_exprs:
            embed.add_field(
                name="Extra r(θ)",
                value=", ".join(f"`{e}`" for e in cfg.additional_exprs),
                inline=False,
            )
        embed.add_field(name="θ symbol", value=f"`{cfg.theta_symbol}`",            inline=True)
        embed.add_field(name="θ range",  value=f"[{cfg.t_min:.4g}, {cfg.t_max:.4g}]", inline=True)

    embed.add_field(
        name="Style",
        value=(f"color `{cfg.line_color}` · "
               f"line `{cfg.line_style}` · "
               f"lw `{cfg.line_width}` · "
               f"marker `{cfg.marker}`"),
        inline=False,
    )

    axis_info_parts = [f"log `{_log_label(cfg)}`", f"grid `{'yes' if cfg.show_grid else 'no'}`"]
    if cfg.fill_below:
        fill_c = cfg.fill_color or "(inherit)"
        axis_info_parts.append(f"fill `{fill_c}`")
    lim_str = _lim_label(cfg)
    if lim_str != "auto":
        axis_info_parts.append(f"limits {lim_str}")
    embed.add_field(name="Axes", value=" · ".join(axis_info_parts), inline=False)

    embed.add_field(
        name="Figure",
        value=(f"{cfg.fig_width}×{cfg.fig_height} in | {cfg.dpi} dpi | "
               f"1-D res {cfg.resolution_1d} | 2-D res {cfg.resolution_2d} | alpha {cfg.alpha}"),
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

class AdditionalExprModal(ui.Modal, title="Additional Expressions"):
    exprs = ui.TextInput(label="Extra f(x) functions (one per line)",
                         style=discord.TextStyle.paragraph,
                         placeholder="cos(x)\nx**2 / 2\nexp(-x)",
                         required=False, max_length=1000)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg = cfg
        self._view = view
        self.exprs.default = "\n".join(cfg.additional_exprs)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        lines = [line.strip() for line in self.exprs.value.split("\n") if line.strip()]
        self._cfg.additional_exprs = lines
        await interaction.response.edit_message(embed=_config_embed(self._cfg), view=self._view)


class AnimationParamModal(ui.Modal, title="Animation Settings"):
    anim_param = ui.TextInput(label="Animation parameter (e.g. 'a')",
                              placeholder="a", required=True, max_length=10)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg = cfg
        self._view = view
        self.anim_param.default = cfg.anim_param or "a"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        self._cfg.anim_param = self.anim_param.value.strip()
        await self._view._render_animation(interaction)


class ExpressionModal(ui.Modal, title="Expressions & Domain"):
    """Collects function expressions and domain/range values."""

    expr_a   = ui.TextInput(label="Expression A  (see hint in embed)",
                             placeholder="e.g. sin(x)/x   or   1+cos(theta)",
                             required=False, max_length=300)
    expr_b   = ui.TextInput(label="Expr B (u/vector, y(t)/parametric, θ-sym/polar)",
                             placeholder="e.g. cos(t)  or  -y  or  theta",
                             required=False, max_length=300)
    expr_c   = ui.TextInput(label="Expr C (v for vector, z(t) for 3-D)",
                             placeholder="e.g. x  or  t/(2*pi)",
                             required=False, max_length=300)
    domain_x = ui.TextInput(label="x / t / θ range  (min, max)",
                             placeholder="-10, 10",
                             required=False, max_length=40)
    domain_y = ui.TextInput(label="y range  (ignored for 1-D / parametric / polar)",
                             placeholder="-10, 10",
                             required=False, max_length=40)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view

        pt = cfg.plot_type
        if pt == "function":
            self.expr_a.default = cfg.expr_main
        elif pt in ("contour", "surface", "wireframe", "heatmap", "implicit", "inequality"):
            self.expr_a.default = cfg.expr_main
            if pt == "implicit":
                self.expr_b.default = str(cfg.implicit_rhs)
            elif pt == "inequality":
                self.expr_b.default = cfg.inequality_op
                self.expr_c.default = str(cfg.implicit_rhs)
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
        elif pt in ("scatter", "scatter-3d", "errorbar"):
            self.expr_a.default = cfg.scatter_xs
            self.expr_b.default = cfg.scatter_ys
            if pt == "scatter-3d":
                self.expr_c.default = cfg.scatter_zs
            elif pt == "errorbar":
                self.expr_c.default = cfg.expr_z
        elif pt == "histogram":
            self.expr_a.default = cfg.scatter_xs
            self.expr_b.default = str(cfg.hist_bins)
        elif pt == "boxplot":
            self.expr_a.default = cfg.scatter_xs
            self.expr_b.default = cfg.box_violin

        elif pt == "riemann":
            self.expr_a.default = cfg.expr_main
            self.expr_b.default = str(cfg.riemann_n)
            self.expr_c.default = cfg.riemann_method

        elif pt == "polar":
            self.expr_a.default = cfg.expr_main
            self.expr_b.default = cfg.theta_symbol

        if pt in ("parametric-2d", "parametric-3d", "polar"):
            self.domain_x.default = f"{cfg.t_min}, {cfg.t_max}"
        else:
            self.domain_x.default = f"{cfg.x_min}, {cfg.x_max}"
        self.domain_y.default = f"{cfg.y_min}, {cfg.y_max}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        pt  = cfg.plot_type

        a = self.expr_a.value.strip()
        b = self.expr_b.value.strip()
        c = self.expr_c.value.strip()

        if pt == "function":
            if a: cfg.expr_main = a
        elif pt in ("contour", "surface", "wireframe", "heatmap", "implicit", "inequality"):
            if a: cfg.expr_main = a
            if pt == "implicit" and b:
                cfg.implicit_rhs = _parse_float(b, cfg.implicit_rhs)
            elif pt == "inequality":
                if b in ("<", "<=", ">", ">="):
                    cfg.inequality_op = b
                if c:
                    cfg.implicit_rhs = _parse_float(c, cfg.implicit_rhs)
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
        elif pt in ("scatter", "scatter-3d", "errorbar"):
            if a: cfg.scatter_xs = a
            if b: cfg.scatter_ys = b
            if c and pt == "scatter-3d": cfg.scatter_zs = c
            if c and pt == "errorbar": cfg.expr_z = c
        elif pt == "histogram":
            if a: cfg.scatter_xs = a
            if b: cfg.hist_bins = max(1, min(500, _parse_int(b, cfg.hist_bins)))
        elif pt == "boxplot":
            if a: cfg.scatter_xs = a
            if b and b.lower() in ("box", "violin"):
                cfg.box_violin = b.lower()
        elif pt == "riemann":
            if a: cfg.expr_main = a
            if b: cfg.riemann_n = max(1, min(500, _parse_int(b, cfg.riemann_n)))
            if c and c.lower().strip() in ("left", "right", "midpoint"):
                cfg.riemann_method = c.lower().strip()

        elif pt == "polar":
            if a: cfg.expr_main    = a
            if b: cfg.theta_symbol = b

        def _range(raw: str, lo: str, hi: str) -> None:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) == 2:
                setattr(cfg, lo, _parse_float(parts[0], getattr(cfg, lo)))
                setattr(cfg, hi, _parse_float(parts[1], getattr(cfg, hi)))

        dx = self.domain_x.value
        dy = self.domain_y.value
        if pt in ("parametric-2d", "parametric-3d", "polar"):
            if dx: _range(dx, "t_min", "t_max")
        else:
            if dx: _range(dx, "x_min", "x_max")
            if dy: _range(dy, "y_min", "y_max")

        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class AppearanceModal(ui.Modal, title="Appearance — Style, Colormap & Theme"):
    line_style = ui.TextInput(
        label="Colour, width, style, marker  (comma-separated)",
        placeholder="#1f77b4, 2.0, solid, none",
        required=False, max_length=80,
    )
    markersize = ui.TextInput(
        label="Marker size  (float, e.g. 6.0)",
        placeholder="6.0", required=False, max_length=10,
    )
    alpha_field = ui.TextInput(
        label="Opacity (alpha, 0.0-1.0)",
        placeholder="0.9", required=False, max_length=6,
    )
    colormap = ui.TextInput(
        label="Colormap  (e.g. viridis, plasma, coolwarm, jet)",
        placeholder="viridis", required=False, max_length=30,
    )
    theme = ui.TextInput(
        label="Theme  (default|dark|academic|cyberpunk|seaborn)",
        placeholder="default", required=False, max_length=20,
    )

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.line_style.default  = f"{cfg.line_color}, {cfg.line_width}, {cfg.line_style}, {cfg.marker}"
        self.markersize.default  = str(cfg.marker_size)
        self.alpha_field.default = str(cfg.alpha)
        self.colormap.default    = cfg.colormap
        self.theme.default       = cfg.theme

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg

        if self.line_style.value.strip():
            parts = [p.strip() for p in self.line_style.value.split(",")]
            if len(parts) >= 1 and parts[0]:
                cfg.line_color = parts[0]
            if len(parts) >= 2 and parts[1]:
                cfg.line_width = _parse_float(parts[1], cfg.line_width)
            if len(parts) >= 3 and parts[2] and parts[2] in LINE_STYLES:
                cfg.line_style = parts[2]
            if len(parts) >= 4 and parts[3] and parts[3] in MARKERS:
                cfg.marker = parts[3]

        if self.markersize.value:
            cfg.marker_size = _parse_float(self.markersize.value, cfg.marker_size)
        if self.alpha_field.value:
            cfg.alpha = max(0.0, min(1.0, _parse_float(self.alpha_field.value, cfg.alpha)))
        if self.colormap.value:
            cfg.colormap = self.colormap.value.strip()
        if self.theme.value and self.theme.value.strip() in THEMES:
            cfg.theme = self.theme.value.strip()

        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class AxesModal(ui.Modal, title="Axes, Labels & Scale"):
    plot_title = ui.TextInput(label="Plot title  (blank = auto)",
                               placeholder="My Beautiful Plot",
                               required=False, max_length=120)
    xl   = ui.TextInput(label="x-axis label", placeholder="x",
                        required=False, max_length=60)
    yl   = ui.TextInput(label="y-axis label", placeholder="y",
                        required=False, max_length=60)
    zl   = ui.TextInput(label="z-axis label  (3-D only)", placeholder="z",
                        required=False, max_length=60)
    log_grid = ui.TextInput(
        label="Log axes & grid  (log=x|y|both|none, grid=yes|no)",
        placeholder="log=none, grid=yes",
        required=False, max_length=40,
    )

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.plot_title.default = cfg.title
        self.xl.default         = cfg.xlabel
        self.yl.default         = cfg.ylabel
        self.zl.default         = cfg.zlabel
        log_val = "both" if cfg.x_log and cfg.y_log else \
                  "x"    if cfg.x_log else \
                  "y"    if cfg.y_log else "none"
        grid_val = "yes" if cfg.show_grid else "no"
        self.log_grid.default = f"log={log_val}, grid={grid_val}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        cfg.title = self.plot_title.value.strip()
        if self.xl.value:   cfg.xlabel = self.xl.value.strip()
        if self.yl.value:   cfg.ylabel = self.yl.value.strip()
        if self.zl.value:   cfg.zlabel = self.zl.value.strip()

        raw = self.log_grid.value.lower()
        for token in re.split(r'[,\s]+', raw):
            if "=" not in token:
                continue
            key, _, val = token.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "log":
                cfg.x_log = val in ("x", "both")
                cfg.y_log = val in ("y", "both")
            elif key == "grid":
                cfg.show_grid = val in ("yes", "true", "1", "on")

        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class AdvancedModal(ui.Modal, title="Advanced Options"):
    cmap       = ui.TextInput(label="Colormap",
                               placeholder="viridis", required=False, max_length=30)
    alpha_lvl  = ui.TextInput(
        label="Opacity & contour levels  (alpha, levels)",
        placeholder="0.9, 20", required=False, max_length=15,
    )
    res_1d     = ui.TextInput(
        label="1-D resolution  (line / parametric, 100–2000)",
        placeholder="800", required=False, max_length=6,
    )
    res_2d     = ui.TextInput(
        label="2-D resolution  (contour / surface, 40–400)",
        placeholder="120", required=False, max_length=6,
    )
    figsize    = ui.TextInput(label="Figure size  (width, height) in inches",
                               placeholder="8, 5", required=False, max_length=15)

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.cmap.default      = cfg.colormap
        self.alpha_lvl.default = f"{cfg.alpha}, {cfg.levels}"
        self.res_1d.default    = str(cfg.resolution_1d)
        self.res_2d.default    = str(cfg.resolution_2d)
        self.figsize.default   = f"{cfg.fig_width}, {cfg.fig_height}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg

        if self.cmap.value:
            cfg.colormap = self.cmap.value.strip()

        if self.alpha_lvl.value:
            parts = [p.strip() for p in self.alpha_lvl.value.split(",")]
            if parts[0]:
                cfg.alpha  = max(0.0, min(1.0, _parse_float(parts[0], cfg.alpha)))
            if len(parts) > 1 and parts[1]:
                cfg.levels = max(2, min(100, _parse_int(parts[1], cfg.levels)))

        if self.res_1d.value:
            cfg.resolution_1d = max(100, min(2000, _parse_int(self.res_1d.value, cfg.resolution_1d)))
        if self.res_2d.value:
            cfg.resolution_2d = max(40, min(400, _parse_int(self.res_2d.value, cfg.resolution_2d)))

        if self.figsize.value:
            parts = self.figsize.value.split(",")
            if len(parts) == 2:
                cfg.fig_width  = max(2.0, min(24.0, _parse_float(parts[0], cfg.fig_width)))
                cfg.fig_height = max(2.0, min(24.0, _parse_float(parts[1], cfg.fig_height)))

        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class FillModal(ui.Modal, title="Fill / Area Shading"):
    enabled    = ui.TextInput(
        label="Enable fill shading?  (yes / no)",
        placeholder="no", required=False, max_length=5,
    )
    fill_color = ui.TextInput(
        label="Fill colour  (hex / name, blank = inherit line colour)",
        placeholder="#1f77b4  or  skyblue  or  leave blank",
        required=False, max_length=40,
    )

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view
        self.enabled.default    = "yes" if cfg.fill_below else "no"
        self.fill_color.default = cfg.fill_color

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        if self.enabled.value:
            cfg.fill_below = _parse_bool(self.enabled.value, cfg.fill_below)
        cfg.fill_color = self.fill_color.value.strip()
        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


class LimitsModal(ui.Modal, title="Axis Limit Overrides"):
    x_min_field = ui.TextInput(
        label="x-axis min  (blank = auto)",
        placeholder="-10  or  auto", required=False, max_length=20,
    )
    x_max_field = ui.TextInput(
        label="x-axis max  (blank = auto)",
        placeholder="10  or  auto", required=False, max_length=20,
    )
    y_min_field = ui.TextInput(
        label="y-axis min  (blank = auto)",
        placeholder="-5  or  auto", required=False, max_length=20,
    )
    y_max_field = ui.TextInput(
        label="y-axis max  (blank = auto)",
        placeholder="5  or  auto", required=False, max_length=20,
    )
    _note = ui.TextInput(
        label="Note  (read-only — just acknowledge & submit)",
        default="Blank / 'auto' = matplotlib chooses that boundary.",
        required=False, max_length=60,
    )

    def __init__(self, cfg: PlotConfig, view: "PlotEngineView") -> None:
        super().__init__()
        self._cfg  = cfg
        self._view = view

        def _fmt(v: Optional[float]) -> str:
            return str(v) if v is not None else ""

        self.x_min_field.default = _fmt(cfg.x_lim_min)
        self.x_max_field.default = _fmt(cfg.x_lim_max)
        self.y_min_field.default = _fmt(cfg.y_lim_min)
        self.y_max_field.default = _fmt(cfg.y_lim_max)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cfg = self._cfg
        cfg.x_lim_min = _parse_optional_float(self.x_min_field.value)
        cfg.x_lim_max = _parse_optional_float(self.x_max_field.value)
        cfg.y_lim_min = _parse_optional_float(self.y_min_field.value)
        cfg.y_lim_max = _parse_optional_float(self.y_max_field.value)
        cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(cfg), view=self._view)


# ─────────────────────────────────────────────────────────────────────────────
# Render  —  PlotConfig → discord.File
# ─────────────────────────────────────────────────────────────────────────────

async def _render(cfg: PlotConfig) -> discord.File:
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    t = sympy.Symbol("t")

    style = cfg.to_style()
    pt    = cfg.plot_type

    if pt == "function":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x)
        additional = []
        for e in cfg.additional_exprs:
            try:
                additional.append(_sympy_expr(_clean_sympy_expr(e), x))
            except Exception:
                pass
        return await plot_function(
            expr, x,
            x_min=cfg.x_min, x_max=cfg.x_max,
            title=cfg.title or str(expr),
            style=style,
            additional_exprs=additional,
            resolution_1d=cfg.resolution_1d,
        )

    elif pt == "contour":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y)
        return await plot_contour(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            levels=cfg.levels,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "implicit":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y)
        return await plot_implicit(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            rhs=cfg.implicit_rhs,
            title=cfg.title,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "inequality":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y)
        return await plot_inequality(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            op=cfg.inequality_op,
            rhs=cfg.implicit_rhs,
            title=cfg.title,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "heatmap":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y)
        return await plot_heatmap(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "vector-field":
        u = _sympy_expr(_clean_sympy_expr(cfg.expr_u), x, y)
        v = _sympy_expr(_clean_sympy_expr(cfg.expr_v), x, y)
        return await plot_vector_field(
            u, v, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            stream=cfg.stream,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "parametric-2d":
        xe = _sympy_expr(_clean_sympy_expr(cfg.expr_x), t)
        ye = _sympy_expr(_clean_sympy_expr(cfg.expr_y), t)
        return await plot_parametric_2d(
            xe, ye, t,
            t_min=cfg.t_min, t_max=cfg.t_max,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            style=style,
            resolution_1d=cfg.resolution_1d,
        )

    elif pt == "surface":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y)
        return await plot_surface(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "wireframe":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x, y)
        return await plot_wireframe(
            expr, x, y,
            x_range=(cfg.x_min, cfg.x_max),
            y_range=(cfg.y_min, cfg.y_max),
            title=cfg.title,
            style=style,
            resolution_2d=cfg.resolution_2d,
        )

    elif pt == "parametric-3d":
        xe = _sympy_expr(_clean_sympy_expr(cfg.expr_x), t)
        ye = _sympy_expr(_clean_sympy_expr(cfg.expr_y), t)
        ze = _sympy_expr(_clean_sympy_expr(cfg.expr_z), t)
        return await plot_parametric_3d(
            xe, ye, ze, t,
            t_min=cfg.t_min, t_max=cfg.t_max,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            zlabel=cfg.zlabel,
            style=style,
            resolution_1d=cfg.resolution_1d,
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

    elif pt == "histogram":
        values = _parse_floatlist(cfg.scatter_xs)
        return await plot_histogram(
            values,
            bins=cfg.hist_bins,
            title=cfg.title,
            xlabel=cfg.xlabel or "value",
            ylabel=cfg.ylabel or "count",
            style=style,
        )

    elif pt == "errorbar":
        xs_list = _parse_floatlist(cfg.scatter_xs)
        ys_list = _parse_floatlist(cfg.scatter_ys)
        err_list = _parse_floatlist(cfg.expr_z)
        return await plot_errorbar(
            xs_list, ys_list, err_list,
            title=cfg.title,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel,
            style=style,
        )

    elif pt == "boxplot":
        groups = [_parse_floatlist(group) for group in cfg.scatter_xs.split("|")]
        return await plot_boxplot(
            groups,
            mode=cfg.box_violin,
            title=cfg.title,
            ylabel=cfg.ylabel or "value",
            style=style,
        )

    elif pt == "riemann":
        expr = _sympy_expr(_clean_sympy_expr(cfg.expr_main), x)
        return await plot_riemann(
            expr, x,
            a=cfg.x_min,
            b=cfg.x_max,
            n=cfg.riemann_n,
            method=cfg.riemann_method,
            title=cfg.title,
            style=style,
            resolution_1d=cfg.resolution_1d,
        )

    elif pt == "polar":
        theta_sym  = sympy.Symbol(cfg.theta_symbol or "theta")
        expr       = _sympy_expr(_clean_sympy_expr(cfg.expr_main), theta_sym)

        extra_exprs = []
        for raw in cfg.additional_exprs:
            try:
                extra_exprs.append(_sympy_expr(_clean_sympy_expr(raw), theta_sym))
            except Exception:
                pass

        return await plot_polar(
            expr, theta_sym,
            theta_min=cfg.t_min,
            theta_max=cfg.t_max,
            title=cfg.title,
            style=style,
            additional_exprs=extra_exprs or None,
            resolution_1d=cfg.resolution_1d,
        )

    else:
        raise ValueError(f"Unknown plot type: `{pt}`")


# ─────────────────────────────────────────────────────────────────────────────
# PlotEngineView — the persistent control panel
# ─────────────────────────────────────────────────────────────────────────────

class PlotEngineView(ui.View):
    def __init__(self, cfg: PlotConfig) -> None:
        super().__init__(timeout=600)
        self.cfg = cfg
        self._message: Optional[discord.Message] = None
        self._add_type_select()

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

    def _add_buttons(self) -> None:
        def _btn(label, style, row, cb, *, emoji=None):
            b = ui.Button(label=label, style=style, row=row, emoji=emoji)
            b.callback = cb
            self.add_item(b)

        _btn("Expressions",   discord.ButtonStyle.primary,   1, self._on_expr)
        _btn("Appearance",    discord.ButtonStyle.primary,   1, self._on_appearance)
        _btn("Axes & Labels", discord.ButtonStyle.primary,   1, self._on_axes)
        _btn("Advanced",      discord.ButtonStyle.secondary, 1, self._on_advanced)
        _btn("Fill",          discord.ButtonStyle.secondary, 1, self._on_fill)

        _btn("🔍+", discord.ButtonStyle.secondary, 2, self._on_zoom_in)
        _btn("🔍-", discord.ButtonStyle.secondary, 2, self._on_zoom_out)
        _btn("⬅️",  discord.ButtonStyle.secondary, 2, self._on_pan_left)
        _btn("➡️",  discord.ButtonStyle.secondary, 2, self._on_pan_right)
        _btn("⬆️",  discord.ButtonStyle.secondary, 2, self._on_pan_up)

        _btn("⬇️",           discord.ButtonStyle.secondary, 3, self._on_pan_down)
        _btn("Limits",       discord.ButtonStyle.secondary, 3, self._on_limits)
        _btn("Syntax Help",  discord.ButtonStyle.secondary, 3, self._on_syntax_help)

        _btn("Reset View",  discord.ButtonStyle.danger,   4, self._on_reset_view)
        _btn("Export",      discord.ButtonStyle.secondary, 4, self._on_export)
        _btn("Render plot", discord.ButtonStyle.success,  4, self._on_render)

        if self.cfg.plot_type == "vector-field":
            # Row 4: Reset View · Export · Render · Stream toggle · Reset
            # Animate is omitted here to keep the row within Discord's 5-item limit;
            # vector-field animation is still reachable via /plot with another type.
            stream_label = "Stream: ON" if self.cfg.stream else "Stream: OFF"
            _btn(stream_label, discord.ButtonStyle.primary, 4, self._on_stream)
        elif self.cfg.plot_type not in ("scatter", "scatter-3d"):
            _btn("Animate",  discord.ButtonStyle.success, 4, self._on_animate)
        _btn("Reset",        discord.ButtonStyle.danger,  4, self._on_reset)

    def _scale_domain(self, factor: float) -> None:
        dx = (self.cfg.x_max - self.cfg.x_min) * factor
        cx = (self.cfg.x_max + self.cfg.x_min) / 2
        self.cfg.x_min = cx - dx / 2
        self.cfg.x_max = cx + dx / 2

        dy = (self.cfg.y_max - self.cfg.y_min) * factor
        cy = (self.cfg.y_max + self.cfg.y_min) / 2
        self.cfg.y_min = cy - dy / 2
        self.cfg.y_max = cy + dy / 2

        if self.cfg.plot_type in ("parametric-2d", "parametric-3d"):
            dt = (self.cfg.t_max - self.cfg.t_min) * factor
            ct = (self.cfg.t_max + self.cfg.t_min) / 2
            self.cfg.t_min = max(0.0, ct - dt / 2)
            self.cfg.t_max = ct + dt / 2
        elif self.cfg.plot_type == "polar":
            dt = (self.cfg.t_max - self.cfg.t_min) * factor
            ct = (self.cfg.t_max + self.cfg.t_min) / 2
            self.cfg.t_min = ct - dt / 2
            self.cfg.t_max = ct + dt / 2

    def _shift_domain(self, x_frac: float, y_frac: float) -> None:
        """Shift by a fraction of the *current* domain width/height so pan
        step stays proportional after any number of zooms."""
        dx = (self.cfg.x_max - self.cfg.x_min) * x_frac
        self.cfg.x_min += dx
        self.cfg.x_max += dx

        dy = (self.cfg.y_max - self.cfg.y_min) * y_frac
        self.cfg.y_min += dy
        self.cfg.y_max += dy

    async def _on_zoom_in(self, interaction: discord.Interaction) -> None:
        self._scale_domain(0.8)
        await self._on_preview(interaction)

    async def _on_zoom_out(self, interaction: discord.Interaction) -> None:
        self._scale_domain(1.25)
        await self._on_preview(interaction)

    async def _on_pan_left(self, interaction: discord.Interaction) -> None:
        self._shift_domain(-0.25, 0)
        await self._on_preview(interaction)

    async def _on_pan_right(self, interaction: discord.Interaction) -> None:
        self._shift_domain(0.25, 0)
        await self._on_preview(interaction)

    async def _on_pan_up(self, interaction: discord.Interaction) -> None:
        self._shift_domain(0, 0.25)
        await self._on_preview(interaction)

    async def _on_pan_down(self, interaction: discord.Interaction) -> None:
        self._shift_domain(0, -0.25)
        await self._on_preview(interaction)

    async def _on_reset_view(self, interaction: discord.Interaction) -> None:
        defaults = PlotConfig()
        self.cfg.x_min = defaults.x_min
        self.cfg.x_max = defaults.x_max
        self.cfg.y_min = defaults.y_min
        self.cfg.y_max = defaults.y_max
        self.cfg.t_min = defaults.t_min
        self.cfg.t_max = defaults.t_max
        self.cfg.x_lim_min = None
        self.cfg.x_lim_max = None
        self.cfg.y_lim_min = None
        self.cfg.y_lim_max = None
        self.cfg.last_error = ""
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def _on_expr(self,     interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(ExpressionModal(self.cfg, self))

    async def _on_appearance(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AppearanceModal(self.cfg, self))

    async def _on_axes(self,     interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AxesModal(self.cfg, self))

    async def _on_advanced(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AdvancedModal(self.cfg, self))

    async def _on_fill(self,     interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(FillModal(self.cfg, self))

    async def _on_limits(self,   interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(LimitsModal(self.cfg, self))

    async def _on_stream(self, interaction: discord.Interaction) -> None:
        self.cfg.stream = not self.cfg.stream
        self.clear_items()
        self._add_type_select()
        self._add_buttons()
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def _on_syntax_help(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(title="SymPy Syntax Guide", color=discord.Color.blurple())
        embed.description = (
            "**Basic Math**\n"
            "`+` `-` `*` `/`\n"
            "`**` for exponents (e.g., `x**2`). We also auto-correct `^` to `**`.\n\n"
            "**Functions**\n"
            "`sin(x)`, `cos(x)`, `tan(x)`\n"
            "`exp(x)` (or `e**x`)\n"
            "`sqrt(x)`\n"
            "`log(x)` (natural log), `log(x, 10)` (base 10)\n\n"
            "**Constants**\n"
            "`pi`, `E`\n\n"
            "**Polar curves (type = polar)**\n"
            "Expression A = r(θ), e.g. `1 + cos(theta)` (limaçon), "
            "`cos(3*theta)` (rose), `theta` (Archimedean spiral).\n"
            "Expression B = θ variable name (default `theta`; use `t` if preferred).\n"
            "θ range (domain field) = min, max in radians, e.g. `0, 6.2832` (0→2π).\n"
            "Use **+ f(x)** → Additional Expressions to overlay extra r(θ) curves.\n\n"
            "**Log-scale axes (Axes & Labels modal)**\n"
            "Enter `log=x`, `log=y`, `log=both`, or `log=none` in the\n"
            "\"Log axes & grid\" field.  Combine with `grid=yes/no`.\n"
            "Example: `log=x, grid=yes`\n\n"
            "**Axis limits (Limits button)**\n"
            "Set exact display bounds.  Leave blank for auto / smart range.\n\n"
            "**Fill shading (Fill button)**\n"
            "Shades the area between the curve and y=0.  Function plots only.\n\n"
            "**Resolution (Advanced modal)**\n"
            "1-D res controls line density; 2-D res controls grid fineness.\n\n"
            "**Examples**\n"
            "`sin(x)*exp(-x)`\n"
            "`sqrt(x**2 + y**2)`\n"
            "`1 + cos(theta)` (polar limaçon)\n"
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _on_export(self, interaction: discord.Interaction) -> None:
        cfg = self.cfg
        pt  = cfg.plot_type

        embed = discord.Embed(
            title="📋 Plot Settings Export",
            description=(
                f"All current settings for your **{pt}** plot.\n"
                "Use the import string at the bottom to restore this exact configuration."
            ),
            color=EMBED_COLOR,
        )

        embed.add_field(name="Plot type", value=f"`{pt}`",               inline=True)
        embed.add_field(name="Title",     value=f"`{cfg.title or '(auto)'}`", inline=True)
        embed.add_field(name="Theme",     value=f"`{cfg.theme}`",         inline=True)

        if pt == "function":
            exprs = [cfg.expr_main] + list(cfg.additional_exprs)
            embed.add_field(
                name="Expressions",
                value="\n".join(f"`{e}`" for e in exprs),
                inline=False,
            )
        elif pt in ("contour", "surface", "wireframe", "heatmap"):
            embed.add_field(name="f(x, y)", value=f"`{cfg.expr_main}`", inline=False)
        elif pt == "implicit":
            embed.add_field(name="Equation", value=f"`{cfg.expr_main} = {cfg.implicit_rhs}`", inline=False)
        elif pt == "inequality":
            embed.add_field(name="Inequality", value=f"`{cfg.expr_main} {cfg.inequality_op} {cfg.implicit_rhs}`", inline=False)
        elif pt == "vector-field":
            embed.add_field(name="u(x, y)", value=f"`{cfg.expr_u}`", inline=True)
            embed.add_field(name="v(x, y)", value=f"`{cfg.expr_v}`", inline=True)
            embed.add_field(name="Streamplot", value="yes" if cfg.stream else "no", inline=True)
        elif pt == "parametric-2d":
            embed.add_field(name="x(t)", value=f"`{cfg.expr_x}`", inline=True)
            embed.add_field(name="y(t)", value=f"`{cfg.expr_y}`", inline=True)
        elif pt == "parametric-3d":
            embed.add_field(name="x(t)", value=f"`{cfg.expr_x}`", inline=True)
            embed.add_field(name="y(t)", value=f"`{cfg.expr_y}`", inline=True)
            embed.add_field(name="z(t)", value=f"`{cfg.expr_z}`", inline=True)
        elif pt in ("scatter", "scatter-3d", "errorbar"):
            embed.add_field(name="xs", value=f"`{cfg.scatter_xs[:80]}`", inline=False)
            embed.add_field(name="ys", value=f"`{cfg.scatter_ys[:80]}`", inline=False)
            if pt == "scatter-3d":
                embed.add_field(name="zs", value=f"`{cfg.scatter_zs[:80]}`", inline=False)
            elif pt == "errorbar":
                embed.add_field(name="y errors", value=f"`{cfg.expr_z[:80]}`", inline=False)
        elif pt == "histogram":
            embed.add_field(name="data", value=f"`{cfg.scatter_xs[:80]}`", inline=False)
            embed.add_field(name="bins", value=f"`{cfg.hist_bins}`", inline=True)
        elif pt == "boxplot":
            embed.add_field(name="groups", value=f"`{cfg.scatter_xs[:80]}`", inline=False)
            embed.add_field(name="mode", value=f"`{cfg.box_violin}`", inline=True)
        elif pt == "polar":
            all_polar = [cfg.expr_main] + list(cfg.additional_exprs)
            embed.add_field(
                name="r(θ) expressions",
                value="\n".join(f"`{e}`" for e in all_polar),
                inline=False,
            )
            embed.add_field(name="θ symbol", value=f"`{cfg.theta_symbol}`", inline=True)

        if pt in ("parametric-2d", "parametric-3d", "polar"):
            embed.add_field(
                name="Domain",
                value=f"θ/t ∈ [{cfg.t_min}, {cfg.t_max}]" if pt == "polar"
                      else f"t ∈ [{cfg.t_min}, {cfg.t_max}]",
                inline=False,
            )
        elif pt not in ("scatter", "scatter-3d", "errorbar", "histogram", "boxplot", "polar"):
            domain_lines = [f"x ∈ [{cfg.x_min}, {cfg.x_max}]"]
            if pt not in ("function",):
                domain_lines.append(f"y ∈ [{cfg.y_min}, {cfg.y_max}]")
            embed.add_field(name="Domain", value="\n".join(domain_lines), inline=False)

        embed.add_field(
            name="Line style",
            value=f"color `{cfg.line_color}`\nstyle `{cfg.line_style}`\nwidth `{cfg.line_width}`",
            inline=True,
        )
        embed.add_field(
            name="Marker",
            value=f"shape `{cfg.marker}`\nsize  `{cfg.marker_size}`",
            inline=True,
        )
        embed.add_field(
            name="Color & opacity",
            value=(
                f"colormap `{cfg.colormap}`\n"
                f"alpha    `{cfg.alpha}`\n"
                f"levels   `{cfg.levels}` *(contour only)*"
            ),
            inline=True,
        )

        embed.add_field(
            name="Axes (Phase 1)",
            value=(
                f"log scale `{_log_label(cfg)}`\n"
                f"grid `{'yes' if cfg.show_grid else 'no'}`\n"
                f"fill `{'on' if cfg.fill_below else 'off'}` color `{cfg.fill_color or '(inherit)'}`\n"
                f"limits {_lim_label(cfg)}"
            ),
            inline=False,
        )

        embed.add_field(
            name="Figure",
            value=(
                f"size `{cfg.fig_width} × {cfg.fig_height}` in\n"
                f"dpi  `{cfg.dpi}`\n"
                f"1-D res `{cfg.resolution_1d}` · 2-D res `{cfg.resolution_2d}`"
            ),
            inline=True,
        )

        if cfg.anim_param:
            embed.add_field(name="Animation param", value=f"`{cfg.anim_param}`", inline=True)

        import_string = cfg.export_config()
        embed.add_field(
            name="Import string  (copy → `/plot_import`)",
            value=f"`{import_string}`",
            inline=False,
        )

        embed.set_footer(text="Settings are ephemeral — only you can see this message.")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _on_theme(self, interaction: discord.Interaction) -> None:
        # Legacy — kept for any external callers; routes to AppearanceModal.
        await interaction.response.send_modal(AppearanceModal(self.cfg, self))

    async def _on_cmap(self, interaction: discord.Interaction) -> None:
        # Legacy — kept for any external callers; routes to AppearanceModal.
        await interaction.response.send_modal(AppearanceModal(self.cfg, self))

    async def _on_preview(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def _on_animate(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AnimationParamModal(self.cfg, self))

    async def _render_animation(self, interaction: discord.Interaction) -> None:
        # The interaction here came from AnimationParamModal.on_submit, which
        # already consumed the response slot with the modal submit.  We must
        # acknowledge via edit_message on the original builder panel, then
        # use followup for the rendered GIF.
        try:
            await interaction.response.edit_message(
                embed=_config_embed(self.cfg), view=self
            )
        except discord.InteractionResponded:
            pass

        try:
            from utils.plotter import plot_animation
            file = await plot_animation(self.cfg)
            self.cfg.last_error = ""
        except Exception as exc:
            self.cfg.last_error = str(exc)
            await interaction.edit_original_response(embed=_config_embed(self.cfg), view=self)
            await interaction.followup.send(f"⚠ Animation failed: {exc}", ephemeral=True)
            return

        await interaction.edit_original_response(embed=_config_embed(self.cfg), view=self)

        embed_out = discord.Embed(
            title=self.cfg.title or f"{self.cfg.plot_type} animation",
            color=EMBED_COLOR,
        )
        embed_out.set_image(url=f"attachment://{file.filename}")
        embed_out.set_footer(
            text=(f"type={self.cfg.plot_type} | theme={self.cfg.theme} | "
                  f"anim_param={self.cfg.anim_param}")
        )
        await interaction.channel.send(embed=embed_out, file=file)

    async def _on_render(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            file = await _render(self.cfg)
            self.cfg.last_error = ""
        except Exception as exc:
            self.cfg.last_error = str(exc)
            await interaction.edit_original_response(embed=_config_embed(self.cfg), view=self)
            await interaction.followup.send(
                f"⚠ Render failed: {exc}", ephemeral=True,
            )
            return

        await interaction.edit_original_response(
            embed=_config_embed(self.cfg),
            view=self,
        )

        embed_out = discord.Embed(
            title=self.cfg.title or f"{self.cfg.plot_type} plot",
            color=EMBED_COLOR,
        )
        embed_out.set_image(url=f"attachment://{file.filename}")
        embed_out.set_footer(
            text=(f"type={self.cfg.plot_type} | cmap={self.cfg.colormap} | "
                  f"alpha={self.cfg.alpha} | 1-D res={self.cfg.resolution_1d} | "
                  f"2-D res={self.cfg.resolution_2d}")
        )
        await interaction.channel.send(embed=embed_out, file=file)

    async def _on_reset(self, interaction: discord.Interaction) -> None:
        """Reset expressions & style but keep the current plot type and domain."""
        defaults = PlotConfig()
        keep = {"plot_type", "x_min", "x_max", "y_min", "y_max",
                "t_min", "t_max", "x_lim_min", "x_lim_max",
                "y_lim_min", "y_lim_max"}
        for k, v in defaults.__dict__.items():
            if k not in keep:
                setattr(self.cfg, k, v)
        self.cfg.last_error = ""
        self.clear_items()
        self._add_type_select()
        self._add_buttons()
        await interaction.response.edit_message(embed=_config_embed(self.cfg), view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        try:
            expired_embed = _config_embed(self.cfg)
            expired_embed.color = discord.Color.greyple()
            expired_embed.set_footer(
                text="Session expired — use /plot to start a new one."
            )
            await self._message.edit(embed=expired_embed, view=self)
        except Exception:
            pass  # message may already be gone


# ─────────────────────────────────────────────────────────────────────────────
# Theme picker — secondary ephemeral view
# ─────────────────────────────────────────────────────────────────────────────

THEMES = ["default", "dark", "academic", "cyberpunk", "seaborn"]

class ThemePickerView(ui.View):
    def __init__(self, cfg: PlotConfig, parent: PlotEngineView) -> None:
        super().__init__(timeout=180)
        self._cfg = cfg
        self._parent = parent
        self._build()

    def _build(self) -> None:
        options = [
            discord.SelectOption(label=t, value=t, default=(t == self._cfg.theme))
            for t in THEMES
        ]
        sel = ui.Select(placeholder="Pick a theme…", options=options)
        sel.callback = self._on_pick
        self.add_item(sel)

    async def _on_pick(self, interaction: discord.Interaction) -> None:
        self._cfg.theme = interaction.data["values"][0]
        self._parent.clear_items()
        self._parent._add_type_select()
        self._parent._add_buttons()
        await interaction.response.edit_message(embed=_config_embed(self._cfg), view=self._parent)


# ─────────────────────────────────────────────────────────────────────────────
# Colormap picker — secondary ephemeral view
# ─────────────────────────────────────────────────────────────────────────────

class ColormapPickerView(ui.View):
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
        await interaction.response.edit_message(embed=_config_embed(self._cfg), view=self._parent)

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
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(
        name="plot_import",
        description="Import a shared PlotEngine configuration.",
    )
    async def plot_import(self, interaction: discord.Interaction, config_string: str) -> None:
        try:
            cfg = PlotConfig.import_config(config_string.strip())
        except Exception as exc:
            await interaction.response.send_message(f"⚠ Invalid config string: {exc}", ephemeral=True)
            return

        view = PlotEngineView(cfg)
        view._add_buttons()

        await interaction.response.send_message(
            embed=_config_embed(cfg),
            view=view,
            ephemeral=True,
        )
        view._message = await interaction.original_response()

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
        view._message = await interaction.original_response()

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
    @app_commands.choices(line_style=[
        app_commands.Choice(name=s, value=s) for s in LINE_STYLES
    ])
    async def quickplot(
        self,
        interaction: discord.Interaction,
        expression: str,
        x_min: float = -10.0,
        x_max: float  = 10.0,
        title: str    = "",
        color: str    = "#1f77b4",
        line_style: Optional[app_commands.Choice[str]] = None,
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


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PlotEngine(bot))