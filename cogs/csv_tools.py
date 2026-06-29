"""
cogs/csv_tools.py — /csv command group for MathFrame.

Commands
--------
/csv upload   file                           Parse & store a CSV attachment.
/csv info                                    Re-display the column summary.
/csv preview  [rows=5]                       Show the first N rows as a table.
/csv stat     operation  column              Run a stat on one column.
/csv stat2    operation  col_x  col_y        Run a two-column stat.
/csv plot                                    Open the interactive plot builder.
/csv clear                                   Drop the active session.

Embed colour: discord.Colour.orange() — visually distinct from the green
math-result embeds produced by the other cogs.
"""

from __future__ import annotations

import asyncio
import csv as _csv
import io
import statistics as _stats
from dataclasses import dataclass, field

import aiohttp
import discord
import matplotlib
import numpy as np
from discord import app_commands, ui
from discord.ext import commands
from scipy import stats as scipy_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.csv_session import (
    CSVSession,
    _MAX_FILE_BYTES,
    _MAX_ROWS,
    _MAX_COLUMNS,
    clear_session,
    get_column_names,
    get_numeric_column,
    get_session,
    store_session,
)
from utils.formatter import error_embed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLOUR = discord.Colour.orange()

_STAT_CHOICES = [
    app_commands.Choice(name="mean",     value="mean"),
    app_commands.Choice(name="median",   value="median"),
    app_commands.Choice(name="mode",     value="mode"),
    app_commands.Choice(name="stdev",    value="stdev"),
    app_commands.Choice(name="variance", value="variance"),
    app_commands.Choice(name="summary",  value="summary"),
]

_STAT2_CHOICES = [
    app_commands.Choice(name="correlation", value="correlation"),
    app_commands.Choice(name="regression",  value="regression"),
]

_PLOT_TYPES = ["scatter", "line", "histogram", "bar", "box", "heatmap"]

_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

_STYLES = ["solid", "dashed", "dotted", "dashdot"]

_AGG_FUNCS = ["mean", "sum", "count"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_embed(title: str, description: str = "", footer: str = "") -> discord.Embed:
    """Orange-tinted CSV embed."""
    em = discord.Embed(title=title, description=description, colour=_COLOUR)
    if footer:
        em.set_footer(text=footer)
    return em


def _session_footer(session: CSVSession) -> str:
    mins = session.minutes_remaining()
    if mins <= 5:
        return f"⚠ Session expires in ~{mins} min — re-upload to refresh."
    return f"Session active · expires in ~{mins} min"


def _infer_type(rows: list[dict[str, str]], col: str) -> str:
    """Infer 'numeric', 'mixed', or 'text' for a column."""
    numeric = text = 0
    for row in rows[:500]:          # sample up to 500 rows for speed
        val = row.get(col, "").strip()
        if not val:
            continue
        try:
            float(val)
            numeric += 1
        except ValueError:
            text += 1
    total = numeric + text
    if total == 0:
        return "text"
    if text == 0:
        return "numeric"
    if numeric == 0:
        return "text"
    return "mixed"


def _sample_values(rows: list[dict[str, str]], col: str, n: int = 2) -> str:
    seen: list[str] = []
    for row in rows:
        v = row.get(col, "").strip()
        if v and v not in seen:
            seen.append(v)
        if len(seen) >= n:
            break
    return ", ".join(seen) if seen else "—"


def _build_summary_embed(session: CSVSession) -> discord.Embed:
    """Build the standard column-summary embed used by /csv upload and /csv info."""
    lines: list[str] = []
    lines.append(f"```")
    header = f"{'Column':<20} {'Type':<10} {'Sample'}"
    lines.append(header)
    lines.append("─" * min(len(header) + 20, 60))

    for col in session.columns:
        col_type = _infer_type(session.rows, col)
        sample   = _sample_values(session.rows, col)
        # truncate long column names
        display_col = col[:18] + ".." if len(col) > 20 else col
        display_smp = sample[:30] + "…" if len(sample) > 30 else sample
        lines.append(f"{display_col:<20} {col_type:<10} {display_smp}")

    lines.append("```")

    em = _csv_embed(
        title=f"📂 {session.filename}",
        description=(
            f"**{session.row_count:,} rows · {len(session.columns)} columns**\n"
            + "\n".join(lines)
        ),
        footer=_session_footer(session),
    )
    return em


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------

async def _column_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    names = get_column_names(interaction.user.id)
    if not names:
        return [app_commands.Choice(name="⚠ No CSV loaded — use /csv upload first", value="__none__")]
    return [
        app_commands.Choice(name=n, value=n)
        for n in names
        if current.lower() in n.lower()
    ][:25]


# ---------------------------------------------------------------------------
# Stat computation helpers
# ---------------------------------------------------------------------------

def _compute_stat(operation: str, arr: np.ndarray, col: str) -> discord.Embed:
    """Run a single-column stat and return a formatted embed."""
    em = _csv_embed(title=f"📊 {operation.title()} — `{col}`")

    if operation == "mean":
        val = float(np.mean(arr))
        em.add_field(name="Mean", value=f"```{val:.6g}```", inline=False)

    elif operation == "median":
        val = float(np.median(arr))
        em.add_field(name="Median", value=f"```{val:.6g}```", inline=False)

    elif operation == "mode":
        from scipy.stats import mode as scipy_mode
        result = scipy_mode(arr, keepdims=True)
        modes = result.mode
        counts = result.count
        vals_str = ", ".join(f"{m:.6g}" for m in modes)
        em.add_field(name="Mode(s)", value=f"```{vals_str}```", inline=False)
        em.add_field(name="Count",   value=f"```{int(counts[0])}```", inline=True)

    elif operation == "stdev":
        val = float(np.std(arr, ddof=1))
        em.add_field(name="Sample Std Dev", value=f"```{val:.6g}```", inline=False)

    elif operation == "variance":
        val = float(np.var(arr, ddof=1))
        em.add_field(name="Sample Variance", value=f"```{val:.6g}```", inline=False)

    elif operation == "summary":
        n      = len(arr)
        mean_  = float(np.mean(arr))
        med    = float(np.median(arr))
        std    = float(np.std(arr, ddof=1))
        var    = float(np.var(arr, ddof=1))
        mn     = float(np.min(arr))
        mx     = float(np.max(arr))
        q1     = float(np.percentile(arr, 25))
        q3     = float(np.percentile(arr, 75))

        em.title = f"📊 Summary — `{col}`"
        em.add_field(name="n",        value=f"`{n}`",           inline=True)
        em.add_field(name="Mean",     value=f"`{mean_:.6g}`",   inline=True)
        em.add_field(name="Median",   value=f"`{med:.6g}`",     inline=True)
        em.add_field(name="Std Dev",  value=f"`{std:.6g}`",     inline=True)
        em.add_field(name="Variance", value=f"`{var:.6g}`",     inline=True)
        em.add_field(name="\u200b",   value="\u200b",           inline=True)
        em.add_field(name="Min",      value=f"`{mn:.6g}`",      inline=True)
        em.add_field(name="Q1",       value=f"`{q1:.6g}`",      inline=True)
        em.add_field(name="Q3",       value=f"`{q3:.6g}`",      inline=True)
        em.add_field(name="Max",      value=f"`{mx:.6g}`",      inline=True)

    em.add_field(name="Column", value=f"`{col}`", inline=True)
    em.add_field(name="n",      value=f"`{len(arr)}`", inline=True)
    return em


def _correlation_label(r: float) -> str:
    direction = "positive" if r >= 0 else "negative"
    abs_r = abs(r)
    if abs_r > 0.8:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"
    return f"{strength} {direction}"


def _regression_plot_bytes(
    xs: np.ndarray, ys: np.ndarray,
    slope: float, intercept: float, r_sq: float,
    col_x: str, col_y: str,
) -> io.BytesIO:
    x_line = np.linspace(xs.min(), xs.max(), 300)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(xs, ys, s=35, zorder=3, label="Data points")
    ax.plot(
        x_line, y_line, color="crimson", linewidth=2,
        label=f"y = {slope:.4g}x + {intercept:.4g}  (R²={r_sq:.4f})",
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Linear Regression: {col_y} vs {col_x}", fontsize=12)
    ax.set_xlabel(col_x, fontsize=11)
    ax.set_ylabel(col_y, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Interactive plot builder
# ---------------------------------------------------------------------------

@dataclass
class CSVPlotConfig:
    """State dataclass for the interactive CSV plot builder."""
    user_id:     int
    plot_type:   str         = "scatter"
    col_x:       str         = ""
    col_y:       str         = ""
    col_extra:   list[str]   = field(default_factory=list)
    title:       str         = ""
    color:       str         = "#1f77b4"
    style:       str         = "solid"
    show_grid:   bool        = True
    show_points: bool        = False
    regression:  bool        = False
    bin_count:   int         = 20
    agg_func:    str         = "mean"
    col_page:    int         = 0      # for >25-column paging


_STYLE_MAP = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}


def _render_csv_plot(cfg: CSVPlotConfig, session: CSVSession) -> io.BytesIO:
    """Synchronous matplotlib render — runs in a thread executor."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ls = _STYLE_MAP.get(cfg.style, "-")

    if cfg.plot_type == "scatter":
        xs = get_numeric_column(session, cfg.col_x)
        ys = get_numeric_column(session, cfg.col_y)
        ax.scatter(xs, ys, color=cfg.color, s=30, zorder=3)
        if cfg.regression and len(xs) > 1:
            slope, intercept, r, *_ = scipy_stats.linregress(xs, ys)
            xl = np.linspace(xs.min(), xs.max(), 300)
            ax.plot(xl, slope * xl + intercept, color="crimson", linewidth=1.5,
                    label=f"y={slope:.3g}x+{intercept:.3g} R²={r**2:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(cfg.col_x); ax.set_ylabel(cfg.col_y)

    elif cfg.plot_type == "line":
        ys = get_numeric_column(session, cfg.col_y)
        xs_raw = [row.get(cfg.col_x, "") for row in session.rows]
        # try numeric X, fall back to index
        try:
            xs = np.array([float(v) for v in xs_raw])
        except ValueError:
            xs = np.arange(len(ys))
            ax.set_xticks(xs)
            short = [v[:12] for v in xs_raw[:len(ys)]]
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.plot(xs[:len(ys)], ys, color=cfg.color, linestyle=ls, linewidth=1.8)
        if cfg.show_points:
            ax.scatter(xs[:len(ys)], ys, color=cfg.color, s=20, zorder=3)
        ax.set_xlabel(cfg.col_x); ax.set_ylabel(cfg.col_y)

    elif cfg.plot_type == "histogram":
        arr = get_numeric_column(session, cfg.col_x)
        ax.hist(arr, bins=cfg.bin_count, color=cfg.color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(cfg.col_x); ax.set_ylabel("Frequency")

    elif cfg.plot_type == "bar":
        ys   = get_numeric_column(session, cfg.col_y)
        cats = [row.get(cfg.col_x, "") for row in session.rows[:len(ys)]]
        # aggregate by category
        from collections import defaultdict
        buckets: dict[str, list[float]] = defaultdict(list)
        for cat, val in zip(cats, ys):
            buckets[cat].append(val)
        labels = list(buckets.keys())
        if cfg.agg_func == "sum":
            vals = [sum(v) for v in buckets.values()]
        elif cfg.agg_func == "count":
            vals = [len(v) for v in buckets.values()]
        else:
            vals = [float(np.mean(v)) for v in buckets.values()]
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, vals, color=cfg.color)
        ax.set_xticks(x_pos)
        short = [lb[:15] for lb in labels]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(cfg.col_x)
        ax.set_ylabel(f"{cfg.agg_func.title()} of {cfg.col_y}")

    elif cfg.plot_type == "box":
        cols = cfg.col_extra if cfg.col_extra else ([cfg.col_x] if cfg.col_x else [])
        arrays = [get_numeric_column(session, c) for c in cols]
        ax.boxplot(arrays, labels=cols, patch_artist=True,
                   boxprops=dict(facecolor=cfg.color, alpha=0.6))
        ax.set_ylabel("Value")

    elif cfg.plot_type == "heatmap":
        # Correlation matrix over all numeric columns
        num_cols = [
            c for c in session.columns
            if _infer_type(session.rows, c) == "numeric"
        ]
        if len(num_cols) < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns.")
        matrix = np.column_stack([get_numeric_column(session, c) for c in num_cols])
        corr   = np.corrcoef(matrix.T)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels([c[:12] for c in num_cols], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c[:12] for c in num_cols], fontsize=8)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(corr[i, j]) < 0.7 else "white")

    title = cfg.title or f"{cfg.plot_type.title()} — {session.filename}"
    ax.set_title(title, fontsize=12)
    if cfg.show_grid and cfg.plot_type not in ("heatmap",):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


class CSVPlotView(ui.View):
    """
    Interactive plot builder for CSV data.

    One ephemeral message; rebuilt in-place on every interaction.
    Mirrors PlotEngineView from plot_engine.py.
    """

    def __init__(self, user_id: int) -> None:
        super().__init__(timeout=600)   # 10-minute timeout
        self.user_id = user_id
        self.cfg = CSVPlotConfig(user_id=user_id)
        self._rebuild()

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _column_options(self, include_empty: bool = False) -> list[discord.SelectOption]:
        names = get_column_names(self.user_id)
        page  = self.cfg.col_page
        chunk = 24                                     # leave 1 slot for "next"
        start = page * chunk
        end   = start + chunk
        options = [discord.SelectOption(label=n, value=n) for n in names[start:end]]
        if include_empty:
            options.insert(0, discord.SelectOption(label="— none —", value=""))
        if end < len(names):
            options.append(discord.SelectOption(label="→ next page", value="__next_page__"))
        if page > 0:
            options.insert(0, discord.SelectOption(label="← prev page", value="__prev_page__"))
        return options

    def _rebuild(self) -> None:
        self.clear_items()
        cfg = self.cfg

        # Row 0 — plot type
        type_select = ui.Select(
            placeholder=f"Plot type: {cfg.plot_type}",
            options=[discord.SelectOption(label=t, value=t, default=(t == cfg.plot_type))
                     for t in _PLOT_TYPES],
            row=0,
        )
        type_select.callback = self._on_type
        self.add_item(type_select)

        # Row 1 — X column (always shown)
        x_opts = self._column_options(include_empty=True)
        x_sel = ui.Select(
            placeholder=f"X column: {cfg.col_x or '—'}",
            options=x_opts,
            row=1,
        )
        x_sel.callback = self._on_col_x
        self.add_item(x_sel)

        # Row 2 — Y column (hidden for histogram / box / heatmap)
        if cfg.plot_type not in ("histogram", "heatmap"):
            y_opts = self._column_options(include_empty=True)
            y_sel = ui.Select(
                placeholder=f"Y column: {cfg.col_y or '—'}",
                options=y_opts,
                row=2,
            )
            y_sel.callback = self._on_col_y
            self.add_item(y_sel)

        # Row 3 — colour + style selects
        colour_sel = ui.Select(
            placeholder="Colour",
            options=[discord.SelectOption(label=c, value=c, default=(c == cfg.color))
                     for c in _COLOURS],
            row=3,
        )
        colour_sel.callback = self._on_colour
        self.add_item(colour_sel)

        style_sel = ui.Select(
            placeholder=f"Style: {cfg.style}",
            options=[discord.SelectOption(label=s, value=s, default=(s == cfg.style))
                     for s in _STYLES],
            row=3,
        )
        style_sel.callback = self._on_style
        self.add_item(style_sel)

        # Row 4 — action buttons
        render_btn = ui.Button(
            label="📊 Render",
            style=discord.ButtonStyle.primary,
            row=4,
        )
        render_btn.callback = self._on_render
        self.add_item(render_btn)

        toggle_grid = ui.Button(
            label=f"Grid: {'on' if cfg.show_grid else 'off'}",
            style=discord.ButtonStyle.secondary,
            row=4,
        )
        toggle_grid.callback = self._on_toggle_grid
        self.add_item(toggle_grid)

        if cfg.plot_type == "scatter":
            regr_btn = ui.Button(
                label=f"Regression: {'on' if cfg.regression else 'off'}",
                style=discord.ButtonStyle.secondary,
                row=4,
            )
            regr_btn.callback = self._on_toggle_regression
            self.add_item(regr_btn)

        close_btn = ui.Button(
            label="✖ Close",
            style=discord.ButtonStyle.danger,
            row=4,
        )
        close_btn.callback = self._on_close
        self.add_item(close_btn)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def _on_type(self, interaction: discord.Interaction) -> None:
        self.cfg.plot_type = interaction.data["values"][0]
        self.cfg.col_page  = 0
        self._rebuild()
        await interaction.response.edit_message(
            content=f"Plot type: **{self.cfg.plot_type}**",
            view=self,
            attachments=[],
        )

    async def _on_col_x(self, interaction: discord.Interaction) -> None:
        val = interaction.data["values"][0]
        if val == "__next_page__":
            self.cfg.col_page += 1
        elif val == "__prev_page__":
            self.cfg.col_page = max(0, self.cfg.col_page - 1)
        else:
            self.cfg.col_x = val
        self._rebuild()
        await interaction.response.edit_message(view=self, attachments=[])

    async def _on_col_y(self, interaction: discord.Interaction) -> None:
        val = interaction.data["values"][0]
        if val == "__next_page__":
            self.cfg.col_page += 1
        elif val == "__prev_page__":
            self.cfg.col_page = max(0, self.cfg.col_page - 1)
        else:
            self.cfg.col_y = val
        self._rebuild()
        await interaction.response.edit_message(view=self, attachments=[])

    async def _on_colour(self, interaction: discord.Interaction) -> None:
        self.cfg.color = interaction.data["values"][0]
        self._rebuild()
        await interaction.response.edit_message(view=self, attachments=[])

    async def _on_style(self, interaction: discord.Interaction) -> None:
        self.cfg.style = interaction.data["values"][0]
        self._rebuild()
        await interaction.response.edit_message(view=self, attachments=[])

    async def _on_toggle_grid(self, interaction: discord.Interaction) -> None:
        self.cfg.show_grid = not self.cfg.show_grid
        self._rebuild()
        await interaction.response.edit_message(view=self, attachments=[])

    async def _on_toggle_regression(self, interaction: discord.Interaction) -> None:
        self.cfg.regression = not self.cfg.regression
        self._rebuild()
        await interaction.response.edit_message(view=self, attachments=[])

    async def _on_close(self, interaction: discord.Interaction) -> None:
        self.stop()
        await interaction.response.edit_message(
            content="Plot builder closed.", view=None, attachments=[]
        )

    async def _on_render(self, interaction: discord.Interaction) -> None:
        session = get_session(self.user_id)
        if session is None:
            await interaction.response.send_message(
                embed=error_embed("Session expired. Use `/csv upload` to reload your file."),
                ephemeral=True,
            )
            return

        # Basic validation
        cfg = self.cfg
        needs_x = cfg.plot_type in ("scatter", "line", "histogram", "bar")
        needs_y = cfg.plot_type in ("scatter", "line", "bar")
        if needs_x and not cfg.col_x:
            await interaction.response.send_message(
                embed=error_embed("Please select an X column first."), ephemeral=True
            )
            return
        if needs_y and not cfg.col_y:
            await interaction.response.send_message(
                embed=error_embed("Please select a Y column first."), ephemeral=True
            )
            return

        await interaction.response.defer()

        loop = asyncio.get_event_loop()
        try:
            buf: io.BytesIO = await loop.run_in_executor(
                None, _render_csv_plot, cfg, session
            )
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"Render failed: {exc}"), ephemeral=True
            )
            return

        f = discord.File(buf, filename="csv_plot.png")
        await interaction.edit_original_response(
            content=f"**{cfg.plot_type.title()}** — `{session.filename}`",
            attachments=[f],
            view=self,
        )

    async def on_timeout(self) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Confirmation view for /csv clear
# ---------------------------------------------------------------------------

class _ClearConfirmView(ui.View):
    def __init__(self, user_id: int) -> None:
        super().__init__(timeout=60)
        self.user_id = user_id

    @ui.button(label="Yes, clear", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: ui.Button) -> None:
        clear_session(self.user_id)
        self.stop()
        await interaction.response.edit_message(
            content="✅ CSV session cleared.", view=None
        )

    @ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: ui.Button) -> None:
        self.stop()
        await interaction.response.edit_message(
            content="Cancelled — session kept.", view=None
        )


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class CSVTools(commands.Cog):
    """
    /csv — Upload a CSV and run statistics or interactive plots on its columns.
    """

    csv_group = app_commands.Group(name="csv", description="CSV upload and analysis commands")

    # ------------------------------------------------------------------ upload

    @csv_group.command(name="upload", description="Upload a CSV file to analyse")
    @app_commands.describe(file="A .csv file (max 2 MB, 10 000 rows, 50 columns)")
    async def csv_upload(
        self,
        interaction: discord.Interaction,
        file: discord.Attachment,
    ) -> None:
        await interaction.response.defer(ephemeral=True)

        # ── validation ────────────────────────────────────────────────────────
        if not file.filename.lower().endswith(".csv"):
            await interaction.followup.send(
                embed=error_embed("Only `.csv` files are supported."), ephemeral=True
            )
            return

        if file.size > _MAX_FILE_BYTES:
            limit_mb = _MAX_FILE_BYTES / (1024 * 1024)
            await interaction.followup.send(
                embed=error_embed(
                    f"File is too large ({file.size / (1024*1024):.1f} MB). "
                    f"Limit is {limit_mb:.0f} MB."
                ),
                ephemeral=True,
            )
            return

        # ── fetch ─────────────────────────────────────────────────────────────
        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(file.url) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    raw_bytes = await resp.read()
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"Could not fetch the attachment: {exc}"), ephemeral=True
            )
            return

        # ── decode & parse ────────────────────────────────────────────────────
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"Could not decode the file: {exc}"), ephemeral=True
            )
            return

        reader = _csv.DictReader(io.StringIO(text))
        try:
            columns = list(reader.fieldnames or [])
        except Exception:
            columns = []

        if not columns:
            await interaction.followup.send(
                embed=error_embed("The CSV file has no header row or appears to be empty."),
                ephemeral=True,
            )
            return

        if len(columns) > _MAX_COLUMNS:
            await interaction.followup.send(
                embed=error_embed(
                    f"The CSV has {len(columns)} columns; the limit is {_MAX_COLUMNS}."
                ),
                ephemeral=True,
            )
            return

        rows: list[dict[str, str]] = []
        try:
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= _MAX_ROWS:
                    break
        except _csv.Error as exc:
            await interaction.followup.send(
                embed=error_embed(f"CSV parse error: {exc}"), ephemeral=True
            )
            return

        if not rows:
            await interaction.followup.send(
                embed=error_embed("The CSV file has a header but no data rows."),
                ephemeral=True,
            )
            return

        # ── store ─────────────────────────────────────────────────────────────
        session = store_session(interaction.user.id, file.filename, columns, rows)
        await interaction.followup.send(
            embed=_build_summary_embed(session), ephemeral=True
        )

    # ------------------------------------------------------------------ info

    @csv_group.command(name="info", description="Re-display the column summary for your loaded CSV")
    async def csv_info(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        session = get_session(interaction.user.id)
        if session is None:
            await interaction.followup.send(
                embed=error_embed("No CSV loaded. Use `/csv upload` first."), ephemeral=True
            )
            return
        await interaction.followup.send(embed=_build_summary_embed(session), ephemeral=True)

    # ------------------------------------------------------------------ preview

    @csv_group.command(name="preview", description="Show the first N rows of your loaded CSV")
    @app_commands.describe(rows="Number of rows to show (1–20, default 5)")
    async def csv_preview(
        self,
        interaction: discord.Interaction,
        rows: app_commands.Range[int, 1, 20] = 5,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        session = get_session(interaction.user.id)
        if session is None:
            await interaction.followup.send(
                embed=error_embed("No CSV loaded. Use `/csv upload` first."), ephemeral=True
            )
            return

        # Build a monospace table
        col_w = 15
        header = " | ".join(c[:col_w].ljust(col_w) for c in session.columns)
        sep    = "-+-".join("-" * col_w for _ in session.columns)
        data_lines: list[str] = []
        for row in session.rows[:rows]:
            line = " | ".join(str(row.get(c, ""))[:col_w].ljust(col_w) for c in session.columns)
            data_lines.append(line)

        table = "\n".join([header, sep] + data_lines)
        # Discord code block limit ~4096 chars
        if len(table) > 3900:
            table = table[:3900] + "\n…(truncated)"

        em = _csv_embed(
            title=f"Preview — {session.filename}",
            description=f"```\n{table}\n```",
            footer=_session_footer(session),
        )
        em.set_footer(text=f"Showing {min(rows, session.row_count)} of {session.row_count:,} rows  ·  {_session_footer(session)}")
        await interaction.followup.send(embed=em, ephemeral=True)

    # ------------------------------------------------------------------ stat

    @csv_group.command(name="stat", description="Run a statistical operation on one CSV column")
    @app_commands.describe(
        operation="Which statistic to compute",
        column="Column name (autocompleted from your loaded CSV)",
    )
    @app_commands.choices(operation=_STAT_CHOICES)
    @app_commands.autocomplete(column=_column_autocomplete)
    async def csv_stat(
        self,
        interaction: discord.Interaction,
        operation: str,
        column: str,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        session = get_session(interaction.user.id)
        if session is None:
            await interaction.followup.send(
                embed=error_embed("No CSV loaded. Use `/csv upload` first."), ephemeral=True
            )
            return

        if column == "__none__" or column not in session.columns:
            await interaction.followup.send(
                embed=error_embed(f"Column `{column}` not found. Use autocomplete to pick a column."),
                ephemeral=True,
            )
            return

        try:
            arr = get_numeric_column(session, column)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return

        em = _compute_stat(operation, arr, column)
        em.set_footer(text=_session_footer(session))
        await interaction.followup.send(embed=em, ephemeral=True)

    # ------------------------------------------------------------------ stat2

    @csv_group.command(name="stat2", description="Run a two-column statistical operation on your CSV")
    @app_commands.describe(
        operation="correlation or regression",
        col_x="Independent column (X)",
        col_y="Dependent column (Y)",
    )
    @app_commands.choices(operation=_STAT2_CHOICES)
    @app_commands.autocomplete(col_x=_column_autocomplete, col_y=_column_autocomplete)
    async def csv_stat2(
        self,
        interaction: discord.Interaction,
        operation: str,
        col_x: str,
        col_y: str,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        session = get_session(interaction.user.id)
        if session is None:
            await interaction.followup.send(
                embed=error_embed("No CSV loaded. Use `/csv upload` first."), ephemeral=True
            )
            return

        for col in (col_x, col_y):
            if col == "__none__" or col not in session.columns:
                await interaction.followup.send(
                    embed=error_embed(f"Column `{col}` not found. Use autocomplete."),
                    ephemeral=True,
                )
                return

        try:
            xs = get_numeric_column(session, col_x)
            ys = get_numeric_column(session, col_y)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return

        # Align lengths
        n = min(len(xs), len(ys))
        xs, ys = xs[:n], ys[:n]
        if n < 2:
            await interaction.followup.send(
                embed=error_embed("Both columns must have at least 2 comparable numeric values."),
                ephemeral=True,
            )
            return

        if operation == "correlation":
            r, p = scipy_stats.pearsonr(xs, ys)
            em = _csv_embed(
                title=f"📊 Correlation — `{col_x}` vs `{col_y}`",
            )
            em.add_field(name="Pearson r",  value=f"`{r:.6f}`",        inline=True)
            em.add_field(name="p-value",    value=f"`{p:.4g}`",        inline=True)
            em.add_field(name="Strength",   value=f"`{_correlation_label(r)}`", inline=True)
            em.add_field(name="n",          value=f"`{n}`",            inline=True)
            em.set_footer(text=_session_footer(session))
            await interaction.followup.send(embed=em, ephemeral=True)

        elif operation == "regression":
            slope, intercept, r, p, se = scipy_stats.linregress(xs, ys)
            r_sq = r ** 2
            em = _csv_embed(title=f"📊 Regression — `{col_y}` ~ `{col_x}`")
            em.add_field(name="Equation",   value=f"`y = {slope:.4g}x + {intercept:.4g}`", inline=False)
            em.add_field(name="R²",         value=f"`{r_sq:.6f}`",      inline=True)
            em.add_field(name="Slope",      value=f"`{slope:.6g}`",     inline=True)
            em.add_field(name="Intercept",  value=f"`{intercept:.6g}`", inline=True)
            em.add_field(name="p-value",    value=f"`{p:.4g}`",         inline=True)
            em.add_field(name="Std Error",  value=f"`{se:.4g}`",        inline=True)
            em.add_field(name="n",          value=f"`{n}`",             inline=True)
            em.set_footer(text=_session_footer(session))

            loop = asyncio.get_event_loop()
            buf = await loop.run_in_executor(
                None, _regression_plot_bytes, xs, ys, slope, intercept, r_sq, col_x, col_y
            )
            f = discord.File(buf, filename="regression.png")
            em.set_image(url="attachment://regression.png")
            await interaction.followup.send(embed=em, file=f, ephemeral=True)

    # ------------------------------------------------------------------ plot

    @csv_group.command(name="plot", description="Open the interactive CSV plot builder")
    async def csv_plot(self, interaction: discord.Interaction) -> None:
        session = get_session(interaction.user.id)
        if session is None:
            await interaction.response.send_message(
                embed=error_embed("No CSV loaded. Use `/csv upload` first."), ephemeral=True
            )
            return

        view = CSVPlotView(interaction.user.id)
        await interaction.response.send_message(
            content=(
                f"📂 **{session.filename}** loaded  "
                f"({session.row_count:,} rows · {len(session.columns)} columns)\n"
                "Choose a plot type and columns, then click **📊 Render**."
            ),
            view=view,
            ephemeral=True,
        )

    # ------------------------------------------------------------------ clear

    @csv_group.command(name="clear", description="Discard your loaded CSV session")
    async def csv_clear(self, interaction: discord.Interaction) -> None:
        session = get_session(interaction.user.id)
        if session is None:
            await interaction.response.send_message(
                content="You have no active CSV session to clear.",
                ephemeral=True,
            )
            return

        view = _ClearConfirmView(interaction.user.id)
        await interaction.response.send_message(
            content=f"Clear your session for **{session.filename}**? This cannot be undone.",
            view=view,
            ephemeral=True,
        )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(CSVTools(bot))