# MathFrame — Codebase Documentation

> Post-migration reference. Reflects the **group-based slash command architecture** introduced by the `GROUPING_MIGRATION.md` plan: 110 subcommands across 16 `app_commands.Group` slots + 1 standalone `/ping`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration Reference](#4-configuration-reference)
5. [Bot Bootstrapping & Lifecycle](#5-bot-bootstrapping--lifecycle)
6. [Command Architecture](#6-command-architecture)
7. [Permission System](#7-permission-system)
8. [Data Layer](#8-data-layer)
   - [Result Cache (`data/cache.py`)](#81-result-cache-datacachepy)
   - [Command History (`data/history.py`)](#82-command-history-datahistorypy)
   - [Guild Permissions (`data/permissions.py`)](#83-guild-permissions-datapermissionspy)
   - [Variable Memory (`data/memory.py`)](#84-variable-memory-datamemorypy)
9. [Expression Parsing](#9-expression-parsing)
   - [Primary Parser (`utils/parser.py`)](#91-primary-parser-utilsparserpy)
   - [Plot Parser (`utils/expr_utils.py`)](#92-plot-parser-utilsexpr_utilspy)
10. [Supporting Utilities](#10-supporting-utilities)
    - [Embed Formatter (`utils/formatter.py`)](#101-embed-formatter-utilsformatterpy)
    - [Step-by-Step Solver (`utils/solver.py`)](#102-step-by-step-solver-utilssolverpy)
    - [LaTeX Renderer (`utils/renderer.py`)](#103-latex-renderer-utilsrendererpy)
    - [Paginator (`utils/paginator.py`)](#104-paginator-utilspaginatorpy)
    - [Plot Generator (`utils/plotter.py`)](#105-plot-generator-utilsplotterpy)
11. [Cog Reference](#11-cog-reference)
    - [/admin — Admin](#111-admin--cogsadminpy)
    - [/alg — Algebra](#112-alg--cogsalgebrapy)
    - [/calc — Calculus](#113-calc--cogscalculuspy)
    - [/stat — Statistics](#114-stat--cogsstatisticspy)
    - [/nt — Number Theory](#115-nt--cogsnumber_theorypy)
    - [/mat — Linear Algebra](#116-mat--cogslinear_algebrapy)
    - [/sym — Symbolic](#117-sym--cogssymbolicpy)
    - [/geo — Geometry](#118-geo--cogsgeometrypy)
    - [/dis — Discrete Math](#119-dis--cogsdiscretepy)
    - [/cx — Complex Numbers](#1110-cx--cogscomplexpy)
    - [/tf — Transforms](#1111-tf--cogstransformspy)
    - [/base — Base Conversion](#1112-base--cogsbase_npy)
    - [/mem — Memory](#1113-mem--cogsmemorypy)
    - [/bot — Bot Utility](#1114-bot--cogsbotpy)
    - [/render — Render](#1115-render--cogsrenderpy)
    - [/plot — Plot Builder](#1116-plot--cogsplot_enginepy)
12. [Complete Command Reference](#12-complete-command-reference)
13. [Data Flow](#13-data-flow)
14. [Security Model](#14-security-model)
15. [Conventions & Patterns](#15-conventions--patterns)
16. [Dependency Reference](#16-dependency-reference)
17. [Codebase Metrics](#17-codebase-metrics)

---

## 1. Project Overview

MathFrame is a Discord mathematics bot built on `discord.py` slash commands. It covers symbolic algebra, calculus, statistics, number theory, geometry, discrete math, linear algebra, complex numbers, integral transforms, base-N arithmetic, interactive plotting, LaTeX rendering, unit conversion, and Wikipedia browsing — all through Discord's slash command interface.

**Key facts:**
- Python 3.12, discord.py ≥ 2.5, SymPy ≥ 1.12, NumPy, SciPy, Matplotlib
- **17 top-level Discord command slots used** (16 groups + `/ping`) out of the 100-slot global limit
- **110 subcommands** total across the 16 groups
- All commands are global Application Commands (slash commands)
- Zero database — all state is in-process memory (lost on restart), except guild permission rules which are persisted to a local JSON file

---

## 2. Repository Structure

```
MathFrame/
├── main.py                   Entry point — bot init, cog loader, global handlers
├── config.py                 Environment-variable configuration constants
├── requirements.txt          Python package dependencies
├── .env                      Secret token (not in VCS)
│
├── cogs/                     Feature modules — each is one Discord group
│   ├── __init__.py
│   ├── admin.py              /admin  — permission management (4 subcommands)
│   ├── algebra.py            /alg    — algebra merged cog (12 subcommands)
│   ├── calculus.py           /calc   — calculus (10 subcommands)
│   ├── statistics.py         /stat   — statistics & probability (14 subcommands)
│   ├── number_theory.py      /nt     — number theory (12 subcommands)
│   ├── linear_algebra.py     /mat    — linear algebra (6 subcommands)
│   ├── symbolic.py           /sym    — symbolic math (5 subcommands)
│   ├── geometry.py           /geo    — geometry (6 subcommands)
│   ├── discrete.py           /dis    — discrete math (7 subcommands)
│   ├── complex.py            /cx     — complex numbers (5 subcommands)
│   ├── transforms.py         /tf     — integral transforms (4 subcommands)
│   ├── base_n.py             /base   — base conversion (4 subcommands)
│   ├── memory.py             /mem    — variable memory (6 subcommands)
│   ├── bot.py                /bot    — utility & wikipedia merged cog (9 subcommands)
│   ├── render.py             /render — LaTeX rendering (2 subcommands)
│   └── plot_engine.py        /plot   — interactive plot builder (4 subcommands)
│
├── data/                     Stateful singletons (all in-process)
│   ├── __init__.py
│   ├── cache.py              TTL result cache (thread-safe singleton)
│   ├── history.py            Per-user command history (thread-safe singleton)
│   ├── permissions.py        Guild permission store (JSON-backed, thread-safe)
│   └── memory.py             Per-user variable memory store (thread-safe singleton)
│
└── utils/                    Pure helper modules — no Discord state
    ├── __init__.py
    ├── parser.py             THE expression parser — all cogs go through this
    ├── expr_utils.py         Lighter parser used exclusively by the plot engine
    ├── solver.py             Step-by-step solution builders (quadratic, cubic, etc.)
    ├── formatter.py          Discord embed builders (math_embed, error_embed)
    ├── renderer.py           LaTeX → PNG via matplotlib (async, thread-pool)
    ├── paginator.py          ◀/▶ paginated embed view
    └── plotter.py            Matplotlib plot generator (14 plot types, async)
```

**Deleted after migration** (contents absorbed into merged cogs):
```
cogs/arithmetic.py    → cogs/algebra.py
cogs/equations.py     → cogs/algebra.py
cogs/inequalities.py  → cogs/algebra.py
cogs/utility.py       → cogs/bot.py
cogs/wiki.py          → cogs/bot.py
```

---

## 3. Installation & Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd MathFrame

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env with your bot token
echo "DISCORD_TOKEN=your_token_here" > .env

# 5. Run
python main.py
```

**Dependencies** (`requirements.txt`):
```
discord.py>=2.5.0
python-dotenv>=1.0.0
sympy>=1.12
latex2sympy2>=0.4.0
cachetools>=5.3.0
matplotlib>=3.8.0
numpy>=1.25.0
scipy>=1.11.0
```

The bot requires **no database**, **no Redis**, **no external services** beyond Discord's API and (optionally) Wikipedia's public REST API used by `/bot wiki` and `/bot wiki_search`.

---

## 4. Configuration Reference

**File:** `config.py`  
Loads all values from environment variables via `python-dotenv`.

| Constant | Type | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | `str` | — | Bot secret token from Discord Developer Portal. Required. |
| `PREFIX` | `str` | `"!"` | Prefix for legacy prefix-style commands. Discord.py requires it even though the bot uses only slash commands. |
| `MAX_EXPR_LENGTH` | `int` | `500` | Maximum character count accepted in any expression input. Inputs longer than this are rejected before parsing to prevent memory abuse. |
| `COMPUTE_TIMEOUT` | `int` | `3` | Seconds allowed for a single SymPy computation. Hard limit against infinite-loop inputs. |
| `CACHE_TTL` | `int` | `300` | Time-to-live in seconds for result cache entries (5 minutes). |
| `CACHE_MAXSIZE` | `int` | `256` | Maximum entries the result cache may hold. Least-recently-used entry evicted when full. |

---

## 5. Bot Bootstrapping & Lifecycle

**File:** `main.py`

### Startup sequence (`on_ready`)

1. Logs the bot user and ID.
2. Records `bot.start_time` (UTC) on the **first** `on_ready` only — reconnects do not reset the uptime clock.
3. Iterates `COGS` list and calls `bot.load_extension(cog)` for each entry. Failed cogs are logged and skipped; the bot continues with whichever cogs loaded successfully.
4. Calls `await bot.tree.sync()` **once per process lifetime** (guarded by `bot._commands_synced`). Discord rate-limits global syncs heavily; syncing on every reconnect would hit those limits.

### COGS load order

```python
COGS: list[str] = [
    "cogs.admin",         # /admin
    "cogs.algebra",       # /alg
    "cogs.calculus",      # /calc
    "cogs.transforms",    # /tf
    "cogs.linear_algebra",# /mat
    "cogs.statistics",    # /stat
    "cogs.number_theory", # /nt
    "cogs.geometry",      # /geo
    "cogs.discrete",      # /dis
    "cogs.symbolic",      # /sym
    "cogs.complex",       # /cx
    "cogs.base_n",        # /base
    "cogs.memory",        # /mem
    "cogs.bot",           # /bot
    "cogs.render",        # /render
    "cogs.plot_engine",   # /plot
]
```

### Global interaction check (`bot.tree.interaction_check`)

Every slash command invocation passes through `_permission_check` before the cog handler runs:

- DMs (`interaction.guild_id is None`) → always allowed.
- Calls `data.permissions.is_command_allowed(guild_id, channel_id, command_name)`.
- If denied → sends an ephemeral error and returns `False`, cancelling invocation.

### Global error handler (`bot.tree.error`)

Handles three cases explicitly:
- `CommandOnCooldown` → ephemeral "slow down, retry in Xs" message.
- `MissingPermissions` → ephemeral "you lack permission" message.
- `BotMissingPermissions` → ephemeral "I lack permission" message.
- Everything else → logs full traceback + ephemeral generic error.

### Built-in command

`/ping` — standalone (not a group subcommand), registered directly on `bot.tree`. Replies with WebSocket latency in milliseconds.

---

## 6. Command Architecture

### Group pattern

Every cog (except `/ping`) uses `app_commands.Group` as a class variable:

```python
class CalcCog(commands.Cog, name="Calculus"):

    calc = app_commands.Group(name="calc", description="Calculus operations.")

    @calc.command(name="diff", description="Differentiate an expression.")
    @app_commands.describe(expression="...")
    @app_commands.checks.cooldown(1, 3.0)
    async def diff(self, interaction: discord.Interaction, expression: str) -> None:
        ...
```

Key rules:
- The group **class variable name** and the group's Discord `name` do not have to match. For `render.py` and `plot_engine.py` they differ (`render_grp`/`plot_grp`) to avoid shadowing method names.
- All subcommand logic, `@app_commands.describe`, and `@app_commands.checks.cooldown` decorators attach to subcommands exactly as they would to top-level commands.
- `admin.py` uses `admin_group` (pre-existing name from before the migration).
- `memory.py` uses `mem` (pre-existing).

### Slot accounting

| Type | Count | Discord limit |
|---|---|---|
| Top-level slots used | 17 | 100 |
| Subcommands per group (max) | 14 (`/stat`) | 25 |
| Total subcommands | 110 | — |
| Remaining top-level slots | 83 | — |

### Standard command pattern

Every subcommand follows the same flow:

```
1. interaction.response.defer()          ← avoid 3-second timeout
2. Resolve $memory references            ← only in algebra/calculus cogs
3. Check result cache                    ← return cached embed if hit
4. Run SymPy computation (async via executor or direct await)
5. Build discord.Embed via math_embed()
6. Store in cache
7. interaction.followup.send(embed=...)
8. Catch ValueError / SympifyError / PolynomialError / NotImplementedError
   and surface them via error_embed()
```

---

## 7. Permission System

**File:** `data/permissions.py`

Guild admins use `/admin enable|disable|reset|status` to control which commands work in which channels.

### Storage schema

Persisted to `data/guild_permissions.json`:

```json
{
  "<guild_id>": {
    "<channel_id | '__all__'>": {
      "<command_name | '__all__'>": true | false
    }
  }
}
```

### Lookup priority (most-specific wins)

1. `guild → channel → command` — channel + command specific
2. `guild → __all__ → command` — guild-wide for this command
3. `guild → channel → __all__` — all commands in this channel
4. `guild → __all__ → __all__` — entire guild default
5. *(no matching rule)* → **allowed** (fail-open)

### Public API

```python
is_command_allowed(guild_id, channel_id, command_name) -> bool
set_permission(guild_id, channel_id, command_name, enabled) -> None
clear_permission(guild_id, channel_id, command_name) -> bool
get_guild_status(guild_id) -> list[dict]   # [{channel, command, enabled}]
```

The in-memory `_data` dict is loaded from disk once at import time and written back on every `set_permission` / `clear_permission` call. A `threading.Lock` makes all reads and writes safe for concurrent cog calls.

---

## 8. Data Layer

### 8.1 Result Cache (`data/cache.py`)

Module-level `TTLCache` singleton from `cachetools`. Thread-safe via `threading.Lock`.

**Parameters** (from `config.py`):
- `maxsize = 256` — max entries before LRU eviction
- `ttl = 300` — seconds until an entry expires

**Public API:**
```python
cache_key(*args) -> str        # "simplify|x**2 + 1"
get(key) -> object | None      # None on miss or expiry
set(key, value) -> None
clear() -> None
info() -> dict                 # {currsize, maxsize, ttl}
```

Cache keys are built by joining all arguments with `|`. Values are typically `discord.Embed` objects. Not every command uses the cache — only deterministic, pure-math commands (simplify, expand, factor, diff, etc.) cache their results. Commands involving user state (history, memory, plotting sessions) do not.

### 8.2 Command History (`data/history.py`)

Per-user `deque[HistoryEntry]` store, capped at **20 entries per user** (oldest entry auto-dropped). Stored in process memory only — lost on restart.

**`HistoryEntry` attributes:**
| Attribute | Type | Description |
|---|---|---|
| `command` | `str` | Command name, e.g. `"circle_area"` |
| `input` | `str` | Short human-readable input summary |
| `result` | `str` | Short human-readable result summary |
| `timestamp` | `datetime` | UTC timestamp of the invocation |

**Public API:**
```python
save_history(user_id, command, input_str, result) -> None
get_history(user_id, limit=20) -> list[HistoryEntry]   # newest first
clear_history(user_id) -> None
```

History is only written by cogs that explicitly call `save_history`. Most cogs do not record history; the feature is surfaced through `/bot history` and `/bot clear`.

### 8.3 Guild Permissions (`data/permissions.py`)

Covered in [§7](#7-permission-system). The only data module backed by a file on disk.

### 8.4 Variable Memory (`data/memory.py`)

Per-user namespace for storing named math values. Keyed by `(guild_id, user_id)` tuples; guild ID `0` is used for DMs.

**Stored types:**

| `MemType` | Description | Example |
|---|---|---|
| `NUMBER` | SymPy expression with no free symbols | `3.14`, `sqrt(2)`, `pi/4` |
| `EXPRESSION` | SymPy expression with ≥1 free symbol | `x^2 + 2*x`, `sin(t)*exp(-t)` |
| `MATRIX` | `sympy.Matrix` object | `[[1,2],[3,4]]` |

**Limits:**
- Max 50 entries per user (`MAX_ENTRIES = 50`)
- Max 32-character name (`MAX_NAME_LEN = 32`)

**`$`-reference syntax:**

In any expression argument, `$name` is substituted with the stored value's string representation before parsing:

```
/mem set  k  3.14
/calc diff  $k * x^2       →  parse_expression("(3.14) * x^2")
```

`MATRIX` entries cannot be inlined via `$` — they raise `ValueError`.

**`MemoryStore` public API:**
```python
memory.set(guild_id, user_id, name, entry) -> None
memory.get(guild_id, user_id, name) -> MemEntry | None
memory.delete(guild_id, user_id, name) -> bool
memory.list_all(guild_id, user_id) -> dict[str, MemEntry]
memory.clear(guild_id, user_id) -> int          # returns count deleted
memory.count(guild_id, user_id) -> int
memory.resolve(guild_id, user_id, raw_str) -> str  # $-substitution
memory.has_refs(raw_str) -> bool
```

The module exposes a single `memory: MemoryStore` singleton at module level.

---

## 9. Expression Parsing

### 9.1 Primary Parser (`utils/parser.py`)

The **single entry point** for all mathematical expression parsing in the bot. No cog calls `sympy.parse_expr` or `latex2sympy` directly — everything goes through `parse_expression`.

**Signature:**
```python
async def parse_expression(
    expr: str,
    local_dict: dict | None = None,
) -> sympy.Expr
```

**Input format auto-detection** (evaluated in order):

| Format | Trigger | Parser used |
|---|---|---|
| LaTeX | Leading `\`, recognised LaTeX macro (e.g. `\frac`, `\int`), or braced exponent `x^{2}` | `latex2sympy2`. Falls back to SymPy `parse_expr` if latex2sympy2 fails. |
| Natural language | Words like "squared", "times", "plus" | Custom token expansion before SymPy parsing |
| Python-style | `x**2`, `math.sin(x)` | SymPy `parse_expr` with standard transformations |
| Plain / default | `x^2 + 2x`, `sin(x)cos(x)` | SymPy `parse_expr` with `implicit_multiplication_application` |

**Validation** (applied before any parsing):
- Length check against `config.MAX_EXPR_LENGTH` (500 chars)
- Forbidden keyword scan: `__`, `import`, `exec`, `eval`, `open`, `os`, `sys`, `subprocess`

**Concurrency:** The blocking SymPy parse runs inside a `ThreadPoolExecutor` (max 4 workers) so the async event loop is never blocked.

**`local_dict` parameter:** Overrides SymPy's default namespace for specific tokens. Used by `algebra.py`'s `solve_sim` command to prevent single-letter names like `E`, `I`, `N`, `O`, `S`, `Q` from being silently interpreted as SymPy built-in constants when the user intends them as variable names.

### 9.2 Plot Parser (`utils/expr_utils.py`)

A lighter, synchronous parser used exclusively by the plot engine. It does not call `latex2sympy2` and does not go async — plots parse expressions inline during modal submission handling.

**Functions:**
```python
_clean_sympy_expr(s: str) -> str    # normalize carets, strip assignment prefixes
_sympy_expr(s: str, *syms) -> sympy.Expr  # sympify with local symbol dict
```

Both functions apply the same `_validate_raw` guard from `parser.py` (imported directly) so the length cap and forbidden-keyword filter apply in all paths.

**Piecewise syntax** supported by `_clean_sympy_expr`:
```
condition: expression | condition: expression
→ Piecewise((expression, condition), (expression, condition))
```

---

## 10. Supporting Utilities

### 10.1 Embed Formatter (`utils/formatter.py`)

All cogs build embeds through these helpers. Never construct a `discord.Embed` manually in a cog.

**`math_embed(title, result, steps=None, footer=None) -> discord.Embed`**

Success embed. Blue (`discord.Color.blurple()`).
- `title` → embed title
- `result` → displayed in the description as a code block
- `steps` → optional `list[tuple[str, str]]` of `(description, expression)` pairs. Rendered as a numbered list inside a single embed field. Auto-truncated at 1024 chars with a notice if needed.
- `footer` → optional footer text

**`error_embed(message: str) -> discord.Embed`**

Red error embed. Used in every cog's `except` blocks.

**`to_readable_text(expr: sympy.Expr) -> str`**

Converts a SymPy expression to a human-readable string. Used in polynomial division output to avoid raw SymPy repr.

### 10.2 Step-by-Step Solver (`utils/solver.py`)

Synchronous functions that return `StepList = list[tuple[str, str]]`. Called by algebra and calculus cogs after the expression is already parsed.

| Function | Used by | Description |
|---|---|---|
| `solve_quadratic_steps(expr, var)` | `/alg solve` | Quadratic formula walkthrough |
| `solve_cubic_steps(expr, var)` | `/alg solve` | Cardano's method steps |
| `solve_quartic_steps(expr, var)` | `/alg solve` | Quartic solution steps |
| `factor_steps(expr)` | `/alg factor` | Factorization walkthrough |
| `differentiate_steps(expr, var, n)` | `/calc diff` | Differentiation rules applied step-by-step |
| `integrate_steps(expr, var)` | `/calc integrate` | Integration technique identification + steps |

Every function wraps its body in a `try/except` and returns `_err(message)` on failure — callers always receive a usable list, never a raised exception.

### 10.3 LaTeX Renderer (`utils/renderer.py`)

Converts LaTeX strings or SymPy expressions into PNG images using matplotlib's `mathtext` renderer.

**Public API:**
```python
async def expr_to_image(latex: str) -> discord.File
async def result_to_image(expr: sympy.Expr) -> discord.File
```

Both functions:
1. Run blocking matplotlib work inside a `ThreadPoolExecutor`
2. Render into an in-memory `io.BytesIO` buffer (no temp files)
3. Return a `discord.File` named `formula.png` ready to attach to any reply

Uses `matplotlib.use("Agg")` (headless backend, set at module import time).

### 10.4 Paginator (`utils/paginator.py`)

**`PaginatorView(pages, timeout=120)`**

A `discord.ui.View` with ◀ and ▶ buttons that edits the original message in-place. Updates the embed footer to `"Page N / Total"` on every turn. Buttons are disabled when at the first or last page.

**`send_paginated(interaction, pages)`** (async convenience function)

- If `len(pages) == 1` → sends a plain `followup.send(embed=pages[0])` (no buttons needed).
- If `len(pages) > 1` → sends with a `PaginatorView`.
- Works with both deferred and non-deferred interactions.

### 10.5 Plot Generator (`utils/plotter.py`)

The rendering back-end for the interactive plot builder. All rendering is async (blocking matplotlib work runs in a `ThreadPoolExecutor`) and thread-isolated via `matplotlib.rc_context`.

**Supported plot types (14):**

| Type | Description |
|---|---|
| `function` | Standard y = f(x) line plot |
| `contour` | Filled contour map of f(x, y) |
| `vector-field` | 2D vector field from u(x,y), v(x,y) |
| `parametric-2d` | x(t), y(t) curve |
| `surface` | 3D surface of f(x, y) |
| `wireframe` | 3D wireframe of f(x, y) |
| `parametric-3d` | x(t), y(t), z(t) space curve |
| `scatter` | 2D or 3D scatter from raw data points |
| `scatter-3d` | 3D scatter |
| `polar` | r(θ) polar curve |
| `implicit` | Implicit equation f(x, y) = c |
| `inequality` | Shaded inequality region f(x, y) ≤/≥ c |
| `riemann` | Riemann sum visualisation (left/right/midpoint/trapezoid) |
| `heatmap` | Color-mapped intensity map of f(x, y) |

**Themes (5):** `default`, `dark`, `academic`, `cyberpunk`, `seaborn`

**Colormaps (19):** `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `coolwarm`, `RdBu`, `seismic`, `Blues`, `Greens`, `Oranges`, `Reds`, `Purples`, `rainbow`, `jet`, `turbo`, `gray`, `bone`, `pink`

---

## 11. Cog Reference

### 11.1 `/admin` — `cogs/admin.py`

**Class:** `AdminCog`  
**Group variable:** `admin_group = app_commands.Group(name="admin", ...)`  
**Guild-only:** Yes (`guild_only=True`)  
**Required permission:** `Manage Guild`

| Subcommand | Parameters | Description |
|---|---|---|
| `/admin enable` | `command?`, `channel?` | Allow a command in a channel or server-wide |
| `/admin disable` | `command?`, `channel?` | Block a command in a channel or server-wide |
| `/admin reset` | `command?`, `channel?` | Remove an existing permission rule |
| `/admin status` | — | Show all active rules for this server |

Both `command` and `channel` are optional. Omitting `command` applies the rule to all commands; omitting `channel` applies it server-wide. Calls `data.permissions.set_permission / clear_permission / get_guild_status`.

---

### 11.2 `/alg` — `cogs/algebra.py`

**Class:** `AlgebraCog`  
**Group variable:** `alg = app_commands.Group(name="alg", ...)`  
**Origin:** Merged from `arithmetic.py` + `equations.py` + `inequalities.py`

Uses the result cache (`data/cache.py`) for pure-math commands. Resolves `$memory` references via `data.memory.memory.resolve()` before parsing.

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/alg simplify` | `expression` | 2s | Simplify expression via `sympy.simplify` |
| `/alg solve` | `expression`, `variable?="x"` | 2s | Solve equation = 0; shows step-by-step for degree 2/3/4 |
| `/alg expand` | `expression` | 2s | Expand via `sympy.expand` |
| `/alg factor` | `expression` | 2s | Factor via `sympy.factor` with step-by-step |
| `/alg solve_sys` | `equations`, `variables?` | 5s | Solve system of equations (linsolve → solve fallback) |
| `/alg table` | `expression`, `start?=-5`, `end?=5`, `step?=1`, `variable?="x"` | 4s | Numeric value table, paginated, max 200 rows |
| `/alg poly_div` | `dividend`, `divisor`, `variable?` | 4s | Polynomial long division via `sympy.div` |
| `/alg verify` | `expr_a`, `expr_b` | 4s | Check equivalence via `sympy.simplify(a - b) == 0` |
| `/alg compare` | `expr_f`, `expr_g` | 4s | Side-by-side comparison of two functions |
| `/alg solve_sim` | `equations`, `variables?` | 4s | Simultaneous equations with reserved-name handling (E, I, N…) |
| `/alg ineq` | `expression`, `variable?` | 2s | Solve inequality via `sympy.solve_univariate_inequality` |
| `/alg ineq_sys` | `expressions`, `variables?` | 4s | System of inequalities via `sympy.reduce_inequalities` |

**Notes:**
- `/alg solve` dispatches to `solve_quadratic_steps`, `solve_cubic_steps`, or `solve_quartic_steps` from `utils/solver.py` based on polynomial degree.
- `/alg solve_sim` pre-scans variable names and overrides SymPy's namespace for any that clash with built-in names (`E`→Euler, `I`→imaginary unit, etc.) to prevent silent mis-parsing.
- `/alg table` uses `sympy.lambdify` + NumPy for fast numeric evaluation over ranges.
- Equations for `/alg solve_sys` and `/alg solve_sim` can use explicit `=` or implicit `= 0` form. Separator is `;` (preferred) or `,`.

---

### 11.3 `/calc` — `cogs/calculus.py`

**Class:** `CalculusCog`  
**Group variable:** `calc = app_commands.Group(name="calc", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/calc diff` | `expression`, `variable?="x"`, `order?=1` | 3s | Differentiate n times; step-by-step via `differentiate_steps` |
| `/calc integrate` | `expression`, `variable?="x"`, `lower?`, `upper?` | 3s | Indefinite or definite integral; step-by-step via `integrate_steps` |
| `/calc limit` | `expression`, `variable?="x"`, `point?="0"`, `direction?="two-sided"` | 3s | Limit via `sympy.limit` |
| `/calc series` | `expression`, `variable?="x"`, `point?=0`, `terms?=6` | 3s | Taylor/Maclaurin series via `sympy.series` |
| `/calc sum` | `expression`, `variable?="n"`, `lower?=1`, `upper?=∞` | 5s | Summation Σ via `sympy.summation` |
| `/calc product` | `expression`, `variable?="n"`, `lower?=1`, `upper?` | 5s | Product Π via `sympy.product` |
| `/calc ode` | `equation`, `function?="y(x)"` | 5s | ODE solving via `sympy.dsolve` |
| `/calc gradient` | `expression`, `variables?="x,y"` | 3s | Gradient vector |
| `/calc divergence` | `u`, `v`, `w?`, `variables?="x,y,z"` | 3s | Divergence of a vector field |
| `/calc curl` | `u`, `v`, `w?`, `variables?="x,y,z"` | 3s | Curl of a vector field |

---

### 11.4 `/stat` — `cogs/statistics.py`

**Class:** `StatisticsCog`  
**Group variable:** `stat = app_commands.Group(name="stat", ...)`  
Uses `scipy.stats` for distribution functions and `matplotlib` for distribution plots.

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/stat mean` | `data` | 2s | Arithmetic mean |
| `/stat median` | `data` | 2s | Median |
| `/stat mode` | `data` | 2s | Mode (most frequent value) |
| `/stat stdev` | `data`, `population?=False` | 2s | Standard deviation |
| `/stat variance` | `data`, `population?=False` | 2s | Variance |
| `/stat zscore` | `value`, `mean`, `stdev` | 2s | Z-score calculation |
| `/stat correlation` | `x_data`, `y_data` | 3s | Pearson correlation coefficient + scatter plot |
| `/stat regression` | `x_data`, `y_data` | 3s | Linear regression + plot |
| `/stat distribution` | `type` (choice), `params...` | 3s | Unified distribution command (choice-based) |
| `/stat normal_pdf` | `x`, `mean?=0`, `stdev?=1` | 2s | Normal PDF with bell curve plot |
| `/stat normal_cdf` | `upper`, `mean?=0`, `stdev?=1` | 2s | Normal CDF (P(X ≤ upper)) with area plot |
| `/stat inv_normal` | `probability`, `mean?=0`, `stdev?=1` | 2s | Inverse normal (quantile) |
| `/stat binomial_cdf` | `n`, `p`, `x` | 2s | Binomial CDF P(X ≤ x) with bar chart |
| `/stat poisson_cdf` | `lam`, `x` | 2s | Poisson CDF P(X ≤ x) with bar chart |

`data` parameters accept comma-separated numbers: `"1.2, 3.4, 5.6"`.

---

### 11.5 `/nt` — `cogs/number_theory.py`

**Class:** `NumberTheoryCog`  
**Group variable:** `nt = app_commands.Group(name="nt", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/nt gcd` | `a`, `b` | 2s | GCD via `sympy.gcd` |
| `/nt lcm` | `a`, `b` | 2s | LCM via `sympy.lcm` |
| `/nt is_prime` | `n` | 2s | Primality test via `sympy.isprime` |
| `/nt factorize` | `n` | 3s | Prime factorization via `sympy.factorint` |
| `/nt primes` | `n` | 3s | All primes up to n (capped; paginated) |
| `/nt modular` | `a`, `operation`, `b`, `mod` | 2s | Modular arithmetic (add/sub/mul/pow/inv) |
| `/nt fibonacci` | `n` | 2s | nth Fibonacci number via `sympy.fibonacci` |
| `/nt totient` | `n` | 2s | Euler's totient φ(n) |
| `/nt divisors` | `n` | 2s | All divisors of n |
| `/nt is_perfect` | `n` | 2s | Perfect number check |
| `/nt mobius` | `n` | 2s | Möbius function μ(n) |
| `/nt crt` | `remainders`, `moduli` | 3s | Chinese Remainder Theorem via `sympy.crt` |

---

### 11.6 `/mat` — `cogs/linear_algebra.py`

**Class:** `LinearAlgebraCog`  
**Group variable:** `mat = app_commands.Group(name="mat", ...)`  
Matrix input format: `[[a,b],[c,d]]`. Vector input: `[a,b,c]`. Parsed via `data.memory.parse_matrix`.

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/mat det` | `matrix` | 3s | Determinant via `sympy.Matrix.det()` |
| `/mat inv` | `matrix` | 3s | Matrix inverse via `sympy.Matrix.inv()` |
| `/mat eigen` | `matrix` | 3s | Eigenvalues and eigenvectors |
| `/mat dot` | `vector_a`, `vector_b` | 2s | Dot product |
| `/mat cross` | `vector_a`, `vector_b` | 2s | Cross product (3D vectors) |
| `/mat rref` | `matrix` | 3s | Reduced row echelon form |

---

### 11.7 `/sym` — `cogs/symbolic.py`

**Class:** `SymbolicCog`  
**Group variable:** `sym = app_commands.Group(name="sym", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/sym latex` | `expression` | 2s | Convert expression to LaTeX string via `sympy.latex` |
| `/sym subs` | `expression`, `variable`, `value` | 2s | Substitute a value into an expression |
| `/sym partial` | `expression`, `variable?` | 3s | Partial fraction decomposition via `sympy.apart` |
| `/sym identify` | `expression` | 3s | Identify the expression (closed form, type, properties) |
| `/sym roots` | `expression`, `variable?` | 3s | All roots (real and complex) via `sympy.roots` |

---

### 11.8 `/geo` — `cogs/geometry.py`

**Class:** `GeometryCog`  
**Group variable:** `geo = app_commands.Group(name="geo", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/geo circle` | `radius` | 2s | Circle area = πr² |
| `/geo circumference` | `radius` | 2s | Circumference = 2πr |
| `/geo triangle` | `base`, `height` (or `a`, `b`, `c` sides) | 2s | Triangle area (base-height or Heron's formula) |
| `/geo pythagorean` | `a?`, `b?`, `c?` | 2s | Pythagorean theorem — provide any two sides |
| `/geo trig` | `function` (choice), `angle`, `mode` (deg/rad) | 2s | Trigonometric function evaluation |
| `/geo distance` | `x1`, `y1`, `x2`, `y2` | 2s | Euclidean distance between two points |

---

### 11.9 `/dis` — `cogs/discrete.py`

**Class:** `DiscreteCog`  
**Group variable:** `dis = app_commands.Group(name="dis", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/dis permutation` | `n`, `r` | 2s | P(n,r) = n!/(n-r)! |
| `/dis combination` | `n`, `r` | 2s | C(n,r) = n!/(r!(n-r)!) |
| `/dis truth_table` | `expression` | 3s | Full truth table for a Boolean expression |
| `/dis set_ops` | `set_a`, `set_b`, `operation` (choice) | 2s | Union, intersection, difference, symmetric difference |
| `/dis binomial` | `n`, `k` | 2s | Binomial coefficient C(n,k) |
| `/dis bool` | `expression`, `form` (choice) | 3s | Boolean simplification (simplified / DNF / CNF) |
| `/dis logic_equiv` | `expr_a`, `expr_b` | 3s | Check logical equivalence of two Boolean expressions |

---

### 11.10 `/cx` — `cogs/complex.py`

**Class:** `ComplexCog`  
**Group variable:** `cx = app_commands.Group(name="cx", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/cx calc` | `expression` | 2s | Evaluate complex expression |
| `/cx polar` | `real`, `imag` | 2s | Rectangular → polar (r, θ) |
| `/cx rect` | `r`, `theta`, `mode` (deg/rad) | 2s | Polar → rectangular (a + bi) |
| `/cx conj` | `expression` | 2s | Complex conjugate |
| `/cx modulus` | `expression` | 2s | Modulus |z| |

---

### 11.11 `/tf` — `cogs/transforms.py`

**Class:** `TransformsCog`  
**Group variable:** `tf = app_commands.Group(name="tf", ...)`  
Uses `sympy.laplace_transform`, `sympy.inverse_laplace_transform`, `sympy.fourier_transform`, `sympy.inverse_fourier_transform`.

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/tf laplace` | `expression`, `variable?="t"`, `transform_var?="s"` | 4s | Laplace transform |
| `/tf inv_laplace` | `expression`, `variable?="s"`, `transform_var?="t"` | 4s | Inverse Laplace transform |
| `/tf fourier` | `expression`, `variable?="t"`, `transform_var?="w"` | 4s | Fourier transform |
| `/tf inv_fourier` | `expression`, `variable?="w"`, `transform_var?="t"` | 4s | Inverse Fourier transform |

---

### 11.12 `/base` — `cogs/base_n.py`

**Class:** `BaseNCog`  
**Group variable:** `base = app_commands.Group(name="base", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/base convert` | `number`, `from_base`, `to_base` | 2s | Convert number between arbitrary bases (2–36) |
| `/base add` | `a`, `b`, `base` | 2s | Add two numbers in a given base |
| `/base logic` | `a`, `b`, `operation` (AND/OR/XOR/NOT), `base?=2` | 2s | Bitwise logic operation |
| `/base table` | `number`, `from_base?=10` | 2s | Display number in binary, octal, decimal, hex simultaneously |

---

### 11.13 `/mem` — `cogs/memory.py`

**Class:** `MemoryCog`  
**Group variable:** `mem = app_commands.Group(name="mem", ...)`

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/mem set` | `name`, `value` | 2s | Store a named variable (number, expression, or matrix) |
| `/mem get` | `name` | 1s | Display the stored variable |
| `/mem list` | — | 2s | List all stored variables |
| `/mem del` | `name` | 2s | Delete a stored variable |
| `/mem clear` | — | 5s | Delete all stored variables (with confirmation) |
| `/mem eval` | `expression` | 2s | Evaluate an expression with `$name` references inline |

---

### 11.14 `/bot` — `cogs/bot.py`

**Class:** `BotCog`  
**Group variable:** `bot_grp = app_commands.Group(name="bot", ...)`  
**Origin:** Merged from `utility.py` + `wiki.py`

The `BotCog` manages an `aiohttp.ClientSession` for Wikipedia requests. The session is created in `cog_load` and closed in `cog_unload`. A `@property` auto-recreates a closed session.

**Unit conversion sub-system** (used by `/bot convert` and `/bot units`):

`/bot convert` handles 10 unit categories with a fast lookup table:
- **Length:** m, km, cm, ft, mile, inch
- **Mass:** kg, g, lb, oz
- **Temperature:** C, F, K (handled via exact-rational Kelvin arithmetic)
- **Time:** s, min, hr, day, week, year
- **Area:** m2, km2, cm2, hectare, acre, sq_ft, sq_mile
- **Volume:** l, ml, m3, cm3, gal, qt, pt, cup, fl_oz
- **Speed:** mps, kph, mph, knot, fps
- **Force:** n, lbf, dyne
- **Energy:** j, kj, cal, kcal, wh, kwh, btu, ev
- **Power:** w, kw, mw, hp

`/bot units` uses `sympy.physics.units.convert_to` for compound/derived units (e.g. `m/s^2`, `kg*m/s^2`). Parses unit expressions by tokenising on `*`, `/`, `^`.

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/bot history` | — | 2s | Paginated calculation history (last 20, 5/page) |
| `/bot clear` | — | 5s | Clear calculation history (ephemeral Yes/No confirmation) |
| `/bot constants` | — | 2s | Reference embed: π, e, φ, √2, i, ∞ with 10-decimal values |
| `/bot help` | — | 2s | Paginated help — one page per cog listing all subcommands |
| `/bot convert` | `value`, `from_unit`, `to_unit` | 2s | Simple unit conversion (10 categories) |
| `/bot units` | `value`, `from_unit`, `to_unit` | 3s | Compound unit conversion via sympy.physics.units |
| `/bot about` | — | 5s | Bot version, library versions, guild count, uptime |
| `/bot wiki` | `topic` | 3s | Fetch Wikipedia article (full sectioned, paginated) |
| `/bot wiki_search` | `topic` | 3s | Search Wikipedia, list top 5 results |

**Wikipedia API used:** `https://en.wikipedia.org/api/rest_v1/` (no key required). Rate limit: 200 req/s per client. `/bot wiki` fetches the mobile-sections endpoint for full content; `/bot wiki_search` uses the `action=query&list=search` endpoint.

---

### 11.15 `/render` — `cogs/render.py`

**Class:** `RenderCog`  
**Group variable:** `render_grp = app_commands.Group(name="render", ...)`

(Group variable is `render_grp` rather than `render` to avoid shadowing the `render` method name at the class level.)

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/render expr` | `latex` | 3s | Render a raw LaTeX string as PNG via `utils/renderer.py` |
| `/render formula` | `expression` | 3s | Parse expression via `utils/parser.py`, then render to PNG |

Both commands return a PNG image as a Discord file attachment.

---

### 11.16 `/plot` — `cogs/plot_engine.py`

**Class:** `PlotEngineCog`  
**Group variable:** `plot_grp = app_commands.Group(name="plot", ...)`

(Group variable is `plot_grp` to avoid shadowing the `plot_start` method name.)

The plot engine is the most complex cog in the project. It provides an **interactive modal-based** plot builder where users configure a plot through a series of Discord modals, then trigger rendering.

**`PlotConfig` dataclass** is the single source of truth for a build session. Key fields:
- `plot_type` — one of 14 types
- `expr_main` — primary expression string
- `x_min`, `x_max`, `y_min`, `y_max`, `t_min`, `t_max` — axis ranges
- `theme`, `colormap`, `line_style`, `marker` — visual style
- `title`, `xlabel`, `ylabel` — labels
- `riemann_n`, `riemann_method` — Riemann sum config
- `additional_exprs` — extra expressions for multi-function plots
- `implicit_rhs`, `inequality_op` — for implicit/inequality plots

`PlotConfig` supports JSON export/import (`export_config()` / `import_config()`) for session sharing and `/plot load`.

**`PlotEngineView`** is the main `discord.ui.View` driving the builder. It holds a `PlotConfig` instance and spawns **modal sub-views** for each configuration section:
- `ExpressionModal` — expression input
- `AxisModal` — domain/range configuration
- `StyleModal` — theme, colormap, labels
- `ScatterModal` — data point input for scatter plots
- `RiemannModal` — Riemann sum parameters
- `PolarModal` — polar plot configuration

| Subcommand | Parameters | Cooldown | Description |
|---|---|---|---|
| `/plot start` | `type` (choice, 14 options) | 5s | Open the interactive plot builder modal for a plot type |
| `/plot quick` | `expression`, `x_min?=-10`, `x_max?=10` | 3s | Fast function plot without the interactive builder |
| `/plot multi` | `expressions`, `x_min?=-10`, `x_max?=10` | 3s | Plot multiple functions on one axes (comma-separated) |
| `/plot load` | `config_string` | 5s | Import a previously exported `PlotConfig` JSON string |

---

## 12. Complete Command Reference

### Standalone
| Command | Description |
|---|---|
| `/ping` | Bot latency in milliseconds |

### /admin (4)
| Command | Description |
|---|---|
| `/admin enable` | Allow a command in a channel or server-wide |
| `/admin disable` | Block a command in a channel or server-wide |
| `/admin reset` | Remove an existing permission rule |
| `/admin status` | Show all active permission rules |

### /alg (12)
| Command | Description |
|---|---|
| `/alg simplify` | Simplify an expression |
| `/alg solve` | Solve equation = 0, step-by-step for degree 2/3/4 |
| `/alg expand` | Expand / distribute |
| `/alg factor` | Factor with steps |
| `/alg solve_sys` | Solve a system of equations |
| `/alg table` | Numeric value table |
| `/alg poly_div` | Polynomial long division |
| `/alg verify` | Check symbolic equivalence of two expressions |
| `/alg compare` | Side-by-side function comparison |
| `/alg solve_sim` | Simultaneous equations (clean x=…, y=… output) |
| `/alg ineq` | Solve a single inequality |
| `/alg ineq_sys` | Solve a system of inequalities |

### /calc (10)
| Command | Description |
|---|---|
| `/calc diff` | Differentiate (nth order, step-by-step) |
| `/calc integrate` | Integrate (indefinite or definite, step-by-step) |
| `/calc limit` | Limit at a point |
| `/calc series` | Taylor/Maclaurin series |
| `/calc sum` | Summation Σ |
| `/calc product` | Product Π |
| `/calc ode` | Ordinary differential equation |
| `/calc gradient` | Gradient vector |
| `/calc divergence` | Divergence of a vector field |
| `/calc curl` | Curl of a vector field |

### /stat (14)
| Command | Description |
|---|---|
| `/stat mean` | Arithmetic mean |
| `/stat median` | Median |
| `/stat mode` | Mode |
| `/stat stdev` | Standard deviation |
| `/stat variance` | Variance |
| `/stat zscore` | Z-score |
| `/stat correlation` | Pearson correlation + scatter plot |
| `/stat regression` | Linear regression + plot |
| `/stat distribution` | Unified distribution command (choice-based) |
| `/stat normal_pdf` | Normal PDF with plot |
| `/stat normal_cdf` | Normal CDF P(X ≤ x) with area plot |
| `/stat inv_normal` | Inverse normal / quantile |
| `/stat binomial_cdf` | Binomial CDF P(X ≤ x) with bar chart |
| `/stat poisson_cdf` | Poisson CDF P(X ≤ x) with bar chart |

### /nt (12)
| Command | Description |
|---|---|
| `/nt gcd` | Greatest common divisor |
| `/nt lcm` | Least common multiple |
| `/nt is_prime` | Primality test |
| `/nt factorize` | Prime factorization |
| `/nt primes` | All primes up to n |
| `/nt modular` | Modular arithmetic |
| `/nt fibonacci` | nth Fibonacci number |
| `/nt totient` | Euler's totient function |
| `/nt divisors` | All divisors |
| `/nt is_perfect` | Perfect number check |
| `/nt mobius` | Möbius function μ(n) |
| `/nt crt` | Chinese Remainder Theorem |

### /mat (6)
| Command | Description |
|---|---|
| `/mat det` | Matrix determinant |
| `/mat inv` | Matrix inverse |
| `/mat eigen` | Eigenvalues and eigenvectors |
| `/mat dot` | Dot product |
| `/mat cross` | Cross product |
| `/mat rref` | Reduced row echelon form |

### /sym (5)
| Command | Description |
|---|---|
| `/sym latex` | Convert expression to LaTeX string |
| `/sym subs` | Substitute a value into an expression |
| `/sym partial` | Partial fraction decomposition |
| `/sym identify` | Identify expression type and properties |
| `/sym roots` | Find all roots |

### /geo (6)
| Command | Description |
|---|---|
| `/geo circle` | Circle area |
| `/geo circumference` | Circle circumference |
| `/geo triangle` | Triangle area |
| `/geo pythagorean` | Pythagorean theorem (any two sides) |
| `/geo trig` | Trigonometric function evaluation |
| `/geo distance` | Distance between two points |

### /dis (7)
| Command | Description |
|---|---|
| `/dis permutation` | Permutation P(n,r) |
| `/dis combination` | Combination C(n,r) |
| `/dis truth_table` | Boolean truth table |
| `/dis set_ops` | Set operations (union, intersection, difference, symmetric difference) |
| `/dis binomial` | Binomial coefficient C(n,k) |
| `/dis bool` | Boolean simplification (simplified / DNF / CNF) |
| `/dis logic_equiv` | Check logical equivalence |

### /cx (5)
| Command | Description |
|---|---|
| `/cx calc` | Evaluate complex expression |
| `/cx polar` | Rectangular → polar |
| `/cx rect` | Polar → rectangular |
| `/cx conj` | Complex conjugate |
| `/cx modulus` | Complex modulus |z| |

### /tf (4)
| Command | Description |
|---|---|
| `/tf laplace` | Laplace transform |
| `/tf inv_laplace` | Inverse Laplace transform |
| `/tf fourier` | Fourier transform |
| `/tf inv_fourier` | Inverse Fourier transform |

### /base (4)
| Command | Description |
|---|---|
| `/base convert` | Convert between bases 2–36 |
| `/base add` | Add in an arbitrary base |
| `/base logic` | Bitwise AND / OR / XOR / NOT |
| `/base table` | Display a number in binary, octal, decimal, hex |

### /mem (6)
| Command | Description |
|---|---|
| `/mem set` | Store a named variable |
| `/mem get` | Retrieve a stored variable |
| `/mem list` | List all stored variables |
| `/mem del` | Delete a stored variable |
| `/mem clear` | Clear all stored variables |
| `/mem eval` | Evaluate an expression with $name references |

### /bot (9)
| Command | Description |
|---|---|
| `/bot history` | Show calculation history |
| `/bot clear` | Clear calculation history |
| `/bot constants` | Mathematical constants reference |
| `/bot help` | Paginated command help |
| `/bot convert` | Simple unit conversion |
| `/bot units` | Compound/derived unit conversion |
| `/bot about` | Bot info and library versions |
| `/bot wiki` | Fetch and browse a Wikipedia article |
| `/bot wiki_search` | Search Wikipedia |

### /render (2)
| Command | Description |
|---|---|
| `/render expr` | Render raw LaTeX as PNG |
| `/render formula` | Parse expression, then render as PNG |

### /plot (4)
| Command | Description |
|---|---|
| `/plot start` | Open interactive plot builder |
| `/plot quick` | Fast single-function plot |
| `/plot multi` | Plot multiple functions |
| `/plot load` | Load a previously exported plot config |

---

## 13. Data Flow

### Standard computation command

```
User types /alg solve x**2 - 4
     │
     ▼
bot.tree.interaction_check
     │  guild_id + channel_id + "solve" → permissions.is_command_allowed()
     │  → allowed
     ▼
AlgebraCog.solve (interaction)
     │
     ├─ interaction.response.defer()
     │
     ├─ memory.resolve(guild_id, user_id, "x**2 - 4")
     │    └─ no $refs → returns unchanged
     │
     ├─ parse_expression("x**2 - 4")          [utils/parser.py]
     │    ├─ _validate_raw() → OK
     │    ├─ detect format → plain
     │    └─ ThreadPoolExecutor → sympy.parse_expr(...) → Expr
     │
     ├─ sympy.solve(expr, x) → [2, -2]
     │
     ├─ solve_quadratic_steps(expr, x)         [utils/solver.py]
     │    └─ [(description, expression), ...]
     │
     ├─ math_embed(title, result, steps)       [utils/formatter.py]
     │    └─ discord.Embed
     │
     └─ interaction.followup.send(embed=embed)
```

### Interactive plot builder

```
User: /plot start function
     │
     ▼
PlotEngineCog.plot_start
     │
     ├─ PlotConfig(plot_type="function")
     │
     └─ PlotEngineView(cfg)
          │  sends initial config embed + button row
          │
          ├─ [User clicks "Set Expression"] → ExpressionModal
          │       └─ modal.on_submit() → cfg.expr_main = user_input
          │                           → view.refresh_embed()
          │
          ├─ [User clicks "Set Axes"] → AxisModal
          │       └─ modal.on_submit() → cfg.x_min/x_max = ...
          │
          ├─ [User clicks "Render"]
          │       └─ _render(cfg)                  [plot_engine.py]
          │             └─ plotter.plot_function()  [utils/plotter.py]
          │                   ├─ lambdify expr
          │                   ├─ np.linspace(x_min, x_max, 800)
          │                   ├─ matplotlib render (rc_context, Agg)
          │                   └─ io.BytesIO → discord.File
          │
          └─ interaction.followup.send(file=plot_file, embed=summary_embed)
```

---

## 14. Security Model

### Input validation

Every expression goes through `_validate_raw` in `utils/parser.py` before any parsing:
- **Length cap:** 500 characters (`config.MAX_EXPR_LENGTH`). Prevents memory abuse from large inputs.
- **Forbidden keywords:** `__`, `import`, `exec`, `eval`, `open`, `os`, `sys`, `subprocess`. These are scanned via regex word-boundary matching and rejected immediately.

### Compute timeout

`config.COMPUTE_TIMEOUT = 3` seconds. SymPy can hang indefinitely on pathological inputs (e.g. a symbolic integral with no closed form). The timeout prevents a single user from monopolising the event loop.

### Permission system

Guild admins can restrict any command to specific channels or disable it server-wide. The check runs **before** the cog handler, so disabled commands never reach SymPy. See [§7](#7-permission-system).

### Rate limiting

Every subcommand carries a per-user `@app_commands.checks.cooldown(rate, per)`. Common values:
- Pure math: `cooldown(1, 2.0)` — 1 use per 2 seconds
- Heavy math (ODE, summation): `cooldown(1, 5.0)` — 1 use per 5 seconds
- Plot build: `cooldown(1, 5.0)`

Cooldown violations are caught by the global error handler and returned as an ephemeral "try again in Xs" message.

### Admin commands

`/admin` subcommands require `Manage Guild` permission, enforced by `AdminCog._require_manage_guild(interaction)` inside each handler (not as a decorator, so the check message can be customised).

### Wikipedia requests

`/bot wiki` and `/bot wiki_search` use the public Wikipedia REST API with a `User-Agent` header identifying the bot. A 10-second `aiohttp.ClientTimeout` prevents hanging on slow responses. The aiohttp session is reused for the bot's lifetime and recreated if closed.

---

## 15. Conventions & Patterns

### Defer immediately

Every non-ephemeral command calls `await interaction.response.defer()` as its first statement. Discord's interaction timeout is 3 seconds; `defer()` extends this to 15 minutes for the followup.

### Cache pattern

```python
key    = cache_key("command_name", *significant_params)
cached = get(key)
if cached is not None:
    await interaction.followup.send(embed=cached)
    return
# ... compute ...
set(key, embed)
```

Only deterministic math commands (simplify, expand, factor, diff, integrate, limit, etc.) use the cache. Commands involving user state (history, memory) or external data (Wikipedia) do not.

### Error handling

Every command handler wraps its body in:
```python
except ValueError as exc:          # raised by validators and parsers
except sympy.SympifyError as exc:  # unparseable expression
except sympy.PolynomialError as exc:
except NotImplementedError:        # SymPy couldn't find closed form
except Exception as exc:           # catch-all
```
Each `except` sends `error_embed(str(exc))` as a followup.

### Memory $-references

Commands in `algebra.py` that accept expression strings call `memory.resolve()` before parsing. Commands that accept variable names (e.g. the `variable` param in `/alg solve`) do **not** pass those through `resolve()` — bare variable names are always treated as free symbols, never as memory lookups.

### Merged cog names

- `algebra.py` → class `AlgebraCog`, group var `alg`
- `bot.py` → class `BotCog`, group var `bot_grp`

### Name-conflict avoidance

When a group's Discord name matches an existing method name in the class:
- `render.py`: group var is `render_grp` (not `render`), Discord group name is still `"render"` 
- `plot_engine.py`: group var is `plot_grp` (not `plot`), Discord group name is still `"plot"`

---

## 16. Dependency Reference

| Package | Version req. | Used for |
|---|---|---|
| `discord.py` | ≥ 2.5.0 | Bot framework, slash commands, modals, views |
| `python-dotenv` | ≥ 1.0.0 | `.env` file loading in `config.py` |
| `sympy` | ≥ 1.12 | All symbolic math (algebra, calculus, number theory, etc.) |
| `latex2sympy2` | ≥ 0.4.0 | LaTeX → SymPy parsing in `utils/parser.py` |
| `cachetools` | ≥ 5.3.0 | `TTLCache` in `data/cache.py` |
| `matplotlib` | ≥ 3.8.0 | Plot rendering (Agg backend), LaTeX → PNG rendering |
| `numpy` | ≥ 1.25.0 | Fast numeric evaluation (`lambdify`), array operations |
| `scipy` | ≥ 1.11.0 | Statistical distributions (`scipy.stats`) |
| `aiohttp` | (discord.py dep) | Wikipedia HTTP requests in `cogs/bot.py` |

---

## 17. Codebase Metrics

| Metric | Value |
|---|---|
| Total Python files | 23 |
| Total lines of code (approx.) | ~8,700 |
| Cogs | 16 |
| Top-level Discord slots | 17 (16 groups + `/ping`) |
| Total subcommands | 110 |
| Discord slots remaining | 83 / 100 |
| Max subcommands remaining (current groups) | 65 across 16 groups |
| Theoretical max future subcommands | ~2,140 (83 new groups × 25 + 65 in existing) |
| In-memory data stores | 4 (`cache`, `history`, `permissions`, `memory`) |
| File-backed data stores | 1 (`guild_permissions.json`) |
| External API dependencies | 2 (Discord, Wikipedia) |
| Plot types | 14 |
| Supported unit categories | 10 |
| Supported base range | 2–36 |
