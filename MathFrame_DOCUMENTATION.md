# MathFrame — Complete Technical Documentation

> A Discord bot for symbolic and numeric mathematics, built on `discord.py` (slash commands), `SymPy`, `matplotlib`, `numpy`, and `scipy`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration Reference](#4-configuration-reference)
5. [Bot Bootstrapping & Lifecycle](#5-bot-bootstrapping--lifecycle)
6. [Permission & Admin System](#6-permission--admin-system)
7. [Data Layer](#7-data-layer)
   - [In-Memory Result Cache (`data/cache.py`)](#71-in-memory-result-cache-datacachepy)
   - [Per-User History (`data/history.py`)](#72-per-user-history-datahistorypy)
   - [Guild Permission Store (`data/permissions.py`)](#73-guild-permission-store-datapermissionspy)
8. [Expression Parsing System](#8-expression-parsing-system)
   - [Primary Parser (`utils/parser.py`)](#81-primary-parser-utilsparserpy)
   - [Plot Parser (`utils/expr_utils.py`)](#82-plot-parser-utilsexpr_utilspy)
9. [Supporting Utilities](#9-supporting-utilities)
   - [Embed Formatter (`utils/formatter.py`)](#91-embed-formatter-utilsformatterpy)
   - [Step-by-Step Solver (`utils/solver.py`)](#92-step-by-step-solver-utilssolverpy)
   - [LaTeX Renderer (`utils/renderer.py`)](#93-latex-renderer-utilsrendererpy)
   - [Paginator (`utils/paginator.py`)](#94-paginator-utilspaginatorpy)
   - [Plot Engine (`utils/plotter.py`)](#95-plot-engine-utilsplotterpy)
   - [Dead Code: `utils/utility.py`](#96-dead-code-utilsutilitypy)
10. [Cog Reference](#10-cog-reference)
    - [Admin (`cogs/admin.py`)](#101-admin-cogsadminpy)
    - [Arithmetic (`cogs/arithmetic.py`)](#102-arithmetic-cogsarithmeticpy)
    - [Calculus (`cogs/calculus.py`)](#103-calculus-cogscalculuspy)
    - [Transforms (`cogs/transforms.py`)](#104-transforms-cogstransformspy)
    - [Linear Algebra (`cogs/linear_algebra.py`)](#105-linear-algebra-cogslinear_algebrapy)
    - [Statistics (`cogs/statistics.py`)](#106-statistics-cogsstatisticspy)
    - [Number Theory (`cogs/number_theory.py`)](#107-number-theory-cogsnumber_theorypy)
    - [Geometry (`cogs/geometry.py`)](#108-geometry-cogsgeometrypy)
    - [Discrete Math (`cogs/discrete.py`)](#109-discrete-math-cogsdiscretepy)
    - [Symbolic (`cogs/symbolic.py`)](#1010-symbolic-cogssymbolicpy)
    - [Equations (`cogs/equations.py`)](#1011-equations-cogsequationspy)
    - [Inequalities (`cogs/inequalities.py`)](#1012-inequalities-cogsinequalitiespy)
    - [Complex Numbers (`cogs/complex.py`)](#1013-complex-numbers-cogscomplexpy)
    - [Base-N Arithmetic (`cogs/base_n.py`)](#1014-base-n-arithmetic-cogsbase_npy)
    - [Utility (`cogs/utility.py`)](#1015-utility-cogsutilitypy)
    - [Rendering (`cogs/render.py`)](#1016-rendering-cogsrenderpy)
    - [Plot Builder (`cogs/plot_engine.py`)](#1017-plot-builder-cogsplot_enginepy)
    - [Wikipedia (`cogs/wiki.py`)](#1018-wikipedia-cogswikipy)
11. [Complete Command Reference](#11-complete-command-reference)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [Security Model](#13-security-model)
14. [Known Issues & Open Items](#14-known-issues--open-items)
15. [Dependency Reference](#15-dependency-reference)
16. [Codebase Metrics](#16-codebase-metrics)

---

## 1. Project Overview

MathFrame is a full-featured Discord mathematics bot that exposes over 60 slash commands covering:

- **Arithmetic & Algebra** — simplification, solving, factoring, expansion, polynomial division
- **Calculus** — differentiation, integration, limits, Taylor/Maclaurin series, summations, products, ODEs
- **Integral Transforms** — Laplace, Inverse Laplace, Fourier, Inverse Fourier
- **Linear Algebra** — determinants, inverses, eigenvalues, dot/cross products, RREF
- **Statistics** — descriptive stats, correlation, regression, and a unified probability distribution command
- **Number Theory** — GCD/LCM, primality, factorization, modular arithmetic, Fibonacci, totient, divisors, Möbius, CRT
- **Geometry** — circle, triangle, Pythagorean theorem, trigonometry, distance
- **Discrete Math** — permutations, combinations, truth tables, set operations, Boolean simplification, logic equivalence
- **Symbolic Math** — LaTeX rendering, substitution, partial fractions, root finding, expression identification
- **Complex Numbers** — evaluation, polar/rectangular conversion
- **Base-N Arithmetic** — base conversion, arithmetic in arbitrary bases, bitwise logic
- **Inequality Solving** — symbolic inequality reduction over reals
- **Interactive Plotting** — 13 plot types with a modal-based interactive builder, animation, and session sharing
- **Utility** — history, unit conversion, constants reference, help
- **Wikipedia** — article browsing and search

The architecture is a classic **cog-based discord.py bot**: `main.py` is the single entry point that loads 18 independent feature modules ("cogs"), each contributing a family of slash commands. All commands are slash (`/`) commands registered globally with Discord's Application Command API.

---

## 2. Repository Structure

```
MathFrame/
├── main.py                     Bot entry point, cog loader, global error handler
├── config.py                   Environment-driven configuration constants
├── requirements.txt            Python package dependencies
├── .gitignore
│
├── data/
│   ├── __init__.py
│   ├── cache.py                In-memory TTL result cache (thread-safe singleton)
│   ├── history.py              In-memory per-user command history (thread-safe singleton)
│   └── permissions.py          Guild-level command permission store (file-backed)
│
├── utils/
│   ├── __init__.py
│   ├── parser.py               THE expression parser — format detection, validation, async parse
│   ├── expr_utils.py           Lighter-weight parser for the plotting subsystem
│   ├── solver.py               Step-by-step solution builders (quadratic, cubic, quartic, diff, integral, factor)
│   ├── formatter.py            Discord embed builders (success / error / info)
│   ├── renderer.py             LaTeX/SymPy → PNG image rendering via matplotlib mathtext
│   ├── paginator.py            Reusable ◀/▶ embed paginator view
│   ├── plotter.py              Low-level plotting engine (2,246 lines of matplotlib figure builders)
│   └── utility.py              ⚠ DEAD CODE — orphaned earlier draft, not loaded anywhere
│
├── cogs/
│   ├── __init__.py
│   ├── admin.py                /admin enable/disable/reset/status — permission management
│   ├── arithmetic.py           /simplify /solve /expand /factor /table /poly_div /verify /compare
│   ├── calculus.py             /diff /integrate /limit /series /sum_series /product_series /ode
│   ├── transforms.py           /laplace /inv_laplace /fourier /inv_fourier
│   ├── linear_algebra.py       /matrix_det /matrix_inv /eigenvalues /dot /cross /rref
│   ├── statistics.py           /mean /median /mode /stdev /variance /zscore /correlation /regression /distribution
│   ├── number_theory.py        /gcd /lcm /is_prime /factorize /primes_up_to /modular /fibonacci
│   │                           /totient /divisors /is_perfect /mobius /chinese_remainder
│   ├── geometry.py             /circle_area /circle_circumference /triangle_area /pythagorean /trig /distance
│   ├── discrete.py             /permutation /combination /truth_table /set_ops /binomial_coeff
│   │                           /simplify_bool /logic_equiv
│   ├── symbolic.py             /latex /subs /partial_fraction /roots /identify
│   ├── equations.py            /solve_sim
│   ├── inequalities.py         /solve_ineq
│   ├── complex.py              /complex_eval /complex_polar /complex_rect
│   ├── base_n.py               /base_convert /base_math /base_logic
│   ├── utility.py              /history /clear_history /constants /help_math /convert /about
│   ├── render.py               /render /formula
│   ├── plot_engine.py          /plot /quickplot /multiplot /plot_import  (1,779 lines)
│   └── wiki.py                 /wiki /wiki_search
│
└── utils/
    └── ChatGPT Image Jun 15, 2026, 11_57_30 AM.png   ⚠ Stray binary; should be removed
```

**Cog loading order** (defined in `main.py::COGS`) matters because earlier cogs could in principle depend on shared utilities. The order is:

```
admin → arithmetic → calculus → transforms → linear_algebra → statistics
→ number_theory → geometry → discrete → symbolic → equations → inequalities
→ complex → base_n → utility → render → plot_engine → wiki
```

---

## 3. Installation & Setup

### Prerequisites

- Python 3.12+ (the codebase uses `list[str]`, `dict[int, deque]`, `int | None` union syntax, and `__slots__`)
- A Discord Application with a Bot token and the **Message Content Intent** enabled in the Developer Portal

### Steps

```bash
# 1. Clone or unzip the repository
cd MathFrame

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env               # if provided, otherwise create manually
# Edit .env and set:
#   DISCORD_TOKEN=your_bot_token_here

# 5. Start the bot
python main.py
```

### `.env` file format

```env
DISCORD_TOKEN=your_discord_bot_token_here
```

No other environment variables are required; all other configuration lives in `config.py` with hardcoded defaults.

### Bot Intents Required

| Intent | Required | Purpose |
|--------|----------|---------|
| `default` | Yes | Basic bot functionality |
| `message_content` | Yes | Reading message content for prefix commands (the `!` prefix is unused but `commands.Bot` requires it) |

### Discord Permissions Required

The bot itself only needs **Send Messages**, **Embed Links**, and **Attach Files** in the channels where it operates. No elevated permissions are required.

---

## 4. Configuration Reference

`config.py` loads all configuration from environment variables via `python-dotenv`, then exposes module-level constants consumed across the codebase.

| Constant | Type | Default | Purpose |
|----------|------|---------|---------|
| `DISCORD_TOKEN` | `str` | *(from env)* | Bot login token. **Never commit this.** |
| `PREFIX` | `str` | `"!"` | Legacy prefix for `commands.Bot` constructor. Slash commands ignore this entirely. |
| `MAX_EXPR_LENGTH` | `int` | `500` | Maximum character count for any math expression input before parsing even begins. Inputs exceeding this are rejected immediately by `_validate_raw()`. |
| `COMPUTE_TIMEOUT` | `int \| float` | `3` | Wall-clock seconds allowed for a single parse/compute job inside `parse_expression()`. SymPy can hang indefinitely on pathological inputs (e.g. symbolic integrals with no closed form); this is the hard safety net. |
| `CACHE_TTL` | `int` | `300` | Time-to-live in seconds for in-memory result cache entries. After 5 minutes, a cached result is considered stale and is recomputed on the next request. |
| `CACHE_MAXSIZE` | `int` | `256` | Maximum number of entries the LRU+TTL cache may hold simultaneously. The least-recently-used entry is evicted to make room when the cache is full. |

---

## 5. Bot Bootstrapping & Lifecycle

### `main.py` — Entry Point

`main.py` performs five roles:

**1. Logging setup**

Configures Python's `logging` module at `INFO` level with a timestamped format before anything else runs. All cogs and utilities use `logging.getLogger(__name__)` which inherits from this root configuration. The logger is passed as `log_handler=None` to `bot.run()` to prevent discord.py from installing its own duplicate handler.

**2. `COGS` list**

A module-level `list[str]` of dot-path strings defining which extensions to load and in what order. Every entry corresponds to a file under `cogs/`.

**3. Bot instantiation**

```python
bot = commands.Bot(
    command_prefix=config.PREFIX,   # "!" — required but unused by slash commands
    intents=discord.Intents.default() | message_content
)
```

**4. `on_ready` event handler**

Called once the bot has connected and is ready. It:

- Logs the bot's username and ID
- Records `bot.start_time = datetime.utcnow()` exactly once (guarded so reconnects don't reset it)
- Iterates `COGS`, calling `await bot.load_extension(cog)` for each. Failures (not found, already loaded, or any exception) are caught individually, logged, and collected into `failed` — a single broken cog never prevents others from loading
- Calls `await bot.tree.sync()` to register all slash commands globally with Discord (this can take up to 1 hour to propagate on Discord's end after the first sync)

**5. Global slash command infrastructure**

Three pieces of global infrastructure sit directly on `bot.tree`:

| Component | Purpose |
|-----------|---------|
| `/ping` command | Replies with `"Pong! X ms"` using `bot.latency`. Useful as a health-check. |
| `@bot.tree.interaction_check` — `_permission_check` | Runs before every slash command. Calls `data.permissions.is_command_allowed(guild_id, channel_id, command_name)`. If denied, sends an ephemeral error and returns `False` (discord.py cancels the invocation). DMs (`guild_id is None`) always pass. |
| `@bot.tree.error` — `on_app_command_error` | Catches all unhandled slash command exceptions. Handles `CommandOnCooldown` (reports retry time), `MissingPermissions`, and `BotMissingPermissions` with specific messages. Falls back to a generic "something went wrong" message for anything else, while logging the full traceback. |

**The `_ephemeral_reply` helper** handles the discord.py complication that once a cog has called `interaction.response.defer()`, you must use `interaction.followup.send()` rather than `interaction.response.send_message()`. This helper selects the correct path automatically.

---

## 6. Permission & Admin System

MathFrame includes a guild-level permission system that allows server administrators to enable or disable any slash command in any channel (or server-wide). This is enforced globally before any cog handler runs.

### Permission Schema

The system is backed by `data/guild_permissions.json`, a nested dictionary:

```json
{
  "<guild_id>": {
    "<channel_id | '__all__'>": {
      "<command_name | '__all__'>": true | false
    }
  }
}
```

### Lookup Order (most-specific wins)

When a command is invoked, `is_command_allowed()` walks through four levels of specificity, returning the first matching rule:

1. `guild → channel → command` — Channel-specific and command-specific rule
2. `guild → __all__ → command` — Guild-wide rule for this specific command
3. `guild → channel → __all__` — Channel rule applying to all commands
4. `guild → __all__ → __all__` — Guild-wide default for all commands
5. *(No match)* — **Allowed** (fail-open behavior — admins must explicitly disable)

### `/admin` Command Group

All four subcommands require the **Manage Guild** Discord permission and are guild-only.

| Command | Parameters | Description |
|---------|-----------|-------------|
| `/admin enable` | `command?` `channel?` | Allow a command in a channel. Omit `command` to enable all commands; omit `channel` to apply server-wide. |
| `/admin disable` | `command?` `channel?` | Block a command in a channel. Same wildcard behavior. |
| `/admin reset` | `command?` `channel?` | Remove an existing rule (reverts to the next less-specific rule, or fail-open if none). |
| `/admin status` | *(none)* | List all active permission rules for the current guild in an embed. |

### Thread Safety

`data/permissions.py` uses a module-level `threading.Lock` around all reads and writes. The JSON file is read once at import time (`_load()`) and written back after every mutation (`_save()`).

---

## 7. Data Layer

### 7.1 In-Memory Result Cache (`data/cache.py`)

A module-level singleton `cachetools.TTLCache` that acts as a shared memoization store across all cogs. The cache is:

- **TTL-based**: entries expire after `config.CACHE_TTL` seconds (default 5 minutes)
- **LRU-evicted**: when full (max `config.CACHE_MAXSIZE` = 256 entries), the least-recently-used entry is evicted
- **Thread-safe**: a `threading.Lock` guards all reads and writes

#### Public API

```python
from data.cache import get, set, cache_key, clear, info

# Build a namespaced cache key
key = cache_key("simplify", "x**2 + 2*x + 1")
# → "simplify|x**2 + 2*x + 1"

# Cache lookup
result = get(key)          # returns None on miss/expiry
if result is None:
    result = expensive_computation()
    set(key, result)       # store for TTL seconds

# Maintenance
clear()                    # flush all entries
stats = info()             # {"currsize": N, "maxsize": 256, "ttl": 300}
```

#### `cache_key(*args)` Format

Arguments are joined with `|`: `cache_key("diff", "sin(x)", "x", "2")` → `"diff|sin(x)|x|2"`. This namespaces results by operation so `cache_key("simplify", "x")` and `cache_key("factor", "x")` never collide.

#### Which Cogs Use the Cache

`/simplify`, `/expand`, `/factor`, `/diff`, `/integrate`, `/limit`, and several other compute-heavy commands build a cache key and return early if a hit is found, bypassing `parse_expression()` and SymPy entirely for repeated identical requests.

---

### 7.2 Per-User History (`data/history.py`)

A module-level singleton `dict[int, deque[HistoryEntry]]` keyed by Discord user ID. This is pure process-memory — nothing is persisted to disk. History is lost on bot restart.

#### `HistoryEntry`

```python
class HistoryEntry:
    __slots__ = ("command", "input", "result", "timestamp")
    command:   str       # e.g. "circle_area"
    input:     str       # e.g. "radius=3"
    result:    str       # e.g. "9*pi"
    timestamp: datetime  # UTC, set at creation
```

#### Public API

```python
from data.history import save_history, get_history, clear_history

save_history(user_id, "circle_area", "radius=3", "9*pi")
entries = get_history(user_id, limit=20)   # list[HistoryEntry], newest first
clear_history(user_id)                     # discard all entries
```

#### Capacity Limits

Each user's deque has `maxlen=20`. When the 21st entry is appended, the oldest is dropped automatically by Python's `deque`. This bounds memory growth to `O(max_users × 20)` entries total.

---

### 7.3 Guild Permission Store (`data/permissions.py`)

Documented fully in [Section 6](#6-permission--admin-system). The `data/` layer also stores `guild_permissions.json` at runtime (created automatically by `_save()` on first write).

#### Public API

```python
from data.permissions import (
    is_command_allowed,   # (guild_id, channel_id, command_name) -> bool
    set_permission,       # (guild_id, channel_id|None, command_name|None, enabled) -> None
    clear_permission,     # (guild_id, channel_id|None, command_name|None) -> bool
    get_guild_status,     # (guild_id) -> list[dict]
)
```

---

## 8. Expression Parsing System

The single most important architectural seam in the codebase is that there are **two separate expression parsers**: the primary one for almost all cogs and a secondary one specifically for the plotting subsystem.

---

### 8.1 Primary Parser (`utils/parser.py`)

This is the authoritative entry point for all symbolic math input. The module docstring states the rule explicitly: **"This is the only module in the project that calls `parse_expr` or `latex2sympy`. Every cog must go through `parse_expression` and work with the returned `sympy.Expr` — never call the underlying parsers directly."**

#### Format Auto-Detection (`_detect_format`)

The parser classifies each input string into one of four formats before choosing a parser:

| Format | Detection heuristic | Parser used |
|--------|--------------------|-----------:|
| `latex` | Leading `\`, known LaTeX macro (`\frac`, `\int`, `\sqrt`, `\sin`, `\alpha`, `\pi`, `\cdot`, etc.), or braced exponent `x^{2}` | `latex2sympy2.latex2sympy` |
| `python` | Contains `**` or starts with `math.` | `sympy.parse_expr` with standard transformations |
| `natural` | Contains prose keywords: "squared", "plus", "minus", "times", "divided by", "over" | Normalizer → `sympy.parse_expr` |
| `plain` | Everything else (default) | Caret-normalized → `sympy.parse_expr` with implicit multiplication |

#### Validation (`_validate_raw`)

Called before any parsing attempt. Raises `ValueError` immediately if:

1. The input string length exceeds `config.MAX_EXPR_LENGTH` (500 chars)
2. The input contains any forbidden keyword: `__`, `import`, `exec`, `eval`, `open`, `os`, `sys`, `subprocess`

The `__` check is a substring test (not word-boundary) because `__builtins__` and similar dunder names should always be rejected. All other keywords are matched at word boundaries via `re.search(r'\bkeyword\b', expr)`.

#### Per-Format Normalization

`_normalize_plain(expr)` handles the plain format:
- `^` → `**`
- Implicit multiplication via `sympy.parsing.sympy_parser.implicit_multiplication_application`
- Standard SymPy transformations applied

`_normalize_natural(expr)` handles the natural language format by string-replacing prose words before passing to the same SymPy pipeline:
- "squared" → `**2`, "cubed" → `**3`
- "plus" → `+`, "minus" → `-`, "times" → `*`, "divided by" → `/`, "over" → `/`

#### LaTeX Fallback

If `latex2sympy2` raises any exception on a LaTeX-classified input, the parser attempts a second parse using the plain/implicit-multiplication pipeline before surfacing an error to the user. This handles cases where input looks like LaTeX (has a `\`) but is actually valid plain SymPy notation.

#### Async Interface (`parse_expression`)

```python
async def parse_expression(expr: str) -> sympy.Expr:
    ...
```

This is the public API. It:

1. Calls `_validate_raw(expr)` — raises `ValueError` on bad input
2. Detects the format via `_detect_format(expr)`
3. Submits the blocking parse to a shared `ThreadPoolExecutor(max_workers=4)` via `loop.run_in_executor`
4. Wraps the executor call in `asyncio.wait_for(..., timeout=config.COMPUTE_TIMEOUT)` (3 seconds)
5. Catches `asyncio.TimeoutError` → reraises as `ValueError("Expression took too long to parse...")`
6. Catches any parse exception → reraises as a user-friendly `ValueError` with the original message

The thread pool (`max_workers=4`) ensures that up to four concurrent parse jobs can run in parallel without blocking the event loop, while still limiting total CPU consumption from runaway inputs.

---

### 8.2 Plot Parser (`utils/expr_utils.py`)

A second, slimmer parser used exclusively by the plotting subsystem (`cogs/plot_engine.py` and `utils/plotter.py`). It bypasses `parser.py` entirely and calls `sympy.sympify()` directly.

#### `_clean_sympy_expr(s)`

Preprocessing before sympify:

1. Strips an optional `name = ` assignment prefix (e.g. `"f(x) = sin(x)"` → `"sin(x)"`)
2. Rewrites `^` → `**`
3. Rewrites `e^x` and `e**x` → `exp(x)` for natural exponential notation
4. Detects and expands piecewise expressions in the form `condition: expr | condition: expr`

#### `_sympy_expr(s, local={})`

Calls `_validate_raw(s)` from `parser.py` (this was added to resolve Known Issue #1 — see Section 14), then calls `sympy.sympify(s, locals=local)` directly. The `locals` dict can inject plot-specific variable names.

#### Remaining Direct-sympify Call Sites

Two isolated call sites still use `sympy.sympify()` without going through either parser:

- `cogs/calculus.py::_parse_point()` — parses limit/series evaluation points. Handles `oo`/`+oo`/`-oo`/`inf` specially, then calls `sympify` on the remainder. This is a narrow single-value case (never a full expression string).
- `cogs/symbolic.py::_parse_substitutions()` — parses the right-hand side of `"x=2, y=pi"` substitution strings, after validating the left-hand side with `str.isidentifier()`.

---

## 9. Supporting Utilities

### 9.1 Embed Formatter (`utils/formatter.py`)

Three embed builders used by every cog's command handlers:

#### `math_embed(title, result, steps=None, footer="")`

Returns a `discord.Embed` with:
- Colour: **blurple** (`discord.Colour.blurple()`)
- `result` displayed in a code block
- Optional `steps` field showing numbered working steps, truncated to fit Discord's 1024-character field limit via `_format_steps()`

#### `error_embed(message)`

Returns a red embed with title `"❌ Error"` and `message` as the description. Called by every cog's `except` block.

#### `info_embed(title, description)`

Returns a gold embed. Used for informational responses like `/about` and `/constants`.

#### `to_readable_text(expr_str)`

A small regex-based post-processor that converts SymPy's default `**` and `*` notation to more readable `^` / implicit-multiplication form for display purposes. Applied when cogs want to show human-readable math rather than raw SymPy output.

---

### 9.2 Step-by-Step Solver (`utils/solver.py`)

Synchronous step generators that take an already-parsed `sympy.Expr` and return a `StepList = list[tuple[str, str]]` — each tuple is `(description, expression_string)` rendered as a numbered "Steps" field by `formatter.math_embed`.

All expression strings in steps are produced via `_expr_str(expr)`, which calls `sympy.pretty(expr, use_unicode=False)` to keep output ASCII-safe across all Discord clients.

#### `solve_quadratic_steps(expr, var)`

For polynomials of exactly degree 2. Extracts `a`, `b`, `c` coefficients, computes the discriminant `Δ = b² - 4ac`, and lists roots using the quadratic formula. Returns `[("Error", "...")]` sentinel for non-degree-2 input rather than raising, so callers can detect the unsupported case without a try/except.

#### `solve_cubic_steps(expr, var)` and `solve_quartic_steps(expr, var)`

Analogous step generators for cubic and quartic polynomials respectively, showing intermediate factoring and root-finding steps.

#### `factor_steps(expr)`

Shows the factoring process, including any intermediate GCD extraction.

#### `differentiate_steps(expr, var, order=1)`

Supports arbitrary order differentiation. Applies `sympy.diff` iteratively, showing the expression after each differentiation pass, then simplifies and shows the simplified form.

#### `integrate_steps(expr, var)`

Runs `sympy.integrate(expr, var)`. Detects and labels unevaluated `sympy.Integral` results (when SymPy cannot find a closed form) rather than returning a misleading expression.

---

### 9.3 LaTeX Renderer (`utils/renderer.py`)

Converts a LaTeX string or SymPy expression into a PNG `discord.File` using matplotlib's `mathtext` rendering engine (headless `Agg` backend).

Two separate `ThreadPoolExecutor(max_workers=2)` pools are maintained — one for rendering, one for parsing — so rendering and parsing workloads never starve each other.

#### `async expr_to_image(latex_str) -> discord.File`

Takes a raw LaTeX string (as-is from the user or from `sympy.latex()`), renders it to PNG, and returns a `discord.File` ready to attach to any interaction reply. Used by `/render`.

#### `async result_to_image(sympy_expr) -> discord.File`

Converts a `sympy.Expr` → `sympy.latex()` first, then renders. Used by `/formula` and `/latex`.

---

### 9.4 Paginator (`utils/paginator.py`)

`PaginatorView(pages: list[discord.Embed], timeout=120)` is a `discord.ui.View` with ◀/▶ navigation buttons.

#### Behavior

- Buttons disable at either end (◀ when on page 1, ▶ when on the last page)
- Each page's embed footer is stamped with `"Page N / Total"` (appended after any existing footer text with a `·` separator)
- The view times out after 120 seconds, after which buttons no longer function

#### `send_paginated(interaction, pages)`

The convenience wrapper used by most cogs. Automatically picks `interaction.response.send_message()` vs `interaction.followup.send()` based on whether the interaction has already been responded to (e.g. after a `defer()`). Skips attaching a view entirely when there is only one page.

#### Known Limitation

`PaginatorView` has **no per-user `interaction_check`** — any user in the channel can navigate another user's paginated result. Only `cogs/utility.py`'s `_ConfirmClearView` implements the correct owner check pattern. See Known Issue #2.

---

### 9.5 Plot Engine (`utils/plotter.py`)

The largest utility module at 2,246 lines. Contains all matplotlib figure builders, animation generators, and the `PlotSpec`/`StyleOptions` data classes. All public functions are async; blocking matplotlib work runs in a `ThreadPoolExecutor(max_workers=4, thread_name_prefix="plotter")`.

#### Key Data Classes

**`StyleOptions`** (dataclass) — carries all visual configuration for a plot:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `color` | `str` | `"#1f77b4"` | Primary line/fill color (hex) |
| `line_width` | `float` | `2.0` | Line width in points |
| `line_style` | `str` | `"-"` | matplotlib line style (`"-"`, `"--"`, `":"`, etc.) |
| `marker` | `Optional[str]` | `None` | matplotlib marker symbol |
| `marker_size` | `float` | `6.0` | Marker size in points |
| `colormap` | `str` | `"viridis"` | matplotlib colormap name |
| `theme` | `str` | `"default"` | matplotlib style theme |
| `alpha` | `float` | `0.9` | Global transparency |
| `show_grid` | `bool` | `True` | Whether to draw axis grid |
| `dpi` | `int` | `150` | Output image DPI |
| `fig_width` | `float` | `8.0` | Figure width in inches |
| `fig_height` | `float` | `4.5` | Figure height in inches |
| `x_log` | `bool` | `False` | Logarithmic x-axis scale |
| `y_log` | `bool` | `False` | Logarithmic y-axis scale |
| `fill_below` | `bool` | `False` | Fill area under curve |
| `fill_color` | `str` | `""` | Fill color (defaults to `color` if empty) |
| `x_lim` | `Optional[Tuple[float, float]]` | `None` | Manual x-axis limits |
| `y_lim` | `Optional[Tuple[float, float]]` | `None` | Manual y-axis limits |

**`PlotSpec`** (dataclass) — carries expression and domain data for one plot:

Includes fields for the main expression, parametric `x(t)/y(t)/z(t)`, vector field `u/v` components, polar expression, plot type identifier, variable names, domain bounds, and resolution settings.

#### Module-Level Constants

| Constant | Value | Purpose |
|---------|-------|---------|
| `PLOT_POINTS` | `800` | x-samples for 1-D line plots |
| `GRID_POINTS` | `120` | Grid resolution for 2-D/3-D surfaces |
| `Y_CLIP` | `1e6` | Hard clip for out-of-range y values |
| `Z_CLIP` | `1e6` | Hard clip for out-of-range z values |
| `PARAM_POINTS` | `1000` | t-samples for parametric curves |
| `ANIM_FRAMES` | `30` | Number of animation frames |
| `ANIM_PARAM_MIN` | `0.0` | Animation parameter sweep start |
| `ANIM_PARAM_MAX` | `10.0` | Animation parameter sweep end |
| `ANIM_GRID_POINTS` | `70` | Grid resolution for animated surfaces |
| `ANIM_PARAM_POINTS` | `400` | t-samples for animated parametric plots |

#### Supported Plot Types and Their Async Functions

| Async function | Plot type | Description |
|---------------|-----------|-------------|
| `plot_function` | `function` | 1-D y=f(x) line plot |
| `plot_riemann` | *(internal)* | Riemann sum visualization |
| `plot_points` | `scatter` | Scatter plot from (x, y) point lists |
| `plot_contour` | `contour` | 2-D contour/level-curve plot of f(x,y) |
| `plot_vector_field` | `vector-field` | 2-D quiver/streamline vector field |
| `plot_parametric_2d` | `parametric-2d` | Parametric 2-D curve (x(t), y(t)) |
| `plot_surface` | `surface` | 3-D surface z=f(x,y) |
| `plot_wireframe` | `wireframe` | 3-D wireframe z=f(x,y) |
| `plot_parametric_3d` | `parametric-3d` | Parametric 3-D curve (x(t), y(t), z(t)) |
| `plot_scatter_3d` | `scatter-3d` | 3-D scatter from (x, y, z) point lists |
| `plot_polar` | `polar` | Polar plot r=f(θ) |
| `plot_implicit` | `implicit` | Implicit curve f(x,y)=0 |
| `plot_inequality` | `inequality` | Shaded region for f(x,y) ≤ 0 |
| `plot_heatmap` | `heatmap` | Color-mapped heatmap of f(x,y) |
| `plot_multi` | *(special)* | Side-by-side subplot panel of up to 4 specs |
| `plot_animation` | *(special)* | Animated GIF over a parameter sweep |

#### Anti-Singularity Logic

Several internal helpers work together to produce clean plots near singularities:

- `_smart_ylim(ys, x_pad)` — computes y-axis limits that exclude extreme outliers while preserving interesting variation
- `_singularities_in_range(expr, var, lo, hi)` — detects poles and discontinuities using SymPy
- `_insert_function_gaps(xs, ys)` — replaces y values near singularities with `NaN`, causing matplotlib to draw visual gaps rather than near-vertical lines

#### Thread Isolation

Each blocking matplotlib call wraps its work in `matplotlib.rc_context(style_overrides)` so that style settings are fully isolated per render call. Concurrent renders never clobber each other's rcParams.

---

### 9.6 Dead Code: `utils/utility.py`

This ~490-line file is an **orphaned earlier draft** of `cogs/utility.py`. It defines a complete `UtilityCog` with `/constants`, `/help_math`, `/convert`, and a `_ConfirmView` — but `main.py`'s `COGS` list only references `"cogs.utility"`, and nothing else in the codebase imports `utils.utility`. Its own module docstring header mislabels it as `cogs/utility.py`. **It is safe to delete.**

---

## 10. Cog Reference

### 10.1 Admin (`cogs/admin.py`)

**Class:** `AdminCog`
**Purpose:** Server administration — enable/disable commands per-channel or server-wide.

All commands are inside the `/admin` group (`app_commands.Group(guild_only=True)`). Every subcommand enforces `Manage Guild` permission via `_require_manage_guild(interaction)` before processing.

| Command | Parameters | Action |
|---------|-----------|--------|
| `/admin enable` | `command?` `channel?` | Calls `set_permission(..., enabled=True)` |
| `/admin disable` | `command?` `channel?` | Calls `set_permission(..., enabled=False)` |
| `/admin reset` | `command?` `channel?` | Calls `clear_permission(...)`, returns `True/False` on whether a rule was found |
| `/admin status` | *(none)* | Calls `get_guild_status(guild_id)`, renders all rules as an embed table |

---

### 10.2 Arithmetic (`cogs/arithmetic.py`)

**Class:** `ArithmeticCog`
**Purpose:** Core symbolic algebra operations.

All commands share a four-step pattern: build a cache key → return early on hit → `parse_expression()` → run SymPy → cache and return result embed. Exception handling catches `ValueError`, `sympy.SympifyError`, `sympy.PolynomialError`, `NotImplementedError`, and a broad `Exception` fallback.

**Module-level helper:** `_parse_equation(raw)` — splits an equation string on bare `=` (not `==`, `<=`, `>=`, `!=` via regex `(?<![<>!=])=(?!=)`), parsing both sides and returning `lhs - rhs`. Falls through to a direct parse if no `=` is present.

| Command | Parameters | SymPy operation |
|---------|-----------|----------------|
| `/simplify` | `expression` | `sympy.simplify()` |
| `/solve` | `expression` `variable?` | `sympy.solve()` + step-by-step via `solver.py` for degree ≤ 4 |
| `/expand` | `expression` | `sympy.expand()` |
| `/factor` | `expression` | `sympy.factor()` + `factor_steps()` |
| `/table` | `expression` `start` `end` `step` | Evaluates `f(x)` at each step; outputs a chunked markdown table via paginator |
| `/poly_div` | `numerator` `denominator` | `sympy.div()` — returns quotient and remainder |
| `/verify` | `expr_a` `expr_b` | `sympy.simplify(a - b) == 0` — true if expressions are equivalent |
| `/compare` | `expr_f` `expr_g` | Side-by-side comparison embed showing both expressions, their domains, and a shared value table |

**Cooldown:** 1 use per 3 seconds for `/simplify`; typically 1 per 2 seconds for others.

---

### 10.3 Calculus (`cogs/calculus.py`)

**Class:** `CalculusCog`
**Purpose:** Differentiation, integration, limits, series, ODEs.

**Module-level helpers:**
- `_ordinal(n)` — formats `1` → `"1st"`, `2` → `"2nd"`, etc. for display
- `_parse_point(point_str)` — handles `oo`/`+oo`/`-oo`/`inf` before calling `sympy.sympify()` (a direct sympify call site)

| Command | Parameters | SymPy operation |
|---------|-----------|----------------|
| `/diff` | `expression` `variable?` `order?` | `sympy.diff()` + `differentiate_steps()` |
| `/integrate` | `expression` `variable?` `lower?` `upper?` | `sympy.integrate()` — definite if both bounds provided, indefinite otherwise |
| `/limit` | `expression` `variable` `point` `direction?` | `sympy.limit()` with `dir="+"`, `"-"`, or `"+-"` |
| `/series` | `expression` `variable?` `point?` `order?` | `sympy.series()` — Taylor/Maclaurin expansion |
| `/sum_series` | `expression` `variable` `lower?` `upper?` | `sympy.summation()` — symbolic Σ |
| `/product_series` | `expression` `variable` `lower?` `upper?` | `sympy.product()` — symbolic Π |
| `/ode` | `expression` `initial_conditions?` | Pre-processes `f'(x)` / `y''` prime notation into `sympy.Derivative` objects; runs `sympy.dsolve()` |

**`/ode` prime notation preprocessing:** The cog rewrites `f'(x)` → `Derivative(f(x), x)` and `y''` → `Derivative(y, x, 2)` via string replacement before the expression reaches SymPy's parser. This allows users to write ODEs in natural notation.

---

### 10.4 Transforms (`cogs/transforms.py`)

**Class:** `TransformsCog`
**Purpose:** Integral transforms — Laplace, Fourier, and their inverses.

All four commands follow the same pattern: `parse_expression()` → create SymPy symbol objects → run transform in thread pool via `loop.run_in_executor()`.

| Command | Parameters | SymPy operation |
|---------|-----------|----------------|
| `/laplace` | `expression` `t?` `s?` | `sympy.laplace_transform(expr, t, s)` |
| `/inv_laplace` | `expression` `s?` `t?` | `sympy.inverse_laplace_transform(expr, s, t)` |
| `/fourier` | `expression` `x?` `k?` | `sympy.fourier_transform(expr, x, k)` |
| `/inv_fourier` | `expression` `k?` `x?` | `sympy.inverse_fourier_transform(expr, k, x)` |

**Cooldown:** 1 use per 3 seconds (transforms can be slow).

---

### 10.5 Linear Algebra (`cogs/linear_algebra.py`)

**Class:** `LinearAlgebraCog`
**Purpose:** Matrix and vector operations.

**Input format:** Matrices and vectors are accepted as **JSON array strings** (e.g. `[[1,2],[3,4]]`, `[1,2,3]`). These are parsed with the standard library `json` module rather than SymPy — a deliberate security choice, since matrix input is structurally a list of numbers, not a symbolic expression.

**Module-level helpers:**
- `_format_matrix(M)` — plain-text grid rendering for embeds, preserving column alignment
- `_parse_json_list(s)` — wraps `json.loads()` with a helpful error message
- `_parse_vector(s)` — calls `_parse_json_list()` and validates the result is a 1-D list

**`parse_matrix(self, s)` (cog method)** — calls `_parse_json_list()`, validates shape (must be 2-D), and returns a `sympy.Matrix`.

| Command | Parameters | SymPy operation |
|---------|-----------|----------------|
| `/matrix_det` | `matrix` | `sympy.Matrix.det()` — exact integer/fraction determinant |
| `/matrix_inv` | `matrix` | `sympy.Matrix.inv()` — exact rational inverse |
| `/eigenvalues` | `matrix` | `sympy.Matrix.eigenvals()` — returns eigenvalue → algebraic multiplicity dict |
| `/dot` | `a` `b` | `sympy.Matrix.dot()` — dot product of two vectors |
| `/cross` | `a` `b` | `sympy.Matrix.cross()` — cross product (3-D vectors only) |
| `/rref` | `matrix` | `sympy.Matrix.rref()` — reduced row-echelon form with pivot columns |

---

### 10.6 Statistics (`cogs/statistics.py`)

**Class:** `StatsCog`
**Purpose:** Descriptive statistics and probability distribution visualization.

**`parse_numbers(s)`** — module-level helper that splits a comma-separated string into `list[float]`. Used by nearly every command in the cog.

**Descriptive stats commands** (all take `data: str` = comma-separated numbers):

| Command | Computation |
|---------|------------|
| `/mean` | `statistics.mean()` |
| `/median` | `statistics.median()` |
| `/mode` | `statistics.multimode()` (returns all modes) |
| `/stdev` | `statistics.stdev()` (sample standard deviation) |
| `/variance` | `statistics.variance()` (sample variance) |
| `/zscore` | `(value - mean) / stdev` |
| `/correlation` | Pearson r via `statistics.correlation()` + `_correlation_label()` human-readable strength |
| `/regression` | `scipy.stats.linregress()` + scatter plot image via `_regression_plot_bytes()` |

**`/distribution` — Unified Distribution Command**

The preferred interface for all probability distributions. Takes `kind` (an autocomplete choice) and `params` (comma-separated parameter list):

| kind | Required params | Description |
|------|----------------|-------------|
| `normal_pdf` | `mean, stdev` | Normal distribution PDF plot |
| `normal_cdf` | `mean, stdev, upper[, lower]` | Normal CDF between bounds |
| `inv_normal` | `probability, mean, stdev` | Inverse normal (percentile) |
| `binomial_cdf` | `n, p, x` | Binomial cumulative probability P(X ≤ x) |
| `poisson_cdf` | `lambda, x` | Poisson cumulative probability P(X ≤ x) |

The `_DIST_PARAMS` registry dict maps each kind to its ordered parameter list, and `_parse_dist_params()` validates count and type before dispatching to the corresponding plot-byte generator.

**Deprecated individual distribution commands** — `/normal_pdf`, `/normal_cdf`, `/inv_normal`, `/binomial_cdf`, `/poisson_cdf` — are retained for backward compatibility with `[DEPRECATED]` in their descriptions.

All plot-byte generators (`_normal_pdf_bytes`, `_normal_cdf_bytes`, etc.) use `scipy.stats` distributions and the headless `Agg` matplotlib backend, run in a thread-pool executor.

---

### 10.7 Number Theory (`cogs/number_theory.py`)

**Class:** `NumberTheoryCog`
**Purpose:** Integer and prime number operations.

This cog has the most helper functions. All commands enforce explicit upper bounds on `n` to keep computation time bounded — none go through `parser.py`'s `COMPUTE_TIMEOUT` guard since they use raw Python/SymPy integer arithmetic.

**Module-level helpers:**

| Helper | Purpose |
|--------|---------|
| `_superscript(n)` | Digit → Unicode superscript for rendering `2³ × 3² × 5¹` |
| `_format_factorisation(factors_dict)` | Formats `{2: 3, 5: 1}` → `"2³ × 5"` |
| `_parse_integers(s)` | Comma-separated string → `list[int]` |
| `_parse_single_integer(s)` | Single integer string → `int` |
| `_validate_int_arg(n, lo, hi, name)` | Range-checks `n ∈ [lo, hi]`, raises `ValueError` otherwise |
| `_list_gcd(nums)` | Iterative pairwise reduction over `math.gcd` (can short-circuit at 1) |
| `_list_lcm(nums)` | Iterative pairwise reduction over `math.lcm` |
| `_fibonacci_list(n)` | Returns first `n` Fibonacci numbers |
| `_format_large_int(n, threshold)` | Falls back to scientific notation above threshold to prevent embed overflow |
| `_paginate_list(items, page_size)` | Generic page-splitting, shared by `/primes_up_to`, `/fibonacci`, `/divisors` |

**Commands and upper bounds:**

| Command | Parameters | Upper bound | SymPy/Python operation |
|---------|-----------|-------------|----------------------|
| `/gcd` | `numbers` | None | `math.gcd` iterative |
| `/lcm` | `numbers` | None | `math.lcm` iterative |
| `/is_prime` | `n` | 10¹² | `sympy.isprime()` |
| `/factorize` | `n` | 10¹⁵ | `sympy.factorint()` |
| `/primes_up_to` | `n` | 10,000 | `sympy.primerange()` |
| `/modular` | `base` `exp` `m` | None | Python `pow(base, exp, m)` |
| `/fibonacci` | `n` | 200 | `_fibonacci_list()` |
| `/totient` | `n` | 10⁹ | `sympy.totient()` |
| `/divisors` | `n` | 10¹² | `sympy.divisors()`, paginated if >30 results |
| `/is_perfect` | `n` | 10¹² | `sum(divisors[:-1]) == n` |
| `/mobius` | `n` | 10¹² | `sympy.mobius()` + plain-language label of −1/0/1 |
| `/chinese_remainder` | `remainders` `moduli` | None | `sympy.ntheory.modular.crt()` (lazy import), validates solution existence first |

---

### 10.8 Geometry (`cogs/geometry.py`)

**Class:** `GeometryCog`
**Purpose:** 2-D and 3-D geometric computations.

**Module-level helpers:**
- `_exact_and_decimal(expr)` — returns both the exact SymPy form and a 6-decimal float approximation; used by most commands
- `_to_exact(value)` — converts Discord's `float` option (which loses exactness) back to `sympy.Rational` via `sympy.nsimplify()`

Commands accept multiple optional parameters and branch internally on which combination was supplied, rather than requiring separate commands per input mode:

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/circle_area` | `radius` | `π r²` |
| `/circle_circumference` | `radius` | `2π r` |
| `/triangle_area` | `base?` `height?` `a?` `b?` `c?` | From base+height or three side lengths (Heron's formula) |
| `/pythagorean` | Any two of `a?` `b?` `c?` | Solves for missing side; validates triangle inequality |
| `/trig` | `function` `angle` | Exact + decimal evaluation; supports degrees/radians toggle |
| `/distance` | `x1` `y1` `x2` `y2` `z1?` `z2?` | 2-D or 3-D Euclidean distance |

---

### 10.9 Discrete Math (`cogs/discrete.py`)

**Class:** `DiscreteCog`
**Purpose:** Combinatorics, boolean logic, truth tables, set operations.

This cog contains the most custom parsing logic. Crucially, it features a **hand-rolled boolean parser that never calls `eval()`** — the only such sandbox in the codebase.

#### Boolean Parsing Pipeline

1. `_tokenize_boolean(s)` — regex tokenizer producing `operator`/`paren`/`variable` tokens
2. `_BoolParser` — recursive-descent parser with precedence (implies → xor → or → and → not → atom)
   - `_parse_implies()` → `_parse_xor()` → `_parse_or()` → `_parse_and()` → `_parse_not()` → `_parse_atom()`
   - Each level calls the next, encoding operator precedence as the call stack
   - Produces a nested-tuple AST
3. `_evaluate_boolean(ast, vars_dict)` — walks the AST against `{variable: bool}` bindings
4. `_ast_to_sympy(ast)` — converts the same AST into SymPy logic (`And`, `Or`, `Not`, `Xor`, `Implies`) for `/simplify_bool` and `/logic_equiv`

**Other helpers:**

| Helper | Purpose |
|--------|---------|
| `_format_large_int(n, threshold)` | Scientific notation fallback for large combinatorial results |
| `_build_truth_table_lines(expr, vars)` | Renders truth table rows as text chunks |
| `_paginate_table(lines, page_size)` | Splits table rows for the paginator |
| `_parse_set_element(s)` | Tries `int` → `float` → `str` in order |
| `_parse_set(s)` | Comma-separated → `set` using `_parse_set_element` |
| `_format_set(s)` | Formats a set as `{a, b, c}` |

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/permutation` | `n` `r` | nPr = n!/(n-r)! |
| `/combination` | `n` `r` | nCr = n!/(r!(n-r)!) |
| `/truth_table` | `expression` | Full truth table; paginated for large variable counts |
| `/set_ops` | `set_a` `set_b` `operation` | union/intersection/difference/symmetric_difference/subset/superset |
| `/binomial_coeff` | `n` | nth row of Pascal's triangle (n ≤ 20) |
| `/simplify_bool` | `expression` `form?` | `form` autocomplete: Simplified (default), DNF, CNF |
| `/logic_equiv` | `expr_a` `expr_b` | XOR satisfiability check; returns a counterexample witness when not equivalent |

---

### 10.10 Symbolic (`cogs/symbolic.py`)

**Class:** `SymbolicCog`
**Purpose:** Symbolic algebra tools — LaTeX, substitution, partial fractions, roots, expression classification.

**Module-level helpers:**
- `_parse_substitutions(s)` — splits `"x=2, y=pi"` on commas, validates left-hand side with `str.isidentifier()`, calls `sympy.sympify()` on right-hand values
- `_root_line(root, multiplicity)` — formats one root as `"root (real/complex, multiplicity k)"`
- `_identify_expression(expr, var)` — classification helper (see below)

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/latex` | `expression` | Parse → `sympy.latex()` → `renderer.result_to_image()` → PNG |
| `/subs` | `expression` `substitutions` | Parse expression → `_parse_substitutions()` → `expr.subs(pairs)` |
| `/partial_fraction` | `expression` | `sympy.apart()` — partial fraction decomposition |
| `/roots` | `expression` | `sympy.roots()` — all roots with multiplicity |
| `/identify` | `expression` `variable?` | `_identify_expression()` — multi-label classifier, 5s timeout |

#### `/identify` — `_identify_expression(expr, var)` Detail

Checks (in order, additively — an expression can carry multiple labels):

| Check | Method |
|-------|--------|
| Polynomial | `sympy.Poly(expr, var)` — extracts degree |
| Rational function | `sympy.cancel()` → non-trivial denominator |
| Trigonometric | Atom scan for `sin`, `cos`, `tan`, `csc`, `sec`, `cot`, `asin`, `acos`, `atan`, `acsc`, `asec`, `acot` |
| Exponential | Atom scan for `exp` |
| Logarithmic | Atom scan for `log`, `ln` |
| Even | `sympy.simplify(f(-x) - f(x)) == 0` |
| Odd | `sympy.simplify(f(-x) + f(x)) == 0` |
| Periodic | `sympy.periodicity(expr, var)` — returns the period or `None` |
| Constant | `expr.free_symbols == set()` |

Runs inside `loop.run_in_executor` with a **5-second timeout** (separate from the 3-second parse timeout) because `sympy.periodicity` and `sympy.simplify` can be slow.

---

### 10.11 Equations (`cogs/equations.py`)

**Class:** `EquationsCog`
**Purpose:** Solving systems of simultaneous equations.

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/solve_sim` | `equations` `variables?` | Comma-separated equations; dynamically determines variables; `sympy.nonlinsolve()` |

`sympy.nonlinsolve()` is used (vs `sympy.solve()`) because it handles non-linear systems and provides broader algebraic solving capabilities. Variable detection uses `sympy.free_symbols` across all parsed equations when `variables` is not explicitly provided.

---

### 10.12 Inequalities (`cogs/inequalities.py`)

**Class:** `InequalitiesCog`
**Purpose:** Symbolic inequality solving over the reals.

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/solve_ineq` | `expression` | Supports single and compound inequalities; `sympy.reduce_inequalities()` over `sympy.S.Reals` |

Contains fallback logic: if `reduce_inequalities` raises, it attempts `sympy.solveset(expr, var, domain=sympy.S.Reals)`. Results are expressed in interval/union-of-intervals notation.

---

### 10.13 Complex Numbers (`cogs/complex.py`)

**Class:** `ComplexCog`
**Purpose:** Complex number evaluation and form conversion.

Post-processing regex normalizes SymPy output into readable `a + bi` formats using `sympy.expand(complex=True)`.

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/complex_eval` | `expression` | Evaluates complex expressions; shows real and imaginary parts |
| `/complex_polar` | `expression` | Converts to polar form `r·e^(iθ)` — shows modulus and argument |
| `/complex_rect` | `expression` | Converts to rectangular form `a + bi` |

---

### 10.14 Base-N Arithmetic (`cogs/base_n.py`)

**Class:** `BaseNCog`
**Purpose:** Integer arithmetic in arbitrary bases.

Notably, `/base_logic` uses Python's native `eval()` wrapped in a restricted environment — the only place in the codebase where `eval()` is intentionally used. It is restricted to bitwise operators and integer literals; string/list/dict builtins are removed from the globals dict.

**Module-level helpers:**
- `_parse_in_base(s, base)` — safely parses a base-N string (base 2–36) using Python's `int(s, base)` with error handling
- `_format_in_base(n, base)` — formats an integer result into a given base representation

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/base_convert` | `number` `from_base` `to_base` | Bases 2–36 supported |
| `/base_math` | `expr` `base` | Evaluates arithmetic expression in a given base; supports `+`, `-`, `*`, `/`, `^` |
| `/base_logic` | `expr` | Evaluates bitwise operations (`&`, `|`, `^`, `~`, `<<`, `>>`) using restricted `eval()` |

---

### 10.15 Utility (`cogs/utility.py`)

**Class:** `UtilityCog`
**Purpose:** Bot housekeeping, user history, unit conversion, and reference information.

**Module-level helpers:**
- `_exact_and_decimal(expr)` — duplicated from `geometry.py` (both cogs define their own copy)
- `_format_uptime(delta)` — renders a `timedelta` as `"1d 2h 3m 4s"`, omitting leading zero units
- `_lib_version(name)` — returns the installed version of a Python package; used by `/about`
- `_temp_to_kelvin(value, unit)` / `_temp_from_kelvin(kelvin, unit)` / `_temp_steps(v, f, t)` — temperature conversion routed through Kelvin as a common intermediate so only `to_kelvin` and `from_kelvin` are needed per unit pair

**`_ConfirmClearView`** — the only `discord.ui.View` in the codebase with a correct **per-user `interaction_check`**. Stores `owner_id` at construction and rejects button presses from other users. `_disable_all()` disables buttons on timeout.

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/history` | *(none)* | Shows last 20 calculations, 5 per page; ephemeral |
| `/clear_history` | *(none)* | Confirmation dialog (5s cooldown); ephemeral |
| `/constants` | *(none)* | π, e, φ, √2, i, ∞ with 10-place decimals |
| `/help_math` | *(none)* | One page per loaded cog listing all its slash commands |
| `/convert` | `value` `from_unit` `to_unit` | 10 categories: length, mass, temperature, time, area, volume, speed, force, energy, power |
| `/about` | *(none)* | Bot info: version, uptime, loaded cogs count, library versions |

---

### 10.16 Rendering (`cogs/render.py`)

**Class:** `RenderCog`
**Purpose:** On-demand LaTeX and expression PNG rendering.

No module-level helpers. Two thin commands that delegate entirely to `utils/renderer.py`:

| Command | Parameters | Notes |
|---------|-----------|-------|
| `/render` | `latex` | Assumes input is already valid LaTeX; skips `parse_expression()` entirely |
| `/formula` | `expression` | Parses via `parse_expression()` first, then renders the resulting SymPy expression |

The distinction exists because `/render` is for users comfortable writing LaTeX directly, while `/formula` is "parse first, render second" — accepting any format that `parser.py` understands.

---

### 10.17 Plot Builder (`cogs/plot_engine.py`)

**Class:** `PlotEngineCog`
**Size:** 1,779 lines — the largest cog.
**Purpose:** A full interactive plot-building experience with 13 plot types, modal UI, pan/zoom controls, animation, and shareable sessions.

#### `PlotConfig` Dataclass

The single source of truth for one in-progress plot session (~50 fields). Key fields:

| Field group | Fields |
|------------|--------|
| Plot type | `plot_type` (one of 13 `PLOT_TYPES`) |
| Expressions | `expr_main`, `expr_u`, `expr_v`, `expr_x`, `expr_y`, `expr_z` |
| Domain | `x_min`, `x_max`, `y_min`, `y_max`, `t_min`, `t_max` |
| Resolution | `resolution_1d` (100–2000), `resolution_2d` (40–400) |
| Style | `color`, `colormap`, `theme`, `alpha`, `fill_below`, `fill_color` |
| Axes | `x_label`, `y_label`, `z_label`, `title` |
| Limits | `x_lim_min/max`, `y_lim_min/max` |
| Animation | `anim_param`, `anim_min`, `anim_max`, `anim_fps` |
| Scatter | `scatter_x`, `scatter_y`, `scatter_z` |
| Additional | `extra_exprs` (for multiplot) |

**Session export:** `export_config()` serializes `PlotConfig` via `json.dumps` → `zlib.compress` → `base64.b64encode`, producing a copy-pasteable string. `import_config(s)` reverses the process. This is how `/plot_import` works — a whole session is a single string.

#### Thirteen Plot Types (`PLOT_TYPES`)

`function`, `contour`, `vector-field`, `parametric-2d`, `surface`, `wireframe`, `parametric-3d`, `scatter`, `scatter-3d`, `polar`, `implicit`, `inequality`, `heatmap`

#### Modal UI System

The bot uses a stack of `discord.ui.Modal` subclasses to collect user input. Each follows the same two-method shape:

- `__init__` — pre-fills `TextInput` fields from current `cfg` (so reopening a modal shows last values)
- `on_submit` — parses each field back into `cfg` using tolerant parsers that fall back to defaults rather than raising, then triggers a re-render

| Modal class | Controls |
|------------|---------|
| `ExpressionModal` | Main expression, vector field u/v, parametric x/y/z |
| `StyleModal` | Color, line width, colormap, theme, alpha |
| `AxesModal` | Domain bounds, axis labels, title |
| `AdvancedModal` | Resolution (1D/2D), scatter data, log scale |
| `FillModal` | Fill toggle and fill color |
| `LimitsModal` | Manual axis limits override |
| `AnimationParamModal` | Animation parameter, range, FPS |
| `AdditionalExprModal` | Extra expressions for multiplot panels |

#### `PlotEngineView` Controls

The persistent view attached to all plot sessions. One handler per concern:

- **Plot type select** — replaces `cfg.plot_type`
- **Expression button** → `ExpressionModal`
- **Style button** → `StyleModal`
- **Axes button** → `AxesModal`
- **Advanced button** → `AdvancedModal`
- **Fill button** → `FillModal`
- **Limits button** → `LimitsModal`
- **Pan buttons (←→↑↓)** → `_shift_domain(cfg, axis, direction, fraction=0.3)`
- **Zoom In / Zoom Out** → `_scale_domain(cfg, factor=0.7 or 1.3)`
- **Preview button** → renders current config as a PNG embed
- **Animate button** → `AnimationParamModal` → `_render_animation(cfg)` → GIF output
- **Render button** → final render; disables the view and edits the message
- **Reset button** → resets `cfg` to defaults and re-renders

**Tolerant field parsers** (module-level):
- `_parse_float(s, default)` — `float(s)` or `default`
- `_parse_int(s, default)` — `int(s)` or `default`
- `_parse_bool(s, default)` — checks `s.lower() in ("true", "1", "yes")` or `default`
- `_parse_floatlist(s, default)` — `[float(x) for x in s.split(",")]` or `default`
- `_parse_optional_float(s)` — `float(s)` or `None`

#### Slash Commands

| Command | Parameters | Description |
|---------|-----------|-------------|
| `/plot` | *(none)* | Opens the full interactive plot builder |
| `/quickplot` | `expression` `domain?` | Renders a function immediately; no UI |
| `/multiplot` | `expr1` `expr2?` `expr3?` `expr4?` | Up to 4 expressions in one side-by-side image |
| `/plot_import` | `config_string` | Loads a shared session string exported from another plot |

#### Known Limitation

`PlotEngineView` has no `interaction_check` — any user in the channel can operate another user's in-progress plot session. See Known Issue #2.

---

### 10.18 Wikipedia (`cogs/wiki.py`)

**Class:** `WikiCog`
**Purpose:** Wikipedia article browsing and search via the Wikipedia REST API.

The only cog that talks to an external HTTP API and imports `aiohttp` directly. Note that `aiohttp` is not in `requirements.txt` — it is available only as a transitive dependency of `discord.py`.

**Session management:** Implements `cog_load()` / `cog_unload()` to create and tear down one shared `aiohttp.ClientSession` for the cog's lifetime via a `session` property. This avoids opening a new HTTP session per request.

**Module-level helpers:**

| Helper | Purpose |
|--------|---------|
| `_fetch_summary(session, title)` | Fetches article thumbnail URL and intro blurb from Wikipedia's summary endpoint |
| `_fetch_sections(session, title)` | Fetches full paragraph-by-paragraph content from the mobile-sections endpoint |
| `_search_wikipedia(session, query)` | Searches Wikipedia and returns a list of matching titles |
| `_strip_html(s)` / `_clean(s)` | Removes MediaWiki HTML markup and normalizes whitespace |
| `_thumbnail_url(data)` | Defensively walks possibly-missing nested dict structure in the API response |
| `_article_pages(data)` | Converts raw API sections into a `list[discord.Embed]` for the paginator |
| `_search_result_embed(results)` | Builds an embed listing search results with numbered entries |

| Command | Parameters | Description |
|---------|-----------|-------------|
| `/wiki` | `title` | Fetches article by title; browse paragraph-by-paragraph via paginator |
| `/wiki_search` | `query` | Searches Wikipedia; returns a list of matching articles |

---

## 11. Complete Command Reference

### Arithmetic

| Command | Description | Cooldown |
|---------|-------------|---------|
| `/simplify expression` | Simplify a mathematical expression | 3s |
| `/solve expression [variable=x]` | Solve expression = 0; step-by-step for degree ≤ 4 polynomials | 3s |
| `/expand expression` | Distribute/expand an expression | 2s |
| `/factor expression` | Factor with step-by-step working | 2s |
| `/table expression start end step` | Generate a value table | 2s |
| `/poly_div numerator denominator` | Polynomial division (quotient & remainder) | 2s |
| `/verify expr_a expr_b` | Check if two expressions are mathematically equivalent | 2s |
| `/compare expr_f expr_g` | Side-by-side function comparison | 3s |

### Calculus

| Command | Description | Cooldown |
|---------|-------------|---------|
| `/diff expression [variable] [order]` | Differentiate with steps | 3s |
| `/integrate expression [variable] [lower] [upper]` | Definite or indefinite integral | 3s |
| `/limit expression variable point [direction]` | Evaluate a limit | 3s |
| `/series expression [variable] [point] [order]` | Taylor/Maclaurin expansion | 3s |
| `/sum_series expression variable [lower] [upper]` | Evaluate a summation (Σ) | 3s |
| `/product_series expression variable [lower] [upper]` | Evaluate a product (Π) | 3s |
| `/ode expression [initial_conditions]` | Solve an ordinary differential equation | 5s |

### Transforms

| Command | Description | Cooldown |
|---------|-------------|---------|
| `/laplace expression [t] [s]` | Laplace transform L{f(t)}(s) | 3s |
| `/inv_laplace expression [s] [t]` | Inverse Laplace transform | 3s |
| `/fourier expression [x] [k]` | Fourier transform | 3s |
| `/inv_fourier expression [k] [x]` | Inverse Fourier transform | 3s |

### Linear Algebra

| Command | Description | Cooldown |
|---------|-------------|---------|
| `/matrix_det matrix` | Determinant (exact) | 2s |
| `/matrix_inv matrix` | Matrix inverse (exact rational) | 2s |
| `/eigenvalues matrix` | Eigenvalues with algebraic multiplicity | 3s |
| `/dot a b` | Dot product | 2s |
| `/cross a b` | Cross product (3-D) | 2s |
| `/rref matrix` | Reduced row-echelon form | 2s |

### Statistics

| Command | Description | Cooldown |
|---------|-------------|---------|
| `/mean data` | Arithmetic mean | 2s |
| `/median data` | Median | 2s |
| `/mode data` | Mode(s) | 2s |
| `/stdev data` | Sample standard deviation | 2s |
| `/variance data` | Sample variance | 2s |
| `/zscore value data` | Standard z-score | 2s |
| `/correlation x y` | Pearson correlation coefficient | 2s |
| `/regression x y` | Linear regression with scatter plot | 3s |
| `/distribution kind params` | **Unified** probability distribution command | 3s |

### Number Theory

| Command | Description | Upper bound |
|---------|-------------|------------|
| `/gcd numbers` | GCD of a list of integers | — |
| `/lcm numbers` | LCM of a list of integers | — |
| `/is_prime n` | Primality test | 10¹² |
| `/factorize n` | Prime factorization | 10¹⁵ |
| `/primes_up_to n` | List all primes up to n | 10,000 |
| `/modular base exp m` | Modular exponentiation | — |
| `/fibonacci n` | First n Fibonacci numbers | 200 |
| `/totient n` | Euler's totient φ(n) | 10⁹ |
| `/divisors n` | All divisors, paginated if >30 | 10¹² |
| `/is_perfect n` | Checks perfect number property | 10¹² |
| `/mobius n` | Möbius μ(n) with plain-language label | 10¹² |
| `/chinese_remainder remainders moduli` | CRT solver | — |

### Geometry

| Command | Description |
|---------|-------------|
| `/circle_area radius` | Area = πr² |
| `/circle_circumference radius` | Circumference = 2πr |
| `/triangle_area [base height] or [a b c]` | Area from base+height or three sides |
| `/pythagorean [any two of a b c]` | Solve for missing right-triangle side |
| `/trig function angle` | Exact + decimal trig evaluation |
| `/distance x1 y1 x2 y2 [z1 z2]` | 2-D or 3-D Euclidean distance |

### Discrete Math

| Command | Description |
|---------|-------------|
| `/permutation n r` | nPr |
| `/combination n r` | nCr |
| `/truth_table expression` | Full boolean truth table |
| `/set_ops set_a set_b operation` | Union/intersection/difference/symmetric_difference/subset/superset |
| `/binomial_coeff n` | nth row of Pascal's triangle (n ≤ 20) |
| `/simplify_bool expression [form]` | Simplify: Simplified / DNF / CNF |
| `/logic_equiv expr_a expr_b` | Logical equivalence check with counterexample |

### Symbolic

| Command | Description |
|---------|-------------|
| `/latex expression` | Render as LaTeX PNG |
| `/subs expression substitutions` | Substitute values (e.g. `"x=2, y=pi"`) |
| `/partial_fraction expression` | Partial fraction decomposition |
| `/roots expression` | All roots with multiplicity |
| `/identify expression [variable]` | Multi-label expression classifier |

### Equations, Inequalities, Complex, Base-N

| Command | Description |
|---------|-------------|
| `/solve_sim equations [variables]` | Solve a system of simultaneous equations |
| `/solve_ineq expression` | Solve a symbolic inequality over ℝ |
| `/complex_eval expression` | Evaluate complex expression |
| `/complex_polar expression` | Convert to polar form |
| `/complex_rect expression` | Convert to rectangular a+bi form |
| `/base_convert number from_base to_base` | Base conversion (bases 2–36) |
| `/base_math expr base` | Arithmetic in a custom base |
| `/base_logic expr` | Bitwise logic evaluation |

### Utility

| Command | Description |
|---------|-------------|
| `/history` | Your last 20 calculations (ephemeral, paginated) |
| `/clear_history` | Clear history with confirmation (ephemeral) |
| `/constants` | π, e, φ, √2, i, ∞ reference |
| `/help_math` | Paginated command listing by category |
| `/convert value from_unit to_unit` | Unit conversion (10 categories) |
| `/about` | Bot info, uptime, library versions |

### Rendering

| Command | Description |
|---------|-------------|
| `/render latex` | LaTeX string → PNG |
| `/formula expression` | Any parseable expression → PNG |

### Plotting

| Command | Description |
|---------|-------------|
| `/plot` | Open the full interactive plot builder (13 plot types) |
| `/quickplot expression [domain]` | Instant single-function plot, no UI |
| `/multiplot expr1 [expr2] [expr3] [expr4]` | Up to 4 functions, side-by-side |
| `/plot_import config_string` | Load a shared plot session |

### Wikipedia

| Command | Description |
|---------|-------------|
| `/wiki title` | Browse an article paragraph-by-paragraph |
| `/wiki_search query` | Search Wikipedia |

### Admin

| Command | Description |
|---------|-------------|
| `/admin enable [command] [channel]` | Enable command in channel or server-wide |
| `/admin disable [command] [channel]` | Disable command in channel or server-wide |
| `/admin reset [command] [channel]` | Remove an existing permission rule |
| `/admin status` | Show all active permission rules |

### Built-in (main.py)

| Command | Description |
|---------|-------------|
| `/ping` | Check bot latency |

---

## 12. Data Flow Diagrams

### A. Typical Slash Command Flow

```
User types /simplify "x^2 + 2x + 1"
           │
           ▼
discord.py receives interaction
           │
           ▼
bot.tree.interaction_check  (_permission_check)
  ├─ guild_id = None → pass (DM)
  ├─ is_command_allowed(guild_id, channel_id, "simplify") = True → pass
  └─ False → ephemeral "disabled" message, stop
           │
           ▼
ArithmeticCog.simplify(interaction, expression="x^2 + 2x + 1")
           │
           ├── await interaction.response.defer()
           │
           ├── key = cache_key("simplify", "x^2 + 2x + 1")
           │
           ├── result = cache.get(key)
           │   ├─ Hit → skip to embed
           │   └─ Miss:
           │       │
           │       ├── expr = await parse_expression("x^2 + 2x + 1")
           │       │     ├── _validate_raw() — length & keywords
           │       │     ├── _detect_format() → "plain"
           │       │     ├── _normalize_plain() → "x**2 + 2*x + 1"
           │       │     └── ThreadPoolExecutor → sympy.parse_expr()
           │       │                             (timeout: 3s)
           │       │
           │       ├── simplified = sympy.simplify(expr)
           │       │                = (x + 1)²
           │       │
           │       └── cache.set(key, simplified)
           │
           ├── embed = math_embed("Simplified", "(x + 1)**2")
           │
           ├── save_history(user_id, "simplify", "x^2 + 2x + 1", "(x + 1)**2")
           │
           └── await interaction.followup.send(embed=embed)
```

### B. Expression Parsing Decision Tree

```
Input string
     │
     ▼
_validate_raw()
  ├─ len > MAX_EXPR_LENGTH → ValueError
  └─ contains FORBIDDEN_KEYWORD → ValueError
     │
     ▼
_detect_format()
  ├─ leading \  OR known LaTeX macro  OR x^{2}  → "latex"
  ├─ contains **  OR starts with math.            → "python"
  ├─ contains "squared"/"plus"/"times"/etc.        → "natural"
  └─ (default)                                     → "plain"
     │
     ▼
┌──────────┬──────────────┬────────────────┬──────────┐
│  latex   │   python     │   natural      │  plain   │
│          │              │                │          │
│latex2sym │_normalize_   │_normalize_     │_normalize│
│py2()     │python()→     │natural()→      │_plain()→ │
│          │parse_expr()  │parse_expr()    │parse_expr│
│ fail?    │              │                │          │
│  ↓       │              │                │          │
│fallback  │              │                │          │
│to plain  │              │                │          │
└──────────┴──────────────┴────────────────┴──────────┘
     │
     ▼
sympy.Expr (returned to cog)
```

### C. Plot Session Flow

```
/plot
  │
  ▼
PlotConfig (defaults)
  │
  ▼
PlotEngineView sent as message
  │
  ├── User selects plot type → cfg.plot_type updated
  │
  ├── User clicks "Expression" → ExpressionModal
  │     └── on_submit → cfg.expr_main = validated string
  │
  ├── User clicks "Preview" → _on_preview()
  │     └── plotter.plot_function(cfg) → PNG → edit message
  │
  ├── User clicks "Animate" → AnimationParamModal
  │     └── on_submit → _render_animation(cfg) → GIF
  │
  └── User clicks "Render" → final render, view disabled
           │
           └── export_config() → base64 string
                  (user can share via /plot_import)
```

---

## 13. Security Model

MathFrame runs user-supplied strings through `sympy.parse_expr()` and `sympy.sympify()`, both of which internally call Python's `eval()`. This is inherently risky. The following defenses are in place:

### Defense Layer 1: Input Length Cap

`config.MAX_EXPR_LENGTH = 500` characters. Rejected immediately by `_validate_raw()` before any parsing begins. Prevents memory-exhaustion attacks from absurdly long inputs.

### Defense Layer 2: Forbidden Keyword Blocklist

`FORBIDDEN_KEYWORDS = ["__", "import", "exec", "eval", "open", "os", "sys", "subprocess"]`

Checked in `_validate_raw()` before parsing. The `__` check is a substring test; all others use word-boundary regex. Prevents the most obvious injection vectors: importing modules, executing shell commands, or accessing the file system.

### Defense Layer 3: Compute Timeout

`config.COMPUTE_TIMEOUT = 3` seconds. Enforced by `asyncio.wait_for()` around the thread pool parse job. SymPy can hang indefinitely on pathological inputs (e.g. integrals of special functions with no closed form). This prevents any single invocation from blocking the event loop indefinitely.

### Defense Layer 4: Thread Pool Isolation

Parsing runs in a `ThreadPoolExecutor(max_workers=4)`. A hung computation is cancelled by `asyncio.wait_for`, but the thread itself may continue running until SymPy eventually returns. The pool cap prevents unbounded thread proliferation.

### Defense Layer 5: Boolean Parser (cogs/discrete.py)

The discrete math cog implements a **hand-rolled recursive-descent parser** for boolean expressions that never calls `eval()` or `sympify()`. This is the gold standard for user input safety and serves as a reference implementation for how the rest of the codebase could eventually be strengthened.

### Defense Layer 6: JSON Matrix Parsing (cogs/linear_algebra.py)

Matrix and vector inputs are parsed via the standard library `json` module, not SymPy. Since matrix input is structurally a list of numbers — not a symbolic expression — this sidesteps the expression-parsing machinery entirely.

### Remaining Vulnerabilities

| Location | Issue | Risk level |
|----------|-------|------------|
| `utils/expr_utils.py::_sympy_expr()` | Calls `sympy.sympify()` directly (now with `_validate_raw()` guard after fix) | Low-Medium |
| `cogs/calculus.py::_parse_point()` | Direct `sympy.sympify()` on single values | Low |
| `cogs/symbolic.py::_parse_substitutions()` | Direct `sympy.sympify()` on RHS values, after `str.isidentifier()` on LHS | Low |
| `cogs/base_n.py` `/base_logic` | Intentional restricted `eval()` with builtins stripped | Low |

The project's own `math_bot_coding_plan.md` states: *"Do NOT skip the validation in `parser.py`. Unsanitized `eval()` is a security hole."*

---

## 14. Known Issues & Open Items

### Issue #1: Plotting Expressions Bypass `parser.py` — ✅ RESOLVED

`utils/expr_utils.py::_sympy_expr()` now calls `_validate_raw()` from `parser.py` before calling `sympify`, applying the same length cap and forbidden-keyword filter as the main expression parser.

**Remaining exposure:** Two isolated direct-`sympify` call sites remain: `cogs/calculus.py::_parse_point()` and the substitution-value parser in `cogs/symbolic.py`. Both are narrow single-value cases (not full expression strings) and are considered lower priority.

---

### Issue #2: No Per-User `interaction_check` on Shared Interactive Views — OPEN

**Affected:** `PlotEngineView` (`cogs/plot_engine.py`) and the `PaginatorView` (`utils/paginator.py`).

**Problem:** Any user in the channel can operate another user's in-progress plot session (changing expression, triggering renders) or page through another user's article paginator.

**Reference implementation:** `cogs/utility.py`'s `_ConfirmClearView` does this correctly — it stores `owner_id` at construction and overrides `interaction_check()` to reject other users. This exact pattern should be applied to both affected views.

**Fix template:**
```python
async def interaction_check(self, interaction: discord.Interaction) -> bool:
    if interaction.user.id != self.owner_id:
        await interaction.response.send_message(
            "This session belongs to another user.", ephemeral=True
        )
        return False
    return True
```

---

### Issue #3: Non-ASCII Characters in Code and UI Labels — OPEN (by design)

`utils/solver.py::_expr_str()` explicitly uses `sympy.pretty(expr, use_unicode=False)` for ASCII-safety. However, this intent is not consistently followed elsewhere:

- `utils/formatter.py::error_embed()` hardcodes `title="❌ Error"` (Unicode emoji)
- `cogs/utility.py`'s constants table uses π/φ/√2/∞/−/→
- `cogs/linear_algebra.py` uses ×/⁻¹/λ in embed footers
- `cogs/number_theory.py` uses ≤/¹²/¹⁵ in command descriptions
- `cogs/wiki.py` uses em dashes in titles
- `utils/paginator.py` uses ◀/▶ for buttons

In practice, Discord's UI handles Unicode well across all modern clients, so this is a low-priority cosmetic inconsistency rather than a functional bug.

---

### Issue #4: `utils/utility.py` is Dead Code — OPEN

A ~490-line near-duplicate of `cogs/utility.py` that is never imported or loaded. Its docstring header mislabels it as `cogs/utility.py`, suggesting it was an earlier draft. **Safe to delete with `git rm utils/utility.py`.**

---

### Issue #5: `.gitignore` Pattern Scope — PARTIALLY RESOLVED

The original `/__pycache__` (leading slash) only matched the repository root. This has been corrected to `__pycache__/` (no leading slash), which now matches all subdirectory `__pycache__` folders.

**Remaining:** A stray ~970 KB PNG file (`utils/ChatGPT Image Jun 15, 2026, 11_57_30 AM.png`) is still committed inside `utils/`. It doesn't belong in the codebase and should be removed with `git rm "utils/ChatGPT Image Jun 15, 2026, 11_57_30 AM.png"`.

---

### Issue #6: Narrow Exception Handling — ✅ RESOLVED

`arithmetic.py`, `calculus.py`, and `symbolic.py` now catch `ValueError`, `sympy.SympifyError`, `sympy.PolynomialError`, `NotImplementedError`, and a broad `Exception` fallback in each command, surfacing specific SymPy error messages rather than the generic global error handler.

---

### Issue #7: `latex2sympy2` Dependency — WATCH

Used for LaTeX-format parsing in `parser.py`. Less actively maintained than core SymPy and has an `antlr4` dependency that has occasionally caused install friction on Windows and in containerized environments. Worth watching if `/latex`-style input or any LaTeX-detected expression ever breaks after a fresh environment setup. Fallback to plain parsing is implemented but may produce different results.

---

### Issue #8: `aiohttp` Not in `requirements.txt` — OPEN

`cogs/wiki.py` imports `aiohttp` directly, but `aiohttp` is only available as a transitive dependency of `discord.py`. If `discord.py` ever drops `aiohttp` as a dependency (or the version constraint changes), `cogs/wiki.py` will fail to load silently (it will appear in the `failed` cogs list). **Add `aiohttp>=3.9.0` to `requirements.txt`.**

---

## 15. Dependency Reference

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

| Package | Used for |
|---------|---------|
| `discord.py` | Bot framework: slash commands, embeds, views, modals, intents |
| `python-dotenv` | Loading `DISCORD_TOKEN` from `.env` file |
| `sympy` | All symbolic mathematics — the computational core |
| `latex2sympy2` | Parsing LaTeX-format math input (wraps antlr4) |
| `cachetools` | `TTLCache` for the in-memory result cache |
| `matplotlib` | Plot generation and LaTeX mathtext rendering (headless Agg backend) |
| `numpy` | Numerical arrays for plot sampling, math evaluation |
| `scipy` | Probability distributions (`scipy.stats`) for statistics commands |
| `aiohttp` | *(transitive via discord.py)* HTTP client for Wikipedia API in `cogs/wiki.py` |

### Python Version

Python **3.12+** is required due to:
- `list[str]`, `dict[int, deque]` generic type hints in assignments
- `int | None` union syntax (PEP 604, Python 3.10+)
- `match` statement usage (if any)
- `datetime | None` default parameter type annotation

---

## 16. Codebase Metrics

### File Sizes (lines of Python source)

| File | Lines |
|------|-------|
| `utils/plotter.py` | 2,246 |
| `cogs/plot_engine.py` | 1,779 |
| `cogs/statistics.py` | 1,044 |
| `cogs/discrete.py` | 878 |
| `cogs/utility.py` | 811 |
| `cogs/number_theory.py` | 724 |
| `cogs/symbolic.py` | 597 |
| `utils/solver.py` | 612 |
| `utils/parser.py` | 352 |
| `utils/paginator.py` | 242 |
| `cogs/arithmetic.py` | 860 |
| `cogs/calculus.py` | ~700 |
| `cogs/linear_algebra.py` | 422 |
| `cogs/geometry.py` | 626 |
| `cogs/wiki.py` | 421 |
| `utils/formatter.py` | 225 |
| `utils/renderer.py` | 174 |
| `cogs/complex.py` | 241 |
| `cogs/equations.py` | 262 |
| `cogs/transforms.py` | 229 |
| `cogs/base_n.py` | ~250 |
| `cogs/inequalities.py` | 176 |
| `cogs/render.py` | 146 |
| `data/permissions.py` | 205 |
| `data/history.py` | 114 |
| `data/cache.py` | 112 |
| `main.py` | 237 |
| `config.py` | 51 |
| **Total (approx.)** | **~15,600** |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total cogs loaded | 18 |
| Total slash commands | 65+ |
| Total source files (non-pycache) | ~30 |
| Dead code files | 1 (`utils/utility.py`) |
| Thread pools | 3 (parser, renderer, plotter) |
| Supported plot types | 13 |
| Supported expression input formats | 4 (latex, python, natural, plain) |
| Unit conversion categories | 10 |
| Max in-memory cache entries | 256 (LRU+TTL) |
| Max history entries per user | 20 |
| Compute timeout | 3 seconds |
| Max expression length | 500 characters |

---

*Documentation generated from MathFrame source code (commit: 2026-06-27).*
