# Math Discord Bot — Python Coding Plan

A complete, ordered coding plan for building a math-heavy Discord bot in Python.
Follow phases top to bottom. Each phase builds on the one before it.

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Discord | `discord.py` (nextcord) | Bot framework, slash commands |
| CAS | `sympy` | Symbolic math engine |
| LaTeX parsing | `latex2sympy2` | LaTeX → SymPy conversion |
| Numerics | `numpy`, `scipy` | Numerical computation |
| Plotting | `matplotlib` | Graph/plot image generation |
| Database | `aiosqlite` | Async SQLite for user history |
| Caching | `cachetools` | In-memory TTL cache |
| Timeout | `concurrent.futures` | Guard against infinite computations |
| Config | `python-dotenv` | Load `.env` secrets |

Install everything:
```
pip install "discord.py[voice]" sympy latex2sympy2 numpy scipy matplotlib aiosqlite cachetools python-dotenv
```

---

## Folder Structure

```
math_bot/
├── main.py                  # Bot entry point
├── config.py                # Tokens, constants, settings
├── .env                     # DISCORD_TOKEN (never commit this)
│
├── cogs/                    # One file per math domain (loaded as Discord cogs)
│   ├── __init__.py
│   ├── arithmetic.py        # Phase 3
│   ├── calculus.py          # Phase 4
│   ├── linear_algebra.py    # Phase 4
│   ├── statistics.py        # Phase 4
│   ├── number_theory.py     # Phase 5
│   ├── geometry.py          # Phase 5
│   ├── discrete.py          # Phase 5
│   └── symbolic.py          # Phase 5
│
├── utils/                   # Shared utilities used by all cogs
│   ├── __init__.py
│   ├── parser.py            # Phase 2 — expression parser
│   ├── renderer.py          # Phase 2 — LaTeX → PNG image
│   ├── plotter.py           # Phase 2 — matplotlib graph generation
│   ├── solver.py            # Phase 2 — step-by-step solver engine
│   ├── formatter.py         # Phase 2 — Discord embed builder
│   └── paginator.py         # Phase 2 — paginated embed views
│
└── data/
    ├── __init__.py
    ├── db.py                # Phase 2 — SQLite async wrapper
    └── cache.py             # Phase 2 — TTL result cache
```

---

## Phase 1 — Project Bootstrap

**Goal:** Bot connects to Discord, responds to a ping, loads cogs.

### `config.py`
```python
import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
PREFIX = "!"
MAX_EXPR_LENGTH = 500        # reject absurdly long inputs
COMPUTE_TIMEOUT = 3          # seconds before killing a computation
CACHE_TTL = 300              # seconds to keep cached results
CACHE_MAXSIZE = 256
```

### `main.py`
```python
import discord
from discord.ext import commands
import config, os

intents = discord.Intents.default()
bot = commands.Bot(command_prefix=config.PREFIX, intents=intents)

COGS = [
    "cogs.arithmetic", "cogs.calculus", "cogs.linear_algebra",
    "cogs.statistics", "cogs.number_theory", "cogs.geometry",
    "cogs.discrete", "cogs.symbolic",
]

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    for cog in COGS:
        await bot.load_extension(cog)
    await bot.tree.sync()   # sync slash commands globally

bot.run(config.DISCORD_TOKEN)
```

**Deliverable:** Bot goes online, `/ping` returns "Pong!".

---

## Phase 2 — Shared Utilities

**Goal:** Build the infrastructure all cogs share. Do this before any math commands.

### `utils/parser.py` — Expression Parser

```python
import re, sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application
)
from latex2sympy2 import latex2sympy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import config

FORBIDDEN_KEYWORDS = ["__", "import", "exec", "eval", "open", "os", "sys", "subprocess"]
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_executor = ThreadPoolExecutor(max_workers=4)

def _validate_raw(expr: str) -> None:
    """Raise ValueError if expression looks dangerous."""
    if len(expr) > config.MAX_EXPR_LENGTH:
        raise ValueError(f"Expression too long (max {config.MAX_EXPR_LENGTH} chars).")
    for kw in FORBIDDEN_KEYWORDS:
        if kw in expr:
            raise ValueError(f"Expression contains forbidden keyword: `{kw}`")

def _detect_format(expr: str) -> str:
    """Return 'latex', 'python', 'natural', or 'plain'."""
    if expr.startswith("\\") or any(k in expr for k in ["\\frac", "\\int", "\\sum", "\\sqrt"]):
        return "latex"
    if "**" in expr or "math." in expr:
        return "python"
    if re.search(r"\b(squared|cubed|plus|minus|times|divided by|sqrt of)\b", expr, re.I):
        return "natural"
    return "plain"

def _normalize_plain(expr: str) -> sympy.Expr:
    expr = re.sub(r'\^', '**', expr)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)  # 2x → 2*x
    return parse_expr(expr, transformations=TRANSFORMATIONS)

def _normalize_natural(expr: str) -> sympy.Expr:
    subs = {
        r'\bsquared\b': '**2', r'\bcubed\b': '**3',
        r'\bplus\b': '+',      r'\bminus\b': '-',
        r'\btimes\b': '*',     r'\bdivided by\b': '/',
        r'\bsquare root of\b': 'sqrt',
    }
    for pat, rep in subs.items():
        expr = re.sub(pat, rep, expr, flags=re.I)
    return _normalize_plain(expr)

def _parse_blocking(raw: str) -> sympy.Expr:
    """Blocking parse — run inside executor to apply timeout."""
    fmt = _detect_format(raw)
    if fmt == "latex":   return latex2sympy(raw)
    if fmt == "python":  return parse_expr(raw)
    if fmt == "natural": return _normalize_natural(raw)
    return _normalize_plain(raw)

async def parse_expression(raw: str) -> sympy.Expr:
    """
    Main entry point. Async-safe, timeout-guarded.
    Raises ValueError with a user-friendly message on failure.
    """
    _validate_raw(raw)
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        expr = await asyncio.wait_for(
            loop.run_in_executor(_executor, _parse_blocking, raw),
            timeout=config.COMPUTE_TIMEOUT
        )
        return expr
    except asyncio.TimeoutError:
        raise ValueError("Computation timed out. Try a simpler expression.")
    except Exception as e:
        raise ValueError(f"Couldn't parse expression: `{e}`")
```

**Key point:** `parse_expression()` is the only function cogs should ever call.
Cogs receive a `sympy.Expr` — they never call `parse_expr` themselves.

---

### `utils/renderer.py` — LaTeX → PNG

```python
import matplotlib
matplotlib.use("Agg")            # headless, no display
import matplotlib.pyplot as plt
import io, discord

async def expr_to_image(latex_str: str) -> discord.File:
    """Render a LaTeX string to a Discord-uploadable PNG."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"${latex_str}$",
            fontsize=22, ha="center", va="center",
            transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=150, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return discord.File(buf, filename="formula.png")
```

---

### `utils/plotter.py` — Graph Generation

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy, io, discord

async def plot_function(
    expr: sympy.Expr,
    var: sympy.Symbol,
    x_min: float = -10,
    x_max: float = 10,
    title: str = "",
) -> discord.File:
    f = sympy.lambdify(var, expr, modules=["numpy"])
    xs = np.linspace(x_min, x_max, 800)
    ys = f(xs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title(title or str(expr))
    ax.set_xlabel(str(var))
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return discord.File(buf, filename="plot.png")
```

---

### `utils/solver.py` — Step-by-Step Engine

```python
import sympy
from typing import List, Tuple

StepList = List[Tuple[str, str]]   # (description, expression string)

def solve_quadratic_steps(expr: sympy.Expr, var: sympy.Symbol) -> StepList:
    """Return human-readable steps for solving a quadratic = 0."""
    steps: StepList = []
    steps.append(("Original expression", str(expr)))

    expanded = sympy.expand(expr)
    steps.append(("Expand", str(expanded)))

    coeffs = sympy.Poly(expanded, var).all_coeffs()
    steps.append(("Coefficients", f"a={coeffs[0]}, b={coeffs[1]}, c={coeffs[2]}"))

    discriminant = coeffs[1]**2 - 4*coeffs[0]*coeffs[2]
    steps.append(("Discriminant b²−4ac", str(sympy.simplify(discriminant))))

    solutions = sympy.solve(expr, var)
    steps.append(("Solutions", ", ".join(str(s) for s in solutions)))

    return steps

def differentiate_steps(expr: sympy.Expr, var: sympy.Symbol) -> StepList:
    steps: StepList = []
    steps.append(("Original", str(expr)))
    derivative = sympy.diff(expr, var)
    steps.append(("Apply d/dx", str(derivative)))
    simplified = sympy.simplify(derivative)
    if simplified != derivative:
        steps.append(("Simplified", str(simplified)))
    return steps
```

---

### `utils/formatter.py` — Embed Builder

```python
import discord
from typing import Optional

COLOUR = discord.Colour.blurple()

def math_embed(
    title: str,
    result: str,
    steps: Optional[list] = None,
    footer: str = "",
) -> discord.Embed:
    embed = discord.Embed(title=title, colour=COLOUR)
    embed.add_field(name="Result", value=f"```{result}```", inline=False)
    if steps:
        step_text = "\n".join(f"**{i+1}. {desc}**\n`{val}`" for i, (desc, val) in enumerate(steps))
        embed.add_field(name="Steps", value=step_text[:1024], inline=False)
    if footer:
        embed.set_footer(text=footer)
    return embed

def error_embed(message: str) -> discord.Embed:
    return discord.Embed(
        title="Parse error",
        description=message,
        colour=discord.Colour.red()
    )
```

---

### `utils/paginator.py` — Paginated Views

```python
import discord
from typing import List

class PaginatorView(discord.ui.View):
    def __init__(self, pages: List[discord.Embed], timeout: float = 120):
        super().__init__(timeout=timeout)
        self.pages = pages
        self.index = 0
        self._update_buttons()

    def _update_buttons(self):
        self.prev_btn.disabled = self.index == 0
        self.next_btn.disabled = self.index == len(self.pages) - 1

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index -= 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self.pages[self.index], view=self)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index += 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self.pages[self.index], view=self)
```

---

### `data/cache.py` — TTL Cache

```python
from cachetools import TTLCache
import config

_cache: TTLCache = TTLCache(maxsize=config.CACHE_MAXSIZE, ttl=config.CACHE_TTL)

def get(key: str):
    return _cache.get(key)

def set(key: str, value):
    _cache[key] = value

def cache_key(*args) -> str:
    return "|".join(str(a) for a in args)
```

---

### `data/db.py` — SQLite History

```python
import aiosqlite, datetime

DB_PATH = "data/math_bot.db"

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT    NOT NULL,
                command   TEXT    NOT NULL,
                input     TEXT    NOT NULL,
                result    TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            )
        """)
        await db.commit()

async def save_history(user_id: int, command: str, input_: str, result: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, command, input, result, timestamp) VALUES (?,?,?,?,?)",
            (str(user_id), command, input_, result, datetime.datetime.utcnow().isoformat())
        )
        await db.commit()

async def get_history(user_id: int, limit: int = 10):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT command, input, result, timestamp FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (str(user_id), limit)
        ) as cursor:
            return await cursor.fetchall()
```

---

## Phase 3 — Core Cog: Arithmetic & Algebra

**Goal:** `/simplify`, `/solve`, `/expand`, `/factor` working end-to-end.

### `cogs/arithmetic.py`

```python
import discord
from discord import app_commands
from discord.ext import commands
import sympy

from utils.parser import parse_expression
from utils.formatter import math_embed, error_embed
from utils.renderer import expr_to_image
from utils.solver import solve_quadratic_steps
from data.cache import get, set, cache_key
from data.db import save_history

class Arithmetic(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="simplify", description="Simplify a math expression")
    @app_commands.describe(expression="e.g. x^2 + 2x + 1 or \\frac{x^2-1}{x-1}")
    async def simplify(self, interaction: discord.Interaction, expression: str):
        await interaction.response.defer()
        try:
            key = cache_key("simplify", expression)
            if cached := get(key):
                return await interaction.followup.send(embed=cached)

            expr = await parse_expression(expression)
            result = sympy.simplify(expr)
            embed = math_embed("Simplify", str(result))
            set(key, embed)
            await save_history(interaction.user.id, "simplify", expression, str(result))
            await interaction.followup.send(embed=embed)
        except ValueError as e:
            await interaction.followup.send(embed=error_embed(str(e)))

    @app_commands.command(name="solve", description="Solve an equation for x")
    @app_commands.describe(
        expression="Left-hand side (assumes = 0, e.g. x^2 - 4)",
        variable="Variable to solve for (default: x)"
    )
    async def solve(self, interaction: discord.Interaction, expression: str, variable: str = "x"):
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            var = sympy.Symbol(variable)
            steps = solve_quadratic_steps(expr, var)
            solutions = sympy.solve(expr, var)
            embed = math_embed(
                f"Solve for {variable}",
                ", ".join(str(s) for s in solutions),
                steps=steps
            )
            await interaction.followup.send(embed=embed)
        except ValueError as e:
            await interaction.followup.send(embed=error_embed(str(e)))

    @app_commands.command(name="expand", description="Expand a math expression")
    async def expand(self, interaction: discord.Interaction, expression: str):
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            result = sympy.expand(expr)
            await interaction.followup.send(embed=math_embed("Expand", str(result)))
        except ValueError as e:
            await interaction.followup.send(embed=error_embed(str(e)))

    @app_commands.command(name="factor", description="Factor a math expression")
    async def factor(self, interaction: discord.Interaction, expression: str):
        await interaction.response.defer()
        try:
            expr = await parse_expression(expression)
            result = sympy.factor(expr)
            await interaction.followup.send(embed=math_embed("Factor", str(result)))
        except ValueError as e:
            await interaction.followup.send(embed=error_embed(str(e)))

async def setup(bot):
    await bot.add_cog(Arithmetic(bot))
```

**Deliverable:** All four commands working, with caching, history, and error embeds.

---

## Phase 4 — Math Engine Cogs

Build these after Phase 3 is solid. Each follows the same pattern as `arithmetic.py`.

### `cogs/calculus.py` — Commands to implement

| Command | SymPy call | Notes |
|---|---|---|
| `/diff` | `sympy.diff(expr, var, n)` | `n` = order, default 1 |
| `/integrate` | `sympy.integrate(expr, var)` | add `(var, a, b)` for definite |
| `/limit` | `sympy.limit(expr, var, point)` | support `oo` for infinity |
| `/series` | `sympy.series(expr, var, n=6)` | Taylor/Maclaurin |
| `/plot` | `plotter.plot_function()` | returns PNG file |

### `cogs/linear_algebra.py` — Commands to implement

| Command | SymPy / NumPy call | Notes |
|---|---|---|
| `/matrix_det` | `sympy.Matrix(m).det()` | parse matrix from string |
| `/matrix_inv` | `sympy.Matrix(m).inv()` | warn if singular |
| `/eigenvalues` | `sympy.Matrix(m).eigenvals()` | |
| `/dot` | `numpy.dot(a, b)` | |
| `/cross` | `numpy.cross(a, b)` | 3D vectors only |
| `/rref` | `sympy.Matrix(m).rref()` | row-reduced echelon form |

**Matrix input format:** Accept `[[1,2],[3,4]]` as a string and `json.loads()` it.

### `cogs/statistics.py` — Commands to implement

| Command | Library | Notes |
|---|---|---|
| `/mean`, `/median`, `/mode` | `statistics` stdlib | |
| `/stdev`, `/variance` | `statistics` stdlib | |
| `/normal_pdf` | `scipy.stats.norm` | plot the curve |
| `/correlation` | `numpy.corrcoef` | |
| `/regression` | `numpy.polyfit` | linear regression |
| `/zscore` | manual formula | |

---

## Phase 5 — Extended Math Cogs

### `cogs/number_theory.py`
- `/gcd`, `/lcm` — `math.gcd`, extend for list of numbers
- `/is_prime` — `sympy.isprime(n)`
- `/factorize` — `sympy.factorint(n)` → formatted as 2³ × 3 × 7
- `/primes_up_to` — `sympy.primerange(2, n)`
- `/modular` — `pow(a, b, m)` for modular exponentiation
- `/fibonacci` — iterative, up to n=200

### `cogs/geometry.py`
- `/circle_area`, `/circle_circumference`
- `/triangle_area` — support base+height and Heron's formula
- `/pythagorean` — find missing side
- `/trig` — sin/cos/tan with degree/radian toggle
- `/distance` — 2D and 3D point distance

### `cogs/discrete.py`
- `/permutation` — `math.perm(n, r)`
- `/combination` — `math.comb(n, r)`
- `/truth_table` — parse logical expression, output all rows
- `/set_ops` — union, intersection, difference from two lists
- `/binomial_coeff` — Pascal's triangle row

### `cogs/symbolic.py`
- `/latex` — convert any expression to LaTeX string + render PNG
- `/subs` — substitute variable values into expression
- `/partial_fraction` — `sympy.apart(expr, var)`
- `/roots` — `sympy.roots(expr, var)` with multiplicity

---

## Phase 6 — History & Utility Commands

### `cogs/utility.py`

```python
@app_commands.command(name="history", description="View your last 10 calculations")
async def history(self, interaction: discord.Interaction):
    rows = await get_history(interaction.user.id)
    if not rows:
        return await interaction.response.send_message("No history yet.")
    pages = []
    for i in range(0, len(rows), 5):
        chunk = rows[i:i+5]
        embed = discord.Embed(title="Your history", colour=discord.Colour.blurple())
        for cmd, inp, res, ts in chunk:
            embed.add_field(name=f"/{cmd} `{inp}`", value=f"→ `{res}`\n*{ts[:10]}*", inline=False)
        pages.append(embed)
    view = PaginatorView(pages)
    await interaction.response.send_message(embed=pages[0], view=view)
```

Other utility commands: `/help_math` (command list embed), `/constants` (π, e, φ, etc.),
`/convert` (unit conversions via `sympy.physics.units`).

---

## Phase 7 — Error Handling & Edge Cases

Add these guards across all cogs before considering the bot production-ready.

```python
# Global error handler in main.py
@bot.tree.error
async def on_app_command_error(interaction, error):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"Slow down! Try again in {error.retry_after:.1f}s.", ephemeral=True)
    else:
        await interaction.response.send_message(
            "Something went wrong. Try again.", ephemeral=True)
        raise error  # still log it
```

**Edge cases to handle per cog:**

| Situation | Response |
|---|---|
| Division by zero in expression | Catch `ZeroDivisionError`, send error embed |
| Complex/imaginary results | Show result, note it is complex |
| Empty result set (no solutions) | "No real solutions found." |
| Matrix not square (for det/inv) | "Matrix must be square." |
| Overflow (e.g. 9999!) | Timeout catches it; "Too large to compute." |
| User types `1/0` literally | Parser returns `zoo` (SymPy's complex infinity) |

---

## Phase 8 — Polish & Deployment

- Add `@app_commands.checks.cooldown(1, 3.0)` to expensive commands
- Add `ephemeral=True` to error responses (only visible to the user who made the error)
- Add logging: `import logging; logging.basicConfig(level=logging.INFO)`
- Add `/about` command showing bot version, library versions, invite link
- Test every command with: empty input, very long input, LaTeX input, plain text input, natural language

### Running in production (Windows)

```bash
python main.py
```

For always-on hosting: deploy to a VPS or use a service like Railway/Render.
They offer free tiers sufficient for a Discord bot.

---

## Build Order Summary

```
Phase 1  →  Bot connects, cog loader works
Phase 2  →  parser.py, renderer.py, plotter.py, formatter.py, db.py, cache.py
Phase 3  →  arithmetic.py (/simplify, /solve, /expand, /factor)  ← test everything here
Phase 4  →  calculus.py, linear_algebra.py, statistics.py
Phase 5  →  number_theory.py, geometry.py, discrete.py, symbolic.py
Phase 6  →  history command, help command, constants
Phase 7  →  error handling, edge cases, cooldowns
Phase 8  →  logging, deployment, invite link
```

Do NOT skip Phase 2. Every cog depends on `parser.py` and `formatter.py`.
Do NOT skip the validation in `parser.py`. Unsanitized `eval()` is a security hole.
