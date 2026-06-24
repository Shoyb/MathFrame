# MathFrame — Codebase Documentation

MathFrame is a Discord bot for symbolic and numeric mathematics, built on
`discord.py` (slash commands) and `SymPy`, with `matplotlib`/`numpy`/`scipy`
powering an interactive plotting subsystem. It is organized as a cog-based
bot: one `main.py` entry point loads a set of independent feature modules
("cogs"), each contributing a family of related `/slash` commands.

This document is a map of the repository as it stands — what each file
does, how data flows between them, every command the bot exposes, and a
running list of known issues worth fixing.

```
MathFrame/
├── main.py                  Bot entry point, cog loader, global error handler
├── config.py                Environment-driven configuration constants
├── requirements.txt
├── math_bot_coding_plan.md  Original design/build-order document
├── data/
│   ├── cache.py              In-memory TTL result cache (singleton)
│   └── history.py            In-memory per-user command history (singleton)
├── utils/
│   ├── parser.py              THE expression parser — format detection + validation
│   ├── expr_utils.py          Separate, lighter-weight parser used only by plotting
│   ├── solver.py               Step-by-step solution builders (quadratic, diff, integral, factor)
│   ├── formatter.py            Discord embed builders (success/error/info)
│   ├── renderer.py             LaTeX/SymPy → PNG image rendering (matplotlib)
│   ├── paginator.py            Reusable ◀/▶ embed paginator view
│   ├── plotter.py               Low-level plotting engine (matplotlib figure builders)
│   └── utility.py               ⚠ Orphaned duplicate of cogs/utility.py — not loaded anywhere
└── cogs/
    ├── arithmetic.py        /simplify /solve /expand /factor
    ├── calculus.py           /diff /integrate /limit /series
    ├── linear_algebra.py     /matrix_det /matrix_inv /eigenvalues /dot /cross /rref
    ├── statistics.py          /mean /median /mode /stdev /variance /zscore /correlation /regression /normal_pdf
    ├── number_theory.py       /gcd /lcm /is_prime /factorize /primes_up_to /modular /fibonacci
    ├── geometry.py             /circle_area /circle_circumference /triangle_area /pythagorean /trig /distance
    ├── discrete.py              /permutation /combination /truth_table /set_ops /binomial_coeff
    ├── symbolic.py               /latex /subs /partial_fraction /roots
    ├── utility.py                  /history /clear_history /constants /help_math /convert /about
    ├── render.py                    /render /formula
    ├── plot_engine.py                 /plot_import /plot /quickplot /multiplot  (interactive plot builder)
    └── wiki.py                         /wiki /wiki_search
```

Total: ~10,800 lines of Python across 12 cogs and 8 utility/data modules
(excluding `__pycache__`, which — see Known Issues — should not be in the
repo at all).

---

## 1. Entry point and bootstrapping

### `main.py`
Creates the `commands.Bot` instance with `message_content` intent enabled,
defines the ordered `COGS` list that's loaded inside `on_ready()`, registers
a `/ping` command, and installs a global `app_commands.error` handler
(`on_app_command_error`) that catches `CommandOnCooldown`,
`MissingPermissions`, and `BotMissingPermissions` with specific user-facing
messages, falling back to a generic "something went wrong" embed for
anything else (with the full traceback logged). This handler is the safety
net for any uncaught exception thrown inside a cog's command body — since
discord.py wraps callback exceptions in `CommandInvokeError`, nothing
propagates to crash the bot, but it does mean unexpected exceptions
(anything not explicitly caught as `ValueError` in a cog) surface to the
user as a generic message rather than the specific SymPy error.

### `config.py`
Loads `.env` via `python-dotenv` and exposes:

| Constant | Value | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | from env | Bot login token |
| `PREFIX` | `"!"` | Legacy prefix (unused by slash commands, required by the `commands.Bot` constructor) |
| `MAX_EXPR_LENGTH` | 500 | Hard cap on expression string length before parsing |
| `COMPUTE_TIMEOUT` | 3 (s) | Wall-clock cap on a single parse/compute job |
| `CACHE_TTL` | 300 (s) | Result cache lifetime |
| `CACHE_MAXSIZE` | 256 | Result cache LRU capacity |

---

## 2. Data layer (`data/`)

Both modules are pure in-memory, process-lifetime singletons — nothing is
persisted to disk or a database (a deliberate choice noted in the code,
since the hosting environment doesn't support one).

**`cache.py`** wraps a `cachetools.TTLCache` behind a `threading.Lock`,
exposing `get`/`set`/`clear`/`info` and a `cache_key(*args)` helper that
joins arguments with `|`. Used by several cogs (e.g. `/simplify`,
`/expand`, `/factor`) to skip recomputation for repeated identical
requests within the TTL window.

**`history.py`** keeps a `dict[user_id -> deque(maxlen=20)]` of
`HistoryEntry(command, input, result, timestamp)` records, also behind a
lock. `save_history` / `get_history` / `clear_history` are the public API,
consumed by `cogs/utility.py`'s `/history` and `/clear_history` commands.

---

## 3. Expression parsing (`utils/parser.py` vs `utils/expr_utils.py`)

This is the most important architectural seam in the codebase, and
currently the most inconsistent one.

**`utils/parser.py`** is documented as *"the only module in the project
that calls `parse_expr` or `latex2sympy`. Every cog must go through
`parse_expression` and work with the returned `sympy.Expr` — never call
the underlying parsers directly."* It implements:

- `_validate_raw()` — rejects input over `MAX_EXPR_LENGTH` or containing
  any of `FORBIDDEN_KEYWORDS` (`__`, `import`, `exec`, `eval`, `open`,
  `os`, `sys`, `subprocess`).
- `_detect_format()` — heuristically classifies input as `latex`
  (leading `\`, known LaTeX macro, or braced exponent `x^{2}`), `python`
  (`**` or `math.` prefix), `natural` (prose keywords like "squared",
  "plus"), or `plain` (default — caret + implicit multiplication).
- Per-format normalizers (`_normalize_plain`, `_normalize_natural`) that
  rewrite the string before handing off to `sympy.parse_expr` with
  `standard_transformations + implicit_multiplication_application`, or to
  `latex2sympy2.latex2sympy` for LaTeX input.
- `parse_expression()` — the async public entry point. Runs the blocking
  parse inside a shared `ThreadPoolExecutor(max_workers=4)`, wrapped in
  `asyncio.wait_for(..., timeout=COMPUTE_TIMEOUT)`, translating timeouts
  and parser exceptions into a single user-friendly `ValueError`.

This is used correctly by `arithmetic.py`, most of `calculus.py`,
`symbolic.py`, and `render.py`.

**`utils/expr_utils.py`** is a second, separate, much thinner parser built
specifically for plotting. `_clean_sympy_expr()` strips an optional
`name = ` assignment prefix, rewrites `^` → `**` and `e^`/`e**` → `exp`,
and detects/expands `condition: expr | condition: expr` piecewise syntax.
`_sympy_expr()` then calls `sympy.sympify(s, locals=local)` **directly**
— with no length check and no `FORBIDDEN_KEYWORDS` check at all. This is
the function that parses every expression typed into the `/plot` builder
modals (main expression, vector field components, parametric `x(t)/y(t)/z(t)`,
polar, animation parameter expressions) in both `cogs/plot_engine.py` and
`utils/plotter.py`.

Two smaller call sites also bypass `parse_expression`:
`cogs/calculus.py::_parse_point()` (limit/series evaluation points) and a
substitution-value parser inside `cogs/symbolic.py`'s `/subs` command —
both call `sympy.sympify()` directly on a single value rather than a full
expression, which narrows but doesn't eliminate the exposure.

> **Why this matters:** `sympy.sympify`/`parse_expr` parse by compiling
> the string and calling Python's `eval()` against a globals dict that is
> not restricted to exclude builtins. The project's own
> `math_bot_coding_plan.md` states this explicitly: *"Do NOT skip the
> validation in `parser.py`. Unsanitized `eval()` is a security hole."*
> The plotting path currently does skip it. See Known Issues §1.

---

## 4. Supporting utilities (`utils/`)

**`solver.py`** — synchronous, takes an already-parsed `sympy.Expr` and
returns a `StepList` (`list[(description, expression_string)]`) that
`formatter.math_embed` renders as a numbered "Steps" field. Four
generators: `solve_quadratic_steps` (extracts a/b/c, shows the
discriminant, lists roots — returns an `[("Error", ...)]` sentinel for
non-degree-2 input rather than raising), `differentiate_steps` (supports
arbitrary order, simplifying after each pass), `integrate_steps` (detects
and labels unevaluated `Integral` results when SymPy can't find a closed
form), `factor_steps`. Internally uses `_expr_str()`, which deliberately
calls `sympy.pretty(expr, use_unicode=False)` "so the output stays
ASCII-safe across all Discord clients" — see Known Issues §3 for why this
intent isn't followed everywhere else.

**`formatter.py`** — three embed builders used by every cog:
`math_embed(title, result, steps=None, footer="")` (blurple, code-blocked
result, optional steps field truncated to fit Discord's 1024-char field
limit via `_format_steps`), `error_embed(message)` (red, title `"❌
Error"`), `info_embed(title, description)` (gold). Also exposes
`to_readable_text()`, a small regex-based converter from SymPy's `**`/`*`
notation to `^`/implicit-multiplication for display.

**`renderer.py`** — turns a LaTeX string or SymPy expression into a PNG
`discord.File` via matplotlib's `mathtext` rendering (headless `Agg`
backend), run inside its own dedicated `ThreadPoolExecutor(max_workers=2)`
separate from the parser's pool so rendering and parsing never starve each
other. Two public async functions: `expr_to_image(latex_str)` and
`result_to_image(sympy_expr)`. Backs `/render`, `/formula`, and `/latex`.

**`paginator.py`** — `PaginatorView(pages, timeout=120)`, a
`discord.ui.View` with ◀/▶ buttons that edit the message in place,
auto-disabling the appropriate button at either end and stamping
`"Page N / Total"` into each embed's footer (appended after existing
footer text via a `·` separator). `send_paginated(interaction, pages)` is
the convenience wrapper most cogs call — it picks
`response.send_message` vs `followup.send` automatically and skips
attaching a view entirely when there's only one page. Note: this view has
no per-user `interaction_check`, so any user in the channel can page
through (though not corrupt) another user's paginated result — see Known
Issues §2.

**`plotter.py`** (2,001 lines, the largest utility module) — the actual
matplotlib figure builders: `plot_function`, `plot_contour`, `plot_surface`,
`plot_wireframe`, `plot_vector_field`, `plot_parametric_2d/3d`,
`plot_polar`, `plot_implicit`, `plot_inequality`, `plot_scatter_3d`,
`plot_heatmap`, `plot_points`, `plot_multi` (side-by-side panels), plus animation-frame
variants used by `plot_engine.py`'s `_render_animation`. Takes a
`PlotSpec`/`StyleOptions` pair (defined here, imported by the cog) carrying
domain bounds, resolution (`resolution_1d` for line/parametric plots,
`resolution_2d` for contour/surface — see §5), color/theme/colormap,
fill/limits overrides, etc. All user expressions reach this module already
parsed via `expr_utils._sympy_expr`, per §3 above.

**`utility.py`** — ⚠ **not loaded by the bot.** This file's own docstring
header reads `"cogs/utility.py — Utility slash commands..."`, and it
defines a complete second `UtilityCog`, its own constants table, its own
unit-conversion helpers, and its own `_ConfirmView` — essentially an
earlier draft of what is now `cogs/utility.py`. `main.py`'s `COGS` list
only references `"cogs.utility"`, and nothing in the codebase imports
`utils.utility`, so this ~490-line file is dead code sitting in the
repository. See Known Issues §4.

---

## 5. The plot builder (`cogs/plot_engine.py`)

The largest and most actively developed cog (1,713 lines). Centers on a
single `@dataclass PlotConfig` that is the source of truth for one
in-progress plot session — plot type, all expression fields
(`expr_main`/`expr_u`/`expr_v`/`expr_x`/`expr_y`/`expr_z`, etc.), domain
bounds, `resolution_1d` (100–2000, line/parametric) and `resolution_2d`
(40–400, contour/surface — these were split from a single non-functional
`resolution` field in an earlier pass), style options, colormap, theme,
fill/limit overrides, and animation parameters. `PlotConfig.export_config`/
`import_config` serialize/deserialize a session to a base64+zlib string so
sessions can be shared between users via `/plot_import`.

Thirteen supported `PLOT_TYPES`: function, contour, vector-field,
parametric-2d, surface, wireframe, parametric-3d, scatter, scatter-3d,
polar, implicit, inequality, heatmap.

UI is built from a stack of `discord.ui.Modal` subclasses (one per concern
— `ExpressionModal`, `StyleModal`, `AxesModal`, `AdvancedModal`,
`FillModal`, `LimitsModal`, `AnimationParamModal`, `AdditionalExprModal`)
opened from buttons/selects on the persistent `PlotEngineView`. The view also
implements pan/zoom (`_scale_domain`, `_shift_domain` and their button
handlers) and an export/preview/animate/render pipeline.

Four slash commands: `/plot` (opens the full builder), `/quickplot`
(renders a function expression immediately, no UI), `/multiplot` (up to 4
expressions in one side-by-side image), `/plot_import` (loads a shared
config string). `PlotEngineView` doesn't define an `interaction_check`, so the builder's
buttons/selects are not restricted to the user who opened it — see Known
Issues §2.

---

## 6. Command reference

All commands are slash commands (`app_commands`); most `defer()`
immediately and reply via `followup.send`. Cooldowns are applied per-command
via `@app_commands.checks.cooldown`.

### Arithmetic (`cogs/arithmetic.py`)
- `/simplify expression` — simplify a mathematical expression
- `/solve expression [variable=x]` — solve `expression = 0`; shows
  step-by-step working for quadratics
- `/expand expression` — distribute/expand
- `/factor expression` — factor, with steps

### Calculus (`cogs/calculus.py`)
- `/diff expression [variable] [order]` — differentiate, with steps
- `/integrate expression [variable] [lower] [upper]` — definite or
  indefinite integral
- `/limit expression variable point [direction]` — evaluate a limit
- `/series expression [variable] [point] [order]` — Taylor/Maclaurin
  expansion

### Linear Algebra (`cogs/linear_algebra.py`)
- `/matrix_det matrix` — determinant
- `/matrix_inv matrix` — inverse (exact fractions preserved)
- `/eigenvalues matrix` — eigenvalues with algebraic multiplicity
- `/dot a b` — dot product
- `/cross a b` — cross product (3-D vectors)
- `/rref matrix` — reduced row-echelon form

### Statistics (`cogs/statistics.py`)
- `/mean`, `/median`, `/mode`, `/stdev`, `/variance` — descriptive stats
  on a comma-separated data set
- `/zscore value data` — standard score for a value within a data set
- `/correlation x y` — Pearson correlation coefficient
- `/regression x y` — linear regression fit, returns a plotted image
- `/normal_pdf mean stdev` — plots a normal distribution's PDF

### Number Theory (`cogs/number_theory.py`)
- `/gcd numbers`, `/lcm numbers` — GCD/LCM of a list of integers
- `/is_prime n` — primality test (n ≤ 10¹²)
- `/factorize n` — prime factorization (n ≤ 10¹⁵)
- `/primes_up_to n` — list primes up to n (capped)
- `/modular base exp m` — fast modular exponentiation
- `/fibonacci n` — first n Fibonacci numbers (capped)

### Geometry (`cogs/geometry.py`)
- `/circle_area radius`, `/circle_circumference radius`
- `/triangle_area` — area from base/height or three sides
- `/pythagorean` — solve for the missing side of a right triangle
- `/trig function angle` — exact + decimal evaluation
- `/distance` — Euclidean distance between two points (2-D or 3-D)

### Discrete Math (`cogs/discrete.py`)
- `/permutation n r`, `/combination n r` — nPr / nCr
- `/truth_table expression` — boolean truth table
- `/set_ops set_a set_b operation` — union/intersection/difference/etc.
  on comma-separated sets
- `/binomial_coeff n` — nth row of Pascal's triangle

### Symbolic (`cogs/symbolic.py`)
- `/latex expression` — render as a LaTeX PNG
- `/subs expression substitutions` — substitute values, e.g.
  `substitutions: "x=2, y=pi"`
- `/partial_fraction expression` — partial fraction decomposition
- `/roots expression` — all roots, set equal to zero

### Utility (`cogs/utility.py`)
- `/history` — show recent calculation history (in-memory, per-user)
- `/clear_history` — clear it (confirmation view, owner-restricted)
- `/constants` — reference list of π, e, φ, √2, i, ∞ with 10-place decimals
- `/help_math` — paginated command listing grouped by cog
- `/convert value from to` — unit conversion (length, mass, temperature)
- `/about` — bot info

### Rendering (`cogs/render.py`)
- `/render latex` — LaTeX string → PNG
- `/formula expression` — any parseable expression → PNG formula image

### Plotting (`cogs/plot_engine.py`)
- `/plot` — open the full interactive plot builder
- `/quickplot expression [domain]` — instant single-function plot, no UI
- `/multiplot expr1 expr2 ...` — up to 4 functions, side-by-side
- `/plot_import config_string` — load a previously exported plot session

### Wikipedia (`cogs/wiki.py`)
- `/wiki title` — fetch and browse an article paragraph-by-paragraph
- `/wiki_search query` — search and list matching articles

---

## 7. Known issues

Ordered by severity / actionability.

**1. Plotting expressions bypass `parser.py`'s validation entirely.**
`utils/expr_utils.py::_sympy_expr()` calls `sympy.sympify()` directly with
no length check and no `FORBIDDEN_KEYWORDS` filter, and is the parser for
every field in the `/plot` builder (300-char modal inputs). `parse_expr`/
`sympify` parse by `eval()`-ing the (transformed) string against a globals
dict that isn't builtins-restricted. `cogs/calculus.py::_parse_point` and
a substitution parser in `cogs/symbolic.py` also call `sympy.sympify`
directly, on narrower single-value input. Fix: route all three through
`parse_expression()`, or at minimum reuse `_validate_raw()` before calling
`sympify` in each case.

**2. No per-user `interaction_check` on shared interactive views.**
`PlotEngineView`
(`cogs/plot_engine.py`) and the wiki paginator have no ownership check, so
any user in the channel can operate another user's in-progress plot
session or article paginator. `cogs/utility.py`'s confirmation view does
this correctly (`interaction_check` comparing `interaction.user.id` to a
stored `owner_id`) — the same pattern should be applied to the others.

**3. "No non-ASCII in code string literals / UI labels" isn't enforced
project-wide.** `utils/solver.py::_expr_str` explicitly uses
`sympy.pretty(expr, use_unicode=False)` for ASCII-safety, but
`utils/formatter.py::error_embed()` — called by every cog's error path —
hardcodes `title="❌ Error"`; `cogs/utility.py`'s constants table and unit
conversion strings use π/φ/√2/∞/−/→ directly; `cogs/linear_algebra.py`
uses ×/⁻¹/λ in footers; `cogs/number_theory.py`'s command *descriptions*
(visible in Discord's slash-command UI) use ≤/¹²/¹⁵; `cogs/wiki.py` uses
em dashes in titles; `utils/paginator.py` uses ◀/▶ for its buttons.

**4. `utils/utility.py` is dead code.** A ~490-line near-duplicate of
`cogs/utility.py` that is never imported or loaded (`main.py`'s `COGS`
list only references `cogs.utility`). Its own docstring header
mislabels it as `cogs/utility.py`, suggesting it was an earlier
draft left behind after the real cog was finished elsewhere. Safe to
delete.

**5. `.gitignore` doesn't actually ignore `__pycache__`.** The pattern is
`/__pycache__` (leading slash → repo-root only), so
`cogs/__pycache__`, `utils/__pycache__`, and `data/__pycache__` are not
matched and their `.pyc` files (23 of them) are tracked in git. A stray
~970 KB PNG (`utils/ChatGPT Image Jun 15, 2026, 11_57_30 AM.png`) is also
committed inside `utils/` and doesn't appear to belong in the codebase.
Fix: change the pattern to `__pycache__` (no leading slash), then
`git rm -r --cached` the tracked cache directories and the stray image.

**6. Narrow exception handling in several cogs.** `arithmetic.py`,
parts of `calculus.py`, and `symbolic.py` only catch `ValueError` around
SymPy calls. Anything else SymPy raises (`PolynomialError`,
`NotImplementedError` on some integrals, etc.) isn't caught locally and
falls through to `main.py`'s generic "something went wrong" handler
instead of a specific, useful error message — not a crash risk (the
global handler catches it), but a UX gap.

**7. `latex2sympy2` dependency.** Used for LaTeX-format parsing in
`parser.py`; less actively maintained than core SymPy and has an
`antlr4` dependency that has occasionally caused install friction on
Windows. Worth keeping in mind if `/latex`-style input ever breaks after
a fresh environment setup.

---

## 8. Per-file design notes

The sections above describe how the modules fit together. This section
goes one level deeper — the internal design of each file: its helper
functions, classes, data structures, and the specific implementation
choices behind them. Files are grouped the same way as the repository
tree (`main.py`/`config.py`, then `data/`, then `utils/`, then `cogs/`),
and within `cogs/`, alphabetically.

### `main.py`

No classes — a flat script. `on_ready()` is idempotent-safe for
reconnects (`if not hasattr(bot, "start_time")` guards against resetting
the uptime clock on every reconnect) and loads cogs in a loop with
per-cog exception handling so one broken cog doesn't prevent the rest
from loading. `_ephemeral_reply()` is a small dispatch helper that picks
`followup.send` vs `response.send_message` based on
`interaction.response.is_done()` — the same "has this interaction been
responded to yet" pattern shows up independently in `utils/paginator.py`
(`_send`). `on_app_command_error` is the bot's single global error
boundary; every cog relies on it as a fallback for anything not caught
locally.

### `config.py`

No functions or classes — six module-level constants read once at import
time via `python-dotenv`'s `load_dotenv()`. Acts as the single source of
tunable limits (`MAX_EXPR_LENGTH`, `COMPUTE_TIMEOUT`, `CACHE_TTL`,
`CACHE_MAXSIZE`) that other modules import rather than hardcoding.

### `data/cache.py`

Five free functions operating on one module-level `TTLCache` singleton
(`_cache`) guarded by one module-level `threading.Lock` (`_lock`).
`cache_key(*args)` is intentionally permissive — it stringifies and joins
any positional arguments with `|`, so callers build keys like
`cache_key("simplify", expression)` without a fixed schema. There's no
class here by design: a single shared cache instance is all the bot
needs, so a module-level singleton avoids the ceremony of a class with
one instance.

### `data/history.py`

One small data class, `HistoryEntry` (uses `__slots__` for memory
efficiency since many short-lived instances accumulate per user), plus
three free functions over a module-level
`dict[user_id -> deque(maxlen=20)]` (`_histories`), also lock-guarded.
`save_history` uses `dict.setdefault` to lazily create a user's deque on
first use and `appendleft` so the deque is always newest-first — the
`maxlen` on the deque itself is what enforces the 20-entry cap (oldest
entries are silently dropped by `deque`, no manual trimming needed).

### `utils/parser.py`

The most carefully validated module in the codebase (see §3 for why this
matters). Structure: `_validate_raw()` (length + forbidden-keyword
gate) → `_detect_format()` (returns one of `latex`/`python`/`natural`/
`plain` via a priority-ordered chain of regex/substring checks) →
format-specific normalizer (`_normalize_plain`, `_normalize_natural`, or
a lazy `latex2sympy2` import for LaTeX) → `_parse_blocking()` (the
synchronous dispatcher that ties detection and normalization together) →
`parse_expression()` (the async public surface, wrapping the blocking
call in `loop.run_in_executor` + `asyncio.wait_for`). The format-detection
regexes are deliberately ordered most-specific-first (LaTeX macros and
braced exponents are checked before the bare `**`/`math.` check for
"python", which is checked before natural-language keywords) so that an
expression matching multiple heuristics resolves predictably rather than
by accident of dict/set iteration order. `FORBIDDEN_KEYWORDS` is a flat
list checked with simple substring containment (`if kw in expr`) — fast,
but also why `_sympy_expr` in `expr_utils.py` not performing the same
check is a meaningful gap (see Known Issues §1).

### `utils/expr_utils.py`

The smaller, second expression parser used only by the plotting code
path (see §3). `_clean_sympy_expr()` handles three things in sequence:
stripping an optional `name = ` assignment prefix via a precompiled
regex (`_ASSIGNMENT_PREFIX_RE`), caret/`e^`/`e**` normalization, and
detecting piecewise syntax (presence of both `:` and `|`) to delegate to
`_clean_piecewise_expr()`, which splits on `|`, requires a `:` in each
segment, and reassembles the pieces into SymPy's
`Piecewise((expr, cond), ...)` constructor call as a string. `_sympy_expr()`
is a thin wrapper around `sympy.sympify()` that builds a `locals` dict
mapping each symbol's string name to the actual `sympy.Symbol` object
(so `x`, `y`, `t`, etc. resolve to the same symbol instances the caller
already created) plus an explicit `Piecewise` entry — but, unlike
`parser.py`, performs no length or keyword validation before calling
`sympify`.

### `utils/solver.py`

Four public step-builders share one shape: build a `StepList`
incrementally, wrap the whole body in `try/except`, and on any exception
return `_err(message)` (a one-element `[("Error", message)]` list)
instead of raising — callers never need a second `try/except` around
these functions, only around the `parse_expression`/SymPy calls that
feed them. `solve_quadratic_steps` explicitly checks `poly.degree() != 2`
and bails out via `_err` rather than attempting to force a non-quadratic
into the discriminant formula. `differentiate_steps` branches its step
labels on `order == 1` vs `order > 1`, using `_ordinal_superscript()`
(a small dict-based lookup for ¹/²/³/⁴, falling back to `^n` for higher
orders) to label repeated-differentiation passes. `factor_steps` adds
its "Expand first" step conditionally — only when
`sympy.simplify(expanded - expr) != 0`, i.e. only when expanding actually
changed something, avoiding a redundant identical step for already-expanded
input. `_expr_str()` is the one place in the file that explicitly chose
ASCII-safe output (`sympy.pretty(..., use_unicode=False)`), though the
step labels surrounding it elsewhere in the same file use Unicode (Δ, ∫, ℤ).

### `utils/formatter.py`

Three public builders, each returning a plain `discord.Embed` — no
classes, since embeds are themselves Discord's data structure and don't
need wrapping. `_format_steps()` implements field-limit truncation
manually: it computes a `budget` equal to `_STEPS_FIELD_LIMIT` minus the
length of a fixed truncation notice, then accumulates step strings one at
a time, stopping (and appending the notice) the moment the next step
would exceed budget — this guarantees the returned string is always
`≤ 1024` characters, Discord's hard field-value limit, without ever
needing to retroactively trim already-formatted text. `to_readable_text()`
is a separate, lighter-weight display transform (not used for
parsing — only for display) that uses a single lookaround regex to strip
multiplication signs between adjacent letters/digits/parens
(`2*x → 2x`) while leaving numeric multiplication (`2*3`) untouched.

### `utils/paginator.py`

One view class, `PaginatorView`, plus two free functions
(`send_paginated`, the public entry point most cogs call, and `_send`,
its internal response-routing helper). The view takes a defensive copy of
every embed at construction time (`_copy_embed`, via
`Embed.from_dict(embed.to_dict())`, since `discord.Embed` doesn't expose
`__copy__`) specifically so that footer-stamping (`_stamp_all_footers`,
which appends `"Page N / Total"` after any existing footer text via a
`·` separator) never mutates embeds the caller still holds a reference
to. `_refresh_buttons()` looks its buttons up by `custom_id` rather than
by list position, which makes the disable-at-the-edges logic robust to
subclassing or reordering. `send_paginated()` short-circuits entirely for
0 or 1 pages (no view is attached for a single page, since there's
nothing to navigate), and only constructs a `PaginatorView` when there
are 2+ pages.

### `utils/renderer.py`

Two public async functions (`expr_to_image`, `result_to_image`) over one
private blocking function (`_render_to_bytes`) and a dedicated
`ThreadPoolExecutor(max_workers=2)` kept separate from `parser.py`'s pool
so a backlog of plot/render jobs can't starve expression parsing or vice
versa. Renders via matplotlib's `mathtext` engine in headless `Agg` mode
(`matplotlib.use("Agg")` is called before `pyplot` is imported, which is
required — importing `pyplot` first would lock in a different backend).
`result_to_image` converts a `sympy.Basic` to LaTeX internally and
delegates to the same code path as `expr_to_image`, so there's exactly
one rendering implementation regardless of whether the caller starts
from a raw LaTeX string or an already-parsed SymPy object.

### `utils/plotter.py`

By far the largest module (2,001 lines). Two `@dataclass`es form its
public type surface: `StyleOptions` (visual styling — color, line
width/style, marker, colormap, theme, alpha, grid, DPI, figure size, log
scales, fill, axis-limit overrides; has one method, `rc_overrides()`,
that maps theme names to matplotlib `rcParams` dict overrides) and
`PlotSpec` (a per-panel descriptor used only by `plot_multi`, bundling a
`kind` string with every possible field a panel might need — expression,
variable(s), domain ranges, parametric/vector-field components,
scatter data, etc. — so one spec shape can describe any of the sixteen
plot kinds without a class hierarchy).

The implementation pattern is consistent across all sixteen plot types:
a private `_plot_<kind>_blocking()` function does the actual matplotlib
work synchronously (lambdify the SymPy expression(s) to a numeric
function via `_lambdify1`/`_lambdify2`/`_lambdify3`, evaluate over a
`numpy` grid via `_eval1`/`_eval2`/`_meshgrid`, build the figure, save to
bytes via `_save_fig_to_bytes`), and a public async wrapper of the same
name without the `_blocking` suffix (`plot_function`, `plot_contour`,
etc.) runs it through `_run_blocking()` — a thin helper around
`loop.run_in_executor` shared by every wrapper, so the executor/threading
boilerplate exists in exactly one place rather than being repeated
sixteen times. `_smart_ylim()` is a shared heuristic (8% padding by
default) used by multiple 1-D plot types to avoid curves touching the
plot edge. The animation functions (`_plot_animation_*_blocking`,
`plot_animation`) follow the same blocking/async split but build a
matplotlib `FuncAnimation` across a fixed parameter sweep
(`_anim_param_values`) and serialize it to GIF bytes
(`_save_animation_to_gif`) rather than a single PNG. `_render_spec_onto_axes`
is the dispatcher `plot_multi` uses to draw one `PlotSpec` onto one subplot
`Axes`, switching on `spec.kind`.

### `utils/utility.py` ⚠ (dead code — not imported anywhere)

Documented in §7. Structurally it mirrors `cogs/utility.py`: a
`_ConfirmView` (Yes/No confirmation, simpler than `cogs/utility.py`'s
`_ConfirmClearView` — no `owner_id`/`interaction_check`, just a generic
`_finish()` helper called by both buttons) and a `UtilityCog` with
`/constants`, `/help_math`, `/convert`, backed by its own private
`_convert_length`/`_convert_mass`/`_convert_temperature`/
`_detect_unit_category` helpers — all independent reimplementations of
logic that also exists (in more complete form, with `/history` and
`/about` added) in `cogs/utility.py`.

### `cogs/arithmetic.py`

No module-level helper functions — the simplest cog in the codebase.
Four commands on `ArithmeticCog`, all following an identical four-step
shape: build a cache key and return early on a hit, call
`parse_expression`, run one SymPy function (`simplify`/`solve`/`expand`/
`factor`), build and cache a `math_embed`. `/solve` is the one command
that doesn't use the result cache (solutions plus step-by-step working
are cheap enough, and the steps depend on `solve_quadratic_steps`'
separate non-cached path) — it instead inspects whether
`solve_quadratic_steps` returned an `[("Error", ...)]` sentinel and
suppresses the steps field entirely when the equation isn't a quadratic,
rather than showing a misleading "Error" steps block alongside a valid
result.

### `cogs/calculus.py`

Two module-level helpers: `_ordinal(n)` (English ordinal suffixes for
display, e.g. "2nd derivative") and `_parse_point(point_str)` (handles
`oo`/`+oo`/`-oo`/`inf` specially before falling back to
`sympy.sympify` for everything else — one of the direct-`sympify` call
sites flagged in Known Issues §1). `CalculusCog` has four commands
(`diff`, `integrate`, `limit`, `series`); `integrate` branches on whether
both `lower` and `upper` were supplied (definite) or left blank
(indefinite) rather than exposing them as separate commands.

### `cogs/discrete.py`

The cog with the most custom parsing logic in the project — and notably,
the one place where a parser was hand-written specifically to *avoid*
`eval()`. `_tokenize_boolean()` splits a boolean expression string into
operator/paren/variable tokens via regex. `_BoolParser` is a small
hand-rolled recursive-descent parser (`_parse_implies` → `_parse_xor` →
`_parse_or` → `_parse_and` → `_parse_not` → `_parse_atom`, each level
calling the next to encode operator precedence, mirroring the classic
grammar-as-call-stack technique) that turns the token list into a nested
tuple AST; `_evaluate_boolean()` then walks that AST recursively against
a `{variable: bool}` dict. The result is a fully sandboxed boolean
evaluator with no `eval()`/`sympify()` anywhere in the path — a useful
contrast with the SymPy-based parsers elsewhere in the project.
Supporting helpers: `_format_large_int()` (falls back to scientific
notation above a configurable digit threshold so `1000!` doesn't blow
out an embed field), `_build_truth_table_lines()` /
`_paginate_table()` (render and then chunk a truth table's text rows to
fit Discord's field limit), and `_parse_set_element()` /`_parse_set()` /
`_format_set()` for the comma-separated set-operation commands
(`_parse_set_element` tries `int` → `float` → raw string, in that order,
so `"1, 2.5, apple"` parses each element as specifically as possible).

### `cogs/geometry.py`

Two helpers: `_exact_and_decimal()` (the shared "show both the exact
SymPy form and a decimal approximation" pattern used across most of this
cog's commands) and `_to_exact()` (converts a `float` slash-command
argument back to an exact `sympy.Rational`/`Integer` via
`sympy.nsimplify`-style logic, since Discord's `float` option type loses
exactness that the rest of the pipeline wants to preserve). Commands like
`triangle_area` and `pythagorean` accept multiple optional parameters
(`base`/`height` vs `a`/`b`/`c`; or any two of `a`/`b`/`c`) and branch
internally on which combination was actually supplied, rather than
requiring separate commands per input mode.

### `cogs/linear_algebra.py`

Three helpers: `_format_matrix()` (plain-text grid rendering for embeds),
`_parse_json_list()` / `_parse_vector()` (matrices and vectors are
accepted as JSON array strings, e.g. `[[1,2],[3,4]]`, parsed with the
standard library `json` module rather than SymPy's parser — a
deliberate choice, since matrix/vector input is structurally a list of
numbers, not a symbolic expression, so it doesn't need the
expression-parsing machinery at all and sidesteps that whole code path's
risks). `LinearAlgebraCog` adds one cog-level method,
`parse_matrix(self, s)`, used by all six matrix/vector commands to go
from the JSON string to a `sympy.Matrix`.

### `cogs/number_theory.py`

The most helper-function-heavy cog. `_superscript()` (digit → Unicode
superscript, used by `_format_factorisation` to render `2³ × 3² × 5¹`),
`_parse_integers()`, `_validate_int_arg()` (generic `[lo, hi]` range
validation reused across most commands), `_list_gcd()`/`_list_lcm()`
(iterative pairwise reduction over `math.gcd`/`lcm` rather than a single
n-ary call, so the running result can be short-circuited early if it
ever hits 1), `_fibonacci_list()`, and `_paginate_list()` (generic
page-splitting used by `/primes_up_to` and `/fibonacci` for large `n`).
Every numeric command enforces an explicit upper bound (`is_prime` ≤
10¹², `factorize` ≤ 10¹⁵, `primes_up_to`/`fibonacci` capped via
module constants) specifically to bound compute time, since none of
these go through `parser.py`'s `COMPUTE_TIMEOUT` guard — they're raw
Python integer arithmetic, not SymPy parsing.

### `cogs/plot_engine.py`

See §5 for the full architectural description. At the design level: the
single `PlotConfig` `@dataclass` (~50 fields) acts as a mutable
"document" that every modal mutates a slice of and every render call
reads from in full — a deliberate single-source-of-truth choice over,
say, passing a dozen separate parameters through the view/modal chain.
`export_config`/`import_config` round-trip a `PlotConfig` through
`json.dumps` → `zlib.compress` → `base64` (and back), which is what
makes `/plot_import` possible — a whole session is just a copy-pasteable
string. Each `ui.Modal` subclass follows the same two-method shape:
`__init__` pre-fills its `TextInput` fields from the current `cfg` (so
reopening a modal shows your last values, not blank fields), and
`on_submit` parses each field back into `cfg` using small tolerant
parsers (`_parse_float`, `_parse_int`, `_parse_bool`, `_parse_floatlist`,
`_parse_optional_float`) that fall back to a default rather than raising
on bad input, then calls back into the view to re-render. `PlotEngineView`
centralizes pan/zoom math in `_scale_domain`/`_shift_domain` (shared by
the zoom in/out and pan left/right/up/down button handlers) and exposes
one private handler per button/select (`_on_expr`, `_on_style`,
`_on_axes`, `_on_advanced`, `_on_fill`, `_on_limits`, `_on_preview`,
`_on_animate`, `_on_render`, `_on_reset`,
etc.) — a flat, one-concern-per-method structure rather than a single
dispatch-by-custom_id handler.

### `cogs/render.py`

No module-level helpers. Two thin commands, both delegating directly to
`utils/renderer.py`: `/render` takes raw LaTeX, `/formula` takes any
format `parser.py` understands and renders the resulting parsed
expression — the two commands exist because `/render` skips
`parse_expression` entirely (the input is assumed to already be valid
LaTeX) while `/formula` is "parse first, render second," giving users two
different mental models for getting to the same image output depending
on whether they're more comfortable writing LaTeX or plain expressions.

### `cogs/statistics.py`

Module-level `parse_numbers()` (comma-separated → `list[float]`, used by
nearly every command in the cog) plus three more specialized helpers:
`_correlation_label()` (maps a Pearson *r* value to a human-readable
strength description — "strong", "moderate", etc.), and
`_regression_plot_bytes()` / `_normal_pdf_bytes()`, which build their
matplotlib figures directly in this cog file rather than through
`utils/plotter.py` — i.e. `statistics.py` has its own small, independent
plotting code path parallel to (not reusing) the much larger plotting
engine, since its needs (a scatter + fitted line, or a single PDF curve)
are simple enough not to warrant going through `PlotSpec`/`StyleOptions`.

### `cogs/symbolic.py`

`_parse_substitutions()` (splits a `"x=2, y=pi"` string on commas, then
each entry on its first `=`, validating that the left side is a legal
Python identifier via `str.isidentifier()` before calling
`sympy.sympify()` on the right-hand value — the second direct-`sympify`
call site noted in Known Issues §1) and `_root_line()` (formats one root
with a "real"/"complex" complexity tag for the `/roots` command's
output). `SymbolicCog` holds four commands; `/subs` is the only one that
needs its own substitution-string grammar rather than reusing
`parse_expression`, since a substitution map isn't itself a single
expression.

### `cogs/utility.py`

The active utility cog (loaded by `main.py`; not to be confused with the
orphaned `utils/utility.py`). Helpers: `_exact_and_decimal()` (duplicated
from `geometry.py` rather than shared — both cogs define their own
copy), `_format_uptime()` (renders a `timedelta` as `"1d 2h 3m 4s"`,
omitting any leading zero units), `_lib_version()` (used by `/about` to
report installed package versions), and a matched trio for `/convert`'s
temperature path — `_temp_to_kelvin`/`_temp_from_kelvin`/`_temp_steps` —
which route every temperature conversion through Kelvin as a common
intermediate (C→F goes through K rather than using a direct C→F formula)
so the conversion table only needs `to_kelvin`/`from_kelvin` per unit
instead of a function for every unit pair. `_ConfirmClearView` is the one
view in the whole codebase with a real `interaction_check` (compares
`interaction.user.id` against a stored `owner_id`, disabling itself via
`_disable_all()` on timeout) — the pattern flagged as missing elsewhere
in Known Issues §2.

### `cogs/wiki.py`

The only cog that talks to an external HTTP API (Wikipedia's REST API),
and the only one that imports `aiohttp` directly — note that `aiohttp`
isn't listed in `requirements.txt`; it's currently available only as a
transitive dependency of `discord.py`, which happens to pull it in.
`WikiCog` implements `cog_load()`/`cog_unload()` to create and tear down
one shared `aiohttp.ClientSession` for the cog's lifetime (rather than
opening a new session per request), exposed via a `session` property.
Module-level helpers: `_fetch_summary()`/`_fetch_sections()` (two
separate Wikipedia endpoints — summary for the thumbnail/intro,
mobile-sections for the full paragraph-by-paragraph content),
`_search_wikipedia()`, `_strip_html()`/`_clean()` (MediaWiki markup
cleanup), `_thumbnail_url()` (defensively walks a possibly-missing nested
dict structure in the API response), and `_article_pages()` /
`_search_result_embed()`, which convert raw API data into the
`list[discord.Embed]` shape `utils/paginator.py::send_paginated` expects
— this cog is `send_paginated`'s primary multi-page consumer in the
codebase, alongside `/help_math` and `/primes_up_to`/`/fibonacci`.

---

## 9. Dependencies (`requirements.txt`)

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
