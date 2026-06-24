# MathFrame — Improvement Plan

This document outlines all planned improvements to the MathFrame Discord bot,
organized by category. Each item includes a description of the problem, the
proposed fix, and implementation notes.

---

## Security

### 1. Route plotting expressions through `parser.py` validation

**Problem:**
`utils/expr_utils.py::_sympy_expr()` calls `sympy.sympify()` directly with no
length check and no `FORBIDDEN_KEYWORDS` filter. This function is the parser for
every expression field in the `/plot` builder — main expression, vector field
components, parametric `x(t)/y(t)/z(t)`, polar, and animation parameter
expressions. The project's own `math_bot_coding_plan.md` explicitly flags
unsanitized `eval()` as a security hole.

**Fix:**
Before calling `sympify()` in `_sympy_expr()`, reuse `_validate_raw()` from
`parser.py` to enforce the length cap (`MAX_EXPR_LENGTH`) and check for
`FORBIDDEN_KEYWORDS`. Alternatively, refactor `_sympy_expr()` to delegate
fully to `parse_expression()` where async context allows, or extract
`_validate_raw()` into a shared `utils/validation.py` module so both parsers
import from the same source.

**Files affected:** `utils/expr_utils.py`, optionally `utils/parser.py`

---

### 2. Fix the two remaining direct `sympify()` call sites

**Problem:**
Two additional locations bypass `parser.py`'s sanitizer:
- `cogs/calculus.py::_parse_point()` — used for limit/series evaluation points
- `cogs/symbolic.py::_parse_substitutions()` — used for the `/subs` command's
  right-hand values

Both call `sympy.sympify()` directly on user input, with no forbidden-keyword
check.

**Fix:**
In `_parse_point()`, add a `_validate_raw()` call before `sympify()` — input
is a single value string so the full `parse_expression()` async path isn't
needed, but keyword filtering must still apply. In `_parse_substitutions()`,
apply the same inline validation before each `sympify()` call on the
right-hand side of each substitution pair.

**Files affected:** `cogs/calculus.py`, `cogs/symbolic.py`

---

## Error Handling

### 6. Broaden exception handling in arithmetic, calculus, and symbolic cogs

**Problem:**
`arithmetic.py`, parts of `calculus.py`, and `symbolic.py` only catch
`ValueError` around SymPy calls. SymPy can also raise `PolynomialError`,
`NotImplementedError` (on some integrals and series), `CoercionFailed`, and
others. These fall through to `main.py`'s generic global handler, which
responds with a vague "something went wrong" message instead of a specific,
actionable one.

**Fix:**
Extend the `except` clauses in each affected command to also catch
`sympy.PolynomialError`, `NotImplementedError`, and `Exception` as a final
fallback, each with a tailored message. For example:
- `NotImplementedError` → `"SymPy couldn't find a closed form for this."`
- `PolynomialError` → `"Expression couldn't be treated as a polynomial."`

**Files affected:** `cogs/arithmetic.py`, `cogs/calculus.py`, `cogs/symbolic.py`

---



## Code Quality

### 10. Extract statistics cog's inline matplotlib code into `plotter.py`

**Problem:**
`cogs/statistics.py` contains its own mini plotting path —
`_regression_plot_bytes()` and `_normal_pdf_bytes()` — that builds matplotlib
figures directly inside the cog file, completely independent of the main
plotting engine in `utils/plotter.py`. This creates two parallel, inconsistent
code paths for producing plot images (different DPI, different styling, no
`StyleOptions` support).

**Fix:**
Move `_regression_plot_bytes()` and `_normal_pdf_bytes()` into `utils/plotter.py`
as `plot_regression()` and `plot_normal_pdf()`, following the same
blocking/async split pattern used by every other plot function in that module
(`_plot_<kind>_blocking` + async wrapper). Update `cogs/statistics.py` to call
the new functions from `plotter.py`.

**Files affected:** `utils/plotter.py`, `cogs/statistics.py`

---

## Features — New Commands

### 12. Per-server command permissions

**Problem:**
There are no admin controls to restrict which commands are available in which
channels or servers. Any user in any channel can run any command, including
computationally expensive ones.

**Fix:**
Add a simple guild-level permissions system:
- Store an allow/deny list per guild and per channel using a JSON file or
  SQLite (one table: `guild_id`, `channel_id`, `command_name`, `enabled`).
- Add an `/admin` cog (`cogs/admin.py`) with commands `/admin enable`,
  `/admin disable`, and `/admin status`, restricted to users with the
  `Manage Guild` Discord permission.
- Check the allow/deny list in a `before_invoke` hook registered in `main.py`
  so the logic applies globally without modifying each cog.

**Files affected:** `main.py`, new `cogs/admin.py`, new `data/permissions.py`

---

### 14. LaTeX rendering fallback

**Problem:**
If `latex2sympy2` fails to parse a LaTeX-format input (which can happen after
library updates or with edge-case LaTeX syntax), `parser.py` raises a
`ValueError` with no fallback. The user gets an error with no alternative path.

**Fix:**
In `parser.py`'s LaTeX branch inside `_parse_blocking()`, wrap the
`latex2sympy2.latex2sympy()` call in a `try/except`. On failure, attempt a
second parse using SymPy's own `parse_expr()` with
`standard_transformations + implicit_multiplication_application` as a fallback,
logging the original `latex2sympy2` error for debugging. If both fail, raise
the user-facing `ValueError`.

**Files affected:** `utils/parser.py`

---

### 20. `/solve` support for systems of equations

**Problem:**
`/solve` only handles a single expression for one variable. Solving systems of
equations (e.g. `x + y = 5, x - y = 1`) requires users to use `/rref`
manually, which returns row-reduced form rather than clean `x = ..., y = ...`
output.

**Fix:**
Add a `/solve_system` command to `cogs/arithmetic.py` (or the planned new
`cogs/equations.py`). Accept multiple equations as a single string separated
by commas or semicolons, parse each through `parse_expression()`, and pass the
list to `sympy.solve()` with the detected free symbols. Return results
formatted as `variable = value` per line, with a fallback to `sympy.linsolve()`
for linear systems for cleaner output.

**Files affected:** `cogs/arithmetic.py` or new `cogs/equations.py`

---

### 21. `/units` command for dimensional analysis

**Problem:**
`/convert` handles unit conversion for a fixed set of hardcoded categories
(length, mass, temperature). It can't handle compound units, derived units, or
unit arithmetic (e.g. `9.8 m/s^2`).

**Fix:**
Add a `/units` command to `cogs/utility.py` backed by
`sympy.physics.units.convert_to()`. Accept an expression with units (e.g.
`9.8 * meter / second**2`) and a target unit string, and return the converted
value. Since `sympy.physics.units` is already available as part of the SymPy
install, this requires no new dependency. Keep the existing `/convert` command
for simple everyday conversions.

**Files affected:** `cogs/utility.py`

---

### 22. Step-by-step polynomial solver up to degree 4

**Problem:**
`/solve` shows step-by-step working only for quadratics via
`solve_quadratic_steps()`. Cubics and quartics get raw SymPy output with no
working shown, and the step builder in `utils/solver.py` explicitly bails out
for non-degree-2 input.

**Fix:**
Add `solve_cubic_steps()` and `solve_quartic_steps()` to `utils/solver.py`,
following the same `StepList` pattern as `solve_quadratic_steps()`. For
cubics: show the depressed cubic substitution and Cardano's method steps. For
quartics: show the resolvent cubic approach. Update `/solve` in
`cogs/arithmetic.py` to dispatch to the correct step builder based on
`poly.degree()`.

**Files affected:** `utils/solver.py`, `cogs/arithmetic.py`

---

### 23. Simultaneous equation solver (`/solve_sim`)

**Problem:**
The 991CW has a dedicated simultaneous equation solver for 2×2 and 3×3 linear
systems that returns clean `x = ..., y = ..., z = ...` output. MathFrame has
no equivalent — `/rref` returns a matrix, not a solution set.

**Fix:**
Add a `/solve_sim` command (to a new `cogs/equations.py` or
`cogs/arithmetic.py`) that accepts equations as a comma-separated string,
parses them, detects the free variables, and calls `sympy.linsolve()` for
linear systems or `sympy.solve()` for non-linear. Format output as one
`variable = value` line per variable, with exact fractions preserved.

**Files affected:** new `cogs/equations.py` or `cogs/arithmetic.py`

---

### 24. Table mode (`/table`)

**Problem:**
The 991CW's TABLE mode generates a value table for f(x) over a range with a
configurable step — one of the most-used calculator features for students.
MathFrame has no equivalent.

**Fix:**
Add a `/table expression start end step` command to `cogs/arithmetic.py` or a
new `cogs/utility.py` section. Parse the expression through `parse_expression()`,
evaluate it at each point using `sympy.lambdify()` over a `numpy.arange()`, and
format the result as a paginated embed table (`x | f(x)` rows). Cap the number
of rows (e.g. 200) to avoid embed flooding, and use the existing
`utils/paginator.py` for multi-page output.

**Files affected:** `cogs/arithmetic.py` or `cogs/utility.py`,
`utils/paginator.py`

---

### 25. Complex number cog (`cogs/complex.py`)

**Problem:**
The 991CW handles complex numbers natively — rectangular/polar form conversion,
argument, conjugate, modulus, and arithmetic. MathFrame has no complex number
commands at all, even though SymPy fully supports them.

**Fix:**
Create a new `cogs/complex.py` cog with the following commands:
- `/complex_calc expression` — evaluate any complex expression (e.g. `(2+3i)*(1-i)`)
- `/complex_polar expression` — convert to polar form (modulus + argument)
- `/complex_rect r theta` — convert polar → rectangular
- `/complex_conjugate expression` — return the conjugate
- `/complex_modulus expression` — return `|z|`

Parse input through `parse_expression()` with `I` mapped to `sympy.I`.
Register the new cog in `main.py`'s `COGS` list.

**Files affected:** new `cogs/complex.py`, `main.py`

---

### 26. Base-N arithmetic cog (`cogs/base_n.py`)

**Problem:**
The 991CW has a dedicated base-N mode for binary, octal, decimal, and
hexadecimal conversions and arithmetic. MathFrame has no equivalent.

**Fix:**
Create a new `cogs/base_n.py` cog with the following commands:
- `/base_convert value from_base to_base` — convert between any two bases (2–36)
- `/base_add a b base` — add two numbers in a given base, show result in same base
- `/base_logic a b operation base` — AND/OR/XOR/NOT on integers in a given base
- `/bases value` — show a decimal value in binary, octal, and hex simultaneously

Use Python's built-in `int(value, base)` and `format(n, 'b'/'o'/'x')` — no
SymPy needed. Register in `main.py`'s `COGS` list.

**Files affected:** new `cogs/base_n.py`, `main.py`

---

### 27. Inequality solver cog (`cogs/inequalities.py`)

**Problem:**
`/plot` can graph inequalities visually but there is no command that *solves*
an inequality symbolically and returns the solution set (e.g.
`x^2 - 3x + 2 < 0` → `1 < x < 2`).

**Fix:**
Create a new `cogs/inequalities.py` cog with the following commands:
- `/solve_ineq expression` — solve a single inequality, return solution set
  as an interval or union of intervals using `sympy.solve_univariate_inequality()`
- `/solve_ineq_system expressions` — solve a system of inequalities using
  `sympy.reduce_inequalities()`

Parse input through `parse_expression()`. Format results using SymPy's
`Interval` and `Union` pretty-printing. Register in `main.py`'s `COGS` list.

**Files affected:** new `cogs/inequalities.py`, `main.py`

---

### Numerical differentiation at a point

**Problem:**
`/diff` performs symbolic differentiation but cannot evaluate `f'(a)` at a
specific numeric point when the symbolic derivative is unavailable or the user
just wants a decimal answer. The 991CW supports this natively.

**Fix:**
Add an optional `at` parameter to `/diff` in `cogs/calculus.py`. When
supplied, substitute the value into the symbolic derivative result using
`sympy.subs()` and return both the symbolic derivative and the numeric
evaluation. If symbolic differentiation fails, fall back to a numerical
estimate using the central difference formula via `numpy` and flag it as
approximate in the embed footer.

**Files affected:** `cogs/calculus.py`

---

### 31. Summation and product commands (`/sum_series`, `/product_series`)

**Problem:**
The 991CW has Σ and Π functions for evaluating finite sums and products.
SymPy has `summation()` and `product()` built in but MathFrame exposes no
commands for them.

**Fix:**
Add two commands to `cogs/calculus.py`:
- `/sum_series expression variable lower upper` — evaluate `Σ f(n)` from
  `lower` to `upper` (or `oo` for infinite series) using `sympy.summation()`
- `/product_series expression variable lower upper` — evaluate `Π f(n)` using
  `sympy.product()`

Parse the expression through `parse_expression()`. For infinite upper bounds,
accept `oo` as a string and map to `sympy.oo`. Display both exact and decimal
results where applicable.

**Files affected:** `cogs/calculus.py`

---

### 32. Polynomial division (`/poly_div`)

**Problem:**
The 991CW can divide two polynomials and show quotient and remainder
separately. MathFrame has no equivalent — `/simplify` and `/factor` don't
expose the quotient/remainder decomposition.

**Fix:**
Add a `/poly_div dividend divisor [variable]` command to `cogs/arithmetic.py`.
Parse both expressions through `parse_expression()`, then call
`sympy.div(dividend, divisor, variable)` which returns `(quotient, remainder)`.
Display both in the embed result, plus a verification line
`dividend = divisor × quotient + remainder`.

**Files affected:** `cogs/arithmetic.py`

---

### 33. Expression equivalence checker (`/verify`)

**Problem:**
Students frequently want to check whether their simplified answer matches the
expected form. There is no command for this — they'd have to `/simplify` both
sides and visually compare.

**Fix:**
Add a `/verify expr_a expr_b` command to `cogs/arithmetic.py`. Parse both
expressions through `parse_expression()`, then evaluate
`sympy.simplify(expr_a - expr_b)`. If the result is `0`, return a green
"✅ Equivalent" embed; otherwise return a red "❌ Not equivalent" embed showing
the simplified difference. Add a note in the footer that equivalence checking
uses symbolic simplification and may time out for complex expressions.

**Files affected:** `cogs/arithmetic.py`

---

### 34. Extended statistics — CDF and inverse CDF (`/cdf`, `/inv_normal`)

**Problem:**
The 991CW has cumulative distribution functions for normal, binomial, and
Poisson distributions, plus inverse normal. MathFrame's `/normal_pdf` only
plots the PDF curve — no CDF, no inverse CDF, and no binomial or Poisson
support at all.

**Fix:**
Add the following commands to `cogs/statistics.py`, backed by `scipy.stats`:
- `/normal_cdf mean stdev lower upper` — P(lower ≤ X ≤ upper) for a normal dist
- `/inv_normal probability mean stdev` — inverse normal (find x given P(X ≤ x))
- `/binomial_cdf n p x` — P(X ≤ x) for Binomial(n, p)
- `/poisson_cdf lambda x` — P(X ≤ x) for Poisson(λ)

`scipy.stats` is already a declared dependency. Return both the probability
value and a small plotted CDF curve image using the existing plotting path in
`utils/plotter.py`.

**Files affected:** `cogs/statistics.py`, `utils/plotter.py`

---

### 35. Differential equations (`/ode`)

**Problem:**
The 991CW doesn't support ODEs, but adding them would make MathFrame
meaningfully more powerful than a physical calculator. SymPy's `dsolve()`
handles first and second order ODEs cleanly, including initial value problems.

**Fix:**
Add a `/ode expression [initial_conditions]` command to `cogs/calculus.py`.
Accept expressions written with `f(x)` and `f'(x)` notation (or `y` and
`y'`), parse the ODE, and call `sympy.dsolve()`. Accept optional initial
conditions as a `"f(0)=1, f'(0)=0"` style string, parsed similarly to
`_parse_substitutions()` in `symbolic.py`. Display the general or particular
solution in the embed, rendered as an image via `utils/renderer.py`.

**Files affected:** `cogs/calculus.py`, `utils/parser.py`

---

### 36. `/quickplot` domain auto-detection

**Problem:**
`/quickplot` defaults to a fixed domain if no range is specified. For
functions with poles, asymptotes, or interesting behaviour in non-default
ranges (e.g. `ln(x)`, `tan(x)`, `1/x`), the default window often produces
an unhelpful or visually broken plot.

**Fix:**
In `cogs/plot_engine.py`'s `/quickplot` handler, before passing the domain to
`plotter.py`, add an auto-detection step:
- Lambdify the expression and sample it over a broad range (`-50` to `50`).
- Find the largest contiguous region where the function is finite and
  well-defined.
- Clip the y-axis to `±10 × median(|f(x)|)` to suppress spike artifacts near
  discontinuities.
- Use the detected range as the default, overrideable by the user's explicit
  `domain` argument.

**Files affected:** `cogs/plot_engine.py`, `utils/plotter.py`

---

### 37. Multiplot legend labels

**Problem:**
`/multiplot` renders up to 4 functions as side-by-side subplots but doesn't
label which curve corresponds to which expression in the image itself. Users
have to mentally match the input order to the panel order.

**Fix:**
In `utils/plotter.py`'s `plot_multi()` function, add a title to each subplot
axes using `ax.set_title(expr_str, fontsize=9, pad=4)`, truncating long
expressions to a configurable character limit (e.g. 40 chars) with an ellipsis.
Optionally also add a colour-coded legend if multiple expressions are drawn on
the same axes panel.

**Files affected:** `utils/plotter.py`

---

## Summary Table

| # | Category | Item | Files |
|---|---|---|---|
| 1 | Security | Route plot expressions through `parser.py` | `utils/expr_utils.py` |
| 2 | Security | Fix direct `sympify()` call sites | `cogs/calculus.py`, `cogs/symbolic.py` |
| 6 | Error Handling | Broaden exception handling in cogs | `cogs/arithmetic.py`, `cogs/calculus.py`, `cogs/symbolic.py` |
| 10 | Code Quality | Extract stats plotting into `plotter.py` | `utils/plotter.py`, `cogs/statistics.py` |
| 12 | Feature | Per-server command permissions | `main.py`, new `cogs/admin.py`, new `data/permissions.py` |
| 14 | Feature | LaTeX rendering fallback | `utils/parser.py` |
| 16 | Error Handling | Computation timeout per cog | `config.py`, multiple cogs |
| 20 | Feature | `/solve` system of equations | `cogs/arithmetic.py` |
| 21 | Feature | `/units` dimensional analysis | `cogs/utility.py` |
| 22 | Feature | Polynomial solver up to degree 4 | `utils/solver.py`, `cogs/arithmetic.py` |
| 23 | Feature | Simultaneous equation solver | new `cogs/equations.py` |
| 24 | Feature | Table mode `/table` | `cogs/arithmetic.py` or `cogs/utility.py` |
| 25 | Feature | Complex number cog | new `cogs/complex.py` |
| 26 | Feature | Base-N arithmetic cog | new `cogs/base_n.py` |
| 27 | Feature | Inequality solver cog | new `cogs/inequalities.py` |
| — | Feature | Numerical differentiation at a point | `cogs/calculus.py` |
| 31 | Feature | Summation and product commands | `cogs/calculus.py` |
| 32 | Feature | Polynomial division `/poly_div` | `cogs/arithmetic.py` |
| 33 | Feature | Expression equivalence checker `/verify` | `cogs/arithmetic.py` |
| 34 | Feature | Extended statistics CDF/inverse CDF | `cogs/statistics.py`, `utils/plotter.py` |
| 35 | Feature | Differential equations `/ode` | `cogs/calculus.py` |
| 36 | Feature | `/quickplot` domain auto-detection | `cogs/plot_engine.py`, `utils/plotter.py` |
| 37 | Feature | Multiplot legend labels | `utils/plotter.py` |
