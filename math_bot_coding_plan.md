# MathFrame — Feature Implementation Plan

> **Scope:** Tier 1 → Tier 2 → Tier 3, ordered by value and dependency.  
> **Codebase baseline:** ~12,000 lines, 16 cogs, 8 utility/data modules.  
> **Convention:** all new code routes expressions through `utils/parser.py::parse_expression()` — never `sympify()` or `parse_expr()` directly.

---

## How to read this document

Each feature block contains:
- **What it is** — plain description
- **Files touched** — every file that needs to change, with reason
- **Implementation steps** — ordered, concrete subtasks
- **Key design decisions** — choices that affect the rest of the codebase
- **Testing checklist** — minimum smoke-test cases

---

# TIER 1 — Quick Wins

---

## T1-1. `/distribution` — collapse all distribution commands

### What it is
A single `/distribution` command that replaces the five separate distribution sub-commands in `statistics.py` (`/normal_pdf`, `/normal_cdf`, `/inv_normal`, `/binomial_cdf`, `/poisson_cdf`). Uses a `type` autocomplete choice to pick the distribution, then gathers parameters dynamically.

### Files touched
| File | Change |
|---|---|
| `cogs/statistics.py` | Add `/distribution` command; mark old commands deprecated (do not delete yet — keep under a `[DEPRECATED]` flag in help text until next major version) |
| `utils/formatter.py` | No change needed; existing `math_embed` handles output |

### Implementation steps

**Step 1 — Define the unified command signature**
```python
@app_commands.command(name="distribution")
@app_commands.describe(
    kind="Distribution type",
    params="Parameters as comma-separated values (see /help_math for order)"
)
@app_commands.choices(kind=[
    app_commands.Choice(name="Normal PDF",         value="normal_pdf"),
    app_commands.Choice(name="Normal CDF",         value="normal_cdf"),
    app_commands.Choice(name="Inverse Normal",     value="inv_normal"),
    app_commands.Choice(name="Binomial CDF",       value="binomial_cdf"),
    app_commands.Choice(name="Poisson CDF",        value="poisson_cdf"),
])
async def distribution(self, interaction, kind: str, params: str): ...
```

**Step 2 — Build a `_DISTRIBUTION_REGISTRY` dict** at module level in `statistics.py`:
```python
_DISTRIBUTION_REGISTRY = {
    "normal_pdf":  {"params": ["value", "mean", "stdev"],  "handler": _dist_normal_pdf},
    "normal_cdf":  {"params": ["value", "mean", "stdev"],  "handler": _dist_normal_cdf},
    "inv_normal":  {"params": ["prob", "mean", "stdev"],   "handler": _dist_inv_normal},
    "binomial_cdf":{"params": ["n", "p", "x"],             "handler": _dist_binomial_cdf},
    "poisson_cdf": {"params": ["lam", "x"],                "handler": _dist_poisson_cdf},
}
```
Each `_dist_*` function is extracted from the body of the existing slash commands — the math stays identical, only the Discord glue moves.

**Step 3 — Write the param parser**
```python
def _parse_dist_params(raw: str, expected: list[str]) -> dict:
    """Parse comma-separated floats into a named dict matching `expected`."""
    values = [v.strip() for v in raw.split(",")]
    if len(values) != len(expected):
        raise ValueError(
            f"Expected {len(expected)} values ({', '.join(expected)}), got {len(values)}."
        )
    return {k: float(v) for k, v in zip(expected, values)}
```

**Step 4 — Wire the command body**
```python
async def distribution(self, interaction, kind, params):
    await interaction.response.defer()
    reg = _DISTRIBUTION_REGISTRY[kind]
    try:
        kwargs = _parse_dist_params(params, reg["params"])
        result_embed, file = await reg["handler"](**kwargs)
    except ValueError as e:
        await interaction.followup.send(embed=error_embed(str(e)))
        return
    await interaction.followup.send(embed=result_embed, file=file)
```

**Step 5 — Update `/help_math`** in `utility.py`: add `/distribution` to the Statistics group, note that old commands are superseded.

### Key design decisions
- Keep old commands alive but mark them deprecated; this avoids breaking any existing server integrations.
- The `params` string approach means no modal is needed; help text in `/help_math` documents argument order per distribution.

### Testing checklist
- `kind=normal_pdf, params="0, 0, 1"` → bell curve image
- `kind=binomial_cdf, params="10, 0.5, 5"` → probability value
- Wrong param count → clean error embed
- Non-numeric param → clean error embed

---

## T1-2. Vector calculus additions to `calculus.py`

### What it is
Three new commands: `/gradient`, `/divergence`, `/curl`. All operate on scalar/vector field expressions passed as strings and return symbolically computed results.

### Files touched
| File | Change |
|---|---|
| `cogs/calculus.py` | Add three commands + two helper functions |
| `utils/parser.py` | No change — existing `parse_expression` handles all inputs |

### Implementation steps

**Step 1 — Add module-level helper `_parse_variable_list`**
```python
def _parse_variable_list(raw: str) -> list[sympy.Symbol]:
    """Parse 'x, y, z' → [Symbol('x'), Symbol('y'), Symbol('z')]."""
    names = [n.strip() for n in raw.split(",")]
    for n in names:
        if not n.isidentifier():
            raise ValueError(f"'{n}' is not a valid variable name.")
    return [sympy.Symbol(n) for n in names]
```

**Step 2 — Add `/gradient`**
```python
@app_commands.command(name="gradient")
@app_commands.describe(
    expression="Scalar field f(x,y,z)",
    variables="Variables to differentiate with respect to, e.g. 'x, y, z'"
)
async def gradient(self, interaction, expression: str, variables: str = "x, y, z"):
    await interaction.response.defer()
    try:
        expr = await parse_expression(expression)
        vars_ = _parse_variable_list(variables)
        components = [sympy.diff(expr, v) for v in vars_]
        result = "(" + ", ".join(sympy.latex(c) for c in components) + ")"
    except ValueError as e:
        ...
    embed = math_embed("Gradient ∇f", result)
    await interaction.followup.send(embed=embed)
```

**Step 3 — Add `/divergence`**

Takes a vector field as comma-separated component expressions (e.g. `"x**2, y*z, z"`), and a variables string. Computes `∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z`.

```python
@app_commands.command(name="divergence")
@app_commands.describe(
    components="Vector field components, comma-separated (e.g. 'x^2, y*z, z')",
    variables="Variables matching each component (e.g. 'x, y, z')"
)
async def divergence(self, interaction, components: str, variables: str = "x, y, z"):
    await interaction.response.defer()
    try:
        vars_ = _parse_variable_list(variables)
        exprs = [await parse_expression(c.strip()) for c in components.split(",")]
        if len(exprs) != len(vars_):
            raise ValueError("Number of components must match number of variables.")
        result = sympy.Add(*[sympy.diff(f, v) for f, v in zip(exprs, vars_)])
    except ValueError as e:
        ...
```

**Step 4 — Add `/curl`**

Restricted to 3-D: `curl F = (∂F₃/∂y − ∂F₂/∂z, ∂F₁/∂z − ∂F₃/∂x, ∂F₂/∂x − ∂F₁/∂y)`. Validate that exactly 3 components and 3 variables are supplied.

**Step 5 — Register all three in `main.py`'s COGS** (already done if they're in `calculus.py`; no new cog file needed).

**Step 6 — Update `/help_math`** with descriptions for all three.

### Key design decisions
- Parse each component expression separately through `parse_expression` — do not join them into one string, as commas inside expressions would break that split. Use `parse_expression` per token after splitting on `", "` (with space), and handle the edge case of commas inside parens by doing a balanced-paren split if needed (start simple, iterate).
- Return results as LaTeX strings inside a code block for readability.

### Testing checklist
- `/gradient x^2 + y^2 + z^2` over `x, y, z` → `(2x, 2y, 2z)`
- `/divergence x, y, z` → `3`
- `/curl -y, x, 0` over `x, y, z` → `(0, 0, 2)`
- Mismatched component/variable count → clear error

---

## T1-3. `/compare f g` — side-by-side function comparison

### What it is
A new command that takes two expressions and produces a single embed (and optionally a side-by-side plot via `plotter.plot_multi`) summarizing: simplified forms, domains (rational functions), common roots, and whether they are algebraically equivalent.

### Files touched
| File | Change |
|---|---|
| `cogs/arithmetic.py` | Best home — add `/compare` here alongside `/verify` |
| `utils/plotter.py` | Already has `plot_multi`; no change needed |
| `utils/formatter.py` | No change — use existing `math_embed` |

### Implementation steps

**Step 1 — Command signature**
```python
@app_commands.command(name="compare")
@app_commands.describe(
    f="First expression",
    g="Second expression",
    variable="Variable (default x)",
    plot="Include a side-by-side plot",
)
async def compare(self, interaction, f: str, g: str,
                  variable: str = "x", plot: bool = True):
```

**Step 2 — Parse and compute comparison fields**
```python
expr_f = await parse_expression(f)
expr_g = await parse_expression(g)
x = sympy.Symbol(variable)

simplified_f = sympy.simplify(expr_f)
simplified_g = sympy.simplify(expr_g)
equivalent = sympy.simplify(expr_f - expr_g) == 0
roots_f = sympy.solve(expr_f, x)
roots_g = sympy.solve(expr_g, x)
common_roots = [r for r in roots_f if r in roots_g]
```

**Step 3 — Build a multi-field embed**
```python
embed = discord.Embed(title="Compare f vs g", color=BLURPLE)
embed.add_field(name="f(x)", value=f"```{sympy.pretty(simplified_f, use_unicode=False)}```", inline=True)
embed.add_field(name="g(x)", value=f"```{sympy.pretty(simplified_g, use_unicode=False)}```", inline=True)
embed.add_field(name="Equivalent?", value="Yes ✓" if equivalent else "No ✗", inline=False)
embed.add_field(name="Roots of f", value=str(roots_f) or "None", inline=True)
embed.add_field(name="Roots of g", value=str(roots_g) or "None", inline=True)
if common_roots:
    embed.add_field(name="Shared roots", value=str(common_roots), inline=False)
```

**Step 4 — Optionally generate a side-by-side plot**

When `plot=True`, call `plotter.plot_multi` with two `PlotSpec` objects for `f` and `g` over the default domain `[-10, 10]`. Attach the resulting image to the followup message.

**Step 5 — Add timeout handling** — both `sympy.solve` calls can be slow. Wrap them in the existing `asyncio.wait_for` pattern used in `parser.py`, with a 5-second timeout and a graceful "Roots could not be computed in time" fallback.

### Key design decisions
- `/verify` already checks equivalence (`sympy.simplify(f-g)==0`); `/compare` is a superset. Do not duplicate logic — factor the equivalence check into a shared helper `_are_equivalent(expr_f, expr_g)` in `arithmetic.py` and call it from both commands.
- Plot is optional and off-by-default for users who just want the symbolic analysis. Default `True` suits most interactive use.

### Testing checklist
- `/compare x^2 x*x` → equivalent, same roots
- `/compare sin(x) cos(x)` → not equivalent, different roots
- `/compare 1/x x` with `plot=True` → image attached
- Invalid expression for `f` → clean error

---

## T1-4. `transforms.py` — new cog for Laplace and Fourier transforms

### What it is
A brand-new cog `cogs/transforms.py` with two commands: `/laplace` and `/fourier`. Both use SymPy's `laplace_transform` and `fourier_transform` respectively.

### Files touched
| File | Change |
|---|---|
| `cogs/transforms.py` | **New file** |
| `main.py` | Add `"cogs.transforms"` to `COGS` list |
| `requirements.txt` | No new deps — SymPy already provides both transforms |

### Implementation steps

**Step 1 — Create `cogs/transforms.py` with boilerplate**
```python
"""cogs/transforms.py — Integral transform commands: Laplace, Fourier."""
import sympy
from discord import app_commands
from discord.ext import commands
from utils.parser import parse_expression
from utils.formatter import math_embed, error_embed

class TransformsCog(commands.Cog, name="Transforms"):
    ...

async def setup(bot):
    await bot.add_cog(TransformsCog(bot))
```

**Step 2 — Implement `/laplace`**
```python
@app_commands.command(name="laplace")
@app_commands.describe(
    expression="f(t) to transform",
    t_var="Time variable (default t)",
    s_var="Frequency variable (default s)",
)
async def laplace(self, interaction, expression: str,
                  t_var: str = "t", s_var: str = "s"):
    await interaction.response.defer()
    try:
        expr = await parse_expression(expression)
        t = sympy.Symbol(t_var, positive=True)  # Laplace needs t > 0
        s = sympy.Symbol(s_var)
        # laplace_transform returns (F(s), convergence_plane, conditions)
        F, plane, _ = sympy.laplace_transform(expr, t, s, noconds=False)
        result = sympy.simplify(F)
    except (ValueError, sympy.IntegralTransformError) as e:
        await interaction.followup.send(embed=error_embed(str(e)))
        return
    embed = math_embed(
        f"L{{f(t)}} — Laplace Transform",
        sympy.pretty(result, use_unicode=False),
        footer=f"Convergence: Re(s) > {plane}"
    )
    await interaction.followup.send(embed=embed)
```

**Step 3 — Implement `/fourier`**

SymPy's `fourier_transform(f, x, k)` computes `∫ f(x) e^{-2πikx} dx`. Expose `x_var` and `k_var` as optional parameters mirroring `/laplace`'s pattern.

```python
@app_commands.command(name="fourier")
@app_commands.describe(
    expression="f(x) to transform",
    x_var="Spatial variable (default x)",
    k_var="Frequency variable (default k)",
)
async def fourier(self, interaction, expression: str,
                  x_var: str = "x", k_var: str = "k"):
    await interaction.response.defer()
    try:
        expr = await parse_expression(expression)
        x = sympy.Symbol(x_var)
        k = sympy.Symbol(k_var)
        result = sympy.fourier_transform(expr, x, k)
        result = sympy.simplify(result)
    except (ValueError, sympy.IntegralTransformError) as e:
        ...
```

**Step 4 — Handle `IntegralTransformError`** gracefully. SymPy raises this when the transform doesn't exist or can't be computed analytically. The error message should tell the user the transform couldn't be computed and suggest checking the function's integrability conditions.

**Step 5 — Register in `main.py`**
```python
COGS = [
    ...
    "cogs.transforms",   # add here, after calculus
]
```

**Step 6 — Wrap in thread-pool executor** — both transforms can be slow. Use the same `asyncio.wait_for` + `loop.run_in_executor` pattern as `parse_expression`, with a 5-second timeout. Extract a helper `_run_sympy(fn, *args)` inside the cog that handles this pattern so it doesn't need to be repeated.

### Key design decisions
- `t` variable declared with `positive=True` for Laplace — without this, SymPy may fail to compute or produce unnecessary Heaviside conditions in the result.
- Both commands accept custom variable names to support non-standard textbook notation.
- Transforms are computationally heavier than differentiation/integration; a longer timeout (5s vs 3s) is justified.

### Testing checklist
- `/laplace e^(-t)` → `1/(s+1)`, convergence `Re(s) > -1`
- `/laplace t^2` → `2/s^3`
- `/fourier e^(-x^2)` → Gaussian in frequency domain
- Non-transformable function → clean `IntegralTransformError` message
- Custom variable names → correct substitution

---

## T1-5. `/convert` expansion — additional unit categories

### What it is
Extend the existing `/convert` command in `cogs/utility.py` to cover more unit categories: area, volume, speed, time, digital storage, pressure, and energy — in addition to the current length, mass, and temperature.

### Files touched
| File | Change |
|---|---|
| `cogs/utility.py` | Extend `_UNIT_TABLE` and the routing logic in `/convert` |

### Implementation steps

**Step 1 — Audit the existing `_convert_length`, `_convert_mass`, `_convert_temperature` helpers**

The current structure routes all conversions through Kelvin for temperature. Length and mass use ratio-based dicts (unit → SI factor). Extend the same pattern.

**Step 2 — Define new unit category tables**
```python
_AREA_UNITS = {   # base: m²
    "m2": 1, "km2": 1e6, "cm2": 1e-4, "mm2": 1e-6,
    "ft2": 0.092903, "in2": 0.00064516, "acre": 4046.86,
    "ha": 10000, "mi2": 2589988.11,
}
_VOLUME_UNITS = {  # base: L
    "l": 1, "ml": 0.001, "m3": 1000, "cm3": 0.001,
    "gal": 3.78541, "qt": 0.946353, "pt": 0.473176,
    "cup": 0.236588, "floz": 0.0295735, "tbsp": 0.0147868, "tsp": 0.00492892,
}
_SPEED_UNITS = {   # base: m/s
    "m/s": 1, "km/h": 1/3.6, "mph": 0.44704, "knot": 0.514444, "ft/s": 0.3048,
}
_TIME_UNITS = {    # base: seconds
    "s": 1, "ms": 0.001, "min": 60, "h": 3600, "d": 86400, "wk": 604800,
    "mo": 2628000, "yr": 31536000,
}
_STORAGE_UNITS = { # base: bytes
    "b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
    "kib": 1024, "mib": 1048576, "gib": 1073741824, "tib": 1099511627776,
}
_PRESSURE_UNITS = { # base: Pa
    "pa": 1, "kpa": 1000, "mpa": 1e6, "bar": 1e5, "atm": 101325,
    "psi": 6894.76, "torr": 133.322, "mmhg": 133.322,
}
_ENERGY_UNITS = {  # base: J
    "j": 1, "kj": 1000, "mj": 1e6, "cal": 4.184, "kcal": 4184,
    "wh": 3600, "kwh": 3600000, "ev": 1.60218e-19, "btu": 1055.06,
}
```

**Step 3 — Extend `_detect_unit_category`**
```python
_CATEGORY_MAP = {
    **{u: "length"    for u in _LENGTH_UNITS},
    **{u: "mass"      for u in _MASS_UNITS},
    **{u: "area"      for u in _AREA_UNITS},
    **{u: "volume"    for u in _VOLUME_UNITS},
    **{u: "speed"     for u in _SPEED_UNITS},
    **{u: "time"      for u in _TIME_UNITS},
    **{u: "storage"   for u in _STORAGE_UNITS},
    **{u: "pressure"  for u in _PRESSURE_UNITS},
    **{u: "energy"    for u in _ENERGY_UNITS},
}
```

Note: temperature stays its own branch since it uses an offset (not a ratio) conversion.

**Step 4 — Add a generic ratio-based converter** (since length and mass already work this way, factor it out):
```python
def _convert_ratio(value: float, from_unit: str, to_unit: str, table: dict) -> float:
    from_factor = table.get(from_unit.lower())
    to_factor   = table.get(to_unit.lower())
    if from_factor is None:
        raise ValueError(f"Unknown unit '{from_unit}'.")
    if to_factor is None:
        raise ValueError(f"Unknown unit '{to_unit}'.")
    return value * from_factor / to_factor
```

Replace the existing `_convert_length` and `_convert_mass` bodies to call `_convert_ratio`, then add the seven new categories as table lookups using the same function.

**Step 5 — Update `/convert` command routing** to dispatch to the new categories and update the command description to list all supported categories.

**Step 6 — Add autocomplete** for `from` and `to` parameters using `app_commands.autocomplete`, returning unit names filtered to the category detected from the `from` input. This prevents typos and improves discoverability.

### Key design decisions
- All new categories use the same ratio-based pattern as length/mass — no new conversion logic, just new tables.
- Unit names are lowercased on input before lookup so `"KB"`, `"kb"`, and `"Kb"` all work.
- Digital storage uses SI (kB=1000B) not IEC (KiB=1024B) as default names, but both are supported explicitly.

### Testing checklist
- `5 km → mi` → `3.10686`
- `1 gal → l` → `3.78541`
- `100 km/h → mph` → `62.1371`
- `1 yr → s` → `31536000`
- `1 kwh → j` → `3600000`
- Unknown unit → error embed listing available units for detected category

---

## T1-6. Number theory gap-fill

### What it is
Add missing number theory commands to `cogs/number_theory.py`: `/totient` (Euler's totient function), `/divisors`, `/is_perfect`, `/mobius`, and `/chinese_remainder`.

### Files touched
| File | Change |
|---|---|
| `cogs/number_theory.py` | Add five commands and supporting helpers |

### Implementation steps

**Step 1 — `/totient n`**

Uses `sympy.totient(n)`. Apply the existing `_validate_int_arg` guard (max ~10⁹ to keep it fast).
```python
@app_commands.command(name="totient")
@app_commands.describe(n="Positive integer")
async def totient(self, interaction, n: int):
    await interaction.response.defer()
    _validate_int_arg(n, lo=1, hi=10**9, name="n")
    result = sympy.totient(n)
    await interaction.followup.send(embed=math_embed(f"φ({n})", str(result)))
```

**Step 2 — `/divisors n`**

Uses `sympy.divisors(n)`. Cap at 10⁹; for large results paginate with `_paginate_list` (already exists in the cog).

**Step 3 — `/is_perfect n`**

A number is perfect if `sum(divisors(n)[:-1]) == n`. SymPy doesn't have a direct `is_perfect` but `sympy.perfect` from `sympy.ntheory` provides it. Alternatively compute from divisors.
```python
from sympy.ntheory import factorint
def _is_perfect(n: int) -> bool:
    return n > 1 and sum(sympy.divisors(n)[:-1]) == n
```

**Step 4 — `/mobius n`**

Uses `sympy.mobius(n)`. Returns -1, 0, or 1. Display result with a plain-language label: `"−1 (odd number of distinct prime factors)"`, `"0 (has squared prime factor)"`, `"1 (even number of distinct prime factors or n=1)"`.

**Step 5 — `/chinese_remainder`**

Takes two slash parameters: `remainders` (comma-separated integers) and `moduli` (comma-separated integers). Uses `sympy.ntheory.modular.crt`.
```python
from sympy.ntheory.modular import crt

@app_commands.command(name="chinese_remainder")
@app_commands.describe(
    remainders="Comma-separated remainders (e.g. '2, 3, 1')",
    moduli="Comma-separated moduli (e.g. '3, 5, 7')"
)
async def chinese_remainder(self, interaction, remainders: str, moduli: str):
    await interaction.response.defer()
    try:
        rs = _parse_integers(remainders)
        ms = _parse_integers(moduli)
        if len(rs) != len(ms):
            raise ValueError("Number of remainders must match number of moduli.")
        result, lcm = crt(ms, rs, symmetric=False)
        if result is None:
            raise ValueError("No solution — moduli may not be pairwise coprime.")
    except ValueError as e:
        ...
    embed = math_embed(
        "Chinese Remainder Theorem",
        f"x ≡ {result} (mod {lcm})",
        footer=f"Remainders: {rs} | Moduli: {ms}"
    )
```

### Key design decisions
- All five commands reuse `_validate_int_arg` and `_parse_integers` — no new parsing utilities needed.
- `/divisors` can return a very long list for numbers with many factors; paginate anything over 30 divisors.

### Testing checklist
- `/totient 12` → `4`
- `/divisors 12` → `1, 2, 3, 4, 6, 12`
- `/is_perfect 6` → Yes; `/is_perfect 12` → No
- `/mobius 30` → `-1`; `/mobius 4` → `0`; `/mobius 1` → `1`
- `/chinese_remainder 2, 3, 1` with moduli `3, 5, 7` → `x ≡ 23 (mod 105)`

---

## T1-7. `/identify` — expression recognition

### What it is
A new command `/identify` that takes an expression and attempts to classify it: polynomial (with degree), rational function, trigonometric, exponential, logarithmic, piecewise, periodic, even/odd, or a named constant (like π or e).

### Files touched
| File | Change |
|---|---|
| `cogs/symbolic.py` | Best home — add alongside `/roots`, `/partial_fraction` |
| `utils/parser.py` | No change |

### Implementation steps

**Step 1 — Build `_identify_expression(expr, var)` helper**
```python
def _identify_expression(expr: sympy.Expr, var: sympy.Symbol) -> dict:
    tags = []

    # Polynomial check
    try:
        poly = sympy.Poly(expr, var)
        tags.append(f"Polynomial (degree {poly.degree()})")
    except sympy.PolynomialError:
        pass

    # Rational function check
    n, d = sympy.fraction(sympy.cancel(expr))
    if d != 1:
        tags.append("Rational function")

    # Trig, exp, log checks using sympy.ask or atom inspection
    atoms = expr.atoms(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)
    if atoms:
        tags.append("Trigonometric")
    if expr.atoms(sympy.exp):
        tags.append("Exponential")
    if expr.atoms(sympy.log):
        tags.append("Logarithmic")

    # Even/odd
    if sympy.simplify(expr.subs(var, -var) - expr) == 0:
        tags.append("Even function f(-x) = f(x)")
    elif sympy.simplify(expr.subs(var, -var) + expr) == 0:
        tags.append("Odd function f(-x) = -f(x)")

    # Periodicity
    period = sympy.periodicity(expr, var)
    if period is not None and period != 0:
        tags.append(f"Periodic (period = {period})")

    # Constant
    if not expr.free_symbols:
        numeric = complex(sympy.N(expr))
        tags.append(f"Constant ≈ {numeric.real:.6g}")

    return tags if tags else ["No standard classification found"]
```

**Step 2 — Add `/identify` command**
```python
@app_commands.command(name="identify")
@app_commands.describe(
    expression="Expression to classify",
    variable="Main variable (default x)"
)
async def identify(self, interaction, expression: str, variable: str = "x"):
    await interaction.response.defer()
    expr = await parse_expression(expression)
    x = sympy.Symbol(variable)
    tags = _identify_expression(expr, x)
    embed = math_embed("Expression Identity", "\n".join(f"• {t}" for t in tags))
    await interaction.followup.send(embed=embed)
```

**Step 3 — Wrap classification in executor** — `sympy.periodicity` and `simplify` can be slow. Wrap inside `loop.run_in_executor` with a 5-second timeout.

### Key design decisions
- `_identify_expression` returns a list of strings, not a dict, so the embed field renders naturally as a bullet list.
- Classifications are additive — an expression can be both trigonometric and periodic.
- Periodicity detection with `sympy.periodicity` may return `None` for functions it can't determine; treat `None` as "unknown/non-periodic" and omit the tag.

### Testing checklist
- `/identify x^2 + 2x + 1` → `Polynomial (degree 2)`, `Even function` (wrong — check), actually neither even nor odd unless x²+2x+1 = (x+1)², so Not even/odd
- `/identify x^2` → `Polynomial (degree 2)`, `Even function`
- `/identify sin(x)` → `Trigonometric`, `Odd function`, `Periodic (period = 2π)`
- `/identify e^x` → `Exponential`
- `/identify 3.14159` → `Constant`

---

## T1-8. Logic expansion — `simplify_bool` and `logic_equiv`

### What it is
Two new commands in `cogs/discrete.py`: `/simplify_bool` (simplify a boolean expression using Quine-McCluskey or SymPy logic) and `/logic_equiv` (check whether two boolean expressions are logically equivalent).

### Files touched
| File | Change |
|---|---|
| `cogs/discrete.py` | Add two commands; reuse existing `_BoolParser` and `_evaluate_boolean` |

### Implementation steps

**Step 1 — Add `/simplify_bool`**

SymPy's `sympy.logic.boolalg` provides `simplify_logic` and `to_dnf`/`to_cnf`. Use the hand-rolled `_BoolParser` from `discrete.py` for parsing (it's already there and safe), then convert the resulting AST into a `sympy.Symbol`-based boolean expression for simplification.

```python
from sympy.logic.boolalg import simplify_logic, to_dnf

@app_commands.command(name="simplify_bool")
@app_commands.describe(
    expression="Boolean expression (e.g. 'A AND (B OR NOT A)')",
    form="Output form: simplified (default), DNF, or CNF"
)
@app_commands.choices(form=[
    app_commands.Choice(name="Simplified", value="simplified"),
    app_commands.Choice(name="DNF (Disjunctive Normal Form)", value="dnf"),
    app_commands.Choice(name="CNF (Conjunctive Normal Form)", value="cnf"),
])
async def simplify_bool(self, interaction, expression: str, form: str = "simplified"):
    await interaction.response.defer()
    try:
        sympy_expr = _bool_str_to_sympy(expression)   # new helper, see below
        if form == "dnf":
            result = sympy.logic.boolalg.to_dnf(sympy_expr, simplify=True)
        elif form == "cnf":
            result = sympy.logic.boolalg.to_cnf(sympy_expr, simplify=True)
        else:
            result = sympy.logic.boolalg.simplify_logic(sympy_expr)
    except Exception as e:
        await interaction.followup.send(embed=error_embed(str(e)))
        return
    await interaction.followup.send(embed=math_embed("Simplified Boolean", str(result)))
```

**Step 2 — Add `_bool_str_to_sympy` helper** (the bridge from the existing tokenizer/parser to SymPy's logic layer):
```python
import sympy
from sympy.logic.boolalg import And, Or, Not, Xor, Implies, Equivalent

def _bool_str_to_sympy(expr_str: str) -> sympy.Basic:
    """Convert a boolean expression string to a SymPy logic expression.
    Uses the existing _BoolParser AST and maps each node type to SymPy."""
    tokens = _tokenize_boolean(expr_str)
    parser = _BoolParser(tokens)
    ast = parser.parse()
    return _ast_to_sympy(ast)

def _ast_to_sympy(node) -> sympy.Basic:
    if isinstance(node, str):  # variable leaf
        return sympy.Symbol(node)
    op, *args = node
    sym_args = [_ast_to_sympy(a) for a in args]
    return {
        "AND": And, "OR": Or, "NOT": lambda x: Not(x),
        "XOR": Xor, "IMPLIES": Implies, "IFF": Equivalent,
    }[op](*sym_args)
```

**Step 3 — Add `/logic_equiv`**
```python
@app_commands.command(name="logic_equiv")
@app_commands.describe(
    expr1="First boolean expression",
    expr2="Second boolean expression"
)
async def logic_equiv(self, interaction, expr1: str, expr2: str):
    await interaction.response.defer()
    try:
        s1 = _bool_str_to_sympy(expr1)
        s2 = _bool_str_to_sympy(expr2)
        # Two boolean exprs are equivalent iff (s1 XOR s2) is unsatisfiable
        diff = sympy.logic.boolalg.Xor(s1, s2)
        equivalent = not sympy.satisfiable(diff)
    except Exception as e:
        await interaction.followup.send(embed=error_embed(str(e)))
        return
    embed = math_embed(
        "Logical Equivalence",
        f"{'✓ Equivalent' if equivalent else '✗ Not equivalent'}",
        footer=f"{expr1}  ↔  {expr2}"
    )
    await interaction.followup.send(embed=embed)
```

### Key design decisions
- Reuse `_BoolParser` rather than calling `sympify` on boolean strings — this maintains the codebase's existing approach of avoiding `eval()` in the discrete math path.
- `sympy.satisfiable` returns a model dict if satisfiable or `False` if not; `not False == True` means not equivalent.

### Testing checklist
- `/simplify_bool A AND (B OR NOT A)` → `A AND B`
- `/simplify_bool A OR NOT A` → `True`
- `/simplify_bool A AND NOT A` → `False`
- `/simplify_bool A AND B` with `form=DNF` → `A & B`
- `/logic_equiv "A AND B" "B AND A"` → Equivalent
- `/logic_equiv "A OR B" "A AND B"` → Not equivalent

---

# TIER 2 — Foundational

---

## T2-1. `/define` — session variable store

### What it is
A per-user, per-session variable store that persists for the lifetime of the bot process. Users can `/define x = sin(pi/3)` and then reference `x` in any subsequent command in the same session. Implemented as a new cog `cogs/session.py`.

### Files touched
| File | Change |
|---|---|
| `cogs/session.py` | **New file** — `/define`, `/undefine`, `/vars` commands |
| `data/session_store.py` | **New file** — in-memory per-user variable store |
| `utils/parser.py` | **Extend** `parse_expression` to accept an optional `user_vars` dict and substitute before parsing |
| `main.py` | Add `"cogs.session"` to `COGS` |

### Implementation steps

**Step 1 — Create `data/session_store.py`**
```python
"""data/session_store.py — Per-user symbolic variable store (in-memory)."""
import threading
from collections import defaultdict

_store: dict[int, dict[str, str]] = defaultdict(dict)  # user_id → {name: expr_str}
_lock = threading.Lock()

def set_var(user_id: int, name: str, expr_str: str) -> None:
    with _lock:
        _store[user_id][name] = expr_str

def get_vars(user_id: int) -> dict[str, str]:
    with _lock:
        return dict(_store[user_id])

def delete_var(user_id: int, name: str) -> bool:
    with _lock:
        return _store[user_id].pop(name, None) is not None

def clear_vars(user_id: int) -> None:
    with _lock:
        _store[user_id].clear()
```

**Step 2 — Create `cogs/session.py`**
```python
@app_commands.command(name="define")
@app_commands.describe(
    name="Variable name (a valid Python identifier)",
    expression="Expression to assign"
)
async def define(self, interaction, name: str, expression: str):
    await interaction.response.defer(ephemeral=True)
    if not name.isidentifier():
        await interaction.followup.send(embed=error_embed(f"'{name}' is not a valid variable name."))
        return
    # Validate the expression parses before storing
    try:
        await parse_expression(expression)
    except ValueError as e:
        await interaction.followup.send(embed=error_embed(str(e)))
        return
    session_store.set_var(interaction.user.id, name, expression)
    await interaction.followup.send(
        embed=info_embed("Variable defined", f"`{name}` = `{expression}`"),
        ephemeral=True
    )

@app_commands.command(name="vars")
async def vars_cmd(self, interaction):
    await interaction.response.defer(ephemeral=True)
    user_vars = session_store.get_vars(interaction.user.id)
    if not user_vars:
        await interaction.followup.send(embed=info_embed("Variables", "No variables defined."), ephemeral=True)
        return
    lines = "\n".join(f"`{k}` = `{v}`" for k, v in user_vars.items())
    await interaction.followup.send(embed=info_embed("Your Variables", lines), ephemeral=True)

@app_commands.command(name="undefine")
async def undefine(self, interaction, name: str):
    ...
```

**Step 3 — Extend `parse_expression` to accept `user_vars`**
```python
async def parse_expression(
    expr_str: str,
    user_vars: dict[str, str] | None = None
) -> sympy.Expr:
    ...
    # Before parsing, substitute user-defined variables
    if user_vars:
        for var_name, var_expr in user_vars.items():
            expr_str = re.sub(rf'\b{re.escape(var_name)}\b', f'({var_expr})', expr_str)
    ...
```

**Step 4 — Thread user vars through cogs that call `parse_expression`**

The cleanest approach is to pass `user_vars` as an optional parameter. This requires updating every cog command that calls `parse_expression`. Since that's many files, implement a convenience wrapper:

```python
# In each cog, at the start of any command body:
user_vars = session_store.get_vars(interaction.user.id)
expr = await parse_expression(expression, user_vars=user_vars)
```

Prioritize `calculus.py`, `arithmetic.py`, and `symbolic.py` — these are highest-value targets. The rest can be updated incrementally.

### Key design decisions
- Responses for `/define`, `/vars`, `/undefine` are ephemeral (only visible to the user) — variable definitions are personal state, not channel content.
- Variables are substituted as strings before parsing, not as SymPy symbol assignments, to avoid polluting the global symbol namespace.
- There's no persistence across bot restarts — consistent with the existing history/cache approach of "no database by design."
- Max 20 variables per user (use `len(_store[user_id]) >= 20` guard in `set_var`) to prevent unbounded memory growth.

### Testing checklist
- `/define a x^2 + 1` then `/simplify a` → `x^2 + 1`
- `/define r 3` then `/circle_area r` → `9π`
- `/vars` → shows all defined variables
- `/undefine a` then `/simplify a` → parses `a` as a free symbol
- Defining 21st variable → error: "Variable limit (20) reached."
- Invalid expression → error before storing

---

## T2-2. CSV tools cog

### What it is
A new cog `cogs/csv_tools.py` with commands for statistical analysis and basic transformation of user-uploaded CSV files: `/csv_stats`, `/csv_plot`, `/csv_filter`.

### Files touched
| File | Change |
|---|---|
| `cogs/csv_tools.py` | **New file** |
| `main.py` | Add `"cogs.csv_tools"` to `COGS` |
| `requirements.txt` | Add `pandas>=2.0.0` |

### Implementation steps

**Step 1 — Create `cogs/csv_tools.py` with boilerplate**

All three commands accept a `discord.Attachment` parameter (the CSV file). Discord.py 2.x supports `discord.Attachment` as a slash parameter type.

**Step 2 — Add `_load_csv_from_attachment` helper**
```python
import io
import pandas as pd

async def _load_csv_from_attachment(attachment: discord.Attachment) -> pd.DataFrame:
    if not attachment.filename.endswith(".csv"):
        raise ValueError("Attachment must be a .csv file.")
    if attachment.size > 5_000_000:  # 5 MB cap
        raise ValueError("CSV file too large (max 5 MB).")
    raw = await attachment.read()
    return pd.read_csv(io.BytesIO(raw))
```

**Step 3 — Implement `/csv_stats file [column]`**
```python
@app_commands.command(name="csv_stats")
@app_commands.describe(
    file="CSV file to analyze",
    column="Optional: analyze a specific column (default: all numeric columns)"
)
async def csv_stats(self, interaction, file: discord.Attachment, column: str = ""):
    await interaction.response.defer()
    df = await _load_csv_from_attachment(file)
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        df = df[[column]]
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric columns found in the file.")
    desc = numeric.describe().round(4)
    # Format as code block table
    result = desc.to_string()
    embed = info_embed(f"Stats — {file.filename}", f"```\n{result}\n```")
    await interaction.followup.send(embed=embed)
```

**Step 4 — Implement `/csv_plot file x_col y_col [plot_type]`**

Generate a scatter or line plot from two columns. Use `plotter.py`'s existing `plot_function` machinery if possible; otherwise build a minimal matplotlib figure directly (since the data is already numeric, no SymPy parsing is needed).
```python
@app_commands.command(name="csv_plot")
@app_commands.choices(plot_type=[
    app_commands.Choice(name="Scatter", value="scatter"),
    app_commands.Choice(name="Line", value="line"),
    app_commands.Choice(name="Bar", value="bar"),
    app_commands.Choice(name="Histogram", value="hist"),
])
async def csv_plot(self, interaction, file: discord.Attachment,
                   x_col: str = "", y_col: str = "",
                   plot_type: str = "scatter"):
```

Use `loop.run_in_executor` to run matplotlib in a thread. Return the PNG as a `discord.File`.

**Step 5 — Implement `/csv_filter file condition`**

`condition` is a pandas query string (e.g. `"age > 30 and salary < 50000"`). Use `df.query(condition)`. Return the first 20 rows of the filtered result as a formatted code block, with row count in the footer.

```python
@app_commands.command(name="csv_filter")
@app_commands.describe(
    file="CSV file",
    condition="Pandas query string (e.g. 'age > 30 and score < 80')"
)
async def csv_filter(self, interaction, file: discord.Attachment, condition: str):
    await interaction.response.defer()
    df = await _load_csv_from_attachment(file)
    try:
        result_df = df.query(condition)
    except Exception as e:
        await interaction.followup.send(embed=error_embed(f"Invalid condition: {e}"))
        return
    preview = result_df.head(20).to_string(index=False)
    embed = info_embed(
        f"Filter Result — {file.filename}",
        f"```\n{preview}\n```",
    )
    embed.set_footer(text=f"{len(result_df)} rows matched")
    await interaction.followup.send(embed=embed)
```

### Key design decisions
- 5 MB attachment cap is consistent with Discord's general limit and prevents memory spikes from very large datasets.
- `df.query` is safe here — pandas' query engine does not execute arbitrary Python the way `eval()` does. If extra caution is needed, validate that `condition` contains only known column names and comparison operators (optional hardening step).
- Run all pandas/matplotlib work in `loop.run_in_executor` to keep the async loop unblocked.
- `pandas` is a new dependency; add it to `requirements.txt`.

### Testing checklist
- Upload a simple CSV, `/csv_stats` → descriptive statistics
- `/csv_stats` with `column` arg → single-column stats
- `/csv_plot` scatter → image attached
- `/csv_filter "value > 5"` → filtered rows + count footer
- File >5 MB → size error
- Non-CSV attachment → type error
- Query with invalid column → error embed

---

## T2-3. Calculus analysis commands

### What it is
Five new commands in `cogs/calculus.py`: `/critical_points`, `/inflection`, `/asymptotes`, `/domain`, `/arc_length`.

### Files touched
| File | Change |
|---|---|
| `cogs/calculus.py` | Add five commands + shared helper `_analyze_function` |

### Implementation steps

**Step 1 — `/critical_points expression [variable]`**
```python
@app_commands.command(name="critical_points")
async def critical_points(self, interaction, expression: str, variable: str = "x"):
    await interaction.response.defer()
    expr = await parse_expression(expression)
    x = sympy.Symbol(variable)
    deriv = sympy.diff(expr, x)
    cps = sympy.solve(deriv, x)
    # Classify each: second derivative test
    d2 = sympy.diff(deriv, x)
    classified = []
    for cp in cps:
        val = d2.subs(x, cp)
        if val.is_positive:
            kind = "local minimum"
        elif val.is_negative:
            kind = "local maximum"
        else:
            kind = "inconclusive (possible inflection)"
        y = expr.subs(x, cp)
        classified.append(f"x={cp}: {kind} (f={sympy.simplify(y)})")
    result = "\n".join(classified) if classified else "No critical points found."
    await interaction.followup.send(embed=math_embed("Critical Points", result))
```

**Step 2 — `/inflection expression [variable]`**

Solve `f''(x) = 0`, check sign change of `f''` around each candidate.
```python
d2 = sympy.diff(expr, x, 2)
candidates = sympy.solve(d2, x)
inflections = []
for c in candidates:
    # Check sign change: evaluate f'' slightly left and right
    left  = d2.subs(x, c - sympy.Rational(1, 100))
    right = d2.subs(x, c + sympy.Rational(1, 100))
    if sympy.sign(left) != sympy.sign(right):
        inflections.append(f"x={c}")
```

**Step 3 — `/asymptotes expression [variable]`**

Detect three types:
- **Vertical:** solve `denominator = 0` after `sympy.cancel`
- **Horizontal:** evaluate `limit(f, x, oo)` and `limit(f, x, -oo)`
- **Oblique:** compute `limit(f/x, x, oo)` → slope `m`; then `limit(f - m*x, x, oo)` → intercept `b`

```python
n, d = sympy.fraction(sympy.cancel(expr))
vertical = sympy.solve(d, x) if d != 1 else []
h_right  = sympy.limit(expr, x, sympy.oo)
h_left   = sympy.limit(expr, x, -sympy.oo)
# Oblique asymptotes (only if horizontal limits are ±oo)
...
```

**Step 4 — `/domain expression [variable]`**

Use SymPy's `sympy.calculus.util.continuous_domain` to find the domain over the reals.
```python
from sympy.calculus.util import continuous_domain
domain = continuous_domain(expr, x, sympy.S.Reals)
```

**Step 5 — `/arc_length expression [variable] lower upper`**

Arc length formula: `∫[a,b] √(1 + (f'(x))²) dx`.
```python
deriv = sympy.diff(expr, x)
integrand = sympy.sqrt(1 + deriv**2)
arc = sympy.integrate(integrand, (x, lower_val, upper_val))
arc = sympy.simplify(arc)
```

For definite numeric results, also compute `sympy.N(arc, 6)` and include both exact and decimal forms.

### Key design decisions
- All five commands run SymPy operations in a thread-pool executor — each can be slow for complex expressions.
- `/asymptotes` should gracefully handle expressions that aren't rational (no denominator) — output "No vertical asymptotes" rather than crashing.
- `/domain` relies on `sympy.calculus.util.continuous_domain`; this is available in SymPy ≥ 1.8, which the project already requires.
- `/arc_length` accepts `lower` and `upper` as slash parameters (floats that are converted to `sympy.Rational` via `sympy.nsimplify` for exactness, using the pattern in `geometry.py::_to_exact`).

### Testing checklist
- `/critical_points x^3 - 3x` → local max at x=-1, local min at x=1
- `/inflection x^3` → inflection at x=0
- `/asymptotes 1/x` → vertical x=0, horizontal y=0
- `/asymptotes (x^2 + 1)/x` → vertical x=0, oblique y=x
- `/domain sqrt(x)` → `[0, ∞)`
- `/domain 1/x` → `(-∞, 0) ∪ (0, ∞)`
- `/arc_length x^2 0 1` → numeric result

---

# TIER 3 — When Ready

---

## T3-1. `/interpolate` — polynomial and spline interpolation

### What it is
A new command that takes a list of `(x, y)` data points and fits an interpolating polynomial (Lagrange/Newton) or a cubic spline, returning both the formula and an optional plot.

### Files touched
| File | Change |
|---|---|
| `cogs/calculus.py` | Add `/interpolate` (thematically fits with series/analysis) |
| `requirements.txt` | `scipy` already in deps; no new packages |

### Implementation steps

**Step 1 — Command signature**
```python
@app_commands.command(name="interpolate")
@app_commands.describe(
    points="Comma-separated (x,y) pairs: '0,1; 1,4; 2,9'",
    method="Interpolation method",
    plot="Generate a plot of the interpolation"
)
@app_commands.choices(method=[
    app_commands.Choice(name="Lagrange polynomial", value="lagrange"),
    app_commands.Choice(name="Newton divided differences", value="newton"),
    app_commands.Choice(name="Cubic spline",  value="spline"),
])
async def interpolate(self, interaction, points: str,
                      method: str = "lagrange", plot: bool = True):
```

**Step 2 — Parse points**
```python
def _parse_xy_pairs(raw: str) -> tuple[list[float], list[float]]:
    """Parse '0,1; 1,4; 2,9' → ([0,1,2], [1,4,9])."""
    pairs = [p.strip() for p in raw.split(";")]
    xs, ys = [], []
    for pair in pairs:
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair: '{pair}'. Expected 'x,y'.")
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))
    if len(set(xs)) != len(xs):
        raise ValueError("x-values must be distinct.")
    return xs, ys
```

**Step 3 — Lagrange interpolation via SymPy**
```python
def _lagrange_poly(xs, ys) -> sympy.Expr:
    x = sympy.Symbol("x")
    poly = sympy.S.Zero
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if j != i:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        poly += term
    return sympy.expand(poly)
```

**Step 4 — Newton divided differences via SymPy or numpy**

Build the divided difference table iteratively, then construct the Newton form polynomial.

**Step 5 — Cubic spline via `scipy.interpolate.CubicSpline`**

For the spline method, use `scipy` (already in requirements) and evaluate the spline on a fine grid for the plot. Since `CubicSpline` doesn't produce a symbolic formula, report the spline's knots and boundary conditions instead.

**Step 6 — Optional plot**

For Lagrange/Newton: convert the SymPy polynomial to a lambda and plot. For spline: plot the scipy spline evaluation. Use `plotter.plot_function` where possible.

### Key design decisions
- Maximum 20 data points for Lagrange (higher degrees become numerically unstable and slow to compute symbolically).
- Lagrange and Newton return a simplified SymPy polynomial — the canonical symbolic output. Spline returns numerical coefficients.
- Plot generation is on by default, skippable with `plot=False`.

### Testing checklist
- Three points `0,0; 1,1; 2,4` → Lagrange: `x^2` (or equivalent simplified)
- Same points with `method=spline` → plot of smooth curve through points
- Duplicate x-values → error
- More than 20 points → error

---

## T3-2. `/piecewise` — define and evaluate piecewise functions

### What it is
A command to define and evaluate piecewise-defined functions. The user provides an expression list in the format `expr1 : condition1 | expr2 : condition2 | ...` (the syntax already supported by `expr_utils.py` for plotting) and optionally a value at which to evaluate.

### Files touched
| File | Change |
|---|---|
| `cogs/calculus.py` | Add `/piecewise` alongside analysis commands |
| `utils/parser.py` | Add `_parse_piecewise_string` — port the piecewise logic from `expr_utils._clean_piecewise_expr` but route through `_validate_raw` |

### Implementation steps

**Step 1 — Port piecewise parsing to `utils/parser.py`**

The plotting path in `expr_utils.py` already parses `"expr1 : cond1 | expr2 : cond2"` into a `Piecewise` string, but it bypasses validation. Add a safe version:

```python
async def parse_piecewise(raw: str) -> sympy.Expr:
    """Parse 'expr1 : cond1 | expr2 : cond2' into a SymPy Piecewise."""
    _validate_raw(raw)  # existing length + forbidden-keyword check
    segments = raw.split("|")
    pairs = []
    for seg in segments:
        if ":" not in seg:
            raise ValueError(f"Each segment needs 'expr : condition'. Got: '{seg}'")
        expr_str, cond_str = seg.split(":", 1)
        expr = await parse_expression(expr_str.strip())
        cond = await parse_expression(cond_str.strip())
        pairs.append((expr, cond))
    return sympy.Piecewise(*pairs)
```

**Step 2 — Add `/piecewise` command**
```python
@app_commands.command(name="piecewise")
@app_commands.describe(
    definition="Piecewise definition: 'expr1 : cond1 | expr2 : cond2'",
    evaluate_at="Optional value to evaluate the function at",
    variable="Variable (default x)"
)
async def piecewise(self, interaction, definition: str,
                    evaluate_at: str = "", variable: str = "x"):
    await interaction.response.defer()
    try:
        pw = await parse_piecewise(definition)
    except ValueError as e:
        await interaction.followup.send(embed=error_embed(str(e)))
        return
    x = sympy.Symbol(variable)
    embed_lines = [f"f({variable}) = {sympy.pretty(pw, use_unicode=False)}"]
    if evaluate_at:
        pt = await parse_expression(evaluate_at)
        val = pw.subs(x, pt)
        embed_lines.append(f"f({evaluate_at}) = {sympy.simplify(val)}")
    await interaction.followup.send(embed=math_embed("Piecewise Function", "\n".join(embed_lines)))
```

**Step 3 — Update `expr_utils._clean_piecewise_expr`** to call `_validate_raw` from `parser.py` before processing (this closes Known Issue #1 for the plotting path simultaneously).

### Key design decisions
- The piecewise syntax `expr : condition | expr : condition` is already established in the plotting subsystem — this command formalizes it as a first-class feature.
- Evaluation is optional; without it, the command is purely a symbolic display/definition tool.
- Fixing `expr_utils.py` to call `_validate_raw` as part of this task addresses Known Issue #1 in the piecewise parsing path specifically.

### Testing checklist
- `"x : x >= 0 | -x : x < 0"` → abs(x) in disguise
- `"x^2 : x < 1 | 2*x - 1 : x >= 1"` → smooth piecewise
- Evaluate at `x=0.5` → `0.25` (first branch)
- Evaluate at `x=2` → `3` (second branch)
- Missing `:` in segment → clear error
- Invalid expression in segment → parser error

---

## T3-3. Quiz cog — basic

### What it is
A new cog `cogs/quiz.py` with a `/quiz` command that starts an interactive math quiz session. The first version covers: arithmetic (integer +-×÷), basic algebra (solve for x), and derivative evaluation. Questions are generated programmatically using random parameters; answers are checked symbolically.

### Files touched
| File | Change |
|---|---|
| `cogs/quiz.py` | **New file** |
| `main.py` | Add `"cogs.quiz"` to `COGS` |

### Implementation steps

**Step 1 — Define question generators**
```python
import random
import sympy

def _gen_arithmetic() -> tuple[str, sympy.Expr]:
    """Returns (question_string, correct_answer)."""
    a, b = random.randint(1, 20), random.randint(1, 20)
    op = random.choice(["+", "-", "*"])
    result = eval(f"{a}{op}{b}")   # safe: only int literals and +-*
    return f"What is {a} {op} {b}?", sympy.Integer(result)

def _gen_algebra() -> tuple[str, sympy.Expr]:
    """Generates 'solve a*x + b = c for x'."""
    a = random.randint(1, 10)
    b = random.randint(-10, 10)
    c = random.randint(-20, 20)
    x = sympy.Symbol("x")
    answer = sympy.Rational(c - b, a)
    return f"Solve: {a}x + {b} = {c}  (for x)", answer

def _gen_derivative() -> tuple[str, sympy.Expr]:
    """Generates 'd/dx of simple polynomial at a point'."""
    n = random.randint(2, 5)
    a = random.randint(1, 5)
    pt = random.randint(0, 3)
    x = sympy.Symbol("x")
    expr = a * x**n
    deriv_at = sympy.diff(expr, x).subs(x, pt)
    return f"What is d/dx of {a}x^{n} at x={pt}?", deriv_at

_GENERATORS = [_gen_arithmetic, _gen_algebra, _gen_derivative]
```

**Step 2 — Build `QuizView` — a `discord.ui.View` with a text input modal**
```python
class AnswerModal(discord.ui.Modal, title="Your Answer"):
    answer = discord.ui.TextInput(label="Answer", placeholder="e.g. 42 or x=3")

    def __init__(self, correct, on_submit_cb):
        super().__init__()
        self.correct = correct
        self.on_submit_cb = on_submit_cb

    async def on_submit(self, interaction):
        await self.on_submit_cb(interaction, self.answer.value, self.correct)
```

**Step 3 — Add `QuizSession` dataclass** to track state: current question, score, total attempted.

**Step 4 — Implement `/quiz`**
```python
@app_commands.command(name="quiz")
@app_commands.describe(
    topic="Topic to quiz on",
    rounds="Number of questions (max 10, default 5)"
)
@app_commands.choices(topic=[
    app_commands.Choice(name="Arithmetic",   value="arithmetic"),
    app_commands.Choice(name="Algebra",      value="algebra"),
    app_commands.Choice(name="Derivatives",  value="derivatives"),
    app_commands.Choice(name="Mixed",        value="mixed"),
])
async def quiz(self, interaction, topic: str = "mixed", rounds: int = 5):
    await interaction.response.defer(ephemeral=True)
    rounds = min(rounds, 10)
    ...
```

**Step 5 — Answer checking**

Parse the user's answer through `parse_expression`, then check `sympy.simplify(user_answer - correct) == 0`. This handles `"6"`, `"2*3"`, and `"sqrt(4)"` all matching `2` correctly.

**Step 6 — Score report**

After all rounds, send a final ephemeral embed with score, percentage, and a breakdown of correct/incorrect by question.

### Key design decisions
- All quiz messages are ephemeral — quizzes are personal and shouldn't clutter channels.
- Answer checking via SymPy's symbolic equality means `"3/2"` and `"1.5"` both match a `Rational(3,2)` answer.
- Only `AnswerModal` is used for input — no open-ended text in the channel, keeping quizzes clean.
- `_gen_arithmetic` uses Python's built-in `eval` but only on a string constructed from `randint` outputs and a safe operator character — this is safe by construction, not a general `eval()` risk.

### Testing checklist
- `/quiz topic=arithmetic rounds=3` → 3 arithmetic questions, score at end
- Correct answer (various formats: int, fraction) → "Correct!" response
- Wrong answer → "Incorrect. The answer was X."
- `/quiz topic=mixed` → mix of all three question types
- `rounds=15` → capped to 10

---

# Cross-cutting tasks (all tiers)

These tasks are not tied to a single feature but should be completed alongside or before the relevant tier.

## X-1. Fix Known Issue #1: route plotting expressions through `_validate_raw`

Before or alongside T3-2, patch `utils/expr_utils.py::_sympy_expr()` to call `parser._validate_raw()` before calling `sympify`. This is a one-file change of ~3 lines but closes the security gap documented in the codebase.

```python
# In utils/expr_utils.py, add at top:
from utils.parser import _validate_raw

# In _sympy_expr(), before sympify:
_validate_raw(s)  # raises ValueError if unsafe
```

Note: `_validate_raw` is currently a module-private function. Either expose it as a public `validate_raw` or move the check inline. Prefer exposing it.

## X-2. Fix Known Issue #4: delete `utils/utility.py`

```bash
git rm utils/utility.py
git commit -m "Remove dead code: utils/utility.py (orphaned draft, never loaded)"
```

## X-3. Fix Known Issue #5: `.gitignore` and stray files

```
# .gitignore — change:
/__pycache__
# to:
__pycache__
```

Then:
```bash
git rm -r --cached cogs/__pycache__ utils/__pycache__ data/__pycache__
git rm "utils/ChatGPT Image Jun 15, 2026, 11_57_30 AM.png"
git commit -m "Fix .gitignore to exclude all __pycache__ dirs; remove stray image"
```

## X-4. Broaden exception handling (Known Issue #6)

In `arithmetic.py`, `calculus.py`, and `symbolic.py`, change:
```python
except ValueError as e:
```
to:
```python
except (ValueError, sympy.PolynomialError, sympy.SympifyError,
        sympy.NotImplementedError, Exception) as e:
```

Or, more precisely, catch the specific SymPy exceptions documented for each function. The goal is to avoid the generic "something went wrong" handler catching SymPy-specific errors that could have a meaningful user-facing message.

---

# Implementation order recommendation

```
Week 1  X-2, X-3, X-4 (housekeeping — low risk, high value)
        T1-8 logic expansion (self-contained, in existing cog)
        T1-7 /identify (new command, existing cog)

Week 2  T1-1 /distribution collapse
        T1-6 number theory gap-fill
        X-1  security fix for expr_utils

Week 3  T1-4 transforms cog (new file, high-value command)
        T1-2 vector calc additions

Week 4  T1-3 /compare
        T1-5 /convert expansion

Week 5  T2-1 /define session store (touches many cogs — plan carefully)

Week 6  T2-3 calculus analysis commands (5 new commands)

Week 7  T2-2 CSV tools cog (new dependency, file upload handling)

Week 8  T3-1 /interpolate
        T3-2 /piecewise + expr_utils security fix (if X-1 not done earlier)

Week 9  T3-3 Quiz cog
```

---

*Last updated: June 2026. Generated from MathFrame codebase documentation.*
