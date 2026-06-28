# MathFrame — Command Grouping Migration

## Problem

Discord enforces a hard limit of **100 global slash commands** per bot.  
MathFrame currently registers **110 top-level commands**, causing silent
truncation: the last ~10 commands never appear in the command picker.

## Solution: App Command Groups

Discord's `app_commands.Group` turns a single top-level slot into a parent
that can hold **up to 25 subcommands**.  The group itself counts as one
command toward the 100 limit.  With 16 groups MathFrame drops to **16
top-level slots** and gains room for ~400 more commands in the future.

```
Before:  /simplify   /solve   /expand   /diff   …  (110 slots used)
After:   /alg simplify
         /alg solve
         /alg expand
         /calc diff
         …                               (16 slots used)
```

---

## Proposed Groups (16 total)

| Group    | Full name       | Commands | Cog file(s)                              |
|----------|-----------------|----------|------------------------------------------|
| `/alg`   | Algebra         | 12       | arithmetic + equations + inequalities → **algebra.py** |
| `/stat`  | Statistics      | 14       | statistics.py                            |
| `/nt`    | Number Theory   | 12       | number_theory.py                         |
| `/calc`  | Calculus        | 10       | calculus.py                              |
| `/dis`   | Discrete Math   | 7        | discrete.py                              |
| `/bot`   | Bot / Utility   | 9        | utility + wiki → **bot.py**              |
| `/geo`   | Geometry        | 6        | geometry.py                              |
| `/mat`   | Linear Algebra  | 6        | linear_algebra.py                        |
| `/mem`   | Memory          | 6        | memory.py ✅ already a group             |
| `/sym`   | Symbolic        | 5        | symbolic.py                              |
| `/cx`    | Complex Numbers | 5        | complex.py                               |
| `/admin` | Admin           | 4        | admin.py                                 |
| `/base`  | Base Conversion | 4        | base_n.py                                |
| `/plot`  | Plotting        | 4        | plot_engine.py                           |
| `/tf`    | Transforms      | 4        | transforms.py                            |
| `/render`| Render          | 2        | render.py                                |

### Notes on the two merges

**`algebra.py`** absorbs `arithmetic.py`, `equations.py`, and `inequalities.py`.
These three cogs already share subject matter (`/solve`, `/solve_sim`, `/solve_ineq`)
and the merge removes two redundant cog files.

**`bot.py`** absorbs `utility.py` and `wiki.py`.
`/wiki` and `/wiki_search` are too small to justify their own top-level slot;
both fit naturally alongside the other bot-utility commands.

---

## Full Command Mapping

### `/alg` — Algebra (12 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/alg simplify`      | `/simplify`          |                                |
| `/alg solve`         | `/solve`             |                                |
| `/alg expand`        | `/expand`            |                                |
| `/alg factor`        | `/factor`            |                                |
| `/alg solve_sys`     | `/solve_system`      | Rename: shorter, consistent    |
| `/alg table`         | `/table`             |                                |
| `/alg poly_div`      | `/poly_div`          |                                |
| `/alg verify`        | `/verify`            |                                |
| `/alg compare`       | `/compare`           |                                |
| `/alg solve_sim`     | `/solve_sim`         | Moved from equations.py        |
| `/alg ineq`          | `/solve_ineq`        | Rename: strip redundant "solve"|
| `/alg ineq_sys`      | `/solve_ineq_system` | Rename: shorter                |

### `/calc` — Calculus (10 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/calc diff`         | `/diff`              |                                |
| `/calc integrate`    | `/integrate`         |                                |
| `/calc limit`        | `/limit`             |                                |
| `/calc series`       | `/series`            |                                |
| `/calc sum`          | `/sum_series`        | Rename: strip "_series"        |
| `/calc product`      | `/product_series`    | Rename: strip "_series"        |
| `/calc ode`          | `/ode`               |                                |
| `/calc gradient`     | `/gradient`          |                                |
| `/calc divergence`   | `/divergence`        |                                |
| `/calc curl`         | `/curl`              |                                |

### `/stat` — Statistics (14 subcommands)

| New command          | Old command          | Notes |
|----------------------|----------------------|-------|
| `/stat mean`         | `/mean`              |       |
| `/stat median`       | `/median`            |       |
| `/stat mode`         | `/mode`              |       |
| `/stat stdev`        | `/stdev`             |       |
| `/stat variance`     | `/variance`          |       |
| `/stat zscore`       | `/zscore`            |       |
| `/stat correlation`  | `/correlation`       |       |
| `/stat regression`   | `/regression`        |       |
| `/stat distribution` | `/distribution`      |       |
| `/stat normal_pdf`   | `/normal_pdf`        |       |
| `/stat normal_cdf`   | `/normal_cdf`        |       |
| `/stat inv_normal`   | `/inv_normal`        |       |
| `/stat binomial_cdf` | `/binomial_cdf`      |       |
| `/stat poisson_cdf`  | `/poisson_cdf`       |       |

### `/nt` — Number Theory (12 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/nt gcd`            | `/gcd`               |                                |
| `/nt lcm`            | `/lcm`               |                                |
| `/nt is_prime`       | `/is_prime`          |                                |
| `/nt factorize`      | `/factorize`         |                                |
| `/nt primes`         | `/primes_up_to`      | Rename: shorter                |
| `/nt modular`        | `/modular`           |                                |
| `/nt fibonacci`      | `/fibonacci`         |                                |
| `/nt totient`        | `/totient`           |                                |
| `/nt divisors`       | `/divisors`          |                                |
| `/nt is_perfect`     | `/is_perfect`        |                                |
| `/nt mobius`         | `/mobius`            |                                |
| `/nt crt`            | `/chinese_remainder` | Rename: standard abbreviation  |

### `/mat` — Linear Algebra (6 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/mat det`           | `/matrix_det`        | Strip redundant "matrix_"      |
| `/mat inv`           | `/matrix_inv`        | Strip redundant "matrix_"      |
| `/mat eigen`         | `/eigenvalues`       | Rename: shorter                |
| `/mat dot`           | `/dot`               |                                |
| `/mat cross`         | `/cross`             |                                |
| `/mat rref`          | `/rref`              |                                |

### `/sym` — Symbolic (5 subcommands)

| New command          | Old command          | Notes |
|----------------------|----------------------|-------|
| `/sym latex`         | `/latex`             |       |
| `/sym subs`          | `/subs`              |       |
| `/sym partial`       | `/partial_fraction`  | Rename: shorter |
| `/sym identify`      | `/identify`          |       |
| `/sym roots`         | `/roots`             |       |

### `/geo` — Geometry (6 subcommands)

| New command          | Old command              | Notes                          |
|----------------------|--------------------------|--------------------------------|
| `/geo circle`        | `/circle_area`           | Rename: area is implicit       |
| `/geo circumference` | `/circle_circumference`  | Strip "circle_"                |
| `/geo triangle`      | `/triangle_area`         | Rename: area is implicit       |
| `/geo pythagorean`   | `/pythagorean`           |                                |
| `/geo trig`          | `/trig`                  |                                |
| `/geo distance`      | `/distance`              |                                |

### `/dis` — Discrete Math (7 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/dis permutation`   | `/permutation`       |                                |
| `/dis combination`   | `/combination`       |                                |
| `/dis truth_table`   | `/truth_table`       |                                |
| `/dis set_ops`       | `/set_ops`           |                                |
| `/dis binomial`      | `/binomial_coeff`    | Rename: strip "_coeff"         |
| `/dis bool`          | `/simplify_bool`     | Rename: strip "simplify_"      |
| `/dis logic_equiv`   | `/logic_equiv`       |                                |

### `/cx` — Complex Numbers (5 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/cx calc`           | `/complex_calc`      | Strip "complex_"               |
| `/cx polar`          | `/complex_polar`     | Strip "complex_"               |
| `/cx rect`           | `/complex_rect`      | Strip "complex_"               |
| `/cx conj`           | `/complex_conjugate` | Rename + strip                 |
| `/cx modulus`        | `/complex_modulus`   | Strip "complex_"               |

### `/tf` — Transforms (4 subcommands)

| New command          | Old command          | Notes |
|----------------------|----------------------|-------|
| `/tf laplace`        | `/laplace`           |       |
| `/tf inv_laplace`    | `/inv_laplace`       |       |
| `/tf fourier`        | `/fourier`           |       |
| `/tf inv_fourier`    | `/inv_fourier`       |       |

### `/base` — Base Conversion (4 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/base convert`      | `/base_convert`      | Strip "base_"                  |
| `/base add`          | `/base_add`          | Strip "base_"                  |
| `/base logic`        | `/base_logic`        | Strip "base_"                  |
| `/base table`        | `/bases`             | Rename: clearer                |

### `/plot` — Plotting (4 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/plot start`        | `/plot`              | Rename: `plot` can't be both group and command |
| `/plot quick`        | `/quickplot`         | Rename: consistent with group  |
| `/plot multi`        | `/multiplot`         | Rename: consistent with group  |
| `/plot load`         | `/plot_import`       | Rename: clearer                |

### `/render` — Render (2 subcommands)

| New command          | Old command          | Notes |
|----------------------|----------------------|-------|
| `/render expr`       | `/render`            | Rename: `render` can't be both group and subcommand |
| `/render formula`    | `/formula`           |       |

### `/bot` — Utility (9 subcommands)

| New command          | Old command          | Notes                          |
|----------------------|----------------------|--------------------------------|
| `/bot history`       | `/history`           |                                |
| `/bot clear`         | `/clear_history`     | Rename: shorter                |
| `/bot constants`     | `/constants`         |                                |
| `/bot help`          | `/help_math`         | Rename: strip "_math"          |
| `/bot convert`       | `/convert`           |                                |
| `/bot units`         | `/units`             |                                |
| `/bot about`         | `/about`             |                                |
| `/bot wiki`          | `/wiki`              | Moved from wiki.py             |
| `/bot wiki_search`   | `/wiki_search`       | Moved from wiki.py             |

### `/admin` — Admin (4 subcommands)

| New command          | Old command          | Notes |
|----------------------|----------------------|-------|
| `/admin enable`      | `/enable`            |       |
| `/admin disable`     | `/disable`           |       |
| `/admin reset`       | `/reset`             |       |
| `/admin status`      | `/status`            |       |

### `/mem` — Memory (already a group ✅)

No changes needed.

---

## Implementation Pattern

The code change per cog is purely structural — **zero logic changes**.

### Before
```python
class CalcCog(commands.Cog, name="Calculus"):

    @app_commands.command(name="diff", description="...")
    @app_commands.describe(expression="...")
    @app_commands.checks.cooldown(1, 3.0)
    async def diff(self, interaction, expression: str, ...):
        ...
```

### After
```python
class CalcCog(commands.Cog, name="Calculus"):

    calc = app_commands.Group(name="calc", description="Calculus operations.")

    @calc.command(name="diff", description="...")          # ← only this line changes
    @app_commands.describe(expression="...")               # ← unchanged
    @app_commands.checks.cooldown(1, 3.0)                 # ← unchanged
    async def diff(self, interaction, expression: str, ...):
        ...                                                # ← unchanged
```

Three things change per command: the decorator `@app_commands.command` → `@<group>.command`.
Everything else — parameters, cooldowns, descriptions, logic — stays identical.

### Pattern for merged cogs

When two cog files are merged, copy both class bodies into one file and share a single Group:

```python
# cogs/algebra.py  (arithmetic + equations + inequalities merged)

class AlgebraCog(commands.Cog, name="Algebra"):

    alg = app_commands.Group(name="alg", description="Algebra and equation solving.")

    # ── from arithmetic.py ────────────────────────────────────────────────
    @alg.command(name="simplify", ...)
    async def simplify(self, ...): ...

    @alg.command(name="solve", ...)
    async def solve(self, ...): ...

    # ── from equations.py ─────────────────────────────────────────────────
    @alg.command(name="solve_sim", ...)
    async def solve_sim(self, ...): ...

    # ── from inequalities.py ──────────────────────────────────────────────
    @alg.command(name="ineq", ...)
    async def solve_ineq(self, ...): ...


async def setup(bot):
    await bot.add_cog(AlgebraCog(bot))
```

---

## Files to Change

| Action  | File(s)                                              | What changes                          |
|---------|------------------------------------------------------|---------------------------------------|
| Merge   | arithmetic + equations + inequalities → `algebra.py` | New file; old three deleted from COGS |
| Merge   | utility + wiki → `bot.py`                            | New file; old two deleted from COGS   |
| Modify  | `calculus.py`                                        | Add `calc` group; wrap all commands   |
| Modify  | `statistics.py`                                      | Add `stat` group; wrap all commands   |
| Modify  | `number_theory.py`                                   | Add `nt` group; wrap all commands     |
| Modify  | `linear_algebra.py`                                  | Add `mat` group; wrap all commands    |
| Modify  | `symbolic.py`                                        | Add `sym` group; wrap all commands    |
| Modify  | `geometry.py`                                        | Add `geo` group; wrap all commands    |
| Modify  | `discrete.py`                                        | Add `dis` group; wrap all commands    |
| Modify  | `complex.py`                                         | Add `cx` group; wrap all commands     |
| Modify  | `transforms.py`                                      | Add `tf` group; wrap all commands     |
| Modify  | `base_n.py`                                          | Add `base` group; wrap all commands   |
| Modify  | `plot_engine.py`                                     | Add `plot` group; wrap all commands   |
| Modify  | `render.py`                                          | Add `render` group; wrap all commands |
| Modify  | `admin.py`                                           | Add `admin` group; wrap all commands  |
| Modify  | `main.py`                                            | Update COGS list (remove merged cogs, add algebra/bot) |
| No-op   | `memory.py`                                          | Already a group — no changes needed   |

---

## Caveats

**`/plot start`** — The original top-level command was `/plot`, but a
subcommand cannot share its name with the parent group.  Renamed to
`/plot start`.  Same applies to `/render expr`.

**`/alg ineq` renaming** — `ineq` is short but unambiguous in context.
If you prefer clarity over brevity, use `solve_ineq` instead.

**Cooldowns on groups** — Cooldowns are set per subcommand with
`@app_commands.checks.cooldown(...)`, same as before.  No change needed.

**`/mem` group** — Already implemented; included in the table for
completeness but requires no migration work.

**Discord sync delay** — After deploying, Discord takes up to an hour to
propagate global command changes.  Use guild-scoped sync during testing:
```python
bot.tree.copy_global_to(guild=discord.Object(id=YOUR_TEST_GUILD_ID))
await bot.tree.sync(guild=discord.Object(id=YOUR_TEST_GUILD_ID))
```
