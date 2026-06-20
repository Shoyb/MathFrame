# MathFrame — Plot & Solver Expansion: Project Plan

**Scope:** Three initiatives — (A) new plot types, (B) auto-generated plot summaries, and (C) a major Solver Engine expansion modeled on `plot_engine.py`'s interactive-builder pattern.

**Out of scope (tracked separately, but relevant):** the parser-security and view-ownership items from the codebase audit. They're called out below wherever a new workstream touches the same surface, since this plan adds several new places that parse user-typed expressions.

---

## Guiding principles

1. **Reuse what already works.** `PlotConfig`'s shape — a single mutable session dataclass, one modal per concern, a persistent view, export/import via base64+zlib — is proven. The Solver Engine should borrow the *pattern*, not the literal modal layout (plotting splits by visual concern; solving should split by problem structure).
2. **Don't force everything into a builder.** Keep `/diff`, `/quickplot`, etc. as fast one-shot commands. The interactive builder is for problem types that genuinely need structured multi-field input — systems, ODEs with initial conditions, multi-inequality regions.
3. **No new direct `sympify` call sites.** Every new modal/field that accepts a math expression routes through `utils/parser.py::parse_expression()`. This is the same gap flagged for `expr_utils.py` in the codebase audit — new work must not repeat it.
4. **One plotting code path.** New plot types are added as another `_plot_<kind>_blocking()` / public-wrapper pair in `utils/plotter.py`, run through the existing `_run_blocking()` helper — not a parallel implementation.

---

## Workstream A — New plot types

### A1. Smart defaults: auto-zoom & asymptote handling
*Do this first — it improves every existing plot type for free.*

- **Auto-domain:** before rendering, compute roots/critical points (`sympy.solve`, `sympy.calculus.util.continuous_domain`) and set the default domain with padding when the user hasn't specified one, instead of a fixed default window.
- **Asymptote-aware breaks:** detect singularities (`sympy.singularities`) and insert gaps (`NaN`) in the sampled array around them, so a function like `1/x` doesn't draw a near-vertical spike across the whole y-range.
- **Files touched:** `utils/plotter.py` (sampling logic shared by `plot_function` and friends), no new `PlotSpec` fields needed.
- **Size:** S–M

### A2. Riemann sum visualization
- Draws shaded rectangles (left/right/midpoint) under a curve between bounds, with the actual curve overlaid — a natural companion to `/integrate`.
- New `_plot_riemann_blocking(expr, a, b, n, method)` in `plotter.py`; reuses existing fill-shading code.
- Hook point: an optional `show_plot` flag on `/integrate`, or accessible via the plot builder as its own kind.
- **Size:** S–M

### A3. Slope / direction fields
- For first-order ODEs `dy/dx = f(x, y)`: draws a grid of short line segments showing the local slope at each point (`matplotlib.pyplot.quiver` or `Line2D` segments).
- New `PLOT_TYPES` entry `"slope-field"`; reuses `ExpressionModal` for `f(x, y)`, needs one new field for arrow-grid density.
- Pairs naturally with the planned `/dsolve` solver feature (Workstream C) but works standalone.
- **Size:** M

### A4. Feasible region for systems of inequalities
- Extends the existing single-inequality `plot_inequality` to accept a *list* of inequalities and shade their intersection.
- UI: an "add inequality" pattern similar to the existing `AdditionalExprModal` used for `/multiplot`.
- **Size:** M

### A5. Domain coloring for complex functions
- Maps `f(z)` over the complex plane: hue = `arg(f(z))`, brightness = `|f(z)|`, rendered via `numpy` meshgrid + HSV→RGB + `imshow`.
- Independent of any future complex-analysis cog, but most valuable once one exists.
- **Risk:** resolution_2d-scale color computation may be slow enough to threaten Discord's interaction-ack window — likely needs the existing defer-then-followup pattern other long-running commands already use.
- **Size:** M–L

---

## Workstream B — Auto-generated plot summary

A short text block accompanying function-type plots: domain, estimated range, intercepts, symmetry (even/odd), and — when cheap to compute — critical points and asymptotes.

- **New module:** `utils/analyzer.py`, exposing `analyze_function(expr) -> FunctionSummary` (a small dataclass: domain, intercepts, symmetry, asymptotes, extrema). Built once, reused in two places: the plot summary now, and a future solver "info" panel later — avoids duplicating the analysis logic.
- **Time-boxing:** must respect `COMPUTE_TIMEOUT` the same way `parser.py` does. Cap what's attempted — e.g. skip global root-finding on high-degree or transcendental expressions and report "not determined" rather than risking a timeout.
- **Display:** appended as a field on the existing plot embed, not a separate message.
- **Size:** M

---

## Workstream C — Solver Engine

A phased build, since this is comparable in scope to `plot_engine.py` (1,713 lines) + `plotter.py` (2,001 lines) combined.

### Phase C1 — Classify-then-dispatch registry (foundation)
This is the prerequisite for everything else in this workstream, and worth doing on its own regardless of whether the interactive builder ships.

- Replace `/solve`'s current "is it a quadratic?" special case with real classification: linear, quadratic, higher-degree polynomial, system, trig, exponential, inequality, ODE.
- Refactor `utils/solver.py` from four standalone functions into a registry: `STEP_GENERATORS: dict[ProblemType, Callable]`, keyed off the classifier's output.
- **Size:** M–L

```python
# illustrative — not final code
class ProblemType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    POLYNOMIAL = "polynomial"
    SYSTEM = "system"
    TRIG = "trig"
    INEQUALITY = "inequality"
    ODE = "ode"

def classify_problem(expr_or_exprs) -> ProblemType: ...

STEP_GENERATORS: dict[ProblemType, Callable[..., StepList]] = {
    ProblemType.QUADRATIC: solve_quadratic_steps,
    ProblemType.LINEAR: solve_linear_steps,
    # ...
}
```

### Phase C2 — `SolverConfig` + input modals
Scope the first pass to the problem types current slash-command parameters genuinely can't express cleanly: **systems of equations** and **ODEs with initial conditions**.

- `SolverConfig` dataclass: `problem_type`, `expressions: list[str]`, `variables: list[str]`, `domain_assumptions`, `initial_conditions: dict`, `method_override`, output prefs (`show_steps`, `decimal_places`, `simplify_level`).
- Two modals, split by *structure* rather than mirroring the plot engine's visual-concern split:
  - **Problem modal** — the equation(s)/ODE itself; supports adding rows for systems, similar to `AdditionalExprModal`.
  - **Constraints & method modal** — domain/assumptions, initial/boundary conditions, technique override.
- Initial-condition syntax: Discord modal fields are effectively single-line; reuse the existing `"x=2, y=pi"`-style grammar from `/subs` (`cogs/symbolic.py::_parse_substitutions`) for something like `"y(0)=1, y'(0)=0"`.
- **All new fields route through `parse_expression()`** — no direct `sympify`.
- **Size:** L

### Phase C3 — Interactive actions
The two features that have no plot-engine equivalent, and the actual differentiators of building this at all:

- **Try another method** — for problems solvable multiple ways (quadratic via factoring vs. completing the square vs. the formula; integrals via substitution vs. parts; systems via substitution vs. elimination), cycle through alternate derivations on the same result instead of re-solving from scratch. Requires scoping which problem types genuinely have multiple natural methods — not all do.
- **Verify** — substitute the solution back into the original equation/system and confirm it collapses to `True`/`0`. Cheap relative to its trust value, since the parsing/SymPy machinery already exists.
- A **toggle steps** button for users who want just the answer.
- **Size:** M

### Phase C4 — Bridge to the plot engine
A "Plot this" action that hands the current `SolverConfig`'s expression to a prefilled `PlotConfig` and opens the existing plot builder (or `/quickplot` for a single function) — the two engines sharing a session object is what makes this a one-click action rather than a retype.

- **Size:** S–M

---

## Cross-cutting concerns

- **Discord modal limits** — 5 fields per modal, 45-char labels, 4000-char paragraph inputs — shape how the Problem/Constraints modals are split, especially for systems with several equations.
- **Caching** — `data/cache.py`'s key scheme currently keys on expression strings; the Solver Engine will need session-shaped cache keys (problem type + expressions + constraints), not just a single expression.
- **Persistence** — saved/named solver sessions are a natural follow-on once this exists, but are a separate ticket; the bot's history/cache layers are explicitly in-memory by current design.
- **Testing** — prioritize coverage on the classify-then-dispatch registry first, since misclassification silently produces wrong step explanations rather than an obvious error.

---

## Suggested build order

| # | Deliverable | Depends on | Size |
|---|---|---|---|
| 1 | Auto-zoom & asymptote handling (A1) | — | S–M |
| 2 | Plot summary / `utils/analyzer.py` (B) | — | M |
| 3 | Riemann sum plot + `/integrate` hook (A2) | — | S–M |
| 4 | Classify-then-dispatch registry (C1) | — | M–L |
| 5 | Feasible region, multi-inequality (A4) | — | M |
| 6 | Slope field plot (A3) | — | M |
| 7 | `SolverConfig` + Problem/Constraints modals (C2) | 4 | L |
| 8 | Try-another-method + Verify (C3) | 4, 7 | M |
| 9 | Domain coloring (A5) | — | M–L |
| 10 | Solver → Plot bridge (C4) | 7 | S–M |

Items 1–3 and 4 have no dependencies on each other and can run in parallel if more than one person is working on this. Everything in the Solver Engine column (4, 7, 8, 10) is sequential.

---

## Open questions

- Which problem types actually have more than one natural solution method worth showing in "Try another method"? Needs a short scoping pass before C3 starts.
- Does domain coloring's render time fit inside the existing defer/followup pattern, or does it need its own longer-timeout path?
- Should the Solver Engine get its own export/import string (mirroring `/plot_import`) from day one, or wait until persistence is revisited?
