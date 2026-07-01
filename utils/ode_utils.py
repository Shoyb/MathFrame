"""
utils/ode_utils.py — Shared ODE parsing helpers for cogs/calculus.py.

Both ``/calc ode`` (symbolic, via ``sympy.dsolve``) and ``/calc ode_numeric``
(numeric, via ``scipy.integrate.solve_ivp``) accept identical expression and
initial-condition syntax — e.g. ``y'' + y = 0`` with ``y(0)=1, y'(0)=0``.
This module holds that shared parsing logic so the two commands can never
silently drift apart in what input they accept.

Public functions
-----------------
parse_ode(expression)
    Parse a raw ODE string into a ``sympy.Eq`` plus the detected dependent
    function (e.g. ``y(x)``).
parse_symbolic_ics(initial_conditions, func_to_solve)
    Parse ``"y(0)=1, y'(0)=0"``-style text into the dict format expected by
    ``sympy.dsolve(..., ics=...)``.
ode_order(eq)
    Highest derivative order appearing in *eq*.
extract_numeric_ics(ics, order)
    Convert a symbolic ICs dict into ``(x0, [y0, y0', y0'', ...])`` for
    ``scipy.integrate.solve_ivp``, which requires every condition at one
    common starting point.
build_numeric_rhs(eq, func_to_solve, indep_var, order)
    Reduce an *order*-th order ODE to a first-order system and return a
    ``rhs(x, state) -> list[float]`` callable for ``solve_ivp``.
"""

from __future__ import annotations

import re

import sympy

from utils.parser import parse_expression

# ---------------------------------------------------------------------------
# Expression preprocessing
# ---------------------------------------------------------------------------


_PLACEHOLDER_PREFIX = "phfn_"
"""
Prefix for synthetic placeholder symbols used during ODE parsing.

SymPy's ``parse_expr`` with ``implicit_multiplication_application`` enabled
(used by :func:`utils.parser.parse_expression`) silently rewrites a call like
``y(x)`` into ``y*x`` whenever ``y`` isn't already known to be a function —
which it never is, since the user is the one defining it via this call.  This
is a real bug (also present in the pre-refactor ``/calc ode`` command) and
bites any ODE written with explicit ``func(var)`` syntax, not just primes.

The fix: replace every ``func(var)`` occurrence with a plain placeholder
*symbol* (no parentheses, so the implicit-multiplication transform has
nothing to misinterpret) before handing the string to the parser, then
substitute the placeholder back to ``sympy.Function(func)(var)`` afterward.
``sympy.Derivative`` stays unevaluated under substitution, so wrapping a
placeholder in ``Derivative(...)`` and substituting later produces the
correct symbolic derivative.
"""


def _preprocess_primes(expr_str: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """
    Rewrite prime notation and explicit function calls into a parser-safe
    placeholder form.

    Handles ``f''(x)`` (explicit independent variable), ``y''`` (implicit,
    assumed a function of ``x``), and bare ``func(var)`` calls for any
    function name already detected via the prime forms above (the
    un-primed, zeroth-order term in an ODE, e.g. the ``y(x)`` in
    ``y'' + y(x) = 0``).

    Returns
    -------
    tuple[str, dict[str, tuple[str, str]]]
        The rewritten string, and a mapping of
        ``placeholder_name -> (function_name, variable_name)`` to be
        substituted back in after parsing via :func:`_restore_placeholders`.
    """
    placeholders: dict[str, tuple[str, str]] = {}

    def _placeholder(func: str, var: str) -> str:
        name = f"{_PLACEHOLDER_PREFIX}{func}_{var}"
        placeholders[name] = (func, var)
        return name

    def repl_f(m: re.Match) -> str:
        func = m.group(1)
        primes = len(m.group(2))
        var = m.group(3)
        ph = _placeholder(func, var)
        args = ", ".join([var] * primes)
        return f"Derivative({ph}, {args})"

    expr_str = re.sub(
        r"\b([a-zA-Z_]\w*)('+)\s*\(\s*([a-zA-Z_]\w*)\s*\)", repl_f, expr_str
    )

    def repl_y(m: re.Match) -> str:
        func = m.group(1)
        primes = len(m.group(2))
        ph = _placeholder(func, "x")
        args = ", ".join(["x"] * primes)
        return f"Derivative({ph}, {args})"

    expr_str = re.sub(r"\b([a-zA-Z_]\w*)('+)(?!\()", repl_y, expr_str)

    # Zeroth-order term: any remaining un-primed func(var) call for a
    # function name already detected above (e.g. the "y(x)" in
    # "y'' + y(x) = 0") must use the *same* placeholder for consistency.
    seen_pairs = {pair for pair in placeholders.values()}
    for func, var in seen_pairs:
        pattern = rf"\b{re.escape(func)}\s*\(\s*{re.escape(var)}\s*\)"
        ph = f"{_PLACEHOLDER_PREFIX}{func}_{var}"
        expr_str = re.sub(pattern, ph, expr_str)

    return expr_str, placeholders


def _restore_placeholders(
    expr: sympy.Basic, placeholders: dict[str, tuple[str, str]]
) -> sympy.Basic:
    """Substitute placeholder symbols back to proper ``Function(var)`` calls."""
    if not placeholders:
        return expr
    subs_map = {
        sympy.Symbol(ph): sympy.Function(func)(sympy.Symbol(var))
        for ph, (func, var) in placeholders.items()
    }
    return expr.subs(subs_map)


# ---------------------------------------------------------------------------
# Equation parsing
# ---------------------------------------------------------------------------


async def parse_ode(expression: str) -> tuple[sympy.Eq, sympy.Function]:
    """
    Parse a user-supplied ODE string into a ``sympy.Eq`` plus the detected
    dependent-function atom (e.g. ``y(x)``).

    Raises
    ------
    ValueError
        If no dependent function could be detected in the equation.
    sympy.SympifyError
        If the expression text itself cannot be parsed.
    """
    expr_str, placeholders = _preprocess_primes(expression)

    if "=" in expr_str:
        lhs_str, rhs_str = expr_str.split("=", 1)
        lhs = _restore_placeholders(await parse_expression(lhs_str), placeholders)
        rhs = _restore_placeholders(await parse_expression(rhs_str), placeholders)
        eq = sympy.Eq(lhs, rhs)
    else:
        parsed = _restore_placeholders(await parse_expression(expr_str), placeholders)
        eq = sympy.Eq(parsed, 0)

    # Identify the dependent function: one that's actually been differentiated
    # somewhere in the equation. Deliberately NOT "any Function atom in eq" —
    # that would also match transcendental functions like sin/cos/exp applied
    # to the dependent variable (e.g. the rhs of `y' = sin(x*y)`), grabbing
    # the wrong "function to solve for" via non-deterministic set iteration.
    funcs_in_deriv = set()
    for atom in eq.atoms(sympy.Derivative):
        if isinstance(atom.expr, sympy.Function):
            funcs_in_deriv.add((str(atom.expr.func), atom.variables))

    # Upgrade any remaining BARE symbol with the same name (e.g. a
    # zeroth-order "+ y" term with no explicit "(x)") to a proper function
    # application, so the whole equation is expressed in terms of y(x).
    subs_dict = {}
    for name, vars_tuple in funcs_in_deriv:
        if len(vars_tuple) > 0:
            var = vars_tuple[0]
            sym = sympy.Symbol(name)
            func = sympy.Function(name)(var)
            subs_dict[sym] = func

    if subs_dict:
        eq = eq.subs(subs_dict)

    # Fallback if no functions were found but we have bare y and x
    if (
        not funcs_in_deriv
        and sympy.Symbol("y") in eq.free_symbols
        and sympy.Symbol("x") in eq.free_symbols
    ):
        eq = eq.subs(sympy.Symbol("y"), sympy.Function("y")(sympy.Symbol("x")))
        funcs_in_deriv.add(("y", (sympy.Symbol("x"),)))

    if not funcs_in_deriv:
        raise ValueError("Could not detect a dependent function in the ODE.")

    name, vars_tuple = next(iter(funcs_in_deriv))
    var = vars_tuple[0] if vars_tuple else sympy.Symbol("x")
    func_to_solve = sympy.Function(name)(var)

    return eq, func_to_solve


# ---------------------------------------------------------------------------
# Initial-condition parsing (symbolic — feeds sympy.dsolve)
# ---------------------------------------------------------------------------


async def parse_symbolic_ics(
    initial_conditions: str, func_to_solve: sympy.Function
) -> dict:
    """
    Parse ``"y(0)=1, y'(0)=0"``-style text into the dict format expected by
    ``sympy.dsolve(eq, func, ics=...)``.

    Returns an empty dict if *initial_conditions* is blank.
    """
    ics: dict = {}
    if not initial_conditions:
        return ics

    indep_var = (
        func_to_solve.args[0] if len(func_to_solve.args) > 0 else sympy.Symbol("x")
    )

    for token in initial_conditions.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid initial condition: {token}")
        lhs_str, rhs_str = token.split("=", 1)
        lhs_str = lhs_str.strip()
        rhs_val = await parse_expression(rhs_str)

        m = re.match(r"^([a-zA-Z_]\w*)('*)\s*\(\s*(.+)\s*\)$", lhs_str)
        if not m:
            raise ValueError(f"Could not parse initial condition LHS: {lhs_str}")
        func_name = m.group(1)
        primes = len(m.group(2))
        val_str = m.group(3)
        val = await parse_expression(val_str)

        func_sym = sympy.Function(func_name)

        if primes == 0:
            lhs_ic = func_sym(val)
        else:
            deriv = sympy.Derivative(
                func_sym(indep_var), *(indep_var for _ in range(primes))
            )
            lhs_ic = deriv.subs(indep_var, val)

        ics[lhs_ic] = rhs_val

    return ics


# ---------------------------------------------------------------------------
# Numeric reduction (feeds scipy.integrate.solve_ivp)
# ---------------------------------------------------------------------------


def ode_order(eq: sympy.Eq) -> int:
    """Return the highest derivative order present in *eq* (0 if none)."""
    orders = [0]
    for d in eq.atoms(sympy.Derivative):
        orders.append(len(d.variables))
    return max(orders)


def extract_numeric_ics(ics: dict, order: int) -> tuple[float, list[float]]:
    """
    Convert a symbolic ICs dict (as produced by :func:`parse_symbolic_ics`)
    into ``(x0, [y0, y0', y0'', ...])`` suitable for
    ``scipy.integrate.solve_ivp``, which requires every initial condition to
    be given at one common starting point.

    Raises
    ------
    ValueError
        If conditions are missing, given at inconsistent starting points,
        or don't cover every derivative order from 0 to ``order - 1``.
    """
    if not ics:
        example = ", ".join(
            f"y{'\u2032' * k}(0)={'1' if k == 0 else '0'}" for k in range(order)
        )
        raise ValueError(
            f"Numeric integration requires {order} initial condition(s) at "
            f"the same point, e.g. `{example}`."
        )

    by_order: dict[int, float] = {}
    x0_values: set[float] = set()

    for lhs, rhs in ics.items():
        try:
            rhs_val = float(rhs)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Initial condition value `{rhs}` is not numeric."
            ) from exc

        if isinstance(lhs, sympy.Subs):
            _deriv_expr, variables, points = lhs.args
            deg = len(variables)
            x0 = points[0]
        else:
            deg = 0
            x0 = lhs.args[0]

        try:
            x0_float = float(x0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Initial condition point `{x0}` is not numeric."
            ) from exc

        x0_values.add(x0_float)
        by_order[deg] = rhs_val

    if len(x0_values) > 1:
        raise ValueError(
            "Numeric integration requires all initial conditions to be "
            f"given at the same point (got multiple: {sorted(x0_values)})."
        )

    x0 = x0_values.pop()

    missing = [k for k in range(order) if k not in by_order]
    if missing:
        names = ", ".join(f"y{chr(0x2032) * k}({x0:g})" for k in missing)
        raise ValueError(f"Missing initial condition(s) for: {names}")

    y0 = [by_order[k] for k in range(order)]
    return x0, y0


def build_numeric_rhs(
    eq: sympy.Eq,
    func_to_solve: sympy.Function,
    indep_var: sympy.Symbol,
    order: int,
):
    """
    Reduce an *order*-th order ODE to a first-order system and return a
    ``rhs(x, state) -> list[float]`` callable for
    ``scipy.integrate.solve_ivp``.

    The state vector is ``[y, y', y'', ..., y^(order-1)]``.

    Raises
    ------
    ValueError
        If the equation cannot be solved for its highest derivative (e.g.
        implicit equations SymPy cannot isolate).
    """
    if order < 1:
        raise ValueError("Cannot numerically integrate a 0th-order equation.")

    highest_deriv = sympy.Derivative(
        func_to_solve, *(indep_var for _ in range(order))
    )
    expr = eq.lhs - eq.rhs
    solutions = sympy.solve(expr, highest_deriv)
    if not solutions:
        raise ValueError(
            "Could not isolate the highest derivative — this equation may be "
            "implicit or otherwise unsupported for numeric integration."
        )
    rhs_expr = solutions[0]

    state_syms = sympy.symbols(f"s0:{order}")
    subs_map = {}
    for k in range(order):
        if k == 0:
            subs_map[func_to_solve] = state_syms[0]
        else:
            deriv_k = sympy.Derivative(
                func_to_solve, *(indep_var for _ in range(k))
            )
            subs_map[deriv_k] = state_syms[k]

    rhs_expr_sub = rhs_expr.subs(subs_map)
    rhs_func = sympy.lambdify((indep_var, *state_syms), rhs_expr_sub, modules=["numpy"])

    def rhs(x: float, state: list) -> list:
        return list(state[1:]) + [float(rhs_func(x, *state))]

    return rhs
