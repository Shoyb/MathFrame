"""
utils/solver.py — Step-by-step symbolic math solvers for the math bot.

Each public function returns a :data:`StepList` — an ordered list of
``(description, expression_string)`` tuples that walk the user through a
computation.  The tuples plug directly into
:func:`utils.formatter.math_embed`'s ``steps`` parameter.

All functions are **synchronous** (they do not call the async parser) and
are intended to be called from within a thread-pool executor or after the
expression has already been parsed by :func:`utils.parser.parse_expression`.

Usage
-----
::

    from utils.solver import differentiate_steps, StepList

    steps: StepList = differentiate_steps(expr, x)
    embed = math_embed("Differentiate", str(steps[-1][1]), steps=steps)
"""

import sympy

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

StepList = list[tuple[str, str]]
"""Ordered list of ``(step_description, expression_string)`` pairs."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _err(message: str) -> StepList:
    """
    Return a single-step list that surfaces *message* as the only entry.

    Used by every public function's ``except`` block so callers never
    receive a raised exception — they always get a usable :data:`StepList`.

    Parameters
    ----------
    message:
        A user-friendly description of what went wrong.
    """
    return [("Error", message)]


def _expr_str(expr: sympy.Basic) -> str:
    """
    Stringify a SymPy expression for embedding in a Discord message.

    Uses ``sympy.pretty`` with ``use_unicode=False`` so the output stays
    ASCII-safe across all Discord clients, then falls back to ``str``
    if pretty-printing raises for any reason.

    Parameters
    ----------
    expr:
        Any SymPy expression or matrix.
    """
    try:
        return sympy.pretty(expr, use_unicode=False)
    except Exception:
        return str(expr)

# ---------------------------------------------------------------------------
# Public step-builders
# ---------------------------------------------------------------------------

def solve_quadratic_steps(
    expr: sympy.Expr,
    var: sympy.Symbol,
) -> StepList:
    """
    Return step-by-step working for solving a quadratic equation in *var*.

    The function treats *expr* as the left-hand side of ``expr = 0`` and
    solves for *var*.  It extracts the standard coefficients ``a``, ``b``,
    ``c`` from the expanded polynomial, shows the discriminant
    ``Δ = b² − 4ac``, and lists all solutions found by SymPy's solver.

    Steps produced
    --------------
    1. **Original expression** — the expression as supplied.
    2. **Expand** — fully expanded polynomial form.
    3. **Identify coefficients** — shows ``a``, ``b``, ``c``.
    4. **Compute discriminant** — ``Δ = b² − 4ac`` evaluated.
    5. **Solutions** — all roots of the equation.

    Parameters
    ----------
    expr:
        A SymPy expression representing the LHS of the equation (= 0).
    var:
        The symbol to solve for.

    Returns
    -------
    StepList
        A :data:`StepList` describing the full solution process, or a
        single error step if something goes wrong.

    Example
    -------
    ::

        x = sympy.Symbol("x")
        expr = sympy.parse_expr("x**2 - 5*x + 6")
        steps = solve_quadratic_steps(expr, x)
        # → [("Original expression", "x**2 - 5*x + 6"),
        #    ("Expand", "x**2 - 5*x + 6"),
        #    ("Identify coefficients", "a = 1, b = -5, c = 6"),
        #    ("Compute discriminant Δ = b² − 4ac", "Δ = 1"),
        #    ("Solutions (x = 0)", "x = 2, x = 3")]
    """
    try:
        steps: StepList = []

        # Step 1 — original
        steps.append(("Original expression", _expr_str(expr)))

        # Step 2 — expand
        expanded = sympy.expand(expr)
        steps.append(("Expand to standard form", _expr_str(expanded)))

        # Step 3 — extract coefficients from the polynomial in var
        poly = sympy.Poly(expanded, var)
        degree = poly.degree()
        if degree != 2:
            return _err(
                f"Expression has degree {degree}; "
                "solve_quadratic_steps requires a degree-2 polynomial."
            )

        # Poly.all_coeffs() returns [a, b, c] for ax² + bx + c
        coeffs = poly.all_coeffs()
        # Pad with zeros in case of a monomial like x²  (coeffs = [1, 0, 0])
        while len(coeffs) < 3:
            coeffs.insert(0, sympy.Integer(0))
        a, b, c = coeffs
        steps.append((
            "Identify coefficients a, b, c",
            f"a = {_expr_str(a)},  b = {_expr_str(b)},  c = {_expr_str(c)}",
        ))

        # Step 4 — discriminant
        discriminant = sympy.expand(b**2 - 4 * a * c)
        steps.append((
            "Compute discriminant  Δ = b² − 4ac",
            f"Δ = {_expr_str(b)}² − 4·{_expr_str(a)}·{_expr_str(c)} = {_expr_str(discriminant)}",
        ))

        # Step 5 — solutions
        solutions = sympy.solve(expanded, var)
        if not solutions:
            sol_str = "No real solutions"
        else:
            sol_str = ",   ".join(
                f"{var} = {_expr_str(s)}" for s in solutions
            )
        steps.append((f"Solutions ({var} = 0)", sol_str))

        return steps

    except sympy.PolynomialError as exc:
        return _err(f"Expression is not a polynomial in {var}: {exc}")
    except Exception as exc:
        return _err(f"Could not solve quadratic: {exc}")


def differentiate_steps(
    expr: sympy.Expr,
    var: sympy.Symbol,
    order: int = 1,
) -> StepList:
    """
    Return step-by-step working for differentiating *expr* with respect to
    *var* up to the given *order*.

    Steps produced
    --------------
    1. **Original expression** — the expression as supplied.
    2. **Apply derivative** — shows the Leibniz notation for the operation.
    3. **Result** — the simplified derivative.

    For *order* > 1 an intermediate step is added for each differentiation
    pass so the user can see the expression evolving.

    Parameters
    ----------
    expr:
        The expression to differentiate.
    var:
        The variable of differentiation.
    order:
        How many times to differentiate (default ``1``).  Must be ≥ 1.

    Returns
    -------
    StepList

    Example
    -------
    ::

        x = sympy.Symbol("x")
        steps = differentiate_steps(sympy.parse_expr("x**3 + 2*x"), x)
        # → [("Original expression", "x**3 + 2*x"),
        #    ("Apply d/dx", "d/dx [x**3 + 2*x]"),
        #    ("Result", "3*x**2 + 2")]
    """
    try:
        if order < 1:
            return _err(f"Differentiation order must be ≥ 1 (got {order}).")

        steps: StepList = []

        # Step 1 — original
        steps.append(("Original expression", _expr_str(expr)))

        current = expr
        for n in range(1, order + 1):
            # Leibniz notation label
            if order == 1:
                op_label = f"Apply  d/d{var}"
                op_expr  = f"d/d{var} [ {_expr_str(current)} ]"
            else:
                sup = _ordinal_superscript(n)
                op_label = f"Apply  d{sup}/d{var}{sup}  (pass {n} of {order})"
                op_expr  = f"d/d{var} [ {_expr_str(current)} ]"

            steps.append((op_label, op_expr))

            current = sympy.diff(current, var)
            current = sympy.simplify(current)

            if order > 1:
                steps.append((f"Result after pass {n}", _expr_str(current)))

        # Final result step (only added once for order == 1)
        if order == 1:
            steps.append(("Result", _expr_str(current)))
        else:
            steps.append((f"Final result  (d^{order}/d{var}^{order})", _expr_str(current)))

        return steps

    except Exception as exc:
        return _err(f"Could not differentiate expression: {exc}")


def integrate_steps(
    expr: sympy.Expr,
    var: sympy.Symbol,
) -> StepList:
    """
    Return step-by-step working for the indefinite integral of *expr* with
    respect to *var*.

    Steps produced
    --------------
    1. **Original expression** — the expression as supplied.
    2. **Apply integral** — shows the integral in ∫ … d*var* notation.
    3. **Result** — the antiderivative, with a note that ``+ C`` is implied.

    If SymPy cannot find a closed-form antiderivative it returns an
    unevaluated ``Integral`` object, which is displayed with a note
    explaining that no elementary form was found.

    Parameters
    ----------
    expr:
        The integrand.
    var:
        The variable of integration.

    Returns
    -------
    StepList

    Example
    -------
    ::

        x = sympy.Symbol("x")
        steps = integrate_steps(sympy.parse_expr("3*x**2 + 2*x"), x)
        # → [("Original expression", "3*x**2 + 2*x"),
        #    ("Apply indefinite integral", "∫ 3*x**2 + 2*x  d x"),
        #    ("Result  (+ C implied)", "x**3 + x**2")]
    """
    try:
        steps: StepList = []

        # Step 1 — original
        steps.append(("Original expression", _expr_str(expr)))

        # Step 2 — show the integral notation
        steps.append((
            f"Apply indefinite integral with respect to {var}",
            f"∫ {_expr_str(expr)}  d{var}",
        ))

        # Step 3 — compute
        result = sympy.integrate(expr, var)

        # Detect unevaluated integrals (SymPy returns an Integral object)
        if result.has(sympy.Integral):
            steps.append((
                "Result  (no elementary antiderivative found)",
                _expr_str(result),
            ))
        else:
            steps.append((
                "Result  (+ C implied)",
                _expr_str(result),
            ))

        return steps

    except Exception as exc:
        return _err(f"Could not integrate expression: {exc}")


def factor_steps(expr: sympy.Expr) -> StepList:
    """
    Return step-by-step working for factoring *expr*.

    Steps produced
    --------------
    1. **Original expression** — the expression as supplied.
    2. **Expand** — fully expanded form (only added when it differs from the
       original, so ``x**2 - 1`` doesn't show a redundant identical step).
    3. **Factored form** — result of ``sympy.factor``.  If the expression is
       already fully factored, a note is appended to that effect.

    Parameters
    ----------
    expr:
        The expression to factor.

    Returns
    -------
    StepList

    Example
    -------
    ::

        x = sympy.Symbol("x")
        steps = factor_steps(sympy.parse_expr("x**2 - 5*x + 6"))
        # → [("Original expression", "x**2 - 5*x + 6"),
        #    ("Factored form", "(x - 2)*(x - 3)")]
    """
    try:
        steps: StepList = []

        # Step 1 — original
        steps.append(("Original expression", _expr_str(expr)))

        # Step 2 — expand if the expanded form differs (catches inputs like
        # "(x+1)*(x-1) + x" that should first be simplified before factoring)
        expanded = sympy.expand(expr)
        if sympy.simplify(expanded - expr) != 0:
            steps.append(("Expand first", _expr_str(expanded)))

        # Step 3 — factor
        factored = sympy.factor(expanded)

        # Detect no-op: factor() returns the expanded poly unchanged when it
        # is already irreducible or not factorable over ℤ.
        if sympy.simplify(factored - expanded) == 0 and str(factored) == str(expanded):
            steps.append((
                "Factored form  (already fully factored / irreducible over ℤ)",
                _expr_str(factored),
            ))
        else:
            steps.append(("Factored form", _expr_str(factored)))

        return steps

    except Exception as exc:
        return _err(f"Could not factor expression: {exc}")

# ---------------------------------------------------------------------------
# Cubic and quartic step-builders
# ---------------------------------------------------------------------------

def solve_cubic_steps(
    expr: sympy.Expr,
    var: sympy.Symbol,
) -> StepList:
    """
    Return step-by-step working for solving a cubic equation in *var*.

    The walk-through follows Cardano's method:

    1. Original expression.
    2. Expand to standard form  ``ax³ + bx² + cx + d = 0``.
    3. Identify coefficients a, b, c, d.
    4. Depress the cubic via the substitution  ``x = t − b/(3a)``
       to obtain ``t³ + pt + q = 0``.
    5. Compute the Cardano discriminant  ``Δ = −4p³ − 27q²``.
    6. Apply Cardano's formula (via SymPy) to find the three roots.

    Parameters
    ----------
    expr:
        A SymPy expression representing the LHS of the equation (= 0).
    var:
        The symbol to solve for.

    Returns
    -------
    StepList
    """
    try:
        steps: StepList = []

        # Step 1 — original
        steps.append(("Original expression", _expr_str(expr)))

        # Step 2 — expand to standard form
        expanded = sympy.expand(expr)
        steps.append(("Expand to standard form  ax³ + bx² + cx + d = 0", _expr_str(expanded)))

        # Step 3 — extract coefficients
        poly = sympy.Poly(expanded, var)
        if poly.degree() != 3:
            return _err(
                f"Expression has degree {poly.degree()}; "
                "solve_cubic_steps requires a degree-3 polynomial."
            )
        # Poly.all_coeffs() returns [a, b, c, d] for ax³ + bx² + cx + d
        coeffs = poly.all_coeffs()
        while len(coeffs) < 4:
            coeffs.insert(0, sympy.Integer(0))
        a, b, c, d = coeffs
        steps.append((
            "Identify coefficients  a, b, c, d",
            f"a = {_expr_str(a)},  b = {_expr_str(b)},  c = {_expr_str(c)},  d = {_expr_str(d)}",
        ))

        # Step 4 — depressed cubic substitution  x = t − b/(3a)
        shift = sympy.Rational(1, 3) * b / a
        t = sympy.Symbol("t")
        depressed_raw = sympy.expand(expr.subs(var, t - shift))
        depressed_poly = sympy.Poly(depressed_raw, t)
        dep_coeffs = depressed_poly.all_coeffs()
        while len(dep_coeffs) < 4:
            dep_coeffs.insert(0, sympy.Integer(0))
        _, _, p, q = dep_coeffs
        p = sympy.simplify(p)
        q = sympy.simplify(q)
        steps.append((
            f"Depress cubic via  {var} = t − b/(3a)",
            f"t³ + pt + q = 0   where  p = {_expr_str(p)},  q = {_expr_str(q)}",
        ))

        # Step 5 — Cardano discriminant  Δ = −4p³ − 27q²
        delta = sympy.expand(-4 * p**3 - 27 * q**2)
        delta_simplified = sympy.simplify(delta)
        disc_note = (
            " (Δ > 0: 3 real roots)"  if delta_simplified.is_positive else
            " (Δ = 0: repeated root)" if delta_simplified == 0 else
            " (Δ < 0: 1 real root)"
        )
        steps.append((
            "Cardano discriminant  Δ = −4p³ − 27q²",
            f"Δ = {_expr_str(delta_simplified)}{disc_note}",
        ))

        # Step 6 — roots via SymPy
        solutions = sympy.solve(expanded, var)
        if not solutions:
            sol_str = "No closed-form roots found"
        else:
            sol_str = "\n".join(f"{var} = {_expr_str(s)}" for s in solutions)
        steps.append(("Apply Cardano's formula — roots", sol_str))

        return steps

    except sympy.PolynomialError as exc:
        return _err(f"Expression is not a polynomial in {var}: {exc}")
    except Exception as exc:
        return _err(f"Could not solve cubic: {exc}")


def solve_quartic_steps(
    expr: sympy.Expr,
    var: sympy.Symbol,
) -> StepList:
    """
    Return step-by-step working for solving a quartic equation in *var*.

    The walk-through follows the resolvent-cubic approach:

    1. Original expression.
    2. Expand to standard form  ``ax⁴ + bx³ + cx² + dx + e = 0``.
    3. Identify coefficients a, b, c, d, e.
    4. Depress the quartic via  ``x = t − b/(4a)`` to obtain
       ``t⁴ + pt² + qt + r = 0``.
    5. Form the resolvent cubic  ``y³ − (p/2)y² − ry + (4pr − q²)/8 = 0``
       and compute its discriminant.
    6. Apply Ferrari's / SymPy's method to find the four roots.

    Parameters
    ----------
    expr:
        A SymPy expression representing the LHS of the equation (= 0).
    var:
        The symbol to solve for.

    Returns
    -------
    StepList
    """
    try:
        steps: StepList = []

        # Step 1 — original
        steps.append(("Original expression", _expr_str(expr)))

        # Step 2 — expand
        expanded = sympy.expand(expr)
        steps.append(("Expand to standard form  ax⁴ + bx³ + cx² + dx + e = 0", _expr_str(expanded)))

        # Step 3 — coefficients
        poly = sympy.Poly(expanded, var)
        if poly.degree() != 4:
            return _err(
                f"Expression has degree {poly.degree()}; "
                "solve_quartic_steps requires a degree-4 polynomial."
            )
        coeffs = poly.all_coeffs()
        while len(coeffs) < 5:
            coeffs.insert(0, sympy.Integer(0))
        a, b, c, d, e = coeffs
        steps.append((
            "Identify coefficients  a, b, c, d, e",
            (
                f"a = {_expr_str(a)},  b = {_expr_str(b)},  c = {_expr_str(c)}\n"
                f"d = {_expr_str(d)},  e = {_expr_str(e)}"
            ),
        ))

        # Step 4 — depress (remove cubic term) via  x = t − b/(4a)
        shift = sympy.Rational(1, 4) * b / a
        t = sympy.Symbol("t")
        depressed_raw = sympy.expand(expr.subs(var, t - shift))
        dep_poly = sympy.Poly(depressed_raw, t)
        dep_coeffs = dep_poly.all_coeffs()
        while len(dep_coeffs) < 5:
            dep_coeffs.insert(0, sympy.Integer(0))
        _, _, p_raw, q_raw, r_raw = dep_coeffs
        p = sympy.simplify(p_raw / a)
        q = sympy.simplify(q_raw / a)
        r = sympy.simplify(r_raw / a)
        steps.append((
            f"Depress quartic via  {var} = t − b/(4a)  →  t⁴ + pt² + qt + r = 0",
            f"p = {_expr_str(p)}\nq = {_expr_str(q)}\nr = {_expr_str(r)}",
        ))

        # Step 5 — resolvent cubic  m³ − (p/2)m² − rm + (4pr − q²)/8 = 0
        res_coeff_b = sympy.Rational(-1, 2) * p
        res_coeff_c = -r
        res_coeff_d = sympy.Rational(1, 8) * (4 * p * r - q**2)
        res_b_s = sympy.simplify(res_coeff_b)
        res_c_s = sympy.simplify(res_coeff_c)
        res_d_s = sympy.simplify(res_coeff_d)
        steps.append((
            "Resolvent cubic  m³ + αm² + βm + γ = 0",
            (
                f"α = {_expr_str(res_b_s)}\n"
                f"β = {_expr_str(res_c_s)}\n"
                f"γ = {_expr_str(res_d_s)}"
            ),
        ))

        # Step 6 — roots via SymPy (Ferrari / algebraic)
        solutions = sympy.solve(expanded, var)
        if not solutions:
            sol_str = "No closed-form roots found"
        else:
            sol_str = "\n".join(f"{var} = {_expr_str(s)}" for s in solutions)
        steps.append(("Apply Ferrari's method — roots", sol_str))

        return steps

    except sympy.PolynomialError as exc:
        return _err(f"Expression is not a polynomial in {var}: {exc}")
    except Exception as exc:
        return _err(f"Could not solve quartic: {exc}")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _ordinal_superscript(n: int) -> str:
    """
    Return a simple text superscript for *n* (used in derivative labels).

    >>> _ordinal_superscript(2)
    '²'
    >>> _ordinal_superscript(5)
    '^5'
    """
    _map = {1: "¹", 2: "²", 3: "³", 4: "⁴"}
    return _map.get(n, f"^{n}")
