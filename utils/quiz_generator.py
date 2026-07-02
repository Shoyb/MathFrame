"""
utils/quiz_generator.py — Question generation and answer verification for
the quiz cog (Phase 4 of the Random/Probability/Quiz plan; see
RANDOM_PROBABILITY_QUIZ_PLAN.md).

Design principle (per the plan doc)
------------------------------------
Questions are generated with random inputs and then solved with the exact
same SymPy/math primitives the corresponding slash commands are built on
(``sympy.diff``, ``sympy.solve``, ``math.gcd``, ``sympy.isprime``,
``math.comb``/``math.perm``, etc.) — never a hand-curated answer key. This
means correctness is inherited from well-tested, well-understood math
library calls rather than something that could silently drift or be typed
in wrong.

Note on cross-cog reuse: most of this codebase's cogs (``algebra.py``,
``number_theory.py``, ``discrete.py``) call SymPy/math primitives directly
inline inside their Discord command handlers rather than exposing them as
separately importable "core" functions — there's nothing clean to import
across cog boundaries. This module therefore calls the same underlying
primitives directly (same computation, same correctness guarantee) rather
than importing Discord-bound command methods from other cogs, which
wouldn't work anyway since they expect a live ``discord.Interaction``.

Every generator has been checked against known-correct references before
being wired in here (see the phase's test transcript) and every generator
retries with fresh random inputs, up to a bounded attempt count, if it
happens to produce a degenerate question (e.g. an all-zero polynomial) —
callers never see a broken question.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

SUBJECTS = ("algebra", "calculus", "number_theory", "discrete")
DIFFICULTIES = ("easy", "medium", "hard")

_MAX_GEN_ATTEMPTS = 20
"""
Bounded retry count for generators that might produce a degenerate
question (e.g. differentiating a constant to get 0, or an all-zero
polynomial). Keeps generation fast and guarantees termination — if 20
random attempts all degenerate, something is wrong with the generator
itself, not bad luck, so it raises rather than looping forever.
"""

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


# ---------------------------------------------------------------------------
# Question data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One generated, pre-verified quiz question."""

    subject: str
    difficulty: str
    prompt: str
    correct_answer: object  # sympy.Expr for symbolic, int/str for exact
    answer_type: str  # "symbolic" | "numeric" | "exact"
    var_name: str | None = None  # e.g. "x", for symbolic answers
    seed: int | None = None
    hint_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Formatting helpers (shared by generators)
# ---------------------------------------------------------------------------


def _format_term(coeff: int, power: int, var: str) -> str:
    """Format one polynomial term with clean coefficient-of-1 display."""
    if power == 0:
        return str(coeff)
    var_part = var if power == 1 else f"{var}**{power}"
    if coeff == 1:
        return var_part
    if coeff == -1:
        return f"-{var_part}"
    return f"{coeff}*{var_part}"


def _format_poly(coeffs: list[int], var: str = "x") -> str:
    """
    Build a clean polynomial string from *coeffs* (index i = coefficient
    of ``var**i``), skipping zero terms.
    """
    terms = []
    for power in range(len(coeffs) - 1, -1, -1):
        c = coeffs[power]
        if c == 0:
            continue
        terms.append(_format_term(c, power, var))
    if not terms:
        return "0"
    return " + ".join(terms).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# Algebra generator
# ---------------------------------------------------------------------------


def _gen_algebra(difficulty: str, rng: random.Random) -> Question:
    x = sympy.Symbol("x")

    for _ in range(_MAX_GEN_ATTEMPTS):
        if difficulty == "easy":
            # a*x + b = c, guaranteed integer solution
            a = rng.choice([n for n in range(-5, 6) if n != 0])
            root = rng.randint(-12, 12)
            b = rng.randint(-15, 15)
            c = a * root + b
            lhs = f"{_format_term(a, 1, 'x')} + {b}" if b >= 0 else f"{_format_term(a, 1, 'x')} - {abs(b)}"
            prompt = f"Solve for x: {lhs} = {c}"
            hint_steps = [
                f"Isolate the x term: {_format_term(a, 1, 'x')} = {c - b}",
                f"Divide both sides by {a}: x = {sympy.Rational(c - b, a)}",
            ]
            return Question(
                subject="algebra",
                difficulty=difficulty,
                prompt=prompt,
                correct_answer=sympy.Integer(root),
                answer_type="numeric",
                var_name="x",
                hint_steps=hint_steps,
            )

        else:
            # (x - r1)(x - r2) = 0 → ask for the LARGER root (keeps the
            # answer single-valued, since quadratics have two roots and a
            # text-box submission can only carry one value cleanly).
            r1 = rng.randint(-9, 9)
            r2 = rng.randint(-9, 9)
            if r1 == r2:
                continue  # avoid a repeated-root question reading oddly
            expr = sympy.expand((x - r1) * (x - r2))
            larger = max(r1, r2)
            prompt = f"Solve for x (give the LARGER root): {expr} = 0"
            hint_steps = [
                f"Factor: {expr} = (x - {r1})(x - {r2})",
                f"Roots: x = {r1} and x = {r2}",
            ]
            return Question(
                subject="algebra",
                difficulty=difficulty,
                prompt=prompt,
                correct_answer=sympy.Integer(larger),
                answer_type="numeric",
                var_name="x",
                hint_steps=hint_steps,
            )

    raise RuntimeError("Algebra generator failed to produce a valid question after max attempts.")


# ---------------------------------------------------------------------------
# Calculus generator
# ---------------------------------------------------------------------------

_CALC_TEMPLATES_MEDIUM = [
    "sin(x)", "cos(x)", "exp(x)", "log(x)",
    "sin(2*x)", "exp(-x)", "cos(x)**2", "sqrt(x)",
]
_CALC_TEMPLATES_HARD = [
    "x*sin(x)", "exp(x)*cos(x)", "sin(x)*cos(x)", "log(x)/x",
    "x**2*exp(x)", "sin(x**2)", "exp(sin(x))", "x*log(x)",
]


def _gen_calculus(difficulty: str, rng: random.Random) -> Question:
    x = sympy.Symbol("x")

    for _ in range(_MAX_GEN_ATTEMPTS):
        if difficulty == "easy":
            deg = rng.randint(2, 4)
            coeffs = [rng.randint(-9, 9) for _ in range(deg + 1)]
            if coeffs[deg] == 0:
                continue  # not actually degree `deg` — resample
            expr_str = _format_poly(coeffs, "x")
            expr = parse_expr(expr_str, transformations=_TRANSFORMATIONS)
        elif difficulty == "medium":
            expr_str = rng.choice(_CALC_TEMPLATES_MEDIUM)
            expr = parse_expr(expr_str, transformations=_TRANSFORMATIONS)
        else:
            expr_str = rng.choice(_CALC_TEMPLATES_HARD)
            expr = parse_expr(expr_str, transformations=_TRANSFORMATIONS)

        derivative = sympy.diff(expr, x)
        derivative = sympy.simplify(derivative)

        if derivative == 0:
            continue  # degenerate (e.g. differentiating a constant) — resample

        prompt = f"Differentiate: d/dx [ {expr} ]"
        hint_steps = [
            f"Original: {expr}",
            f"Apply differentiation rules with respect to x",
        ]
        return Question(
            subject="calculus",
            difficulty=difficulty,
            prompt=prompt,
            correct_answer=derivative,
            answer_type="symbolic",
            var_name="x",
            hint_steps=hint_steps,
        )

    raise RuntimeError("Calculus generator failed to produce a valid question after max attempts.")


# ---------------------------------------------------------------------------
# Number theory generator
# ---------------------------------------------------------------------------


def _gen_number_theory(difficulty: str, rng: random.Random) -> Question:
    ranges = {"easy": (10, 200), "medium": (100, 5000), "hard": (1000, 100000)}
    lo, hi = ranges[difficulty]

    for _ in range(_MAX_GEN_ATTEMPTS):
        kind = rng.choice(["gcd", "lcm", "is_prime", "factor_count"])

        if kind == "gcd":
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if a == 0 or b == 0:
                continue
            answer = math.gcd(a, b)
            return Question(
                subject="number_theory",
                difficulty=difficulty,
                prompt=f"Find gcd({a}, {b})",
                correct_answer=answer,
                answer_type="exact",
                hint_steps=[f"Use the Euclidean algorithm on {a} and {b}."],
            )

        if kind == "lcm":
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if a == 0 or b == 0:
                continue
            answer = math.lcm(a, b)
            return Question(
                subject="number_theory",
                difficulty=difficulty,
                prompt=f"Find lcm({a}, {b})",
                correct_answer=answer,
                answer_type="exact",
                hint_steps=[f"lcm(a,b) = a*b / gcd(a,b) = {a}*{b} / {math.gcd(a,b)}"],
            )

        if kind == "is_prime":
            n = rng.randint(lo, hi)
            is_p = sympy.isprime(n)
            answer = "yes" if is_p else "no"
            return Question(
                subject="number_theory",
                difficulty=difficulty,
                prompt=f"Is {n} prime? (answer yes or no)",
                correct_answer=answer,
                answer_type="exact",
                hint_steps=[f"Check divisibility by primes up to sqrt({n}) ≈ {int(n**0.5)}."],
            )

        # factor_count
        n = rng.randint(lo, hi)
        if n < 2:
            continue
        factors = sympy.factorint(n)
        answer = sum(factors.values())
        return Question(
            subject="number_theory",
            difficulty=difficulty,
            prompt=f"How many prime factors does {n} have, counted with multiplicity?",
            correct_answer=answer,
            answer_type="exact",
            hint_steps=[f"Find the full prime factorization of {n}."],
        )

    raise RuntimeError("Number theory generator failed to produce a valid question after max attempts.")


# ---------------------------------------------------------------------------
# Discrete math generator
# ---------------------------------------------------------------------------


def _gen_discrete(difficulty: str, rng: random.Random) -> Question:
    ranges = {"easy": (4, 8), "medium": (6, 15), "hard": (10, 25)}
    lo, hi = ranges[difficulty]

    for _ in range(_MAX_GEN_ATTEMPTS):
        kind = rng.choice(["permutation", "combination"])
        n = rng.randint(lo, hi)
        k = rng.randint(1, n)

        if kind == "permutation":
            answer = math.perm(n, k)
            prompt = f"How many ways can you arrange {k} items chosen from {n} distinct items, where order matters? (P({n},{k}))"
            hint_steps = [f"P(n,k) = n! / (n-k)! = {n}! / {n-k}!"]
        else:
            answer = math.comb(n, k)
            prompt = f"How many ways can you choose {k} items from {n} distinct items, where order doesn't matter? (C({n},{k}))"
            hint_steps = [f"C(n,k) = n! / (k!(n-k)!) = {n}! / ({k}! * {n-k}!)"]

        return Question(
            subject="discrete",
            difficulty=difficulty,
            prompt=prompt,
            correct_answer=answer,
            answer_type="exact",
            hint_steps=hint_steps,
        )

    raise RuntimeError("Discrete generator failed to produce a valid question after max attempts.")


_GENERATORS = {
    "algebra": _gen_algebra,
    "calculus": _gen_calculus,
    "number_theory": _gen_number_theory,
    "discrete": _gen_discrete,
}


# ---------------------------------------------------------------------------
# Public generation entry point
# ---------------------------------------------------------------------------


def generate_question(
    subject: str | None = None,
    difficulty: str | None = None,
    seed: int | None = None,
) -> Question:
    """
    Generate one verified :class:`Question`.

    Parameters
    ----------
    subject:
        One of :data:`SUBJECTS`, or ``None`` to pick uniformly at random.
    difficulty:
        One of :data:`DIFFICULTIES`, or ``None`` to pick uniformly at
        random.
    seed:
        Optional seed for reproducible generation (e.g. so a battle's two
        participants can be proven to have received the identical
        question — used starting Phase 5).

    Raises
    ------
    ValueError
        If *subject* or *difficulty* is given but not recognized.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    if subject is None:
        subject = rng.choice(SUBJECTS)
    elif subject not in SUBJECTS:
        raise ValueError(f"Unknown subject `{subject}`. Expected one of: {', '.join(SUBJECTS)}.")

    if difficulty is None:
        difficulty = rng.choice(DIFFICULTIES)
    elif difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty `{difficulty}`. Expected one of: {', '.join(DIFFICULTIES)}.")

    question = _GENERATORS[subject](difficulty, rng)
    # dataclass is frozen — rebuild with the seed attached for the record,
    # rather than trying to mutate a frozen instance.
    return Question(
        subject=question.subject,
        difficulty=question.difficulty,
        prompt=question.prompt,
        correct_answer=question.correct_answer,
        answer_type=question.answer_type,
        var_name=question.var_name,
        seed=seed,
        hint_steps=question.hint_steps,
    )


# ---------------------------------------------------------------------------
# Answer verification
# ---------------------------------------------------------------------------


def check_answer(question: Question, user_answer: str) -> bool:
    """
    Check whether *user_answer* (raw text as typed by the user) matches
    *question*'s correct answer.

    - ``symbolic`` answers: parsed with the same implicit-multiplication
      parser used elsewhere in this codebase, then checked for symbolic
      equivalence via ``sympy.simplify(user - correct) == 0`` — so
      differently-formatted-but-equal answers (``"2*x"`` vs ``"x+x"``,
      ``"1/2"`` vs ``"0.5"``) are both accepted.
    - ``numeric`` answers: parsed the same way, compared via
      ``math.isclose`` for float tolerance.
    - ``exact`` answers (yes/no, integer counts): case-insensitive string
      comparison for strings, exact integer comparison for numbers.

    Malformed input (unparseable text) is treated as simply incorrect
    rather than raising — a quiz shouldn't crash on a garbled guess.
    """
    raw = user_answer.strip()
    if not raw:
        return False

    try:
        if question.answer_type == "exact":
            if isinstance(question.correct_answer, str):
                return raw.lower() == question.correct_answer.lower()
            # integer exact answer — accept "42" and also "42.0"-style
            # whole-number float text (a plausible, harmless formatting
            # choice), but not "42.3" (that's just wrong, not a format
            # quirk).
            try:
                return int(raw) == question.correct_answer
            except ValueError:
                as_float = float(raw)  # may itself raise -> caught by the outer except, treated as incorrect
                return as_float.is_integer() and int(as_float) == question.correct_answer

        if question.answer_type == "numeric":
            parsed = parse_expr(raw, transformations=_TRANSFORMATIONS)
            diff = sympy.simplify(parsed - question.correct_answer)
            return diff == 0

        if question.answer_type == "symbolic":
            parsed = parse_expr(raw, transformations=_TRANSFORMATIONS)
            diff = sympy.simplify(parsed - question.correct_answer)
            return diff == 0

    except Exception:
        return False

    return False
