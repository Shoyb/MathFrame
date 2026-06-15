"""
cogs/geometry.py — Geometry and trigonometry slash commands for the math bot.

Commands
--------
/circle_area            radius                              Area of a circle:  A = π r²
/circle_circumference   radius                              Circumference of a circle:  C = 2π r
/triangle_area          base height  -or-  a b c            Area via ½·base·height or Heron's formula
/pythagorean            a b c  (exactly one omitted)         Solve for the missing side of a right triangle
/trig                   angle func unit                      Evaluate sin/cos/tan/asin/acos/atan
/distance               x1 y1 x2 y2 [z1 z2]                  Euclidean distance between two points (2D/3D)

All commands defer immediately, validate their inputs, and surface errors
through a consistent red error embed.

Wherever a result can be irrational (π, √, etc.) it is shown both as an
exact SymPy expression and as a decimal approximation, courtesy of
:func:`_exact_and_decimal`.
"""

import sympy
from discord import app_commands
from discord.ext import commands
import discord

from utils.formatter import math_embed, error_embed

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DECIMAL_DIGITS = 6   # significant figures shown in decimal approximations

_FORWARD_TRIG = {"sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan}
_INVERSE_TRIG = {"asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan}


def _exact_and_decimal(expr: sympy.Basic, digits: int = _DECIMAL_DIGITS) -> tuple[str, str]:
    """
    Render *expr* as an ``(exact, decimal)`` pair of display strings.

    Parameters
    ----------
    expr:
        Any SymPy expression (typically already simplified).
    digits:
        Number of significant figures for the decimal approximation
        (default :data:`_DECIMAL_DIGITS`).

    Returns
    -------
    tuple[str, str]
        ``(exact_str, decimal_str)``.  If numeric evaluation fails for any
        reason the decimal string falls back to ``"—"``.

    Example
    -------
    ::

        >>> _exact_and_decimal(sympy.pi * 9)
        ('9*pi', '28.2743')
    """
    exact_str = str(expr)
    try:
        decimal_str = str(sympy.N(expr, digits))
    except Exception:
        decimal_str = "—"
    return exact_str, decimal_str


def _to_exact(value: float) -> sympy.Rational:
    """
    Convert a user-supplied ``float`` slash-command argument to an exact
    :class:`sympy.Rational`.

    Going through ``str(value)`` (rather than ``sympy.Rational(value)``
    directly) avoids binary floating-point artefacts — e.g.
    ``Rational(str(0.1))`` is exactly ``1/10``, while ``Rational(0.1)``
    would produce an ugly denominator close to ``2**56``.
    """
    return sympy.Rational(str(value))


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class GeometryCog(commands.Cog, name="Geometry"):
    """Geometry and trigonometry commands: circles, triangles, trig, and distance."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /circle_area
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="circle_area",
        description="Compute the area of a circle from its radius.",
    )
    @app_commands.describe(radius="Radius of the circle (must be positive)")
    @app_commands.checks.cooldown(1, 1.0)
    async def circle_area(
        self,
        interaction: discord.Interaction,
        radius: float,
    ) -> None:
        """Compute A = π r² for *radius*, showing exact and decimal forms."""
        await interaction.response.defer()

        try:
            if radius <= 0:
                raise ValueError(f"`radius` must be positive (got {radius}).")

            r = _to_exact(radius)
            area = sympy.pi * r ** 2
            exact_str, decimal_str = _exact_and_decimal(area)

            steps = [
                ("Formula", "A = π · r²"),
                ("Substitute r", f"A = π · ({r})²"),
                ("Exact value", exact_str),
                ("Decimal approximation", decimal_str),
            ]

            embed = math_embed(
                title="Circle Area",
                result=f"{exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=f"r = {radius}",
            )

            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /circle_circumference
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="circle_circumference",
        description="Compute the circumference of a circle from its radius.",
    )
    @app_commands.describe(radius="Radius of the circle (must be positive)")
    @app_commands.checks.cooldown(1, 1.0)
    async def circle_circumference(
        self,
        interaction: discord.Interaction,
        radius: float,
    ) -> None:
        """Compute C = 2π r for *radius*, showing exact and decimal forms."""
        await interaction.response.defer()

        try:
            if radius <= 0:
                raise ValueError(f"`radius` must be positive (got {radius}).")

            r = _to_exact(radius)
            circumference = 2 * sympy.pi * r
            exact_str, decimal_str = _exact_and_decimal(circumference)

            steps = [
                ("Formula", "C = 2π · r"),
                ("Substitute r", f"C = 2π · {r}"),
                ("Exact value", exact_str),
                ("Decimal approximation", decimal_str),
            ]

            embed = math_embed(
                title="Circle Circumference",
                result=f"{exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=f"r = {radius}",
            )

            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /triangle_area
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="triangle_area",
        description="Compute the area of a triangle.",
    )
    @app_commands.describe(
        base="Base length — use together with `height`",
        height="Height length — use together with `base`",
        a="Side a — use together with `b` and `c` for Heron's formula",
        b="Side b — use together with `a` and `c`",
        c="Side c — use together with `a` and `b`",
    )
    @app_commands.checks.cooldown(1, 1.0)
    async def triangle_area(
        self,
        interaction: discord.Interaction,
        base: float = None,
        height: float = None,
        a: float = None,
        b: float = None,
        c: float = None,
    ) -> None:
        """
        Compute a triangle's area.

        Two mutually exclusive modes are supported:

        * ``base`` + ``height``  →  ``A = ½ · base · height``
        * ``a`` + ``b`` + ``c``  →  Heron's formula, with a triangle-inequality check.
        """
        await interaction.response.defer()

        try:
            bh_given  = (base is not None, height is not None)
            abc_given = (a is not None, b is not None, c is not None)

            use_bh  = any(bh_given)
            use_abc = any(abc_given)

            if use_bh and use_abc:
                raise ValueError(
                    "Provide either `base` and `height`, or `a`, `b`, and `c` — not both."
                )

            if use_bh:
                if not all(bh_given):
                    raise ValueError(
                        "Both `base` and `height` are required for this mode."
                    )
                if base <= 0 or height <= 0:
                    raise ValueError("`base` and `height` must both be positive.")

                b_val = _to_exact(base)
                h_val = _to_exact(height)
                area  = sympy.Rational(1, 2) * b_val * h_val
                exact_str, decimal_str = _exact_and_decimal(area)

                steps = [
                    ("Formula", "A = ½ · base · height"),
                    ("Substitute values", f"A = ½ · {b_val} · {h_val}"),
                    ("Exact value", exact_str),
                    ("Decimal approximation", decimal_str),
                ]
                footer    = f"base = {base}, height = {height}"

            elif use_abc:
                if not all(abc_given):
                    raise ValueError(
                        "All three sides `a`, `b`, and `c` are required for Heron's formula."
                    )
                if a <= 0 or b <= 0 or c <= 0:
                    raise ValueError("All side lengths must be positive.")
                if a + b <= c or a + c <= b or b + c <= a:
                    raise ValueError(
                        f"Sides {a}, {b}, {c} do not satisfy the triangle inequality "
                        "— no such triangle exists."
                    )

                a_val, b_val, c_val = _to_exact(a), _to_exact(b), _to_exact(c)
                s    = (a_val + b_val + c_val) / 2
                area = sympy.sqrt(s * (s - a_val) * (s - b_val) * (s - c_val))

                exact_str, decimal_str = _exact_and_decimal(area)
                s_exact, s_decimal     = _exact_and_decimal(s)

                steps = [
                    ("Formula (Heron's)", "A = √( s·(s−a)·(s−b)·(s−c) ),   s = (a + b + c) / 2"),
                    (
                        "Compute semi-perimeter s",
                        f"s = ({a_val} + {b_val} + {c_val}) / 2 = {s_exact}  ≈ {s_decimal}",
                    ),
                    (
                        "Substitute into Heron's formula",
                        f"A = √( {s_exact}·({s_exact}−{a_val})·({s_exact}−{b_val})·({s_exact}−{c_val}) )",
                    ),
                    ("Exact value", exact_str),
                    ("Decimal approximation", decimal_str),
                ]
                footer    = f"a = {a}, b = {b}, c = {c}"

            else:
                raise ValueError(
                    "Provide either `base` and `height`, or all three sides "
                    "`a`, `b`, and `c`."
                )

            embed = math_embed(
                title="Triangle Area",
                result=f"{exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=footer,
            )

            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /pythagorean
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="pythagorean",
        description="Solve for the missing side of a right triangle (a² + b² = c²).",
    )
    @app_commands.describe(
        a="Leg a — omit this argument to solve for it",
        b="Leg b — omit this argument to solve for it",
        c="Hypotenuse c — omit this argument to solve for it",
    )
    @app_commands.checks.cooldown(1, 1.0)
    async def pythagorean(
        self,
        interaction: discord.Interaction,
        a: float = None,
        b: float = None,
        c: float = None,
    ) -> None:
        """Solve ``a² + b² = c²`` for whichever of ``a``, ``b``, ``c`` was omitted."""
        await interaction.response.defer()

        try:
            provided = {"a": a, "b": b, "c": c}
            missing  = [name for name, val in provided.items() if val is None]

            if len(missing) != 1:
                raise ValueError(
                    "Exactly one of `a`, `b`, or `c` must be omitted "
                    f"(got {3 - len(missing)} value(s) provided)."
                )

            for name, val in provided.items():
                if val is not None and val <= 0:
                    raise ValueError(f"`{name}` must be positive (got {val}).")

            target = missing[0]

            if target == "c":
                a_val, b_val = _to_exact(a), _to_exact(b)
                result       = sympy.sqrt(a_val ** 2 + b_val ** 2)
                formula      = "c = √(a² + b²)"
                substitution = f"c = √( {a_val}² + {b_val}² )"

            elif target == "b":
                a_val, c_val = _to_exact(a), _to_exact(c)
                if c_val <= a_val:
                    raise ValueError(
                        f"`c` ({c}) must be greater than `a` ({a}) "
                        "for a valid right triangle."
                    )
                result       = sympy.sqrt(c_val ** 2 - a_val ** 2)
                formula      = "b = √(c² − a²)"
                substitution = f"b = √( {c_val}² − {a_val}² )"

            else:  # target == "a"
                b_val, c_val = _to_exact(b), _to_exact(c)
                if c_val <= b_val:
                    raise ValueError(
                        f"`c` ({c}) must be greater than `b` ({b}) "
                        "for a valid right triangle."
                    )
                result       = sympy.sqrt(c_val ** 2 - b_val ** 2)
                formula      = "a = √(c² − b²)"
                substitution = f"a = √( {c_val}² − {b_val}² )"

            exact_str, decimal_str = _exact_and_decimal(result)

            steps = [
                ("Formula", formula),
                ("Substitute known values", substitution),
                ("Exact value", exact_str),
                ("Decimal approximation", decimal_str),
            ]

            known = ", ".join(
                f"{name} = {val}" for name, val in provided.items() if val is not None
            )

            embed = math_embed(
                title=f"Pythagorean Theorem — solve for {target}",
                result=f"{target} = {exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=f"Given {known}",
            )

            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /trig
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="trig",
        description="Evaluate a trigonometric function with exact and decimal results.",
    )
    @app_commands.describe(
        angle="Angle (for sin/cos/tan) or ratio in [-1, 1] (for asin/acos/atan)",
        func="Trigonometric function to evaluate",
        unit="Angle unit: degrees (default) or radians",
    )
    @app_commands.choices(
        func=[
            app_commands.Choice(name="sin",  value="sin"),
            app_commands.Choice(name="cos",  value="cos"),
            app_commands.Choice(name="tan",  value="tan"),
            app_commands.Choice(name="asin", value="asin"),
            app_commands.Choice(name="acos", value="acos"),
            app_commands.Choice(name="atan", value="atan"),
        ],
        unit=[
            app_commands.Choice(name="Degrees (deg)", value="deg"),
            app_commands.Choice(name="Radians (rad)", value="rad"),
        ],
    )
    @app_commands.checks.cooldown(1, 1.0)
    async def trig(
        self,
        interaction: discord.Interaction,
        angle: float,
        func: app_commands.Choice[str],
        unit: app_commands.Choice[str] = None,
    ) -> None:
        """
        Evaluate a trig function, preferring exact SymPy values.

        For ``sin``/``cos``/``tan``, *angle* is interpreted as an angle in
        *unit* and the result is a dimensionless ratio.

        For ``asin``/``acos``/``atan``, *angle* is interpreted as a ratio
        and the result is an angle, displayed in *unit*.
        """
        await interaction.response.defer()

        try:
            func_name = func.value
            unit_name = unit.value if unit is not None else "deg"

            value = _to_exact(angle)

            # -----------------------------------------------------------
            # Forward functions: sin / cos / tan
            # -----------------------------------------------------------
            if func_name in _FORWARD_TRIG:
                sym_func = _FORWARD_TRIG[func_name]

                if unit_name == "deg":
                    arg       = sympy.rad(value)
                    arg_label = f"{angle}°"
                else:
                    arg       = value
                    arg_label = f"{angle} rad"

                arg_simplified = sympy.simplify(arg)
                result = sympy.simplify(sym_func(arg_simplified))

                if result == sympy.zoo or not result.is_finite:
                    raise ValueError(
                        f"`{func_name}({arg_label})` is undefined (division by zero)."
                    )

                exact_str, decimal_str = _exact_and_decimal(result)

                steps = []
                if unit_name == "deg":
                    steps.append(("Convert to radians", f"{arg_label}  →  {arg_simplified} rad"))
                steps.append((
                    "Evaluate",
                    f"{func_name}({arg_simplified}) = {exact_str}",
                ))

                title  = f"{func_name}({arg_label})"
                footer = f"Exact value: {exact_str}"

            # -----------------------------------------------------------
            # Inverse functions: asin / acos / atan
            # -----------------------------------------------------------
            else:
                sym_func = _INVERSE_TRIG[func_name]

                if func_name in ("asin", "acos") and abs(value) > 1:
                    raise ValueError(
                        f"`{func_name}` requires an input in [-1, 1] (got {angle})."
                    )

                rad_result = sympy.simplify(sym_func(value))

                if not rad_result.is_real:
                    raise ValueError(f"`{func_name}({angle})` has no real solution.")

                if unit_name == "deg":
                    display_result = sympy.simplify(sympy.deg(rad_result))
                    unit_label = "°"
                else:
                    display_result = rad_result
                    unit_label = " rad"

                exact_str, decimal_str = _exact_and_decimal(display_result)

                steps = [
                    ("Evaluate (radians)", f"{func_name}({value}) = {rad_result} rad"),
                ]
                if unit_name == "deg":
                    steps.append(("Convert to degrees", f"{rad_result} rad  →  {exact_str}°"))

                exact_str   = f"{exact_str}{unit_label}"
                decimal_str = f"{decimal_str}{unit_label}"

                title  = f"{func_name}({angle})"
                footer = f"Result shown in {'degrees' if unit_name == 'deg' else 'radians'}"

            embed = math_embed(
                title=title,
                result=f"{exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=footer,
            )

            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /distance
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="distance",
        description="Compute the Euclidean distance between two points (2D or 3D).",
    )
    @app_commands.describe(
        x1="x-coordinate of the first point",
        y1="y-coordinate of the first point",
        x2="x-coordinate of the second point",
        y2="y-coordinate of the second point",
        z1="z-coordinate of the first point (omit for 2D)",
        z2="z-coordinate of the second point (omit for 2D)",
    )
    @app_commands.checks.cooldown(1, 1.0)
    async def distance(
        self,
        interaction: discord.Interaction,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        z1: float = None,
        z2: float = None,
    ) -> None:
        """
        Compute the distance between two points.

        Automatically uses the 2D formula ``√((x₂−x₁)² + (y₂−y₁)²)`` unless
        both ``z1`` and ``z2`` are supplied, in which case the 3D formula
        ``√((x₂−x₁)² + (y₂−y₁)² + (z₂−z₁)²)`` is used.
        """
        await interaction.response.defer()

        try:
            if (z1 is None) != (z2 is None):
                raise ValueError(
                    "For a 3D distance, provide both `z1` and `z2`. "
                    "For a 2D distance, omit both."
                )

            is_3d = z1 is not None

            x1v, y1v, x2v, y2v = (_to_exact(v) for v in (x1, y1, x2, y2))
            dx = x2v - x1v
            dy = y2v - y1v

            if is_3d:
                z1v, z2v = _to_exact(z1), _to_exact(z2)
                dz = z2v - z1v

                sum_sq       = dx ** 2 + dy ** 2 + dz ** 2
                formula      = "d = √( (x₂−x₁)² + (y₂−y₁)² + (z₂−z₁)² )"
                substitution = f"d = √( ({dx})² + ({dy})² + ({dz})² )"
                point_a, point_b = f"({x1}, {y1}, {z1})", f"({x2}, {y2}, {z2})"
            else:
                sum_sq       = dx ** 2 + dy ** 2
                formula      = "d = √( (x₂−x₁)² + (y₂−y₁)² )"
                substitution = f"d = √( ({dx})² + ({dy})² )"
                point_a, point_b = f"({x1}, {y1})", f"({x2}, {y2})"

            simplified_sq = sympy.expand(sum_sq)
            result        = sympy.sqrt(simplified_sq)
            exact_str, decimal_str = _exact_and_decimal(result)

            steps = [
                ("Formula", formula),
                ("Substitute coordinates", substitution),
                ("Simplify under the root", f"d = √({simplified_sq})"),
                ("Exact value", exact_str),
                ("Decimal approximation", decimal_str),
            ]

            embed = math_embed(
                title="Distance" + (" (3D)" if is_3d else " (2D)"),
                result=f"{exact_str}   ≈ {decimal_str}",
                steps=steps,
                footer=f"Between {point_a} and {point_b}",
            )

            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the GeometryCog into *bot*."""
    await bot.add_cog(GeometryCog(bot))