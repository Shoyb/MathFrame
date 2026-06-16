"""
cogs/linear_algebra.py — Linear algebra slash commands for the math bot.

Commands
--------
/matrix_det   matrix                   Determinant of a square matrix.
/matrix_inv   matrix                   Inverse of a square, non-singular matrix.
/eigenvalues  matrix                   Eigenvalues (with multiplicity) of a square matrix.
/dot          vec_a  vec_b             Dot product of two vectors.
/cross        vec_a  vec_b             Cross product of two 3-D vectors.
/rref         matrix                   Reduced row-echelon form of a matrix.

Matrix / vector input format
----------------------------
Matrices are supplied as JSON 2-D arrays: ``[[1,2],[3,4]]``.
Vectors are supplied as JSON 1-D arrays: ``[1,2,3]``.
"""

import json

import numpy as np
import sympy
from discord import app_commands
from discord.ext import commands
import discord

from utils.formatter import math_embed, error_embed

# ---------------------------------------------------------------------------
# Module-level formatting helpers
# ---------------------------------------------------------------------------

def _format_matrix(mat: sympy.Matrix) -> str:
    """
    Render *mat* as a plain-text grid of rows, one per line.

    Each entry is converted with ``str()`` so exact fractions and symbolic
    values are preserved rather than being rounded.

    Example output for a 2×2 matrix::

        [ 1   0 ]
        [ 0   1 ]
    """
    rows = []
    for i in range(mat.rows):
        entries = "   ".join(str(mat[i, j]) for j in range(mat.cols))
        rows.append(f"[ {entries} ]")
    return "\n".join(rows)


def _parse_json_list(raw: str, name: str = "input") -> list:
    """
    Parse *raw* as a JSON array and return the resulting Python list.

    Parameters
    ----------
    raw:
        Raw user-supplied string.
    name:
        Label used in error messages (e.g. ``"vec_a"``).

    Raises
    ------
    ValueError
        If *raw* is not valid JSON or not a list.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"`{name}` is not valid JSON: {exc}\n"
            "Use bracket notation, e.g. `[1, 2, 3]` or `[[1,2],[3,4]]`."
        ) from exc
    if not isinstance(data, list):
        raise ValueError(f"`{name}` must be a JSON array, got {type(data).__name__}.")
    return data


def _parse_vector(raw: str, name: str = "vector") -> list:
    """
    Parse *raw* as a 1-D JSON array of numbers.

    Raises
    ------
    ValueError
        If not a flat list, or if any element is not a real number.
    """
    data = _parse_json_list(raw, name)
    if data and isinstance(data[0], list):
        raise ValueError(
            f"`{name}` must be a 1-D vector array like `[1, 2, 3]`, "
            "not a nested 2-D array."
        )
    for i, v in enumerate(data):
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"`{name}[{i}]` is not a number (got {type(v).__name__!r})."
            )
    return data


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class LinearAlgebraCog(commands.Cog, name="Linear Algebra"):
    """Matrix and vector commands: determinant, inverse, eigenvalues, dot, cross, RREF."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # Shared matrix parser (instance method so subclasses can override)
    # -----------------------------------------------------------------------

    def parse_matrix(self, s: str) -> sympy.Matrix:
        """
        Parse a JSON 2-D array string into a :class:`sympy.Matrix`.

        Parameters
        ----------
        s:
            User-supplied string, e.g. ``"[[1,2],[3,4]]"``.

        Returns
        -------
        sympy.Matrix

        Raises
        ------
        ValueError
            If the string is not valid JSON, not a 2-D list, not rectangular,
            or contains non-numeric entries.
        """
        data = _parse_json_list(s, "matrix")

        # Must be a list of lists (2-D)
        if not data:
            raise ValueError("Matrix cannot be empty.")
        if not isinstance(data[0], list):
            raise ValueError(
                "Matrix must be a 2-D array like `[[1,2],[3,4]]`. "
                "For a vector, use /dot or /cross instead."
            )

        # Rectangular check
        row_len = len(data[0])
        for i, row in enumerate(data):
            if not isinstance(row, list):
                raise ValueError(f"Row {i} is not a list.")
            if len(row) != row_len:
                raise ValueError(
                    f"Matrix is not rectangular: row 0 has {row_len} columns "
                    f"but row {i} has {len(row)}."
                )
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Entry [{i}][{j}] is not a number (got {type(val).__name__!r})."
                    )

        try:
            return sympy.Matrix(data)
        except Exception as exc:
            raise ValueError(f"Could not build matrix: {exc}") from exc

    # -----------------------------------------------------------------------
    # /matrix_det
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="matrix_det",
        description="Compute the determinant of a square matrix.",
    )
    @app_commands.describe(matrix='Square matrix as a JSON 2-D array, e.g. [[1,2],[3,4]]')
    @app_commands.checks.cooldown(1, 3.0)
    async def matrix_det(
        self,
        interaction: discord.Interaction,
        matrix: str,
    ) -> None:
        await interaction.response.defer()
        try:
            mat = self.parse_matrix(matrix)
            if mat.rows != mat.cols:
                raise ValueError(
                    f"Determinant requires a square matrix "
                    f"(got {mat.rows}×{mat.cols})."
                )
            det = mat.det()
            embed = math_embed(
                title="Matrix Determinant",
                result=str(det),
                footer=f"det(A)  |  {mat.rows}×{mat.cols} matrix",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /matrix_inv
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="matrix_inv",
        description="Compute the inverse of a square, non-singular matrix.",
    )
    @app_commands.describe(matrix='Square matrix as a JSON 2-D array, e.g. [[1,2],[3,4]]')
    @app_commands.checks.cooldown(1, 3.0)
    async def matrix_inv(
        self,
        interaction: discord.Interaction,
        matrix: str,
    ) -> None:
        await interaction.response.defer()
        try:
            mat = self.parse_matrix(matrix)
            if mat.rows != mat.cols:
                raise ValueError(
                    f"Inverse requires a square matrix (got {mat.rows}×{mat.cols})."
                )
            if mat.det() == 0:
                await interaction.followup.send(
                    embed=error_embed(
                        "Matrix is singular (det = 0) — the inverse does not exist."
                    )
                )
                return

            inv = mat.inv()
            embed = math_embed(
                title="Matrix Inverse",
                result=_format_matrix(inv),
                footer=f"A⁻¹  |  {mat.rows}×{mat.cols} matrix  |  exact fractions preserved",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /eigenvalues
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="eigenvalues",
        description="Find the eigenvalues (with multiplicity) of a square matrix.",
    )
    @app_commands.describe(matrix='Square matrix as a JSON 2-D array, e.g. [[1,2],[3,4]]')
    @app_commands.checks.cooldown(1, 3.0)
    async def eigenvalues(
        self,
        interaction: discord.Interaction,
        matrix: str,
    ) -> None:
        await interaction.response.defer()
        try:
            mat = self.parse_matrix(matrix)
            if mat.rows != mat.cols:
                raise ValueError(
                    f"Eigenvalues require a square matrix (got {mat.rows}×{mat.cols})."
                )

            # .eigenvals() returns {eigenvalue: algebraic_multiplicity, ...}
            eigs = mat.eigenvals()
            if not eigs:
                raise ValueError("No eigenvalues could be computed for this matrix.")

            lines = [
                f"λ = {val}   (multiplicity {mult})"
                for val, mult in sorted(eigs.items(), key=lambda kv: sympy.re(kv[0]))
            ]
            result_str = "\n".join(lines)

            embed = math_embed(
                title="Eigenvalues",
                result=result_str,
                footer=f"{mat.rows}×{mat.cols} matrix  |  algebraic multiplicities shown",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /dot
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="dot",
        description="Compute the dot product of two vectors.",
    )
    @app_commands.describe(
        vec_a="First vector as a JSON array, e.g. [1,2,3]",
        vec_b="Second vector as a JSON array, e.g. [4,5,6]",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def dot(
        self,
        interaction: discord.Interaction,
        vec_a: str,
        vec_b: str,
    ) -> None:
        await interaction.response.defer()
        try:
            a = _parse_vector(vec_a, "vec_a")
            b = _parse_vector(vec_b, "vec_b")
            if len(a) != len(b):
                raise ValueError(
                    f"Vectors must have the same length "
                    f"(vec_a has {len(a)} elements, vec_b has {len(b)})."
                )
            result = float(np.dot(a, b))
            # Format as int if the result is a whole number
            result_str = str(int(result)) if result == int(result) else str(result)

            embed = math_embed(
                title="Dot Product",
                result=result_str,
                footer=f"a · b  |  {len(a)}-dimensional vectors",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /cross
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="cross",
        description="Compute the cross product of two 3-D vectors.",
    )
    @app_commands.describe(
        vec_a="First 3-D vector as a JSON array, e.g. [1,0,0]",
        vec_b="Second 3-D vector as a JSON array, e.g. [0,1,0]",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def cross(
        self,
        interaction: discord.Interaction,
        vec_a: str,
        vec_b: str,
    ) -> None:
        await interaction.response.defer()
        try:
            a = _parse_vector(vec_a, "vec_a")
            b = _parse_vector(vec_b, "vec_b")
            if len(a) != 3:
                raise ValueError(
                    f"Cross product requires 3-D vectors (vec_a has {len(a)} elements)."
                )
            if len(b) != 3:
                raise ValueError(
                    f"Cross product requires 3-D vectors (vec_b has {len(b)} elements)."
                )

            result = np.cross(a, b).tolist()
            # Format each component; drop .0 for whole numbers
            def _fmt(v: float) -> str:
                return str(int(v)) if v == int(v) else str(v)

            result_str = f"[ {',   '.join(_fmt(v) for v in result)} ]"

            embed = math_embed(
                title="Cross Product",
                result=result_str,
                footer="a × b  |  3-D vectors",
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))

    # -----------------------------------------------------------------------
    # /rref
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="rref",
        description="Compute the reduced row-echelon form (RREF) of a matrix.",
    )
    @app_commands.describe(matrix='Matrix as a JSON 2-D array, e.g. [[1,2,3],[4,5,6]]')
    @app_commands.checks.cooldown(1, 3.0)
    async def rref(
        self,
        interaction: discord.Interaction,
        matrix: str,
    ) -> None:
        await interaction.response.defer()
        try:
            mat = self.parse_matrix(matrix)

            # .rref() returns (reduced_matrix, tuple_of_pivot_column_indices)
            rref_mat, pivot_cols = mat.rref()

            pivot_str = (
                ", ".join(f"col {c}" for c in pivot_cols)
                if pivot_cols
                else "none (zero matrix)"
            )
            rank = len(pivot_cols)

            embed = math_embed(
                title="Reduced Row-Echelon Form (RREF)",
                result=_format_matrix(rref_mat),
                footer=(
                    f"Pivot columns: {pivot_str}  |  "
                    f"Rank: {rank}  |  "
                    f"Original: {mat.rows}×{mat.cols}"
                ),
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the LinearAlgebraCog into *bot*."""
    await bot.add_cog(LinearAlgebraCog(bot))
