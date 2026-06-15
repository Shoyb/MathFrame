"""
utils/renderer.py — LaTeX → PNG renderer for the math bot.

Converts LaTeX strings (or SymPy expressions) into PNG images and returns
them as ``discord.File`` objects ready to attach to any interaction reply.

Both public functions are async and safe to call concurrently: the
blocking matplotlib work runs inside a thread-pool executor so it never
stalls the event loop.

Usage
-----
::

    from utils.renderer import expr_to_image, result_to_image

    # From a raw LaTeX string:
    file = await expr_to_image(r"\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}")
    await interaction.followup.send(file=file)

    # From a SymPy expression:
    import sympy
    expr = sympy.parse_expr("x**2 + 2*x + 1")
    file = await result_to_image(expr)
    await interaction.followup.send(file=file)
"""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import discord
import matplotlib
import sympy

matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)

# ---------------------------------------------------------------------------
# Module-level executor
# ---------------------------------------------------------------------------

# Separate from the parser's executor so that rendering and parsing can
# proceed concurrently without one pool starving the other.
_render_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="renderer")

# ---------------------------------------------------------------------------
# Internal blocking implementation
# ---------------------------------------------------------------------------

def _render_to_bytes(latex_str: str) -> io.BytesIO:
    """
    Render *latex_str* to a PNG and return the raw bytes in a BytesIO buffer.

    This function is **synchronous** and designed to run inside an executor.
    It must not be called directly from async code.

    Parameters
    ----------
    latex_str:
        A LaTeX expression string, *without* surrounding ``$`` delimiters —
        those are added here so matplotlib treats the text as math mode.

    Returns
    -------
    io.BytesIO
        Seeked to position 0, ready for reading or passing to discord.File.
    """
    fig, ax = plt.subplots(figsize=(6, 1.2))

    # White background on both the figure and the axes patch.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.text(
        0.5, 0.5,
        f"${latex_str}$",
        fontsize=22,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=150,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)   # release figure memory immediately

    buf.seek(0)
    return buf

# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def expr_to_image(latex_str: str) -> discord.File:
    """
    Render a LaTeX string to a PNG and return it as a :class:`discord.File`.

    The heavy matplotlib work runs in a background thread so the Discord
    event loop is never blocked.

    Parameters
    ----------
    latex_str:
        A LaTeX expression string without surrounding ``$`` delimiters,
        e.g. ``r"\\frac{1}{2}"`` or ``"x^{2} + 2x + 1"``.

    Returns
    -------
    discord.File
        A file named ``formula.png``, suitable for passing directly to
        ``interaction.followup.send(file=...)``.

    Raises
    ------
    ValueError
        If matplotlib raises a ``RuntimeError`` while parsing the LaTeX
        (e.g. unmatched braces or unknown commands) the exception is
        caught and re-raised as a :class:`ValueError` with a friendly
        message so cogs can display it via :func:`~utils.formatter.error_embed`.
    """
    loop = asyncio.get_event_loop()
    try:
        buf = await loop.run_in_executor(_render_executor, _render_to_bytes, latex_str)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Could not render LaTeX to image: {exc}\n"
            "Check that the expression contains valid LaTeX syntax."
        ) from exc

    return discord.File(buf, filename="formula.png")


async def result_to_image(sympy_expr: sympy.Basic) -> discord.File:
    """
    Convert a SymPy expression to a PNG image via its LaTeX representation.

    This is a thin convenience wrapper: it calls ``sympy.latex()`` to obtain
    the LaTeX string, then delegates to :func:`expr_to_image`.

    Parameters
    ----------
    sympy_expr:
        Any SymPy expression, e.g. the return value of
        :func:`utils.parser.parse_expression` or any SymPy computation.

    Returns
    -------
    discord.File
        A file named ``formula.png``.

    Raises
    ------
    ValueError
        Propagated from :func:`expr_to_image` if rendering fails.

    Example
    -------
    ::

        result = sympy.integrate(sympy.parse_expr("x**2"), sympy.Symbol("x"))
        file = await result_to_image(result)
        await interaction.followup.send(file=file)
    """
    latex_str = sympy.latex(sympy_expr)
    return await expr_to_image(latex_str)
