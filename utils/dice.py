"""
utils/dice.py — Shared dice-notation parser.

Parses standard "NdM+K" dice notation (e.g. ``"2d6+3"``, ``"d20"``,
``"4d6-1"``) into its components. One parser, two consumers:
``cogs/random_tools.py``'s ``/rand dice`` (roll it and return an outcome)
and the Phase 3 probability cog's ``/prob dice_sum`` (compute the exact
distribution of outcomes) both import this rather than each maintaining
their own copy of the same regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DICE_RE = re.compile(
    r"^\s*(\d*)\s*[dD]\s*(\d+)\s*([+-]\s*\d+)?\s*$"
)


@dataclass(frozen=True)
class DiceSpec:
    """A parsed dice expression: roll *count* dice with *sides* sides each, plus *modifier*."""

    count: int
    sides: int
    modifier: int

    @property
    def min_total(self) -> int:
        return self.count * 1 + self.modifier

    @property
    def max_total(self) -> int:
        return self.count * self.sides + self.modifier


def parse_dice(notation: str) -> DiceSpec:
    """
    Parse dice notation into a :class:`DiceSpec`.

    Accepts ``"NdM"``, ``"dM"`` (implicit count of 1), and an optional
    trailing ``+K``/``-K`` modifier, with flexible whitespace and either
    case of ``d``.

    Raises
    ------
    ValueError
        If *notation* doesn't match the expected dice-notation shape, or
        if the parsed count/sides are out of a sane range.
    """
    m = _DICE_RE.match(notation)
    if not m:
        raise ValueError(
            f"`{notation}` isn't valid dice notation. Expected something "
            'like `"2d6"`, `"d20"`, or `"3d6+2"`.'
        )

    count_str, sides_str, modifier_str = m.groups()
    count = int(count_str) if count_str else 1
    sides = int(sides_str)
    modifier = int(modifier_str.replace(" ", "")) if modifier_str else 0

    if count < 1 or count > 100:
        raise ValueError("Dice count must be between 1 and 100.")
    if sides < 2 or sides > 1000:
        raise ValueError("Dice must have between 2 and 1000 sides.")

    return DiceSpec(count=count, sides=sides, modifier=modifier)
