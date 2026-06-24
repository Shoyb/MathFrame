"""
cogs/base_n.py — Base-N arithmetic and conversions.

Commands
--------
/base_convert value from_base to_base  Convert between any two bases (2-36).
/base_add a b base                     Add two numbers in a given base.
/base_logic a b operation base         AND/OR/XOR/NOT on integers in a given base.
/bases value                           Show a decimal value in binary, octal, and hex simultaneously.
"""

from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from utils.formatter import math_embed, error_embed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_base(val: str, base: int) -> int:
    """Parse string value in given base to decimal integer."""
    try:
        # Strip common prefixes if they match the base, e.g. 0x for hex, 0b for bin, 0o for oct
        val = val.strip().lower()
        if base == 16 and val.startswith("0x"): val = val[2:]
        if base == 2 and val.startswith("0b"): val = val[2:]
        if base == 8 and val.startswith("0o"): val = val[2:]
        
        return int(val, base)
    except ValueError:
        raise ValueError(f"`{val}` is not a valid base-{base} number.")

def _format_base(val: int, base: int) -> str:
    """Format decimal integer to string in given base."""
    if val == 0:
        return "0"
        
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Handle negative numbers
    is_negative = val < 0
    val = abs(val)
    
    res = ""
    while val > 0:
        res = chars[val % base] + res
        val //= base
        
    if is_negative:
        res = "-" + res
        
    # Add standard prefixes for common bases to make them clear
    if base == 2:
        return "0b" + res
    elif base == 8:
        return "0o" + res
    elif base == 16:
        return "0x" + res
        
    return res

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class BaseNCog(commands.Cog, name="Base-N"):
    """Base-N conversions and logic (Binary, Octal, Hex, and more)."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -----------------------------------------------------------------------
    # /base_convert
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="base_convert",
        description="Convert a number from one base to another (bases 2-36).",
    )
    @app_commands.describe(
        value="The number to convert",
        from_base="Source base (e.g. 2 for binary, 10 for decimal, 16 for hex)",
        to_base="Target base"
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def base_convert(
        self,
        interaction: discord.Interaction,
        value: str,
        from_base: int,
        to_base: int,
    ) -> None:
        await interaction.response.defer()
        try:
            if not (2 <= from_base <= 36) or not (2 <= to_base <= 36):
                raise ValueError("Bases must be between 2 and 36.")
                
            decimal_val = _parse_base(value, from_base)
            result_str = _format_base(decimal_val, to_base)
            
            embed = math_embed(
                title=f"Base-{from_base} to Base-{to_base} Conversion",
                result=result_str,
                steps=[("Decimal (Base-10)", str(decimal_val))] if from_base != 10 and to_base != 10 else []
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /base_add
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="base_add",
        description="Add two numbers in a specific base.",
    )
    @app_commands.describe(
        a="First number",
        b="Second number",
        base="The base of both numbers (2-36)"
    )
    @app_commands.checks.cooldown(1, 2.0)
    async def base_add(
        self,
        interaction: discord.Interaction,
        a: str,
        b: str,
        base: int,
    ) -> None:
        await interaction.response.defer()
        try:
            if not (2 <= base <= 36):
                raise ValueError("Base must be between 2 and 36.")
                
            val_a = _parse_base(a, base)
            val_b = _parse_base(b, base)
            
            total = val_a + val_b
            result_str = _format_base(total, base)
            
            steps = [
                ("Base-10 equivalent", f"{val_a} + {val_b} = {total}")
            ]
            
            embed = math_embed(
                title=f"Base-{base} Addition",
                result=result_str,
                steps=steps
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /base_logic
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="base_logic",
        description="Perform bitwise logic operations (AND, OR, XOR, NOT).",
    )
    @app_commands.describe(
        operation="The logic operation to perform",
        a="First number",
        b="Second number (leave blank for NOT)",
        base="The base of the inputs (default: 2 for binary)"
    )
    @app_commands.choices(operation=[
        app_commands.Choice(name="AND", value="and"),
        app_commands.Choice(name="OR", value="or"),
        app_commands.Choice(name="XOR", value="xor"),
        app_commands.Choice(name="NOT", value="not"),
    ])
    @app_commands.checks.cooldown(1, 2.0)
    async def base_logic(
        self,
        interaction: discord.Interaction,
        operation: str,
        a: str,
        b: str = "",
        base: int = 2,
    ) -> None:
        await interaction.response.defer()
        try:
            if not (2 <= base <= 36):
                raise ValueError("Base must be between 2 and 36.")
                
            val_a = _parse_base(a, base)
            
            if operation == "not":
                # For NOT, we perform a bitwise inversion (two's complement)
                # Display it in binary so it's clear what happened
                result_val = ~val_a
                
                # We show the bits up to the length of the original number (min 8 bits)
                # To do this safely for NOT we can just output the binary representation
                result_str = _format_base(result_val, base)
                expr_str = f"NOT {a}"
                
            else:
                if not b:
                    raise ValueError(f"Second number `b` is required for {operation.upper()} operation.")
                val_b = _parse_base(b, base)
                
                if operation == "and":
                    result_val = val_a & val_b
                    expr_str = f"{a} AND {b}"
                elif operation == "or":
                    result_val = val_a | val_b
                    expr_str = f"{a} OR {b}"
                elif operation == "xor":
                    result_val = val_a ^ val_b
                    expr_str = f"{a} XOR {b}"
                else:
                    raise ValueError("Unknown operation.")
                    
                result_str = _format_base(result_val, base)
                
            embed = math_embed(
                title=f"Base-{base} Logic: {operation.upper()}",
                result=result_str,
                steps=[("Operation", expr_str)]
            )
            await interaction.followup.send(embed=embed)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

    # -----------------------------------------------------------------------
    # /bases
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="bases",
        description="Show a value in Decimal, Binary, Octal, and Hexadecimal simultaneously.",
    )
    @app_commands.describe(value="The value to display (must be an integer)")
    @app_commands.checks.cooldown(1, 2.0)
    async def bases(
        self,
        interaction: discord.Interaction,
        value: str,
    ) -> None:
        await interaction.response.defer()
        try:
            # We accept inputs with prefixes like 0x, 0b, etc., so we guess the base if not decimal
            value_lower = value.strip().lower()
            if value_lower.startswith("0x"):
                decimal_val = int(value_lower, 16)
            elif value_lower.startswith("0b"):
                decimal_val = int(value_lower, 2)
            elif value_lower.startswith("0o"):
                decimal_val = int(value_lower, 8)
            else:
                decimal_val = int(value_lower, 10)
                
            steps = [
                ("Decimal (Base 10)", str(decimal_val)),
                ("Binary (Base 2)", _format_base(decimal_val, 2)),
                ("Octal (Base 8)", _format_base(decimal_val, 8)),
                ("Hexadecimal (Base 16)", _format_base(decimal_val, 16).upper().replace("0X", "0x")),
            ]
            
            embed = math_embed(
                title="Base-N Display",
                result=str(decimal_val),
                steps=steps
            )
            await interaction.followup.send(embed=embed)
        except ValueError:
            await interaction.followup.send(embed=error_embed(f"`{value}` is not a valid integer."))
        except Exception as exc:
            await interaction.followup.send(embed=error_embed(f"An unexpected error occurred: {exc}"))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the BaseNCog into *bot*."""
    await bot.add_cog(BaseNCog(bot))
