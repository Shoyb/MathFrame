"""
cogs/quiz.py — Quiz/Battle system, Phase 4 foundation (solo practice only;
see RANDOM_PROBABILITY_QUIZ_PLAN.md Phase 4). Phase 5 will add
``/quiz battle``, ``/quiz leaderboard``, ``/quiz daily``, and ``/quiz hint``
on top of the generator/verification engine built here.

Commands
--------
/quiz practice  [subject] [difficulty]   Generate one question; answer via a modal button.
/quiz stats     [user]                   Solved/wrong/streak/rating for a user.

Design note (accepted tradeoff, documented in the plan doc)
-------------------------------------------------------------
Nothing stops a player from running the corresponding solver command
(``/alg solve``, ``/calc diff``, etc.) themselves and pasting the answer
in — the quiz bot and the solver bot are the same bot. This is accepted:
the goal is engagement and personal practice, not tamper-proof competitive
integrity, and building anti-cheat measures for a single-server hobby
project isn't a good use of the remaining build time.
"""

from __future__ import annotations

import discord
from discord import app_commands, ui
from discord.ext import commands

from utils.formatter import math_embed, error_embed
from utils.quiz_generator import generate_question, check_answer, SUBJECTS, DIFFICULTIES
from data.quiz_store import get_record, record_result

_SUBJECT_LABELS = {
    "algebra": "Algebra",
    "calculus": "Calculus",
    "number_theory": "Number Theory",
    "discrete": "Discrete Math",
}

_ANSWER_TIMEOUT = 120  # seconds — generous enough for real problem-solving, not indefinite


# ---------------------------------------------------------------------------
# Answer submission modal
# ---------------------------------------------------------------------------


class AnswerModal(ui.Modal, title="Submit Your Answer"):
    answer = ui.TextInput(
        label="Your answer",
        placeholder="e.g. 5, cos(x), yes",
        required=True,
        max_length=200,
    )

    def __init__(self, question, view: "PracticeView") -> None:
        super().__init__()
        self._question = question
        self._view = view

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self._view.handle_answer(interaction, self.answer.value)


# ---------------------------------------------------------------------------
# Practice view — one button that opens the answer modal
# ---------------------------------------------------------------------------


class PracticeView(ui.View):
    def __init__(self, question, guild_id: int, user_id: int) -> None:
        super().__init__(timeout=_ANSWER_TIMEOUT)
        self.question = question
        self.guild_id = guild_id
        self.user_id = user_id
        self.message: discord.Message | None = None
        self._answered = False

    @ui.button(label="Submit Answer", style=discord.ButtonStyle.primary)
    async def submit_button(self, interaction: discord.Interaction, button: ui.Button) -> None:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                embed=error_embed("This isn't your question to answer."), ephemeral=True
            )
            return
        if self._answered:
            await interaction.response.send_message(
                embed=error_embed("This question has already been answered."), ephemeral=True
            )
            return
        await interaction.response.send_modal(AnswerModal(self.question, self))

    async def handle_answer(self, interaction: discord.Interaction, raw_answer: str) -> None:
        if self._answered:
            await interaction.response.send_message(
                embed=error_embed("This question has already been answered."), ephemeral=True
            )
            return
        self._answered = True

        correct = check_answer(self.question, raw_answer)
        record = record_result(self.guild_id, self.user_id, self.question.subject, correct)

        for child in self.children:
            child.disabled = True
        self.stop()

        if correct:
            title = "Correct!"
            body = (
                f"Your answer: `{raw_answer.strip()}`\n"
                f"Current streak: {record['streak_current']}  |  "
                f"Best streak: {record['streak_best']}"
            )
        else:
            title = "Not quite"
            body = (
                f"Your answer: `{raw_answer.strip()}`\n"
                f"Correct answer: `{self.question.correct_answer}`\n"
                f"Streak reset to 0."
            )

        embed = math_embed(
            title=title,
            result=body,
            footer=f"{_SUBJECT_LABELS[self.question.subject]}  |  {self.question.difficulty}  |  "
            f"rating: {record['rating']}",
        )
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self) -> None:
        if self._answered:
            return
        for child in self.children:
            child.disabled = True
        if self.message is not None:
            try:
                timeout_embed = error_embed(
                    f"Time's up! No answer submitted.\n"
                    f"Correct answer was: `{self.question.correct_answer}`"
                )
                await self.message.edit(embed=timeout_embed, view=self)
            except discord.HTTPException:
                pass  # message may have been deleted — nothing more we can do


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class QuizCog(commands.Cog, name="Quiz"):
    """Solo math practice with generated, auto-verified questions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    quiz = app_commands.Group(name="quiz", description="Math quiz — practice and stats.")

    # -----------------------------------------------------------------------
    # /quiz practice
    # -----------------------------------------------------------------------

    @quiz.command(name="practice", description="Generate a practice question and answer it.")
    @app_commands.describe(
        subject="Subject to practice (random if omitted)",
        difficulty="Difficulty (random if omitted)",
    )
    @app_commands.choices(
        subject=[app_commands.Choice(name=_SUBJECT_LABELS[s], value=s) for s in SUBJECTS],
        difficulty=[app_commands.Choice(name=d.title(), value=d) for d in DIFFICULTIES],
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def practice(
        self,
        interaction: discord.Interaction,
        subject: str | None = None,
        difficulty: str | None = None,
    ) -> None:
        await interaction.response.defer()
        try:
            question = generate_question(subject, difficulty)
        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
            return
        except RuntimeError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Couldn't generate a question this time — try again. ({exc})")
            )
            return

        guild_id = interaction.guild_id or 0
        view = PracticeView(question, guild_id, interaction.user.id)

        embed = math_embed(
            title=f"{_SUBJECT_LABELS[question.subject]} — {question.difficulty.title()}",
            result=question.prompt,
            footer=f"You have {_ANSWER_TIMEOUT} seconds — tap Submit Answer when ready.",
        )
        await interaction.followup.send(embed=embed, view=view)
        view.message = await interaction.original_response()

    # -----------------------------------------------------------------------
    # /quiz stats
    # -----------------------------------------------------------------------

    @quiz.command(name="stats", description="View quiz stats for yourself or another user.")
    @app_commands.describe(user="User to look up (defaults to you)")
    @app_commands.checks.cooldown(1, 3.0)
    async def stats(
        self,
        interaction: discord.Interaction,
        user: discord.User | None = None,
    ) -> None:
        await interaction.response.defer()
        target = user or interaction.user
        guild_id = interaction.guild_id or 0
        record = get_record(guild_id, target.id)

        total = record["solved"] + record["wrong"]
        accuracy = (record["solved"] / total * 100) if total > 0 else 0.0

        subject_lines = []
        for s in SUBJECTS:
            stats_s = record["subject_stats"].get(s, {"solved": 0, "wrong": 0})
            subject_lines.append(f"{_SUBJECT_LABELS[s]:15} {stats_s['solved']} correct, {stats_s['wrong']} wrong")

        result = (
            f"Rating: {record['rating']}\n"
            f"Solved: {record['solved']}   Wrong: {record['wrong']}   "
            f"Accuracy: {accuracy:.1f}%\n"
            f"Current streak: {record['streak_current']}   Best streak: {record['streak_best']}\n\n"
            + "\n".join(subject_lines)
        )

        embed = math_embed(
            title=f"Quiz Stats — {target.display_name}",
            result=result,
        )
        await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    """Load the QuizCog into *bot*."""
    await bot.add_cog(QuizCog(bot))
