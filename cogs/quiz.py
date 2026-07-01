"""
cogs/quiz.py — Quiz/Battle system.

Phase 4 (solo foundation): /quiz practice, /quiz stats.
Phase 5 (social layer, this update): /quiz battle, /quiz leaderboard,
/quiz daily, /quiz hint, plus achievement announcements — all built on
top of Phase 4's generator/verification engine (utils/quiz_generator.py)
and Phase 4's persistence (data/quiz_store.py). See
RANDOM_PROBABILITY_QUIZ_PLAN.md, Phase 5, for the design doc this follows.

Commands
--------
/quiz practice    [subject] [difficulty]   Generate one question; answer via a modal button.
/quiz stats       [user]                   Solved/wrong/streak/rating/achievements for a user.
/quiz battle      opponent [subject] [difficulty]   Challenge another player to a race.
/quiz leaderboard [subject]                Top players in this server by rating (or subject).
/quiz daily                                One question, same for everyone, once per UTC day.
/quiz hint                                 Reveal the next hint step for your active question.

Design note (accepted tradeoff, documented in the plan doc)
-------------------------------------------------------------
Nothing stops a player from running the corresponding solver command
(``/alg solve``, ``/calc diff``, etc.) themselves and pasting the answer
in — the quiz bot and the solver bot are the same bot. This is accepted:
the goal is engagement and personal practice, not tamper-proof competitive
integrity, and building anti-cheat measures for a single-server hobby
project isn't a good use of build time.

Active-question tracking (new in Phase 5, needed by /quiz hint)
------------------------------------------------------------------
``/quiz hint`` has no parameters — it operates on "your current active
question", which means something has to remember which question that is.
Every other data module in this codebase that only needs to survive for
the life of a live interaction (not a bot restart) is a plain in-memory
dict keyed by ``(guild_id, user_id)`` — the same pattern
``utils/rng.py``'s per-user seed store and ``data/memory.py`` use — so
that's what ``_active_questions`` below is. It is intentionally NOT part
of ``data/quiz_store.py``'s JSON-backed persistence: a hint about a
question that no longer exists after a restart is meaningless, so there's
nothing worth persisting.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import discord
from discord import app_commands, ui
from discord.ext import commands

from utils.formatter import math_embed, error_embed, info_embed
from utils.quiz_generator import generate_question, check_answer, SUBJECTS, DIFFICULTIES, Question
from data.quiz_store import (
    get_record,
    record_result,
    get_leaderboard,
    apply_battle_result,
    apply_hint_cost,
    get_daily_date_str,
    get_daily_seed,
    has_answered_daily,
    record_daily_result,
    get_daily_leaderboard,
    ACHIEVEMENT_LABELS,
    HINT_COST,
)

_SUBJECT_LABELS = {
    "algebra": "Algebra",
    "calculus": "Calculus",
    "number_theory": "Number Theory",
    "discrete": "Discrete Math",
}

_ANSWER_TIMEOUT = 120     # seconds — generous enough for real problem-solving, not indefinite
_BATTLE_TIMEOUT = 180     # seconds — a bit longer than solo practice; two people need to coordinate
_CHALLENGE_TIMEOUT = 60   # seconds — how long an opponent has to Accept/Decline


# ---------------------------------------------------------------------------
# Active-question tracking (in-memory, per (guild_id, user_id) — see module
# docstring). Populated when a question is shown, cleared once it's answered
# or times out. Consumed by /quiz hint.
# ---------------------------------------------------------------------------


@dataclass
class _ActiveQuestion:
    question: Question
    hints_revealed: int = 0


_active_questions: dict[tuple[int, int], _ActiveQuestion] = {}


def _register_active(guild_id: int, user_id: int, question: Question) -> None:
    _active_questions[(guild_id, user_id)] = _ActiveQuestion(question)


def _clear_active(guild_id: int, user_id: int) -> None:
    _active_questions.pop((guild_id, user_id), None)


def _new_achievements(before: dict, after: dict) -> list[str]:
    """Diff two record snapshots' ``achievements`` lists to find newly-earned ones."""
    return [a for a in after.get("achievements", []) if a not in set(before.get("achievements", []))]


def _achievement_note(new_ids: list[str]) -> str:
    if not new_ids:
        return ""
    labels = "\n".join(f"• {ACHIEVEMENT_LABELS.get(a, a)}" for a in new_ids)
    return f"\n\n**🏅 Achievement unlocked!**\n{labels}"


def _mention_embed(title: str, description: str, colour: discord.Colour = discord.Colour.blurple()) -> discord.Embed:
    """
    Plain (non-code-block) embed for content that needs to render Discord
    mentions (``<@id>``) — ``math_embed`` wraps its result in a code block,
    which would print mentions as literal text instead of resolving them.
    """
    embed = discord.Embed(title=title, description=description, colour=colour)
    return embed


# ---------------------------------------------------------------------------
# Answer submission modal (shared by practice, battle, and daily views)
# ---------------------------------------------------------------------------


class AnswerModal(ui.Modal, title="Submit Your Answer"):
    answer = ui.TextInput(
        label="Your answer",
        placeholder="e.g. 5, cos(x), yes",
        required=True,
        max_length=200,
    )

    def __init__(self, question: Question, handler) -> None:
        super().__init__()
        self._question = question
        self._handler = handler

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self._handler.handle_answer(interaction, self.answer.value)


# ---------------------------------------------------------------------------
# Practice view — one button that opens the answer modal
# ---------------------------------------------------------------------------


class PracticeView(ui.View):
    def __init__(self, question: Question, guild_id: int, user_id: int) -> None:
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
        _clear_active(self.guild_id, self.user_id)

        before = get_record(self.guild_id, self.user_id)
        correct = check_answer(self.question, raw_answer)
        record = record_result(self.guild_id, self.user_id, self.question.subject, correct)
        new_achievements = _new_achievements(before, record)

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
        body += _achievement_note(new_achievements)

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
        _clear_active(self.guild_id, self.user_id)
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
# Daily challenge view — one attempt, enforced by data/quiz_store.py
# ---------------------------------------------------------------------------


class DailyView(ui.View):
    def __init__(self, question: Question, guild_id: int, user_id: int, date_str: str) -> None:
        super().__init__(timeout=_ANSWER_TIMEOUT)
        self.question = question
        self.guild_id = guild_id
        self.user_id = user_id
        self.date_str = date_str
        self.message: discord.Message | None = None
        self._answered = False

    @ui.button(label="Submit Answer", style=discord.ButtonStyle.primary)
    async def submit_button(self, interaction: discord.Interaction, button: ui.Button) -> None:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                embed=error_embed("This isn't your daily challenge to answer."), ephemeral=True
            )
            return
        if self._answered:
            await interaction.response.send_message(
                embed=error_embed("You've already submitted an answer."), ephemeral=True
            )
            return
        await interaction.response.send_modal(AnswerModal(self.question, self))

    async def handle_answer(self, interaction: discord.Interaction, raw_answer: str) -> None:
        if self._answered:
            await interaction.response.send_message(
                embed=error_embed("You've already submitted an answer."), ephemeral=True
            )
            return
        self._answered = True
        _clear_active(self.guild_id, self.user_id)

        before = get_record(self.guild_id, self.user_id)
        correct = check_answer(self.question, raw_answer)

        try:
            record = record_daily_result(self.guild_id, self.user_id, self.question.subject, correct, self.date_str)
        except ValueError as exc:
            # Only reachable on a genuine double-submit race (e.g. two rapid
            # clicks); the button-disable above prevents the common case.
            for child in self.children:
                child.disabled = True
            self.stop()
            await interaction.response.edit_message(embed=error_embed(str(exc)), view=self)
            return

        new_achievements = _new_achievements(before, record)

        for child in self.children:
            child.disabled = True
        self.stop()

        if correct:
            title = "Correct! 🎉"
            body = (
                f"Your answer: `{raw_answer.strip()}`\n"
                f"Daily streak: {record['daily_streak_current']}  |  "
                f"Best: {record['daily_streak_best']}"
            )
        else:
            title = "Not quite"
            body = (
                f"Your answer: `{raw_answer.strip()}`\n"
                f"Correct answer: `{self.question.correct_answer}`\n"
                f"Daily streak reset to 0."
            )
        body += _achievement_note(new_achievements)

        embed = math_embed(
            title=title,
            result=body,
            footer=f"Daily Challenge  |  {_SUBJECT_LABELS[self.question.subject]}  |  rating: {record['rating']}",
        )
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self) -> None:
        if self._answered:
            return
        _clear_active(self.guild_id, self.user_id)
        for child in self.children:
            child.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(
                    embed=error_embed(
                        "Time's up! Daily challenge not answered — no penalty, "
                        "run `/quiz daily` again to retry (still today)."
                    ),
                    view=self,
                )
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Battle challenge view — Accept/Decline, sent to the opponent
# ---------------------------------------------------------------------------


class ChallengeView(ui.View):
    def __init__(
        self,
        challenger: discord.abc.User,
        opponent: discord.abc.User,
        subject: str | None,
        difficulty: str | None,
    ) -> None:
        super().__init__(timeout=_CHALLENGE_TIMEOUT)
        self.challenger = challenger
        self.opponent = opponent
        self.subject = subject
        self.difficulty = difficulty
        self.message: discord.Message | None = None
        self._resolved = False

    @ui.button(label="Accept", style=discord.ButtonStyle.success)
    async def accept(self, interaction: discord.Interaction, button: ui.Button) -> None:
        if interaction.user.id != self.opponent.id:
            await interaction.response.send_message(
                embed=error_embed("Only the challenged player can respond to this."), ephemeral=True
            )
            return
        if self._resolved:
            return
        self._resolved = True
        for child in self.children:
            child.disabled = True
        self.stop()

        seed = random.SystemRandom().randint(1, 2**31 - 1)
        try:
            question = generate_question(self.subject, self.difficulty, seed=seed)
        except (ValueError, RuntimeError) as exc:
            await interaction.response.edit_message(
                embed=error_embed(f"Couldn't start the battle: {exc}"), view=self
            )
            return

        guild_id = interaction.guild_id or 0
        battle_view = BattleView(question, guild_id, self.challenger.id, self.opponent.id, question.subject)

        embed = math_embed(
            title=f"⚔️ Battle! {_SUBJECT_LABELS[question.subject]} — {question.difficulty.title()}",
            result=question.prompt,
            footer=f"First correct answer wins!  {self.challenger.display_name} vs {self.opponent.display_name}  "
            f"|  {_BATTLE_TIMEOUT}s",
        )
        await interaction.response.edit_message(content=None, embed=embed, view=battle_view)
        battle_view.message = await interaction.original_response()
        _register_active(guild_id, self.challenger.id, question)
        _register_active(guild_id, self.opponent.id, question)

    @ui.button(label="Decline", style=discord.ButtonStyle.danger)
    async def decline(self, interaction: discord.Interaction, button: ui.Button) -> None:
        if interaction.user.id != self.opponent.id:
            await interaction.response.send_message(
                embed=error_embed("Only the challenged player can respond to this."), ephemeral=True
            )
            return
        if self._resolved:
            return
        self._resolved = True
        for child in self.children:
            child.disabled = True
        self.stop()
        await interaction.response.edit_message(
            content=None,
            embed=error_embed(f"{self.opponent.display_name} declined the challenge."),
            view=self,
        )

    async def on_timeout(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        for child in self.children:
            child.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(
                    content=None,
                    embed=error_embed("Challenge expired — no response in time."),
                    view=self,
                )
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Battle view — shared question, race to answer first
# ---------------------------------------------------------------------------


class BattleView(ui.View):
    def __init__(self, question: Question, guild_id: int, challenger_id: int, opponent_id: int, subject: str) -> None:
        super().__init__(timeout=_BATTLE_TIMEOUT)
        self.question = question
        self.guild_id = guild_id
        self.player_ids = (challenger_id, opponent_id)
        self.subject = subject
        self.message: discord.Message | None = None
        self._answered: dict[int, bool] = {}   # user_id -> was their answer correct
        self._winner: int | None = None
        self._resolved = False
        self._async_lock = asyncio.Lock()

    @ui.button(label="Submit Answer", style=discord.ButtonStyle.primary)
    async def submit_button(self, interaction: discord.Interaction, button: ui.Button) -> None:
        if interaction.user.id not in self.player_ids:
            await interaction.response.send_message(
                embed=error_embed("You're not part of this battle."), ephemeral=True
            )
            return
        if interaction.user.id in self._answered:
            await interaction.response.send_message(
                embed=error_embed("You've already submitted your answer."), ephemeral=True
            )
            return
        await interaction.response.send_modal(AnswerModal(self.question, self))

    async def handle_answer(self, interaction: discord.Interaction, raw_answer: str) -> None:
        user_id = interaction.user.id
        correct = check_answer(self.question, raw_answer)

        # Server-received-order tie-break: whichever submission acquires the
        # lock first and finds no winner yet claims the win. Concurrent
        # submissions serialize here rather than racing on wall-clock time.
        async with self._async_lock:
            if self._resolved:
                await interaction.response.send_message(
                    embed=error_embed("This battle has already been decided."), ephemeral=True
                )
                return
            self._answered[user_id] = correct
            if correct and self._winner is None:
                self._winner = user_id
            should_finish = self._winner is not None or len(self._answered) >= len(self.player_ids)
            if should_finish:
                self._resolved = True

        _clear_active(self.guild_id, user_id)

        if not should_finish:
            await interaction.response.send_message(
                embed=math_embed(
                    title="Answer received",
                    result="Waiting to see if your opponent answers first...",
                ) if correct else error_embed("Not correct — waiting on your opponent's attempt."),
                ephemeral=True,
            )
            return

        await self._finish(interaction)

    async def _finish(self, interaction: discord.Interaction) -> None:
        for uid in self.player_ids:
            _clear_active(self.guild_id, uid)
        for child in self.children:
            child.disabled = True
        self.stop()

        challenger_id, opponent_id = self.player_ids

        if self._winner is not None:
            winner_id = self._winner
            loser_id = opponent_id if winner_id == challenger_id else challenger_id

            record_result(self.guild_id, winner_id, self.subject, True)
            if loser_id in self._answered:
                record_result(self.guild_id, loser_id, self.subject, self._answered[loser_id])

            w_rec, l_rec, delta_w, delta_l = apply_battle_result(self.guild_id, winner_id, loser_id)

            description = (
                f"🏆 <@{winner_id}> wins the battle!\n\n"
                f"<@{winner_id}>: **{w_rec['rating']}** rating ({'+' if delta_w >= 0 else ''}{delta_w})\n"
                f"<@{loser_id}>: **{l_rec['rating']}** rating ({delta_l})\n\n"
                f"Correct answer: `{self.question.correct_answer}`"
            )
            title = "Battle Result"
        else:
            # Draw — everyone who submitted was wrong, no rating change.
            for uid, correct in self._answered.items():
                record_result(self.guild_id, uid, self.subject, correct)
            description = (
                f"Nobody answered correctly — no rating change.\n\n"
                f"Correct answer: `{self.question.correct_answer}`"
            )
            title = "Battle Result — Draw"

        embed = _mention_embed(title, description)
        embed.set_footer(text=f"{_SUBJECT_LABELS[self.subject]}  |  {self.question.difficulty}")
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        for uid in self.player_ids:
            _clear_active(self.guild_id, uid)
        for child in self.children:
            child.disabled = True

        for uid, correct in self._answered.items():
            record_result(self.guild_id, uid, self.subject, correct)

        if self.message is not None:
            try:
                embed = error_embed(
                    f"Time's up! No winner — no rating change.\n"
                    f"Correct answer: `{self.question.correct_answer}`"
                )
                await self.message.edit(embed=embed, view=self)
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class QuizCog(commands.Cog, name="Quiz"):
    """Math practice, head-to-head battles, and daily challenges with auto-verified questions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    quiz = app_commands.Group(name="quiz", description="Math quiz — practice, battle, and stats.")

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
            footer=f"You have {_ANSWER_TIMEOUT} seconds — tap Submit Answer when ready. "
            f"Use /quiz hint if you get stuck.",
        )
        await interaction.followup.send(embed=embed, view=view)
        view.message = await interaction.original_response()
        _register_active(guild_id, interaction.user.id, question)

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
            f"Current streak: {record['streak_current']}   Best streak: {record['streak_best']}\n"
            f"Battle record: {record.get('battle_wins', 0)}W - {record.get('battle_losses', 0)}L\n"
            f"Daily streak: {record.get('daily_streak_current', 0)}   "
            f"Best: {record.get('daily_streak_best', 0)}\n\n"
            + "\n".join(subject_lines)
        )

        achievements = record.get("achievements", [])
        if achievements:
            result += "\n\n🏅 " + ", ".join(ACHIEVEMENT_LABELS.get(a, a) for a in achievements)

        embed = math_embed(
            title=f"Quiz Stats — {target.display_name}",
            result=result,
        )
        await interaction.followup.send(embed=embed)

    # -----------------------------------------------------------------------
    # /quiz battle
    # -----------------------------------------------------------------------

    @quiz.command(name="battle", description="Challenge another player to a head-to-head quiz race.")
    @app_commands.describe(
        opponent="User to challenge",
        subject="Subject (random if omitted)",
        difficulty="Difficulty (random if omitted)",
    )
    @app_commands.choices(
        subject=[app_commands.Choice(name=_SUBJECT_LABELS[s], value=s) for s in SUBJECTS],
        difficulty=[app_commands.Choice(name=d.title(), value=d) for d in DIFFICULTIES],
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def battle(
        self,
        interaction: discord.Interaction,
        opponent: discord.User,
        subject: str | None = None,
        difficulty: str | None = None,
    ) -> None:
        if opponent.id == interaction.user.id:
            await interaction.response.send_message(
                embed=error_embed("You can't battle yourself — challenge someone else!"), ephemeral=True
            )
            return
        if opponent.bot:
            await interaction.response.send_message(
                embed=error_embed("You can't battle a bot."), ephemeral=True
            )
            return

        view = ChallengeView(interaction.user, opponent, subject, difficulty)
        embed = _mention_embed(
            "Quiz Battle Challenge",
            f"{interaction.user.mention} has challenged {opponent.mention} to a quiz battle!",
        )
        embed.set_footer(
            text=f"Subject: {_SUBJECT_LABELS.get(subject, 'Random')}  |  "
            f"Difficulty: {(difficulty or 'random').title()}  |  "
            f"{opponent.display_name} has {_CHALLENGE_TIMEOUT}s to respond."
        )
        await interaction.response.send_message(content=opponent.mention, embed=embed, view=view)
        view.message = await interaction.original_response()

    # -----------------------------------------------------------------------
    # /quiz leaderboard
    # -----------------------------------------------------------------------

    @quiz.command(name="leaderboard", description="Top players in this server by rating (or by subject).")
    @app_commands.describe(subject="Rank by this subject's solved count instead of overall rating")
    @app_commands.choices(
        subject=[app_commands.Choice(name=_SUBJECT_LABELS[s], value=s) for s in SUBJECTS],
    )
    @app_commands.checks.cooldown(1, 5.0)
    async def leaderboard(
        self,
        interaction: discord.Interaction,
        subject: str | None = None,
    ) -> None:
        await interaction.response.defer()
        guild_id = interaction.guild_id
        if guild_id is None:
            await interaction.followup.send(
                embed=error_embed("Leaderboards are server-specific — use this command inside a server.")
            )
            return

        entries = get_leaderboard(guild_id, subject)
        if not entries:
            await interaction.followup.send(
                embed=info_embed(
                    "Quiz Leaderboard",
                    "No one has played a quiz in this server yet — be the first with `/quiz practice`!",
                )
            )
            return

        lines = []
        for i, e in enumerate(entries, start=1):
            if subject:
                s = e["subject_stats"].get(subject, {"solved": 0, "wrong": 0})
                value = f"{s['solved']} solved"
            else:
                value = f"{e['rating']} rating"
            lines.append(f"**{i}.** <@{e['user_id']}> — {value}")

        title = f"Quiz Leaderboard — {_SUBJECT_LABELS[subject]}" if subject else "Quiz Leaderboard — Overall Rating"
        embed = _mention_embed(title, "\n".join(lines))
        await interaction.followup.send(embed=embed)

    # -----------------------------------------------------------------------
    # /quiz daily
    # -----------------------------------------------------------------------

    @quiz.command(name="daily", description="Today's daily challenge — same question for everyone, once per day.")
    @app_commands.checks.cooldown(1, 3.0)
    async def daily(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        guild_id = interaction.guild_id or 0
        date_str = get_daily_date_str()

        if has_answered_daily(guild_id, interaction.user.id, date_str):
            await interaction.followup.send(
                embed=error_embed("You've already answered today's daily challenge. Come back after 00:00 UTC!")
            )
            return

        seed = get_daily_seed(date_str)
        try:
            question = generate_question(seed=seed)
        except RuntimeError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Couldn't generate today's challenge — try again shortly. ({exc})")
            )
            return

        view = DailyView(question, guild_id, interaction.user.id, date_str)
        embed = math_embed(
            title=f"📅 Daily Challenge — {_SUBJECT_LABELS[question.subject]} ({question.difficulty.title()})",
            result=question.prompt,
            footer=f"One attempt only today  |  {_ANSWER_TIMEOUT}s  |  same question for the whole server.",
        )
        await interaction.followup.send(embed=embed, view=view)
        view.message = await interaction.original_response()
        _register_active(guild_id, interaction.user.id, question)

    # -----------------------------------------------------------------------
    # /quiz hint
    # -----------------------------------------------------------------------

    @quiz.command(name="hint", description="Reveal the next hint for your active question (costs rating points).")
    @app_commands.checks.cooldown(1, 3.0)
    async def hint(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id or 0
        key = (guild_id, interaction.user.id)
        active = _active_questions.get(key)

        if active is None:
            await interaction.response.send_message(
                embed=error_embed(
                    "You don't have an active question. Start one with `/quiz practice`, "
                    "`/quiz battle`, or `/quiz daily`."
                ),
                ephemeral=True,
            )
            return

        steps = active.question.hint_steps
        if active.hints_revealed >= len(steps):
            await interaction.response.send_message(
                embed=error_embed("No more hints available for this question."), ephemeral=True
            )
            return

        record = apply_hint_cost(guild_id, interaction.user.id)
        hint_text = steps[active.hints_revealed]
        active.hints_revealed += 1

        embed = math_embed(
            title=f"Hint {active.hints_revealed}/{len(steps)}",
            result=hint_text,
            footer=f"-{HINT_COST} rating (now {record['rating']})",
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    """Load the QuizCog into *bot*."""
    await bot.add_cog(QuizCog(bot))
