# Random / Probability / Stats Summary / Quiz-Battle — Implementation Plan

**Goal of this document:** a phase-by-phase build plan for four pieces of new
functionality, ordered so each phase's output becomes infrastructure the
next phase reuses rather than duplicates. Quiz/Battle is the largest and
comes last because it consumes the random-generation utilities, the
probability/stats math, *and* every existing solver cog (`alg`, `calc`,
`nt`, `dis`) as its question generators and answer-verifiers.

---

## Conceptual split: Random vs Probability

These two are easy to blur together, so the dividing line is made explicit
up front and should be the test applied to every new command idea before it
gets added to either cog:

| | Random (`/rand`) | Probability (`/prob`) |
|---|---|---|
| Question it answers | "Give me **a** random thing." | "What's the **likelihood/distribution** of an outcome?" |
| Output | One concrete outcome (a number, a card, a shuffled list). | A probability, a distribution, a histogram, an expected value. |
| Example | `/rand dice 2d6` → rolls and returns `9`. | `/prob dice_sum 2d6 target:9` → returns `P(sum=9) = 4/36 ≈ 0.111`. |

If a command's output is "here's what happened," it's `/rand`. If its
output is "here's how likely / how distributed," it's `/prob`.

---

## Shared infrastructure (build this first, used by every phase below)

### `utils/rng.py` — new file

A single shared randomness module so `/rand`, `/prob`, and the quiz
generator never duplicate RNG logic or drift out of sync.

```python
import random
import secrets

def get_rng(seed: int | None = None) -> random.Random:
    """Returns a random.Random instance, seeded if a seed is given."""
    return random.Random(seed) if seed is not None else random.Random()

def secure_token(length: int) -> str:
    """Cryptographically secure token — uses `secrets`, never `random`."""
    return secrets.token_urlsafe(length)
```

Per-user seed state (for reproducible "regenerate the same roll/question")
lives in a small in-memory store mirroring `data/memory.py`'s
`(guild_id, user_id)` keying pattern — no disk persistence needed here,
since reproducibility only matters within a live session.

### `utils/dice.py` — new file

Shared dice-notation parser (`"2d6+3"` → `(count=2, sides=6, modifier=3)`),
used by both `/rand dice` (roll it) and `/prob dice_sum` (compute its
distribution). One regex, one parser, two consumers — avoids the
copy-pasted-regex trap that bit the ODE parsing earlier in this project.

---

## Phase 1 — Random cog (`cogs/random_tools.py`, group `/rand`)

**Estimated time: 0.5–1 day.** Pure utility generation, no persistence,
lowest risk phase — good to build first and builds confidence/momentum.

| Command | Params | Behavior |
|---|---|---|
| `/rand int` | `min`, `max` | Random integer in `[min, max]` inclusive. |
| `/rand float` | `min`, `max`, `decimals=2` | Random float, rounded for display. |
| `/rand choice` | `options` (comma list) | Pick one item. |
| `/rand shuffle` | `items` (comma list) | Return the list reordered. |
| `/rand sample` | `items` (comma list), `k` | Sample `k` without replacement. |
| `/rand dice` | `notation` (e.g. `"2d6+3"`) | Parse via `utils/dice.py`, roll, show total + individual die results. |
| `/rand coin` | `bias=0.5` | Single weighted flip → heads/tails. |
| `/rand matrix` | `rows`, `cols`, `min=-9`, `max=9`, `integer=True` | Random matrix — feeds `/mat` commands and quiz question generation. |
| `/rand vector` | `dim`, `min=-9`, `max=9` | Random vector — feeds `/mat dot`/`cross`. |
| `/rand poly` | `degree`, `min_coeff=-9`, `max_coeff=9`, `var="x"` | Random polynomial as a SymPy-parseable string — feeds `/alg` and `/calc`, and is the core engine behind quiz algebra/calculus questions. |
| `/rand prime` | `min`, `max` | Random prime in range via `sympy.randprime`. |
| `/rand token` | `length=16` | Secure token via `secrets` (genuinely security-sensitive — must not use `random`). |
| `/rand seed` | `value` (optional; omit to clear) | Set/clear a per-user reproducible seed for this session. |

**Testing checklist:** range edges (`min == max`, `min > max` should error
cleanly), `/rand poly degree=0` (constant — should still work), dice
notation edge cases (`"d20"` with no count, `"3d6-2"` negative modifier),
empty `options`/`items` lists should error with a clear message rather
than crash.

---

## Phase 2 — `/stat summary` (added to existing `cogs/statistics.py`)

**Estimated time: 0.5 day.** Quick, standalone, no new cog needed — slots
into the existing `stat` group alongside `/stat stdev`/`/stat variance`.

`/stat summary data` — same `data: "comma-separated numbers"` input
convention as the existing commands. Single embed returns:

n, sum, sum of squares, mean, median, mode, sample variance, population
variance, sample stdev, population stdev, min, max, range, Q1, Q3, IQR,
skewness, kurtosis.

Implementation note: factor the actual computation into a standalone
`compute_summary(nums: list[float]) -> dict` function in `utils/` (not
inline in the command) — **Phase 3's `/prob sample` reuses this exact
function** to summarize generated distribution samples, so it shouldn't be
locked inside the Discord command handler.

**Testing checklist:** n=1 (variance/stdev undefined — error cleanly,
don't divide by zero), all-identical values (variance=0, mode=that value),
even vs odd n (median/IQR interpolation).

---

## Phase 3 — Probability cog (`cogs/probability.py`, group `/prob`)

**Estimated time: 1.5–2 days.** Heaviest math-content phase. Reuses
`utils/dice.py` from Phase 1 and `compute_summary` from Phase 2.

| Command | Params | Behavior |
|---|---|---|
| `/prob sample` | `distribution` (normal/binomial/poisson/uniform/exponential), `n`, distribution params | Draw `n` samples, show histogram (reuse existing plotter pattern from `statistics.py`) + `compute_summary` output. This is the "set generator" feature — a generated sample set with full statistical characterization, not just a single draw. |
| `/prob dice_sum` | `notation`, `target` | Exact `P(sum = target)` for the given dice notation, computed combinatorially (not simulated). |
| `/prob bayes` | `prior`, `sensitivity` (P(B\|A)), `false_positive_rate` (P(B\|¬A)) | Bayes' theorem: returns `P(A\|B)`. |
| `/prob conditional` | small 2×2 contingency table input | `P(A\|B)`, `P(B\|A)`, independence check. |
| `/prob card_draw` | `hand_size`, `target_count`, `target_cards`, `deck_size=52` | General hypergeometric calculator — "probability of drawing exactly `target_count` of `target_cards` in a `hand_size`-card hand." |
| `/prob urn` | `red`, `blue`, `draws` | Hypergeometric urn draw — exact probability + Monte Carlo cross-check shown side by side (nice trust-building visual: exact math vs simulation converging). |
| `/prob birthday` | `n_people`, `simulate=False` | Exact birthday-paradox formula; optional Monte Carlo overlay if `simulate=True`. |
| `/prob set_sample` | `items` (comma list), `k`, `replacement=False`, `trials=1000` | Repeated sampling from a defined set, returns empirical outcome-frequency distribution — the other half of the "set generator" request, distinguished from `/rand sample` by reporting the *distribution over many trials* rather than one outcome. |
| `/prob monte_carlo_pi` | `trials` | Estimate π via random point-in-circle sampling; convergence plot (reuses plotter). |
| `/prob buffon` | `needle_length`, `line_spacing`, `trials` | Buffon's needle π estimation. |

**Testing checklist:** `/prob sample` distribution param validation (e.g.
binomial needs `n` and `p`, reject missing params with a clear message
listing what's needed — mirror the existing `/stat distribution` command's
`app_commands.choices` + conditional-param pattern), `/prob card_draw` with
`target_count > target_cards` or `> deck_size` (should error, not crash),
`/prob bayes` with probabilities outside `[0,1]`.

---

## Phase 4 — Quiz/Battle cog, foundation (`cogs/quiz.py`, group `/quiz`)

**Estimated time: 2–3 days.** This is where the project gets genuinely
ambitious. Foundation phase = solo practice only, no multiplayer yet,
because the answer-verification and question-generation engine needs to be
solid before anything social is layered on top.

### New files

- `cogs/quiz.py` — the Discord-facing commands.
- `utils/quiz_generator.py` — question generation + answer verification.
  Generates problems by calling `/rand poly`/`/rand int`-style generators
  from Phase 1, then runs them through the **same underlying solver
  functions** `/alg`, `/calc`, `/nt`, `/dis` already use (not duplicate
  logic) to get a verified-correct answer. This is the key design choice:
  correctness is inherited from code that's already been tested in this
  project, not hand-curated.
- `data/quiz_store.py` — persistent storage. **Important:** every other
  data module in this codebase (`memory.py`, `history.py`,
  `csv_session.py`) is in-memory only and wiped on bot restart.
  `data/permissions.py` is the one existing module that persists to disk
  (plain JSON file, `threading.Lock` around read/write). Quiz ratings and
  streaks need to survive restarts, so `quiz_store.py` should follow
  `permissions.py`'s exact pattern rather than introducing a new
  persistence paradigm (e.g. sqlite) into the codebase.

### Data model (`data/quiz_store.py`, JSON-backed)

```json
{
  "users": {
    "<guild_id>:<user_id>": {
      "rating": 1200,
      "solved": 0,
      "wrong": 0,
      "streak_current": 0,
      "streak_best": 0,
      "subject_stats": {
        "algebra": {"solved": 0, "wrong": 0},
        "calculus": {"solved": 0, "wrong": 0},
        "number_theory": {"solved": 0, "wrong": 0},
        "discrete": {"solved": 0, "wrong": 0}
      },
      "last_daily_date": null,
      "achievements": []
    }
  }
}
```

### Question generation per subject

| Subject | Generator approach | Verified via |
|---|---|---|
| Algebra | `/rand poly`-style random polynomial equation | The actual `/alg solve` core function |
| Calculus | Random expression (built from a small template set: polynomials, trig, exp) | `differentiate_steps`/`integrate_steps` core functions |
| Number theory | Random integers in a difficulty-scaled range | `gcd`/`lcm`/`is_prime`/`factorize` core functions |
| Discrete | Random `n`, `k` for permutation/combination | The existing permutation/combination core functions |

`Question` dataclass: `subject`, `difficulty`, `prompt_text`,
`correct_answer` (SymPy expr or number), `answer_type`
(`symbolic`/`numeric`/`exact`), `seed` (for reproducibility/debugging),
`generated_at`.

### Answer verification

- Symbolic: `sympy.simplify(parsed_user_answer - correct_answer) == 0` —
  same equivalence-checking pattern already used elsewhere in this
  project, so `"2x"` and `"x*2"` and `"x+x"` are all accepted as correct.
- Numeric: tolerance-based (`math.isclose`) for floating-point answers.
- Submission UX: a "Submit Answer" button opening a
  `discord.ui.Modal` with one text input — more reliable than
  message-watching (`wait_for`), and consistent with this codebase's
  existing modal usage in the CSV plot builder.

### Commands (Phase 4 — solo only)

| Command | Params | Behavior |
|---|---|---|
| `/quiz practice` | `subject` (optional — random if omitted), `difficulty` (optional) | Generate one question, show it with a timer and a "Submit" button → modal. |
| `/quiz stats` | `user` (optional, defaults to self) | Solved/wrong/streak/rating for one user — read-only, no cross-user ranking yet (that's Phase 5). |

**Testing checklist:** generator must never produce an unsolvable/degenerate
problem (zero polynomial, division-by-zero in generated coefficients,
calculus templates that hit `NotImplementedError` in `/calc integrate` —
generator should retry with new random params rather than surface a broken
question to the user), verify the equivalence checker against deliberately
tricky equal-but-differently-formatted answers (`"1/2"` vs `"0.5"`, `"sin(x)^2 + cos(x)^2"` vs `"1"`).

---

## Phase 5 — Quiz/Battle cog, social layer

**Estimated time: 2–3 days.** Builds directly on Phase 4's generator and
verification engine — nothing here re-solves "how do I generate/verify a
question," it's purely the multiplayer/competitive/retention layer on top.

| Command | Params | Behavior |
|---|---|---|
| `/quiz battle` | `opponent` (user mention), `subject` (optional), `difficulty` (optional) | Sends an Accept/Decline button challenge to `opponent`. On accept, both players race the same question (same seed, so it's provably identical); first correct submission wins. Simple ELO-style rating delta applied to both players afterward. |
| `/quiz leaderboard` | `subject` (optional) | Top N by rating, scoped to the current server (`guild_id` partition — never leaks across servers). |
| `/quiz daily` | — | One question, identical for everyone, once per UTC day. Separate daily leaderboard. Streak tracked via `last_daily_date` in the data model. |
| `/quiz hint` | — (used during an active question) | Costs rating points, reveals the next step from the question's already-computed step-by-step solution (reuses `/alg solve`/`/calc diff`'s existing step-by-step output — another place this design avoids inventing new logic). |

### Achievements

Simple rule-based checks evaluated after each answered question (e.g.
`streak_current >= 7` → award `"week_streak"` if not already held). Stored
as a list of string IDs in the user's `quiz_store.py` record. Cheap to add
once the rating/streak system exists — a few `if` checks, not a new
subsystem.

### Design tradeoff worth noting explicitly

Nothing stops a player from running `/alg solve` themselves and pasting
the answer into `/quiz practice`/`/quiz battle` — the quiz bot and the
solver bot are the same bot. This is treated as an accepted tradeoff
rather than something to engineer around: the goal is server engagement
and your own practice, not tamper-proof competitive integrity. Not worth
spending phase time on anti-cheat measures for a single-server hobby
project.

**Testing checklist:** battle where both players answer simultaneously
(race condition on "who submitted first" — needs a clear, consistent
tie-breaking rule, e.g. server-received timestamp), daily challenge reset
exactly at UTC midnight (test near the boundary), leaderboard with zero
participants (empty state message, not a crash), rating delta math doesn't
let ratings go negative.

---

## Suggested week schedule

| Day | Work |
|---|---|
| 1 | Shared infra (`utils/rng.py`, `utils/dice.py`) + Phase 1 (Random cog) + Phase 2 (`/stat summary`) |
| 2–3 | Phase 3 (Probability cog) |
| 3–5 | Phase 4 (Quiz foundation: generator, verification, solo practice) |
| 5–7 | Phase 5 (Battle mode, leaderboard, daily challenge, achievements) + buffer |

## Open decisions to confirm before building

1. **Quiz group structure** — one `/quiz` group with subcommands
   (`practice`, `battle`, `leaderboard`, `daily`, `stats`, `hint`) keeps
   this to a single new top-level command group, well within Discord's
   per-group subcommand limits. Confirm this naming is what you want
   before Phase 4 starts, since renaming later means re-syncing commands.
2. **ELO formula specifics** — standard chess-style ELO (K-factor, starting
   rating 1200) is the default assumption above; flag now if you want
   something simpler (e.g. flat +10/-5 per win/loss).
3. **Persistence scale** — JSON-file persistence (matching
   `permissions.py`) is appropriate at single-server hobby-project scale.
   If this ever needs to scale to many servers/users, that's a future
   migration to something like SQLite, not a Phase 4/5 concern.
