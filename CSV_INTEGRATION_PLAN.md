# MathFrame — CSV Integration Plan

## Overview

Add a `/csv` command group that lets users upload a CSV file and run
statistical operations and interactive plots directly from column data,
eliminating the need to manually paste numbers into every command.

The feature has three parts:

1. **Session manager** (`data/csv_session.py`) — in-memory per-user store
   that holds the parsed CSV so it is available across commands.
2. **CSV cog** (`cogs/csv_tools.py`) — the `/csv` command group covering
   upload, inspection, stat operations, and the interactive plot builder.
3. **Column autocomplete** — a shared autocomplete callback that reads the
   active session and populates Discord's dropdown with live column names,
   applied to every `/csv` command that takes a column argument.

No external additions to `requirements.txt` are needed. The implementation
uses `csv` (stdlib), `numpy`, and `scipy` — all already present.

---

## 1. Session Manager — `data/csv_session.py`

Pattern: follow `data/history.py` exactly (in-memory dict, threading.Lock,
bounded size). No database, no disk writes.

```
_sessions: dict[int, CSVSession]   # keyed by user_id
```

### `CSVSession` dataclass

| Field          | Type                        | Notes                              |
|----------------|-----------------------------|------------------------------------|
| `user_id`      | `int`                       |                                    |
| `filename`     | `str`                       | original filename for display      |
| `columns`      | `list[str]`                 | header row, order preserved        |
| `rows`         | `list[dict[str, str]]`      | raw strings; conversion on demand  |
| `uploaded_at`  | `datetime`                  | UTC; used for TTL check            |
| `row_count`    | `int`                       | cached `len(rows)`                 |

### Limits

| Constraint     | Value     | Rationale                               |
|----------------|-----------|-----------------------------------------|
| Max file size  | 2 MB      | Discord attachment ceiling is 8 MB      |
| Max rows       | 10 000    | keeps numpy operations sub-100 ms       |
| Max columns    | 50        | Discord Select options cap at 25; split |
| Session TTL    | 30 min    | avoids stale data after user walks away |

### Public API

```python
def store_session(user_id: int, filename: str,
                  columns: list[str], rows: list[dict]) -> CSVSession

def get_session(user_id: int) -> CSVSession | None
    # returns None if expired or not found; expires lazily on access

def clear_session(user_id: int) -> None

def get_numeric_column(session: CSVSession, col: str) -> np.ndarray
    # raises ValueError if col not found or contains non-numeric values

def get_column_names(user_id: int) -> list[str]
    # returns [] if no session; used by autocomplete
```

### Parsing

Done in the cog (not in the session manager) so the cog can send progress
feedback. Steps:

1. Validate file extension (`.csv`) and content-type.
2. `aiohttp` GET on `attachment.url` (discord.Attachment provides it).
3. Decode as UTF-8 (with `errors="replace"` for dirty files).
4. `csv.DictReader` → header row becomes `columns`, remaining rows become
   `rows` (list of `dict[str, str]`).
5. Validate limits, then call `store_session(...)`.

---

## 2. CSV Cog — `cogs/csv_tools.py`

Command group: `/csv`
Add to `COGS` list in `main.py` as `"cogs.csv_tools"`.

### 2.1 Command list

#### `/csv upload <file>`

```
Attachment: file   (discord.Attachment, required)
```

- Fetch and parse the CSV (see §1 Parsing).
- On success: send an ephemeral embed showing filename, row/column count,
  and a column summary table (name · inferred type · sample values).
- On error: send ephemeral error embed (file too large, not CSV, etc.).
- Replaces any existing session for that user silently.

**Inferred column type** (shown in the summary, not stored):
- `numeric` — all non-empty values parse as float
- `mixed`   — some numeric, some not
- `text`    — none numeric

Example embed:

```
📂 sales_data.csv  (1 243 rows · 6 columns)

Column          Type      Sample
──────────────────────────────────────
date            text      2024-01-01
revenue         numeric   4320.5, 3891.2
units_sold      numeric   142, 98
region          text      North, South
discount_pct    numeric   0.05, 0.12
notes           text      promo, —
```

---

#### `/csv info`

Shows the same summary embed as upload, using the existing session.
Useful if the user forgot what columns they have loaded.

---

#### `/csv preview [rows=5]`

```
Optional int: rows   (1–20, default 5)
```

Renders the first N rows as a monospace table inside a code block embed.
Truncates wide columns to 15 characters to stay within Discord's 4096-char
embed description limit.

---

#### `/csv stat <operation> <column>`

```
Choice: operation   (see list below)
str:    column      (autocompleted from session columns)
```

Reads the named column as a `np.ndarray` of floats, runs the operation,
and sends the same styled embed as the equivalent `/stat` command.

Internally delegates to the same logic already in `statistics.py`:

| `operation` choice | Equivalent existing command | What it calls          |
|--------------------|-----------------------------|------------------------|
| `mean`             | `/stat mean`                | `statistics.mean()`    |
| `median`           | `/stat median`              | `statistics.median()`  |
| `mode`             | `/stat mode`                | `statistics.multimode()`|
| `stdev`            | `/stat stdev`               | `statistics.stdev()`   |
| `variance`         | `/stat variance`            | `statistics.variance()` |
| `summary`          | *(new — no existing equiv)* | all five at once       |

`summary` is a new operation unique to the CSV path: runs mean, median,
mode, stdev, variance, min, max, and quartiles in a single embed. This is
the most useful addition since running five separate commands on the same
column was the main pain point.

---

#### `/csv stat2 <operation> <col_x> <col_y>`

Two-column version for operations that require paired series:

```
Choice: operation   (correlation | regression)
str:    col_x       (autocompleted)
str:    col_y       (autocompleted)
```

| `operation`   | Equivalent existing command | Notes                  |
|---------------|----------------------------|------------------------|
| `correlation` | `/stat correlation`        | Pearson r              |
| `regression`  | `/stat regression`         | linear, renders scatter|

Both share the same column-pair validation: equal length required, both
numeric.

---

#### `/csv plot`

Opens the interactive `CSVPlotView` (see §3). No extra parameters — the
plot type and columns are chosen inside the view.

---

#### `/csv clear`

Drops the user's session. Sends a confirmation ephemeral message.
Mirrors `/bot clear` which uses a confirmation view — do the same here
to avoid accidental clears mid-session.

---

### 2.2 Column autocomplete

```python
async def _column_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    names = get_column_names(interaction.user.id)
    return [
        app_commands.Choice(name=n, value=n)
        for n in names
        if current.lower() in n.lower()
    ][:25]
```

Applied with `@app_commands.autocomplete(column=_column_autocomplete)` on
every command that takes a `column` parameter. If the user has no active
session, the dropdown returns one placeholder entry:
`"⚠ No CSV loaded — use /csv upload first"`.

---

### 2.3 Error handling

All commands must check for an active session before doing anything:

```python
session = get_session(interaction.user.id)
if session is None:
    await interaction.followup.send(
        embed=error_embed("No CSV loaded. Use `/csv upload` first."),
        ephemeral=True,
    )
    return
```

Numeric column extraction (`get_numeric_column`) raises `ValueError` with a
descriptive message if the column contains non-numeric values — caught and
forwarded as an error embed with the problematic cell shown.

---

## 3. Interactive CSV Plot View — `CSVPlotView`

Modeled after `PlotEngineView` in `plot_engine.py`: a `discord.ui.View`
subclass that rebuilds its component tree on every state change and edits
the same ephemeral message in-place.

### 3.1 Plot types

| Type        | Axes needed    | Description                              |
|-------------|----------------|------------------------------------------|
| `scatter`   | X (num), Y (num) | Points with optional regression line   |
| `line`      | X (any), Y (num) | Line chart; X can be text (categorical)|
| `histogram` | X (num) only   | Frequency distribution with bin control |
| `bar`       | X (text/cat), Y (num) | Grouped bar; aggregates Y by X  |
| `box`       | One or more num columns | Side-by-side box-and-whisker    |
| `heatmap`   | All num columns | Pearson correlation matrix, colour-coded|

### 3.2 UI flow

The view lives in a single ephemeral message and uses Discord's Select
menus and Buttons. Max 5 rows of components per message.

```
Row 0:  [Select: Plot type ▾]
Row 1:  [Select: X column  ▾]   [Select: Y column ▾]  (Y hidden for hist/box/heatmap)
Row 2:  [Select: Colour ▾]      [Select: Style ▾]
Row 3:  [🎨 More options]       [📊 Render]            [✖ Close]
```

"More options" opens a secondary ephemeral view (same pattern as
`ThemePickerView` in `plot_engine.py`) with:
- Title text input (Modal)
- Show grid toggle
- Show data points overlay (scatter on line chart)
- Bin count (histogram only)
- Regression line toggle (scatter only)
- Aggregate function for bar chart (mean / sum / count)

### 3.3 State dataclass — `CSVPlotConfig`

```python
@dataclass
class CSVPlotConfig:
    user_id:    int
    plot_type:  str  = "scatter"
    col_x:      str  = ""
    col_y:      str  = ""
    col_extra:  list[str] = field(default_factory=list)  # box plot: multiple cols
    title:      str  = ""
    color:      str  = "#1f77b4"
    show_grid:  bool = True
    show_points: bool = False   # overlay raw points on line chart
    regression: bool = False    # scatter only
    bin_count:  int  = 20       # histogram only
    agg_func:   str  = "mean"   # bar chart: mean | sum | count
```

### 3.4 Rendering pipeline

Same pattern as `_render()` in `plot_engine.py`:

1. `loop.run_in_executor(None, _render_csv_plot, cfg, session)` — runs
   matplotlib in the thread pool so the event loop is not blocked.
2. `_render_csv_plot` builds a `Figure`, plots to it, saves to `io.BytesIO`
   as PNG, returns the buffer.
3. `discord.File(buf, filename="csv_plot.png")` attached to the edit.
4. On error: ephemeral error embed without closing the view, so the user can
   fix their selection and try again.

### 3.5 Column Select with >25 columns

Discord limits Select options to 25. If the CSV has more than 25 columns,
add a `[Select: Column page ▾]` control in row 1 that pages through groups
of 24 columns (last option is always "→ next page"). The active page index
is stored in `CSVPlotView` state and cleared on plot type change.

In practice most CSVs used in a Discord math bot context will have ≤25
columns, so this is a graceful edge-case fallback, not the main path.

---

## 4. Integration with existing `/stat` commands

The existing `/stat` commands are **not modified**. They remain typed-string
input only. The CSV path lives entirely in `/csv stat` and `/csv stat2`.

This keeps the existing commands simple and avoids adding optional parameters
that change the meaning of a command based on whether a session happens to be
active. Clean separation:

```
User wants to type numbers?       →  /stat mean  data:"1,2,3,4"
User has a CSV loaded?            →  /csv stat   operation:mean  column:revenue
User wants two-column analysis?   →  /csv stat2  operation:regression  col_x:units  col_y:revenue
User wants a chart?               →  /csv plot   (interactive)
```

---

## 5. File and folder changes

```
MathFrame/
├── data/
│   └── csv_session.py          ← NEW
├── cogs/
│   └── csv_tools.py            ← NEW
└── main.py                     ← ADD "cogs.csv_tools" to COGS list
```

No changes to existing cogs. No changes to `requirements.txt`.

---

## 6. Implementation order

### Phase 1 — Foundation (do this first, nothing else depends on it)
1. Write `data/csv_session.py` with full API and tests.
2. Write `/csv upload` and `/csv info` + `/csv preview`.
3. Verify session store/retrieve/expire cycle manually.

### Phase 2 — Stat integration
4. Write `/csv stat` with all six operations including `summary`.
5. Write `/csv stat2` (correlation + regression).
6. Wire up column autocomplete to both.

### Phase 3 — Interactive plot
7. Write `CSVPlotConfig` dataclass and `_render_csv_plot()`.
8. Build `CSVPlotView` starting with scatter and histogram only.
9. Add remaining plot types one at a time (line → bar → box → heatmap).
10. Add "More options" secondary view.

### Phase 4 — Polish
11. `/csv clear` with confirmation view.
12. Session expiry warning: if session is >25 min old, add a footer note
    "⚠ Session expires in ~5 min — re-upload to refresh."
13. Embed colour convention: use `discord.Colour.orange()` for all CSV
    embeds to visually distinguish them from the green math-result embeds.
14. Add `cogs.csv_tools` to `/bot help` automatically (no changes needed —
    the fixed help command already recurses into all cog groups).

---

## 7. Discord API constraints to keep in mind

| Constraint                        | Value   | Impact                               |
|-----------------------------------|---------|--------------------------------------|
| Select menu max options           | 25      | Column paging needed for wide CSVs   |
| Max components per message        | 5 rows  | View layout must stay within 5 rows  |
| Embed description limit           | 4 096   | Preview table must truncate          |
| Embed field value limit           | 1 024   | Summary stat values must be concise  |
| Attachment max size (free server) | 8 MB    | Enforce 2 MB limit in upload command |
| Attachment URL lifetime           | short   | Fetch immediately on upload, not later|
| Interaction timeout               | 15 min  | CSVPlotView timeout set to 10 min    |
| Ephemeral message editability     | yes     | PlotView edits work as expected      |

---

## 8. Example user flows

### Flow A — Quick column stat
```
/csv upload  file: [attach sales.csv]
→ "Loaded sales.csv · 1 243 rows · 6 columns"

/csv stat  operation: summary  column: revenue
→ Embed: mean=4 120.3  median=3 891.2  stdev=812.4  min=201  max=9 840
         Q1=3 412  Q3=4 901  n=1 243
```

### Flow B — Regression from CSV
```
/csv upload  file: [attach data.csv]

/csv stat2  operation: regression  col_x: units_sold  col_y: revenue
→ Embed: y = 28.4x + 312.7  r²=0.91
→ Scatter plot image with regression line
```

### Flow C — Interactive plot
```
/csv upload  file: [attach data.csv]

/csv plot
→ Ephemeral view opens:
   [Select: Plot type ▾]  → user picks "histogram"
   [Select: X column ▾]   → autocomplete shows column names → user picks "revenue"
   [📊 Render]
→ Message updates with histogram image
   User changes bins via More Options → re-renders in place
```
