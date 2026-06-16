"""
cogs/wiki.py — Wikipedia article browser for the math bot.

Commands
--------
/wiki   topic:str   Fetch a Wikipedia article and display it paragraph by
                    paragraph with ◀ / ▶ navigation.

/wiki_search topic:str
                    Search Wikipedia for matching articles and let the user
                    pick one to open.

API used
--------
Wikipedia REST API v1 (no key required):
  Summary  : GET https://en.wikipedia.org/api/rest_v1/page/summary/{title}
  Sections : GET https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}
  Search   : GET https://en.wikipedia.org/w/api.php?action=query&list=search&...

Rate limits: 200 req/s per client — well within any Discord bot's needs.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from utils.formatter import error_embed
from utils.paginator import send_paginated

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE          = "https://en.wikipedia.org"
_REST_BASE     = f"{_BASE}/api/rest_v1/page"
_API_BASE      = f"{_BASE}/w/api.php"
_TIMEOUT       = aiohttp.ClientTimeout(total=10)

# Characters per embed description — Discord hard limit is 4096
_PAGE_CHARS    = 1800
# Max search results shown in /wiki_search
_SEARCH_LIMIT  = 5
# Thumbnail size hint for Wikipedia image URLs
_THUMB_WIDTH   = 480
# Colour used on all wiki embeds
_COLOUR        = discord.Colour.from_rgb(255, 255, 255)   # Wikipedia white

# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """
    Strip MediaWiki artefacts from plain-text article content.

    Removes:
    * Reference markers like ``[1]``, ``[note 2]``
    * Leftover ``\\n`` runs (collapse to single blank line)
    * Leading / trailing whitespace per line
    """
    text = re.sub(r"\[\d+\]",          "",   text)   # numeric refs
    text = re.sub(r"\[[a-z]+ \d+\]",   "",   text)   # named refs
    text = re.sub(r"\[note \d+\]",     "",   text)
    text = re.sub(r" {2,}",            " ",  text)   # collapse spaces
    text = re.sub(r"\n{3,}",           "\n\n", text) # max one blank line
    return text.strip()


def _thumbnail_url(summary: dict[str, Any]) -> str | None:
    """Extract the best available thumbnail URL from a summary response."""
    original = summary.get("originalimage") or {}
    thumb    = summary.get("thumbnail") or {}
    url      = original.get("source") or thumb.get("source")
    if url and "width=" in url:
        # Request a sensible width so we don't embed a 4000 px image
        url = re.sub(r"/\d+px-", f"/{_THUMB_WIDTH}px-", url)
    return url


async def _fetch_summary(session: aiohttp.ClientSession, title: str) -> dict[str, Any]:
    """
    Fetch the Wikipedia summary for *title*.

    Returns the parsed JSON dict on success.

    Raises
    ------
    ValueError
        If the page does not exist (404) or the request fails.
    """
    url = f"{_REST_BASE}/summary/{aiohttp.helpers.quote(title, safe='')}"
    async with session.get(url, timeout=_TIMEOUT) as resp:
        if resp.status == 404:
            raise ValueError(
                f"No Wikipedia article found for **{title}**. "
                "Try `/wiki_search` to find the right title."
            )
        if resp.status != 200:
            raise ValueError(
                f"Wikipedia returned HTTP {resp.status}. Please try again."
            )
        return await resp.json()


async def _fetch_sections(
    session: aiohttp.ClientSession,
    title: str,
) -> list[dict[str, Any]]:
    """
    Fetch all sections of *title* via the mobile-sections API.

    Returns a list of section dicts, each with at least ``"title"``
    and ``"text"`` keys (plain text, already stripped of most markup).

    Falls back to an empty list if the endpoint fails — the summary
    paragraph is always shown even when sections can't be fetched.
    """
    url = f"{_REST_BASE}/mobile-sections/{aiohttp.helpers.quote(title, safe='')}"
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []

    sections: list[dict[str, Any]] = []

    # Lead section lives at data["lead"]["sections"][0]
    lead_sections = (data.get("lead") or {}).get("sections") or []
    for s in lead_sections:
        text = _clean(_strip_html(s.get("text") or ""))
        if text:
            sections.append({"title": "Introduction", "text": text})

    # Remaining sections live under data["remaining"]["sections"]
    remaining = (data.get("remaining") or {}).get("sections") or []
    for s in remaining:
        text = _clean(_strip_html(s.get("text") or ""))
        if text:
            heading = s.get("line") or s.get("title") or "Section"
            heading = _clean(_strip_html(heading))
            sections.append({"title": heading, "text": text})

    return sections


async def _search_wikipedia(
    session: aiohttp.ClientSession,
    query: str,
    limit: int = _SEARCH_LIMIT,
) -> list[dict[str, str]]:
    """
    Search Wikipedia for *query* and return up to *limit* results.

    Each result is a dict with ``"title"`` and ``"snippet"`` keys.
    """
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  str(limit),
        "format":   "json",
        "utf8":     "1",
    }
    async with session.get(_API_BASE, params=params, timeout=_TIMEOUT) as resp:
        if resp.status != 200:
            raise ValueError(
                f"Wikipedia search returned HTTP {resp.status}. Please try again."
            )
        data   = await resp.json()
        hits   = (data.get("query") or {}).get("search") or []
        return [{"title": h["title"], "snippet": _clean(_strip_html(h.get("snippet", "")))}
                for h in hits]


def _strip_html(text: str) -> str:
    """Remove all HTML tags from *text*, replacing ``<br>`` with newlines."""
    text = re.sub(r"<br\s*/?>",  "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>",   "",   text)
    text = re.sub(r"&amp;",      "&",  text)
    text = re.sub(r"&lt;",       "<",  text)
    text = re.sub(r"&gt;",       ">",  text)
    text = re.sub(r"&nbsp;",     " ",  text)
    text = re.sub(r"&#\d+;",     "",   text)
    return text

# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------

def _article_pages(
    title: str,
    wiki_url: str,
    sections: list[dict[str, Any]],
    thumbnail: str | None,
) -> list[discord.Embed]:
    """
    Convert a list of article sections into a list of Discord embeds,
    splitting long sections across multiple pages so no embed exceeds
    Discord's description limit.

    Parameters
    ----------
    title:
        Article title used in every embed's title field.
    wiki_url:
        Link to the full article (shown in the first embed only).
    sections:
        List of ``{"title": str, "text": str}`` dicts.
    thumbnail:
        Optional image URL; set on the first embed only.

    Returns
    -------
    list[discord.Embed]
        At least one embed; ready to pass to :func:`~utils.paginator.send_paginated`.
    """
    pages: list[discord.Embed] = []

    for sec_idx, section in enumerate(sections):
        heading = section["title"]
        text    = section["text"]

        # Split section text into chunks that fit the embed description limit
        chunks = textwrap.wrap(
            text,
            width=_PAGE_CHARS,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        if not chunks:
            continue

        for chunk_idx, chunk in enumerate(chunks):
            # Section heading only on the first chunk of each section
            display_heading = heading if chunk_idx == 0 else f"{heading} (cont.)"

            embed = discord.Embed(
                title=f"Wikipedia — {title}",
                colour=discord.Colour.blurple(),
            )
            embed.add_field(
                name=display_heading,
                value=chunk,
                inline=False,
            )

            # First page gets the article link and thumbnail
            if not pages:
                embed.url = wiki_url
                if thumbnail:
                    embed.set_thumbnail(url=thumbnail)

            pages.append(embed)

    if not pages:
        # Shouldn't happen, but guard against completely empty articles
        embed = discord.Embed(
            title=f"Wikipedia — {title}",
            description="*(article has no readable content)*",
            colour=discord.Colour.blurple(),
            url=wiki_url,
        )
        pages.append(embed)

    return pages


def _search_result_embed(query: str, results: list[dict[str, str]]) -> discord.Embed:
    """Build a single embed listing search results for *query*."""
    embed = discord.Embed(
        title=f"Wikipedia Search: \"{query}\"",
        description=(
            "Here are the closest matches. "
            "Use `/wiki <title>` with the exact title to open an article."
        ),
        colour=discord.Colour.gold(),
    )
    for i, result in enumerate(results, start=1):
        snippet = result["snippet"]
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        embed.add_field(
            name=f"{i}. {result['title']}",
            value=snippet or "*(no snippet available)*",
            inline=False,
        )
    embed.set_footer(text=f"{len(results)} result(s) found  |  Powered by Wikipedia")
    return embed

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class WikiCog(commands.Cog, name="Wiki"):
    """Wikipedia article browser — read articles paragraph by paragraph in Discord."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot     = bot
        # Reuse one aiohttp session for the lifetime of the cog
        self._session: aiohttp.ClientSession | None = None

    async def cog_load(self) -> None:
        """Create the shared HTTP session when the cog is loaded."""
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "MathBot/1.0 (Discord bot; educational use)"}
        )

    async def cog_unload(self) -> None:
        """Close the HTTP session cleanly when the cog is unloaded."""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "MathBot/1.0 (Discord bot; educational use)"}
            )
        return self._session

    # -----------------------------------------------------------------------
    # /wiki
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="wiki",
        description="Fetch a Wikipedia article and browse it paragraph by paragraph.",
    )
    @app_commands.describe(
        topic="Article title or topic to look up, e.g. 'Pythagorean theorem'",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def wiki(
        self,
        interaction: discord.Interaction,
        topic: str,
    ) -> None:
        await interaction.response.defer()
        try:
            summary   = await _fetch_summary(self.session, topic)
            title     = summary.get("title", topic)
            wiki_url  = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
            thumbnail = _thumbnail_url(summary)

            # Try to get full sectioned content; fall back to summary extract
            sections = await _fetch_sections(self.session, title)

            if not sections:
                # Graceful fallback: use the summary extract as a single section
                extract = _clean(summary.get("extract") or "*(no content)*")
                sections = [{"title": "Summary", "text": extract}]

            pages = _article_pages(title, wiki_url, sections, thumbnail)
            await send_paginated(interaction, pages)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except aiohttp.ClientError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Network error fetching Wikipedia: {exc}")
            )

    # -----------------------------------------------------------------------
    # /wiki_search
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="wiki_search",
        description="Search Wikipedia and see a list of matching articles.",
    )
    @app_commands.describe(
        topic="Search query, e.g. 'Fourier transform' or 'prime numbers'",
    )
    @app_commands.checks.cooldown(1, 3.0)
    async def wiki_search(
        self,
        interaction: discord.Interaction,
        topic: str,
    ) -> None:
        await interaction.response.defer()
        try:
            results = await _search_wikipedia(self.session, topic)

            if not results:
                await interaction.followup.send(
                    embed=error_embed(
                        f"No Wikipedia articles found for **{topic}**. "
                        "Try different keywords."
                    )
                )
                return

            embed = _search_result_embed(topic, results)
            await interaction.followup.send(embed=embed)

        except ValueError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)))
        except aiohttp.ClientError as exc:
            await interaction.followup.send(
                embed=error_embed(f"Network error searching Wikipedia: {exc}")
            )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    """Load the WikiCog into *bot*."""
    await bot.add_cog(WikiCog(bot))
