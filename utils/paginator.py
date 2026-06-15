"""
utils/paginator.py — Embed paginator for the Discord math bot.

Provides a reusable :class:`PaginatorView` (a ``discord.ui.View`` with ◀ / ▶
navigation buttons) and the :func:`send_paginated` convenience helper that
picks the right send strategy automatically.

Usage
-----
::

    from utils.paginator import send_paginated

    pages = [embed1, embed2, embed3]
    await send_paginated(interaction, pages)

Or, if you need a handle on the view (e.g. to await it):
::

    from utils.paginator import PaginatorView

    view = PaginatorView(pages)
    await interaction.followup.send(embed=view.current_embed, view=view)
"""

import discord

# ---------------------------------------------------------------------------
# PaginatorView
# ---------------------------------------------------------------------------

class PaginatorView(discord.ui.View):
    """
    A :class:`discord.ui.View` that pages through a list of embeds with
    ◀ and ▶ buttons.

    Each button click edits the original message in-place.  The active
    embed's footer is updated on every turn to show ``"Page N/Total"``.

    Parameters
    ----------
    pages:
        Ordered list of :class:`discord.Embed` objects to page through.
        The embeds are copied internally so that footer mutations do not
        affect the caller's originals.
    timeout:
        Seconds of inactivity after which the view stops listening for
        interactions (default ``120``).  Buttons are *not* automatically
        disabled on timeout; add an ``on_timeout`` override if you need that.

    Attributes
    ----------
    current_index : int
        Zero-based index of the currently displayed page.

    Example
    -------
    ::

        view = PaginatorView(pages, timeout=60)
        await interaction.response.send_message(embed=view.current_embed, view=view)
    """

    def __init__(self, pages: list[discord.Embed], timeout: float = 120) -> None:
        super().__init__(timeout=timeout)

        # Defensive copy so footer updates don't mutate the caller's embeds.
        self._pages: list[discord.Embed] = [_copy_embed(e) for e in pages]
        self.current_index: int = 0

        # Stamp footers and set initial button state.
        self._stamp_all_footers()
        self._refresh_buttons()

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    @property
    def current_embed(self) -> discord.Embed:
        """Return the :class:`discord.Embed` for the active page."""
        if not self._pages:
            # Edge case: return a blank embed rather than raising IndexError.
            return discord.Embed(description="*(no pages)*")
        return self._pages[self.current_index]

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, custom_id="prev")
    async def prev_button(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        """Navigate to the previous page."""
        if self.current_index > 0:
            self.current_index -= 1
        self._refresh_buttons()
        await interaction.response.edit_message(
            embed=self.current_embed,
            view=self,
        )

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, custom_id="next")
    async def next_button(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        """Navigate to the next page."""
        if self.current_index < len(self._pages) - 1:
            self.current_index += 1
        self._refresh_buttons()
        await interaction.response.edit_message(
            embed=self.current_embed,
            view=self,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_buttons(self) -> None:
        """
        Enable / disable ◀ and ▶ based on the current position.

        * ◀ is disabled when on the first page (index 0) or when the
          page list is empty.
        * ▶ is disabled when on the last page or when the page list is
          empty.

        The buttons are looked up by their ``custom_id`` so this method
        is robust to subclasses that add extra children.
        """
        empty = len(self._pages) == 0
        last  = len(self._pages) - 1

        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.custom_id == "prev":
                child.disabled = empty or self.current_index == 0
            elif child.custom_id == "next":
                child.disabled = empty or self.current_index >= last

    def _stamp_all_footers(self) -> None:
        """
        Write ``"Page N / Total"`` into the footer of every embed.

        If an embed already has footer text, the page indicator is
        appended after a separator (``" · "``), so existing attribution
        strings are preserved.
        """
        total = len(self._pages)
        for i, embed in enumerate(self._pages):
            indicator = f"Page {i + 1} / {total}"
            existing  = embed.footer.text if embed.footer else None
            if existing:
                new_text = f"{existing} · {indicator}"
            else:
                new_text = indicator
            embed.set_footer(text=new_text)


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

async def send_paginated(
    interaction: discord.Interaction,
    pages: list[discord.Embed],
) -> None:
    """
    Send one or more embeds to *interaction*, adding pagination controls
    only when there is more than one page.

    If the interaction response has not been sent yet
    (``interaction.response.is_done()`` is ``False``) the first message
    is sent via ``interaction.response.send_message``; otherwise it falls
    back to ``interaction.followup.send``.  Cogs that call
    ``await interaction.response.defer()`` before doing heavy work will
    therefore still receive the result correctly.

    Parameters
    ----------
    interaction:
        The slash-command interaction to reply to.
    pages:
        Ordered list of embeds.  An empty list sends a plain "no results"
        message with no view attached.

    Examples
    --------
    Single page — sent without buttons::

        await send_paginated(interaction, [single_embed])

    Multiple pages — ◀ / ▶ buttons attached::

        await send_paginated(interaction, [page1, page2, page3])
    """
    if not pages:
        await _send(interaction, content="*(no results)*")
        return

    if len(pages) == 1:
        await _send(interaction, embed=pages[0])
        return

    view = PaginatorView(pages)
    await _send(interaction, embed=view.current_embed, view=view)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

async def _send(
    interaction: discord.Interaction,
    **kwargs,
) -> None:
    """
    Route a send call to ``response.send_message`` or ``followup.send``
    depending on whether the interaction response has already been used.
    """
    if interaction.response.is_done():
        await interaction.followup.send(**kwargs)
    else:
        await interaction.response.send_message(**kwargs)


def _copy_embed(embed: discord.Embed) -> discord.Embed:
    """
    Return a shallow copy of *embed* so mutations (footer text) are
    isolated from the caller's original.

    :class:`discord.Embed` does not implement ``__copy__``; the canonical
    way to clone one is ``discord.Embed.from_dict(embed.to_dict())``.
    """
    return discord.Embed.from_dict(embed.to_dict())
