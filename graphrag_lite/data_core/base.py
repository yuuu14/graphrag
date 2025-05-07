
"""Base data structure."""

from dataclasses import dataclass


@dataclass
class Identified:
    """A protocol for an item with an ID."""

    id: str
    """The ID of the item."""

    readable_id: str | None
    """Human readable ID used to refer to this community in prompts or texts displayed to users, such as in a report text (optional)."""


@dataclass
class Named(Identified):
    """A protocol for an item with a name/title."""

    alias: str
    """The name/title of the item."""
