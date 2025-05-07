
"""run模块里用到的类型, 包括RunContext, RunStorage等."""


import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from langchain_core.caches import BaseCache

from graphrag.logger.base import ProgressLogger

# Run contains multiple RunOp.
# RunOp is execution of operation. 

class RunStorage(metaclass=ABCMeta):
    """Provide a storage interface for the Run. This is where the Run will store its output data."""

    @abstractmethod
    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        progress: ProgressLogger | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find files in the storage using a file pattern, as well as a custom filter function."""

    @abstractmethod
    async def get(
        self, key: str, as_bytes: bool | None = None, encoding: str | None = None
    ) -> Any:
        """Get the value for the given key.

        Args
        ----
            - key: The key to get the value for.
            - as_bytes: Whether or not to return the value as bytes.

        Returns
        -------
            - output: The value for the given key.
        """

    @abstractmethod
    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set the value for the given key.

        Args:
            - key - The key to set the value for.
            - value - The value to set.
        """

    @abstractmethod
    async def has(self, key: str) -> bool:
        """Return True if the given key exists in the storage.

        Args
        ----
            - key - The key to check for.

        Returns
        -------
            - output - True if the key exists in the storage, False otherwise.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete the given key from the storage.

        Args:
            - key - The key to delete.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear the storage."""
    

    @abstractmethod
    def keys(self) -> list[str]:
        """List all keys in the storage."""

    @abstractmethod
    async def get_creation_date(self, key: str) -> str:
        """Get the creation date for the given key.

        Args:
            - key - The key to get the creation date for.

        Returns
        -------
            - output - The creation date for the given key.
        """


def get_timestamp_formatted_with_local_tz(timestamp: datetime) -> str:
    """Get the formatted timestamp with the local time zone."""
    creation_time_local = timestamp.astimezone()

    return creation_time_local.strftime("%Y-%m-%d %H:%M:%S %z")


@dataclass
class RunStats:
    """Pipeline running stats."""

    total_runtime: float = field(default=0)
    """Float representing the total runtime."""

    num_documents: int = field(default=0)
    """Number of documents."""

    input_load_time: float = field(default=0)
    """Float representing the input load time."""

    workflows: dict[str, dict[str, float]] = field(default_factory=dict)
    """A dictionary of workflows."""

@dataclass
class RunContext:
    """Provides the context for the current Run."""

    stats: RunStats
    storage: RunStorage
    "Long-term storage for pipeline verbs to use. Items written here will be written to the storage provider."
    cache: BaseCache
    "Cache instance for reading previous LLM responses."
    callbacks: Callbacks
    "Callbacks to be called during the pipeline run."
