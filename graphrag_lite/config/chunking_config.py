# Copyright (c) 2025 @ SUPCON.
"""Chunking configuration."""

from pydantic import BaseModel, Field

from graphrag_lite._typing.enums import ChunkingStrategyType


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    chunk_size: int = Field(
        default=1200,
        description="Chunk Size."
    )
    overlap: int = Field(
        default=100,
        description="Overlap between adjacent chunks."
    )
    group_by_columns: list[str] = Field(
        default=["id"],
        description="Columns to group by.",
    )
    strategy: ChunkingStrategyType = Field(
        default=ChunkingStrategyType.Tokens,
        description="The chunking strategy.",
    )
    encoding_model: str = Field(
        default="cl100k_base",
        description="The encoding model.",
    )