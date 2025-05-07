# Copyright (c) 2025 @ SUPCON.
"""Embedding Model configuration."""

from devtools import pformat
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    """Contains default configurations for EmbeddingModel."""

    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json(indent=4)
    
    model_api: str | None = Field(
        default=None,
        description="Embedding model api, if not initialized then raise NotImplementError.",
    )

    model_name: str

    normalized: bool = Field(
        default=False,
        description="是否对向量进行归一化",
    )
