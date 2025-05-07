# Copyright (c) 2025 @ SUPCON.
"""GraphRAG configuration."""

from dataclasses import asdict
from pathlib import Path

from devtools import pformat
from pydantic import BaseModel, Field, model_validator


class GraphRAGConfig(BaseModel):
    """Contains default configurations for GraphRAGLite."""

    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json(indent=4)