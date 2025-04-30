
"""LLM configuration"""

from dataclasses import asdict
from pathlib import Path

from devtools import pformat
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


class LLMParameters(TypedDict):
    """LLM参数."""

    temperature: float = 0.7
    top_p: float = 0.2
    max_tokens: int = 1200


class LLMConfig(BaseModel):
    """Contains default configurations for LLM."""

    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json(indent=4)
    
    model_api: str | None = Field(
        default=None,
        description="LLM api",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name.",
    )
    model_parameters: LLMParameters = Field(
        default=LLMParameters(),
        description="LLM参数",
    )

    max_retries = Field(
        default=3,
        description="LLM最大重试次数.",
    )