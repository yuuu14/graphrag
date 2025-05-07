# Copyright (c) 2025 @ SUPCON.
"""LLM configuration."""

from devtools import pformat
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from graphrag_lite._typing.errors import TooManyRetriesError


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

    model_id: int | None = Field(
        default=None,
        description="SupChat后端传的模型ID.",
    )

    model_name: str | None = Field(
        default=None,
        description="LLM name.",
    )

    model_parameters: LLMParameters = Field(
        default_factory=LLMParameters,
        description="LLM参数.",
    )

    max_retries: int = Field(
        default=3,
        description="LLM最大重试次数.",
    )

    def _validate_max_retries(self):
        """Make sure max_retries <= 5."""
        if self.max_retries > 5:
            raise TooManyRetriesError(max_retries=self.max_retries)
        
    @model_validator(mode="after")
    def _validate_model(self):
        self._validate_max_retries()
        return self

