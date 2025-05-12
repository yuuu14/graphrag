# Copyright (c) 2025 @ SUPCON.
"""LLM configuration."""

from devtools import pformat
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

from graphrag_lite._typing.errors import TooManyRetriesError


class LLMParameters(BaseModel):
    """LLM参数."""

    temperature: float = 0.7
    top_p: float = 0.2
    max_tokens: int = 1200

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )


class LLMConfig(BaseModel):
    """LLM配置."""

    model_api: str = Field(default="http://127.0.0.1:13443/v1/chat/completions")
    model_id: str | int | None = Field(default=None, description="SupChat后端传的模型ID.")
    model_name: str | None = Field(default=None, alias="model", description="模型名称.")
    model_parameters: LLMParameters = Field(default_factory=LLMParameters, description="模型参数.")
    max_retries: int = Field(default=3, description="最大重试次数.")
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        validate_assignment=True,
        extra="allow",
        exclude_none=True,
    )
        
    def _validate_max_retries(self):
        """Make sure max_retries <= 5."""
        if self.max_retries > 5:
            raise TooManyRetriesError(max_retries=self.max_retries)
        
    @model_validator(mode="after")
    def _validate_model(self):
        self._validate_max_retries()
        return self
