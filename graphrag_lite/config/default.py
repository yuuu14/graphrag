# Copyright (c) 2025 @ SUPCON.
"""Default configurations."""

from dataclasses import asdict
from pathlib import Path

from devtools import pformat
from pydantic import BaseModel, Field, model_validator

from graphrag_lite.config.embedding_config import EmbeddingConfig
from graphrag_lite.config.llm_config import LLMConfig, LLMParameters


class GraphRAGDefaultConfig(BaseModel):
    """Contains default configurations for GraphRAGLite."""

    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json(indent=4)
    

QWEN_MODEL_API = "http://10.16.11.41:11112/v1/chat/completions"

default_qwen_llm_parameters = LLMParameters(
    temperature=0.7,
    top_p=0.1,
    max_tokens=1500,
)

default_qwen_llm_config = LLMConfig(
    model_api=QWEN_MODEL_API,
    model_name="local",
    model_parameters=default_qwen_llm_parameters,
    max_retries=3,
)

EMBEDDING_API = "http://10.16.11.40:1114/v1/embeddings"

default_supcon_embedding_config = EmbeddingConfig(
    model_api=EMBEDDING_API,
    model_name="m3e",
)