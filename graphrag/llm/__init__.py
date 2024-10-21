# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Datashaper OpenAI Utilities package."""

from .base import BaseLLM, CachingLLM, RateLimitingLLM
from .errors import RetriesExhaustedError
from .factory import EmbeddingLLMFactory, LLMFactory
from .limiting import (
    CompositeLLMLimiter,
    LLMLimiter,
    NoopLLMLimiter,
    TpmRpmLLMLimiter,
    create_tpm_rpm_limiters,
)
from .mock import MockChatLLM, MockCompletionLLM
from .openai import (
    OpenAIChatLLM,
    OpenAIClientTypes,
    OpenAIConfiguration,
    OpenAIEmbeddingsLLM,
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embedding_llm,
)
from .types import (
    LLM,
    CompletionInput,
    CompletionLLM,
    CompletionOutput,
    EmbeddingInput,
    EmbeddingLLM,
    EmbeddingOutput,
    ErrorHandlerFn,
    IsResponseValidFn,
    LLMCache,
    LLMConfig,
    LLMInput,
    LLMInvocationFn,
    LLMInvocationResult,
    LLMOutput,
    OnCacheActionFn,
    TextListSplitter,
    TextSplitter,
)

__all__ = [  # noqa: RUF022
    # LLM Types
    "LLM",
    "BaseLLM",
    "CachingLLM",
    "CompletionInput",
    "CompletionLLM",
    "CompletionOutput",
    "CompositeLLMLimiter",
    "EmbeddingInput",
    "EmbeddingLLM",
    "EmbeddingOutput",
    "TextListSplitter",
    "TextSplitter",
    # Callbacks
    "ErrorHandlerFn",
    "IsResponseValidFn",
    # Cache
    "LLMCache",
    "LLMConfig",
    # LLM I/O Types
    "LLMInput",
    "LLMInvocationFn",
    "LLMInvocationResult",
    "LLMLimiter",
    "LLMOutput",
    "MockChatLLM",
    # Mock
    "MockCompletionLLM",
    "NoopLLMLimiter",
    "OnCacheActionFn",
    "OpenAIChatLLM",
    "OpenAIClientTypes",
    # OpenAI
    "OpenAIConfiguration",
    "OpenAIEmbeddingsLLM",
    "RateLimitingLLM",
    # Errors
    "RetriesExhaustedError",
    "TpmRpmLLMLimiter",
    "create_openai_chat_llm",
    "create_openai_client",
    "create_openai_embedding_llm",
    # Limiters
    "create_tpm_rpm_limiters",
    # LLM Factory
    "LLMFactory",
    "EmbeddingLLMFactory",
]
