# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Load llm utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from graphrag.config.enums import LLMType

from .limiting import (
    LLMLimiter,
    create_tpm_rpm_limiters,
)
from .mock import MockCompletionLLM
from .openai import (
    OpenAIConfiguration,
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embedding_llm,
)

if TYPE_CHECKING:
    from datashaper import VerbCallbacks

    from graphrag.index.cache import PipelineCache
    from graphrag.index.typing import ErrorHandlerFn

    from .types import (
        CompletionLLM,
        EmbeddingLLM,
        LLMCache,
    )

log = logging.getLogger(__name__)

_semaphores: dict[str, asyncio.Semaphore] = {}
_rate_limiters: dict[str, LLMLimiter] = {}


class LLMFactory:
    """A factory class for creating llm's."""

    llm_types: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, llm_type: str, llm: type):
        """Register a LLM type."""
        cls.llm_types[llm_type] = llm

    @classmethod
    def create_llm(
        cls,
        name: str,
        llm_type: LLMType,
        callbacks: VerbCallbacks,
        cache: PipelineCache | None,
        llm_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> CompletionLLM:
        """Get the vector store type from a string."""
        on_error = _create_error_handler(callbacks)
        if cache is not None:
            cache = cache.child(name)
        match llm_type:
            case LLMType.AzureOpenAIChat:
                return _load_openai_chat_llm(
                    on_error, cache, llm_config or {}, is_azure=True
                )  # type: ignore
            case LLMType.OpenAIChat:
                return _load_openai_chat_llm(
                    on_error, cache, llm_config or {}, is_azure=False
                )  # type: ignore
            case LLMType.StaticResponse:
                return _load_static_response(on_error, cache, llm_config or {})  # type: ignore
            case _:
                if llm_type in cls.llm_types:
                    return cls.llm_types[llm_type](**kwargs)
                msg = f"Unknown llm type: {llm_type}"
                raise ValueError(msg)


class EmbeddingLLMFactory:
    """A factory class for creating embedding llm's."""

    llm_types: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, llm_type: str, llm: type):
        """Register a LLM type."""
        cls.llm_types[llm_type] = llm

    @classmethod
    def create_llm(
        cls,
        name: str,
        llm_type: LLMType,
        callbacks: VerbCallbacks,
        cache: PipelineCache | None,
        llm_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> EmbeddingLLM:
        """Get the embedding llm type from a string."""
        on_error = _create_error_handler(callbacks)
        if cache is not None:
            cache = cache.child(name)
        match llm_type:
            case LLMType.AzureOpenAIEmbedding:
                return _load_openai_embeddings_llm(
                    on_error, cache, llm_config or {}, is_azure=True
                )  # type: ignore
            case LLMType.OpenAIEmbedding:
                return _load_openai_embeddings_llm(on_error, cache, llm_config or {})  # type: ignore
            case LLMType.StaticResponse:
                return _load_static_response(on_error, cache, llm_config or {})  # type: ignore
            case _:
                if llm_type in cls.llm_types:
                    return cls.llm_types[llm_type](**kwargs)
                msg = f"Unknown llm type: {llm_type}"
                raise ValueError(msg)


def _create_error_handler(callbacks: VerbCallbacks) -> ErrorHandlerFn:
    def on_error(
        error: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ) -> None:
        callbacks.error("Error calling LLM", error, stack, details)

    return on_error


def _load_openai_chat_llm(
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    config: dict[str, Any],
    is_azure=False,
):
    return _create_openai_chat_llm(
        OpenAIConfiguration({
            # Set default values
            **_get_base_config(config),
            "model": config.get("model", "gpt-4-turbo-preview"),
            "deployment_name": config.get("deployment_name"),
            "temperature": config.get("temperature", 0.0),
            "frequency_penalty": config.get("frequency_penalty", 0),
            "presence_penalty": config.get("presence_penalty", 0),
            "top_p": config.get("top_p", 1),
            "max_tokens": config.get("max_tokens"),
            "n": config.get("n"),
        }),
        on_error,
        cache,
        is_azure,
    )


def _create_openai_chat_llm(
    configuration: OpenAIConfiguration,
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    is_azure=False,
) -> CompletionLLM:
    """Create an openAI chat llm."""
    client = create_openai_client(configuration=configuration, is_azure=is_azure)
    limiter = _create_limiter(configuration)
    semaphore = _create_semaphore(configuration)
    return create_openai_chat_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


def _load_openai_embeddings_llm(
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    config: dict[str, Any],
    is_azure=False,
):
    # TODO: Inject Cache
    return _create_openai_embeddings_llm(
        OpenAIConfiguration({
            **_get_base_config(config),
            "model": config.get(
                "embeddings_model", config.get("model", "text-embedding-3-small")
            ),
            "deployment_name": config.get("deployment_name"),
        }),
        on_error,
        cache,
        is_azure,
    )


def _create_openai_embeddings_llm(
    configuration: OpenAIConfiguration,
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    is_azure=False,
) -> EmbeddingLLM:
    """Create an openAI embeddings llm."""
    client = create_openai_client(configuration=configuration, is_azure=is_azure)
    limiter = _create_limiter(configuration)
    semaphore = _create_semaphore(configuration)
    return create_openai_embedding_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


def _get_base_config(config: dict[str, Any]) -> dict[str, Any]:
    api_key = config.get("api_key")

    return {
        # Pass in all parameterized values
        **config,
        # Set default values
        "api_key": api_key,
        "api_base": config.get("api_base"),
        "api_version": config.get("api_version"),
        "organization": config.get("organization"),
        "proxy": config.get("proxy"),
        "max_retries": config.get("max_retries", 10),
        "request_timeout": config.get("request_timeout", 60.0),
        "model_supports_json": config.get("model_supports_json"),
        "concurrent_requests": config.get("concurrent_requests", 4),
        "encoding_model": config.get("encoding_model", "cl100k_base"),
        "cognitive_services_endpoint": config.get("cognitive_services_endpoint"),
    }


def _load_static_response(
    _on_error: ErrorHandlerFn, _cache: PipelineCache, config: dict[str, Any]
) -> CompletionLLM:
    return MockCompletionLLM(config.get("responses", []))


def _create_limiter(configuration: OpenAIConfiguration) -> LLMLimiter:
    limit_name = configuration.model or configuration.deployment_name or "default"
    if limit_name not in _rate_limiters:
        tpm = configuration.tokens_per_minute
        rpm = configuration.requests_per_minute
        log.info("create TPM/RPM limiter for %s: TPM=%s, RPM=%s", limit_name, tpm, rpm)
        _rate_limiters[limit_name] = create_tpm_rpm_limiters(configuration)
    return _rate_limiters[limit_name]


def _create_semaphore(configuration: OpenAIConfiguration) -> asyncio.Semaphore | None:
    limit_name = configuration.model or configuration.deployment_name or "default"
    concurrency = configuration.concurrent_requests

    # bypass the semaphore if concurrency is zero
    if not concurrency:
        log.info("no concurrency limiter for %s", limit_name)
        return None

    if limit_name not in _semaphores:
        log.info("create concurrency limiter for %s: %s", limit_name, concurrency)
        _semaphores[limit_name] = asyncio.Semaphore(concurrency)

    return _semaphores[limit_name]
