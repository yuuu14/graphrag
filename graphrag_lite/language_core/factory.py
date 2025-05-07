# Copyright (c) 2025 @ SUPCON.
"""LLM和Embedding Model工厂."""

from langchain_core.runnables import Runnable
from langchain_core.runnables.retry import RunnableRetry

from graphrag_lite._typing.enums import ChatLLMType, EmbeddingModelType
from graphrag_lite.config.embedding_config import EmbeddingConfig
from graphrag_lite.config.llm_config import LLMConfig
from graphrag_lite.language_core import chat_llms, embedding_models
from graphrag_lite.language_core.types import EmbeddingModel


def create_chat_runnable(
    chat_llm_type: str | ChatLLMType,
    config: LLMConfig
) -> Runnable:
    """返回具有重试机制的ChatLLM."""
    if isinstance(chat_llm_type, str):
        if chat_llm_type not in [_type.value for _type in ChatLLMType]:
            msg = f"无效的chat_llm_type: {chat_llm_type}"
            raise NotImplementedError(msg)
        chat_llm_type = ChatLLMType(chat_llm_type)
    llm_cls = chat_llms[chat_llm_type]
    llm = llm_cls(config=config)
    return llm.with_retry(stop_after_attempt=config.max_retries)

def create_embedding_model(
    embedding_model_type: str | EmbeddingModelType,
    config: EmbeddingConfig
) -> EmbeddingModel:
    """返回embedding模型."""
    if isinstance(embedding_model_type, str):
        if embedding_model_type not in [_type.value for _type in EmbeddingModelType]:
            msg = f"无效的embedding_model_type: {embedding_model_type}"
            raise NotImplementedError(msg)
        embedding_model_type = EmbeddingModelType(embedding_model_type)
    embedding_model_cls = embedding_models[embedding_model_type]
    return embedding_model_cls(config)

def test_chat_llm_functional():
    ...