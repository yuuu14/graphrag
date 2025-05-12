# Copyright (c) 2025 @ SUPCON.
"""Custom Large Language Models and Embedding Models based on Langchain."""

from graphrag_lite._typing.enums import ChatLLMType, EmbeddingModelType
from graphrag_lite.language_core.mock_chat_llm import MockChatLLM
from graphrag_lite.language_core.openai_chat_llm import OpenaiChatLLM
from graphrag_lite.language_core.qwen_chat_llm import QwenChatLLM
from graphrag_lite.language_core.supcon_embedding_model import SupconEmbeddingModel
from graphrag_lite.language_core.types import ChatLLM, EmbeddingModel

chat_llms: dict[ChatLLMType, type[ChatLLM]] = {
    ChatLLMType.QwenChat: QwenChatLLM,
    ChatLLMType.MockChat: MockChatLLM,
}
embedding_models: dict[EmbeddingModelType, type[EmbeddingModel]] = {
    EmbeddingModelType.SupconEmbedding: SupconEmbeddingModel,
}

__all__ = [
    "MockChatLLM",
    "OpenaiChatLLM",
    "QwenChatLLM",
    "SupconEmbeddingModel",
    "chat_llms",
    "embedding_models",
]