# Copyright (c) 2025 @ SUPCON.
"""定义的Enum类型."""

from enum import Enum


class CustomStrEnumType(str, Enum):
    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'
    

class ChatLLMType(CustomStrEnumType):
    """ChatLLM Type enum class definition."""

    OpenAIChat = "openai_chat"
    QwenChat = "qwen_chat"
    SupconChat = "supcon_chat"
    MockChat = "mock_chat"
    

class EmbeddingModelType(CustomStrEnumType):
    """Embedding Model Type enum class definition."""

    OpenAIEmbedding = "openai_embedding"
    SupconEmbedding = "supcon_embedding"
    MockEmbedding = "mock_embedding"


class ChunkingStrategyType(CustomStrEnumType):
    """ChunkStrategy class definition."""

    Tokens = "tokens"
    Sentences = "sentences"
