


from enum import Enum


class ModelType(str, Enum):
    """LLMType enum class definition."""

    # Embeddings
    OpenAIEmbedding = "openai_embedding"
    SupconEmbedding = "supcon_embedding"

    # Chat Completion
    OpenAIChat = "openai_chat"
    QwenChat = "qwen_chat"
    SupconChat = "supcon_chat"


    # Debug
    MockChat = "mock_chat"
    MockEmbedding = "mock_embedding"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'