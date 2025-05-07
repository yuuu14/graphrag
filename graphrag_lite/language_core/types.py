# Copyright (c) 2025 @ SUPCON.
"""LLM和Embedding Model的模型类型, 输入以及返回类型."""

from typing import ClassVar, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import Field
from typing_extensions import TypedDict

from graphrag_lite._typing.types import SubscriptableBaseModel

ChatLLM = LLM
EmbeddingModel = Embeddings


class SupconEmbedResult(TypedDict):
    embedding: list[float]
    index: int
    object: str = "embedding"

class SupconEmbedResponse(SubscriptableBaseModel):
    """Response returned by embed requests."""

    data: list[SupconEmbedResult]
    model: str
    object: str
    usage: dict

# Chat input & output
class Context(TypedDict):
    """定义的Text格式."""

    type: Literal["text", "think"] = "text"
    context: str

class SupconChatHistory(TypedDict):
    """定义的聊天历史格式."""
    
    queryList: list[Context]
    answerList: list[Context]

class SupconChatModelParameters(TypedDict, total=False):
    """由后端确定的模型参数."""

    id: int = Field(description="后端传的模型ID")
    temperature: float = 0.7
    top_p: float = 0.2
    max_tokens: int = 1200
    stream: bool = True
    model: str | None = None

class SupconChatData(SubscriptableBaseModel):
    """特殊的Chat-Completion Input格式."""

    queryList: list[Context]  # noqa: N815
    history: ClassVar[list[SupconChatHistory]] = []
    model: SupconChatModelParameters

class SupconChatResponse(SubscriptableBaseModel):
    """Response returned by custom chat-completion requests."""
    
    answerList: list[Context]  # noqa: N815
    finishFlag: bool  # noqa: N815
    algorithmSpecMap: dict | None = Field(  # noqa: N815
        default=None,
        description="传tenantId OR 返回源文件实体 etc.",
    )
    errCode: str | int | None  # noqa: N815
    errMsg: str | None  # noqa: N815


__all__ = [
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatLLM",
    "Context",
    "EmbeddingModel",
    "SupconChatData",
    "SupconChatHistory",
    "SupconChatModelParameters",
    "SupconChatResponse",
]