from typing import Any
from langchain_core.runnables import Runnable
from .types import ChatLLM


def create_chat_runnable(config: dict[str, Any]) -> Runnable:
    llm = ChatLLM(config)
    if config.get("max_retries", 0) > 0:
        return llm.with_retry()
    return llm
