
from graphrag_lite.callbacks.callback_handler import AsyncOperationCallbackHandler
from pydantic import Field, ConfigDict
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import BaseStore, InMemoryStore

from graphrag_lite.utils.progress import ProgressHolder

class OperationRunnableConfig(RunnableConfig):
    store: BaseStore | None = Field(default=None)
    callback_handler: AsyncOperationCallbackHandler | None = Field(default=None)
    progress_holders: dict[str, ProgressHolder] | None = Field(default=None)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )