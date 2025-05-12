import asyncio
from pydantic import Field
from langchain_core.runnables import RunnableSerializable
from typing import Any
from langchain_core.runnables.utils import Input, Output

from graphrag_lite.config.operation_config import OperationRunnableConfig
from graphrag_lite.workflow.operation import AsyncOperation, AsyncCompositeOperation
from graphrag_lite.utils.progress import ProgressHolder


class AsyncProcessingChain(RunnableSerializable):
    operations: list[AsyncOperation | AsyncCompositeOperation] = Field(default_factory=list)

    def invoke(self, input, config=None, **kwargs):
        return asyncio.run(self.ainvoke(input, config, **kwargs))
    async def ainvoke(
        self,
        input: Input,
        config: OperationRunnableConfig | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:

        current_input = input
        results = {}
        config = config or OperationRunnableConfig()
        progress_id = kwargs.get("progress_id") or "MAIN_CHAIN_PROGRESS"
        progress_holder = config["progress_holders"].get(progress_id)
        acallback = config["callback_handler"]
        for op in self.operations:
            # 必须传递完整的 config 对象
            result = await op.ainvoke(current_input, config=config)
            
            results[op.operation_id] = result  # 这里现在可以正确访问 operation_id
            current_input = result

            # if acallback and progress_holder:
            #     await acallback.on_chain_progress(
            #         operation_id="主流程",
            #         progress=progress_holder,
            #         message=f"Completed {progress_holder.processed}/{progress_holder.total} operations"
            #     )

        return results