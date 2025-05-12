
"""Operation class."""

from typing import Any
from collections.abc import AsyncGenerator
from pydantic import Field
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Input, Output
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager

import asyncio
from langchain_core.stores import BaseStore, InMemoryStore

from graphrag_lite.config.operation_config import OperationRunnableConfig


class AsyncOperation(RunnableSerializable[Input, Output]):
    operation_id: str = Field(alias="operation_id")  # 显式声明 Pydantic 字段
    progress_id: str = Field(alias="progress_id", description="所在Chain的Progress ID")
    """所在Chain的Progress ID, 同一层Chain的operation有同样的progress_id"""
    chain_id: str = Field(default="CHAIN", description="Operation所在Chain ID")

    class Config:  # noqa: D106
        arbitrary_types_allowed = True  # 允许自定义类型
        populate_by_name = True

    def __init__(self, operation_id: str, progress_id: str = "MAIN_CHAIN_PROGRESS", **kwargs):
        super().__init__(operation_id=operation_id, progress_id=progress_id, **kwargs)

    async def _process(
        self,
        input: Input,
        config: OperationRunnableConfig) -> Output:
        """处理logging/stats信息."""
        raise NotImplementedError

    def invoke(self, input, config=None, **kwargs):
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def ainvoke(
        self,
        input: Input,
        config: OperationRunnableConfig | None = None,
        **kwargs: Any
    ) -> Output:
        # 获取上下文配置
        store = config["store"]
        acallback = config["callback_handler"]
        progress_holder = config["progress_holders"].get(self.progress_id)
        print(f"Operation {self.operation_id}, Progress ID: {self.progress_id}")
        try:
            if acallback:
                await acallback.on_chain_start(self.operation_id, {"input": input})

            # 从 store 加载数据
            # 处理逻辑
            result = await self._process(input, config)
            
            # 保存结果到 store
            await store.amset([("result_key", result)])

            # 进度更新
            await acallback.on_chain_progress(
                operation_id=self.operation_id,
                chain_id=self.chain_id,
                progress=progress_holder,
                message="Processing completed",
            )

            # 触发完成回调
            if acallback:
                await acallback.on_chain_end(self.operation_id, {"result": result})

        except Exception as e:
            # TODO: define OperationError
            await acallback.on_chain_error(self.operation_id, e)
            raise
        else:
            return result
        

class AsyncCompositeOperation(RunnableSerializable):
    """支持嵌套子操作的复合操作基类."""

    operation_id: str = Field(..., alias="operation_id")
    sub_operations: list[RunnableSerializable] = Field(default_factory=list)

    async def _process_children(
        self,
        input: Any,
        config: OperationRunnableConfig,
    ) -> AsyncGenerator[Any, Any, Any]:
        """执行子操作流水线"""
        current_input = input
        acallback = config["callback_handler"]
        for idx, op in enumerate(self.sub_operations):
            # 执行子操作
            result = await op.ainvoke(current_input, config=config)
            yield result
            current_input = result

            # 更新进度
            if acallback:
                progress = (idx + 1) / len(self.sub_operations) * 100
                await acallback.on_chain_progress(
                    self.operation_id,
                    progress,
                    f"子操作 {op.operation_id} 完成"
                )

    async def _aggregate_results(self, results: list[Any]) -> Any:
        """聚合子操作结果（可被子类覆盖）"""
        return {op.operation_id: res for op, res in zip(self.sub_operations, results, strict=True)}