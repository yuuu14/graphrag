
from typing import Any
import logging
from collections.abc import AsyncGenerator
from pydantic import Field
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Input, Output
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager

from graphrag_lite.utils.progress import ProgressHolder


logger = logging.getLogger(__name__)

class AsyncOperationCallbackHandler(AsyncCallbackHandler):
    async def on_chain_start(
        self,
        operation_id: str,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """异步处理操作开始"""
        msg = f"🚀 Operation {operation_id} started with inputs: {inputs}"
        logger.info(msg)
        print(msg)

    async def on_chain_progress(
        self,
        operation_id: str,
        chain_id: str,
        progress: ProgressHolder | None = None,
        message: str | None = None,
        **kwargs: Any,
    ) -> None:
        """异步处理进度更新"""
        if progress:
            progress.processed += 1
            progress.progress = (progress.processed / progress.total) * 100
            msg = f"📈 当前操作为 {operation_id}, Chain: {chain_id}, 进度: {progress.progress:.2f}% - {message}"
            logger.info(msg)
            print(msg)
        else:
            logger.warning("No ProgressHolder set.")

    async def on_chain_error(
        self,
        operation_id: str,
        error: BaseException,  # try to trace line in operation
        **kwargs: Any,
    ) -> None:
        """log错误"""
        msg = f"🔥!!! Error in {operation_id}: {error!s}"
        logger.error(msg)
        print(msg)
        # write to log file
        

    async def on_chain_end(
        self,
        operation_id: str,
        outputs: dict[str, Any],
        **kwargs: Any
    ) -> None:
        msg = f"✅ Operation {operation_id} 完成. Outputs: {outputs}"
        logger.info(msg)
        print(msg)

    

    
    # asyncio.gather 处理多个chunk

# LLM Performance Callback
from langchain.callbacks.base import BaseCallbackHandler
import time
import os
import tiktoken

tiktoken_cache_dir = "/home/xiaoyu/tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

encoding_fn = tiktoken.get_encoding("cl100k_base")


class PerformanceCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.generated_tokens = 0

    # 记录LLM开始生成的时间
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()

    # 捕获第一个Token的到达时间
    def on_llm_new_token(self, token, **kwargs):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.generated_tokens += len(encoding_fn.encode(token))

    # 记录生成结束时间并计算指标
    def on_llm_end(self, response, **kwargs):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        ttft = self.first_token_time - self.start_time if self.first_token_time else 0
        token_rate = self.generated_tokens / total_time if total_time > 0 else 0

        print(f"""
        [性能报告]
        首Token响应时间(TTFT): {ttft:.3f}s
        总生成时间: {total_time:.3f}s
        生成Token总数: {self.generated_tokens}
        Token生成速率: {token_rate:.1f} tokens/s
        """)

