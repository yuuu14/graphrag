# Copyright (c) 2025 @ SUPCON.
"""定义的QwenChatLLM."""
# https://python.langchain.com/docs/how_to/custom_llm/
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import GenerationChunk

from graphrag_lite.config.llm_config import LLMConfig
from graphrag_lite.language_core.base import AsyncChatClient, ChatClient
from graphrag_lite.language_core.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatLLM,
)


class QwenChatLLM(ChatLLM):
    """本地部署的Qwen2.5-72B."""
    
    config: LLMConfig

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        """Async version of the _call method.

        The default implementation delegates to the synchronous _call method using
        `run_in_executor`. Subclasses that need to provide a true async implementation
        should override this method to reduce the overhead of using `run_in_executor`.

        Args
        ----
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.
                - *history_messages*: List, e.g. [{"role: "assistant", "content": "ANSWER"}]

        Returns
        -------
            The model output as a string. SHOULD NOT include the prompt.
        """
        history = kwargs.get("history_messages", [])
        messages = [*history, {"role": "user", "content": prompt}]
        stream = False
        args = {
            "data": {
                "messages": messages,
                "model": self.config.model_name,
                "stream": stream,
                **self.config.model_parameters,
            },
            "url": self.config.model_api,
            "stream": stream,
        }

        aclient = AsyncChatClient(response_cls=ChatCompletion)
        response: ChatCompletion = await aclient.chat(**args)
        return response.choices[0].message.content

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args
        ----
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at
                the first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments.
            - *history_messages*: List, e.g. [{"role: "assistant", "content": "ANSWER"}]

        Returns
        -------
            The model output as a string.
        """
        if stop is not None:
            msg = "stop kwargs are not permitted."
            raise ValueError(msg)
        
        history = kwargs.get("history_messages", [])
        messages = [*history, {"role": "user", "content": prompt}]
        stream = False
        args = {
            "data": {
                "messages": messages,
                "model": self.config.model_name,
                "stream": stream,
                **self.config.model_parameters,
            },
            "url": self.config.model_api,
            "stream": stream,
        }

        client = ChatClient(response_cls=ChatCompletion)
        response: ChatCompletion = client.chat(**args)
        return response.choices[0].message.content

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream the LLM on the given prompt in Async mode.

        The default implementation uses the synchronous _stream method and wraps it in
        an async iterator. Subclasses that need to provide a true async implementation
        should override this method.

        Args
        ----
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.
                - *history_messages*: List, e.g. [{"role: "assistant", "content": "ANSWER"}]

        Returns
        -------
            An async iterator of GenerationChunks.
        """
        history = kwargs.get("history_messages", [])
        messages = [*history, {"role": "user", "content": prompt}]
        stream = True
        args = {
            "data": {
                "messages": messages,
                "model": self.config.model_name,
                "stream": stream,
                **self.config.model_parameters,
            },
            "url": self.config.model_api,
            "stream": stream,
        }
        # make sure stream==True
        if not stream:
            print("WARNING: CANNOT stream data when `stream==False`, setting `stream:=True`now")  # noqa: T201
            args["data"]["stream"] = True

        aclient = AsyncChatClient(response_cls=ChatCompletionChunk)

        chat_responses: AsyncIterator[ChatCompletionChunk] = await aclient.chat(**args)
        async for response in chat_responses:
            choice = response.choices[0]
            if choice.delta.content is None:
                continue
            chunk = GenerationChunk(text=choice.delta.content)
            
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
        
    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at
                the first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments.
                - *history_messages*: List, e.g. [{"role: "assistant", "content": "ANSWER"}]

        Returns:
            An iterator of GenerationChunks.
        """  # noqa: D406, D407
        history = kwargs.get("history_messages", [])
        messages = [*history, {"role": "user", "content": prompt}]
        stream = True
        args = {
            "data": {
                "messages": messages,
                "model": self.config.model_name,
                "stream": stream,
                **self.config.model_parameters,
            },
            "url": self.config.model_api,
            "stream": stream,
        }

        client = ChatClient(response_cls=ChatCompletionChunk)

        chat_responses: Iterator[ChatCompletionChunk] = client.chat(**args)
        for response in chat_responses:
            choice = response.choices[0]
            if choice.delta.content is None:
                continue
            chunk = GenerationChunk(text=choice.delta.content)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "QwenChatLLM"
        }
    
    @property
    def _llm_string(self, stop: list[str] | None = None) -> str:
        """Return a string representation of the LLM configuration."""
        params = self.dict()
        params["stop"] = stop
        return str(sorted(params.items()))

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "Qwen2.5-72B"