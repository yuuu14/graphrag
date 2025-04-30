
"""Custom Client for LLM and Embedding Models."""
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import (
    TypeVar,
)

import aiohttp
import httpx

T = TypeVar("T")

class AsyncBaseClient:
    """异步BaseClient."""

    client_timeout = aiohttp.ClientTimeout(connect=10, sock_read=45)

    async def _arequest_raw(self, **kwargs):
        """Non-streaming async request."""
        async with aiohttp.ClientSession(
            timeout=self.client_timeout,
            raise_for_status=True,
        ) as session, session.request(**kwargs) as r:
            try:
                response = await r.json()
            except Exception as e:
                msg = f"LLM异步请求失败：{type(e)}, {e}"
                raise Exception(msg)
        return response
    
    async def _request(
        self,
        cls: type[T],
        stream: bool = False,
        **kwargs,
    ) -> T | AsyncIterator[T]:
        
        if stream:
            async def inner():
                async with aiohttp.ClientSession(
                    timeout=self.client_timeout,
                    raise_for_status=True,
                ) as session, session.request(**kwargs) as r:
                        # async for chunk in r.content.iter_any():
                        # ref
                        # - https://stackoverflow.com/questions/59681726/how-to-read-lines-of-a-streaming-api-with-aiohttp-in-python
                        async for line in r.content:
                            data_list = line.decode().strip().split("data:")
                            for part in data_list:
                                if not part or part.strip() == "[DONE]":
                                    continue
                                try:
                                    part: dict = json.loads(part)
                                except Exception as e:
                                    print(f"加载JSON失败！CHUNK: {part}，报错：{type(e), e}")
                                    raise Exception(e)
                                if err := part.get('error'):
                                    raise Exception(f"返回字段包含error：{err}")
                                try:
                                    yield cls(**part)
                                except Exception as e:
                                    raise Exception(f"LLM异步流式返回结果：{part}，报错：{type(e)}, {e}")
            return inner()
        
        response = await self._arequest_raw(**kwargs)
        return cls(**response)


class BaseClient:
    """Derived from ollama Client."""

    client_timeout = httpx.Timeout(timeout=60, connect=10, read=45, write=30)

    def _request_raw(self, **kwargs):

        r = httpx.request(timeout=self.client_timeout, **kwargs)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Exception(e.response.text, e.response.status_code) from e
        return r

    def _request(
        self,
        cls: type[T],
        stream: bool = False,
        **kwargs,
    ) -> T | Iterator[T]:
        if stream:
            def inner():
                with httpx.stream(timeout=self.client_timeout, **kwargs) as r:
                    try:
                        r.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        e.response.read()
                        raise Exception(e.response.text, e.response.status_code) from e
                    for line in r.iter_lines():
                        data_ = line.removeprefix("data:")
                        if not data_ or data_.strip() == "[DONE]":
                            continue
                        part: dict = json.loads(data_)
                        if err := part.get("error"):
                            print(f"返回字段包含error：{err}")
                            raise Exception(err)
                        try:
                            yield cls(**part)
                        except Exception as e:
                            msg = f"LLM流式返回结果：{part}，报错：{type(e)}, {e}"
                            raise Exception(msg) from e
            return inner()
        
        return cls(**self._request_raw(**kwargs).json())
    

@dataclass
class AsyncChatClient(AsyncBaseClient):
    """异步ChatClient."""

    response_cls: type[T]

    async def chat( # prompt, history_messages,
        self,
        data: dict,
        url: str = "",
        headers: dict | None = None,
        stream: bool = True,
    ) -> T | AsyncIterator[T]:
        """Create a chat response. If `stream==True`, return a ChatResponse AsyncIterator."""
        if headers is None:
            headers = {"Content-Type": "application/json"}
        return await self._request(
            self.response_cls,
            method="POST",
            url=url,
            headers=headers,
            data=json.dumps(data),
            stream=stream,
        )
    

@dataclass
class ChatClient(BaseClient):
    """Custom ChatClient."""

    response_cls: type[T]

    def chat( # prompt, history_messages,
        self,
        data: dict,
        url: str = "",
        headers: dict | None = None,
        stream: bool = True,
    ) -> T | Iterator[T]:
        """Create a chat response. If `stream==True`, return a ChatResponse Iterator."""
        if headers is None:
            headers = {"Content-Type": "application/json"}
        return self._request(
            self.response_cls,
            method="POST",
            url=url,
            headers=headers,
            data=json.dumps(data),
            stream=stream,
        )


@dataclass
class AsyncEmbedClient(AsyncBaseClient):
    """异步EmbedClient."""

    response_cls: type[T]

    def embed(
        self,
        data: dict,
        url: str,
        headers: dict | None = None,
        stream: bool = True,
    ) -> T | Iterator[T]:
        if headers is None:
            headers = {"Content-Type": "application/json"}
        return self._request(
            self.response_cls,
            method="POST",
            url=url,
            headers=headers,
            data=json.dumps(data),
            stream=stream,
        )


@dataclass
class EmbedClient(BaseClient):
    """Custom EmbedClient."""

    response_cls: type[T]

    def embed(
        self,
        data: dict,
        url: str,
        headers: dict | None = None,
        stream: bool = True,
    ) -> T | Iterator[T]:
        if headers is None:
            headers = {"Content-Type": "application/json"}
        return self._request(
            self.response_cls,
            method="POST",
            url=url,
            headers=headers,
            data=json.dumps(data),
            stream=stream,
        )
