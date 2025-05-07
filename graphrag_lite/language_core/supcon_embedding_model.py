# Copyright (c) 2025 @ SUPCON.
"""定义的本地Embedding Model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphrag_lite.config.embedding_config import EmbeddingConfig
from graphrag_lite.language_core.base import AsyncEmbedClient, EmbedClient
from graphrag_lite.language_core.types import EmbeddingModel, SupconEmbedResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@dataclass
class SupconEmbeddingModel(EmbeddingModel):
    """本地部署的Embedding model."""

    config: EmbeddingConfig

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args
        ----
            texts: List of text to embed.

        Returns
        -------
            List of embeddings.
        """
        args = {
            "data": {
                "input": texts,
                "model": self.config.model_name,
                "isNorm": self.config.normalized,
            },
            "url": self.config.model_api,
            "headers": {"Content-Type": "application/json"},
            "stream": True,
        }
        client = EmbedClient(response_cls=SupconEmbedResponse)
        embed_response: Iterator[SupconEmbedResponse] = client.embed(**args)

        embedding_list = [
            embed_result["embedding"]
            for response in embed_response  # embed_response
            for embed_result in response.data
        ]
        return embedding_list  # noqa: RET504
    
    async def aembed_documents(self, texts):
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """  # noqa: D406, D407
        args = {
            "data": {
                "input": texts,
                "model": self.config.model_name,
                "isNorm": self.config.normalized,
            },
            "url": self.config.model_api,
            "headers": {"Content-Type": "application/json"},
            "stream": True,
        }
        aclient = AsyncEmbedClient(response_cls=SupconEmbedResponse)
        embed_response: AsyncIterator[SupconEmbedResponse] = await aclient.embed(**args)

        embedding_list = [
            embed_result["embedding"]
            async for response in embed_response  # embed_response
            for embed_result in response.data
        ]
        return embedding_list  # noqa: RET504

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """  # noqa: D406, D407
        return self.embed_documents([text])[0]
    
    async def aembed_query(self, text):
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """  # noqa: D406, D407
        get_aembed_documents = await self.aembed_documents([text])
        return get_aembed_documents[0]
    