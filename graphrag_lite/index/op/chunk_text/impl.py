# Copyright (c) 2025 @ SUPCON.
"""Implementation for chunk_text."""
from typing import Any, cast
from uuid import UUID
import pandas as pd
from graphrag_lite._typing.enums import ChunkingStrategyType
from graphrag_lite.config.chunking_config import ChunkingConfig
from graphrag.logger.progress import ProgressTicker, progress_ticker
from langchain_core.callbacks import BaseCallbackHandler
# impls



# Progress in Context
# Callback get Progress in Context
# https://python.langchain.com/docs/how_to/callbacks_attach/
class LoggingHandler(BaseCallbackHandler):
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Run when the Retriever starts running.

        Args
        ----
            serialized (Dict[str, Any]): The serialized Retriever.
            query (str): The query.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        print(f"Chain {serialized.get('name')} started, current run_id: {run_id}")

        return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running.

        Args:
            outputs (Dict[str, Any]): The outputs of the chain.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """



def chunk_text(
    input_df: pd.DataFrame,
    text_column: str,
    chunk_size: int,
    overlap: int,
    encoding_model: str,
    strategy: ChunkingStrategyType,
):

    chunking_fn = load_chunking_fn(strategy)

    input_count = len(input_df[text_column])

    config = ChunkingConfig(chunk_size=chunk_size, overlap=overlap, encoding_model=encoding_model)

    #!!! TODO: implement logging_callback
    tick = progress_ticker(..., input_count)

    return cast(
        "pd.Series",
        input_df.apply(
            lambda x: [res.text_chunk for res in chunking_fn((x), config, tick)],
            axis="columns",
        )
    )


# get chunking function by strategy
def load_chunking_fn(strategy: ChunkingStrategyType):
    """Load strategy method definition."""
    match strategy:
        case ChunkingStrategyType.Tokens:
            from graphrag_lite.index.op.chunk_text.strategies import run_tokens

            return run_tokens
        case ChunkingStrategyType.Sentences:
            # NLTK
            from graphrag.index.operations.chunk_text.bootstrap import bootstrap
            from graphrag_lite.index.op.chunk_text.strategies import run_sentences

            # TODO: `poetry add` all needed nltk models
            bootstrap()
            return run_sentences
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)


