# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cache functions for the GraphRAG update module."""

from graphrag.config.pipeline import (
    PipelineCacheConfigTypes,
    PipelineMemoryCacheConfig,
)
from graphrag.index.cache import load_cache
from graphrag.index.cache.pipeline_cache import PipelineCache


def _create_cache(
    config: PipelineCacheConfigTypes | None, root_dir: str
) -> PipelineCache:
    return load_cache(config or PipelineMemoryCacheConfig(), root_dir=root_dir)
