from pathlib import Path
from typing import Any

import pytest

import graphrag.config2.defaults as defs
from graphrag.config2 import (
    CacheConfig,
    ChunkingConfig,
    ClaimExtractionConfig,
    ClusterGraphConfig,
    CommunityReportsConfig,
    EmbedGraphConfig,
    EntityExtractionConfig,
    GlobalSearchConfig,
    GraphRagConfig,
    InputConfig,
    LLMParameters,
    LocalSearchConfig,
    ParallelizationParameters,
    ReportingConfig,
    SnapshotsConfig,
    StorageConfig,
    SummarizeDescriptionsConfig,
    TextEmbeddingConfig,
    UmapConfig,
)

MOCK_API_KEY = "1234567890ABCD"


def _get_default_reporting_config() -> ReportingConfig:
    args: dict[str, Any] = {
        "type": defs.REPORTING_TYPE,
        "base_dir": defs.REPORTING_BASE_DIR.replace("$$", "$"),
        "connection_string": None,
        "container_name": None,
        "storage_account_blob_url": None,
    }

    return ReportingConfig(**args)


def _get_default_storage_config() -> StorageConfig:
    args: dict[str, Any] = {
        "type": defs.STORAGE_TYPE,
        "base_dir": defs.STORAGE_BASE_DIR.replace("$$", "$"),
        "connection_string": None,
        "container_name": None,
        "storage_account_blob_url": None,
    }

    return StorageConfig(**args)


def _get_default_cache_config() -> CacheConfig:
    args: dict[str, Any] = {
        "type": defs.CACHE_TYPE,
        "base_dir": defs.CACHE_BASE_DIR,
        "connection_string": None,
        "container_name": None,
        "storage_account_blob_url": None,
    }

    return CacheConfig(**args)


def _get_default_input_config() -> InputConfig:
    args: dict[str, Any] = {
        "type": defs.INPUT_TYPE,
        "file_type": defs.INPUT_FILE_TYPE,
        "base_dir": defs.INPUT_BASE_DIR,
        "connection_string": None,
        "storage_account_blob_url": None,
        "container_name": None,
        "encoding": defs.INPUT_FILE_ENCODING,
        "file_pattern": defs.INPUT_TEXT_PATTERN.replace("$$", "$"),
        "file_filter": None,
        "source_column": None,
        "timestamp_column": None,
        "timestamp_format": None,
        "text_column": defs.INPUT_TEXT_COLUMN,
        "title_column": None,
        "document_attribute_columns": [],
    }

    return InputConfig(**args)


def _get_default_embed_graph_config() -> EmbedGraphConfig:
    args: dict[str, Any] = {
        "enabled": defs.NODE2VEC_ENABLED,
        "num_walks": defs.NODE2VEC_NUM_WALKS,
        "walk_length": defs.NODE2VEC_WALK_LENGTH,
        "window_size": defs.NODE2VEC_WINDOW_SIZE,
        "iterations": defs.NODE2VEC_ITERATIONS,
        "random_seed": defs.NODE2VEC_RANDOM_SEED,
        "strategy": None,
    }

    return EmbedGraphConfig(**args)


def _get_default_llm_parameters() -> LLMParameters:
    args: dict[str, Any] = {
        "api_key": None,
        "type": defs.LLM_TYPE,
        "model": defs.LLM_MODEL,
        "max_tokens": defs.LLM_MAX_TOKENS,
        "temperature": defs.LLM_TEMPERATURE,
        "top_p": defs.LLM_TOP_P,
        "n": defs.LLM_N,
        "request_timeout": defs.LLM_REQUEST_TIMEOUT,
        "api_base": None,
        "api_version": None,
        "organization": None,
        "proxy": None,
        "cognitive_services_endpoint": None,
        "deployment_name": None,
        "model_supports_json": None,
        "tokens_per_minute": defs.LLM_TOKENS_PER_MINUTE,
        "requests_per_minute": defs.LLM_REQUESTS_PER_MINUTE,
        "max_retries": defs.LLM_MAX_RETRIES,
        "max_retry_wait": defs.LLM_MAX_RETRY_WAIT,
        "sleep_on_rate_limit_recommendation": defs.LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION,
        "concurrent_requests": defs.LLM_CONCURRENT_REQUESTS,
    }

    return LLMParameters(**args)


def _get_default_parallelization_parameters() -> ParallelizationParameters:
    args: dict[str, Any] = {
        "stagger": defs.PARALLELIZATION_STAGGER,
        "num_threads": defs.PARALLELIZATION_NUM_THREADS,
    }

    return ParallelizationParameters(**args)


def _get_default_text_embedding_config() -> TextEmbeddingConfig:
    args: dict[str, Any] = {
        "llm": _get_default_llm_parameters(),
        "parallelization": _get_default_parallelization_parameters(),
        "async_mode": defs.ASYNC_MODE,
        "batch_size": defs.EMBEDDING_BATCH_SIZE,
        "batch_max_tokens": defs.EMBEDDING_BATCH_MAX_TOKENS,
        "target": defs.EMBEDDING_TARGET,
        "skip": [],
        "vector_store": None,
        "strategy": None,
    }

    return TextEmbeddingConfig(**args)


def _get_default_chunk_config() -> ChunkingConfig:
    args: dict[str, Any] = {
        "size": defs.CHUNK_SIZE,
        "overlap": defs.CHUNK_OVERLAP,
        "group_by_columns": defs.CHUNK_GROUP_BY_COLUMNS,
        "strategy": None,
        "encoding_model": None,
    }

    return ChunkingConfig(**args)


def _get_default_snapshot_config() -> SnapshotsConfig:
    args: dict[str, Any] = {
        "graphml": defs.SNAPSHOTS_GRAPHML,
        "raw_entities": defs.SNAPSHOTS_RAW_ENTITIES,
        "top_level_nodes": defs.SNAPSHOTS_TOP_LEVEL_NODES,
    }

    return SnapshotsConfig(**args)


def _get_default_entity_extraction_config() -> EntityExtractionConfig:
    args: dict[str, Any] = {
        "llm": _get_default_llm_parameters(),
        "parallelization": _get_default_parallelization_parameters(),
        "async_mode": defs.ASYNC_MODE,
        "prompt": None,
        "entity_types": defs.ENTITY_EXTRACTION_ENTITY_TYPES,
        "max_gleanings": defs.ENTITY_EXTRACTION_MAX_GLEANINGS,
        "strategy": None,
        "encoding_model": None,
    }

    return EntityExtractionConfig(**args)


def _get_default_summarize_descriptions_config() -> SummarizeDescriptionsConfig:
    args: dict[str, Any] = {
        "llm": _get_default_llm_parameters(),
        "parallelization": _get_default_parallelization_parameters(),
        "async_mode": defs.ASYNC_MODE,
        "prompt": None,
        "max_length": defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
        "strategy": None,
    }

    return SummarizeDescriptionsConfig(**args)


def _get_default_community_reports_config() -> CommunityReportsConfig:
    args: dict[str, Any] = {
        "llm": _get_default_llm_parameters(),
        "parallelization": _get_default_parallelization_parameters(),
        "async_mode": defs.ASYNC_MODE,
        "prompt": None,
        "max_length": defs.COMMUNITY_REPORT_MAX_LENGTH,
        "max_input_length": defs.COMMUNITY_REPORT_MAX_INPUT_LENGTH,
        "strategy": None,
    }

    return CommunityReportsConfig(**args)


def _get_default_claim_extraction_config() -> ClaimExtractionConfig:
    args: dict[str, Any] = {
        "llm": _get_default_llm_parameters(),
        "parallelization": _get_default_parallelization_parameters(),
        "async_mode": defs.ASYNC_MODE,
        "enabled": defs.CLAIM_EXTRACTION_ENABLED,
        "prompt": None,
        "description": defs.CLAIM_DESCRIPTION,
        "max_gleanings": defs.CLAIM_MAX_GLEANINGS,
        "strategy": None,
        "encoding_model": None,
    }

    return ClaimExtractionConfig(**args)


def _get_default_cluster_graph_config() -> ClusterGraphConfig:
    args: dict[str, Any] = {
        "max_cluster_size": defs.MAX_CLUSTER_SIZE,
        "strategy": None,
    }

    return ClusterGraphConfig(**args)


def _get_default_umap_config() -> UmapConfig:
    args: dict[str, Any] = {
        "enabled": defs.UMAP_ENABLED,
    }

    return UmapConfig(**args)


def _get_default_local_search_config() -> LocalSearchConfig:
    args: dict[str, Any] = {
        "text_unit_prop": defs.LOCAL_SEARCH_TEXT_UNIT_PROP,
        "community_prop": defs.LOCAL_SEARCH_COMMUNITY_PROP,
        "conversation_history_max_turns": defs.LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS,
        "top_k_entities": defs.LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES,
        "top_k_relationships": defs.LOCAL_SEARCH_TOP_K_RELATIONSHIPS,
        "temperature": defs.LOCAL_SEARCH_LLM_TEMPERATURE,
        "top_p": defs.LOCAL_SEARCH_LLM_TOP_P,
        "n": defs.LOCAL_SEARCH_LLM_N,
        "max_tokens": defs.LOCAL_SEARCH_MAX_TOKENS,
        "llm_max_tokens": defs.LOCAL_SEARCH_LLM_MAX_TOKENS,
    }

    return LocalSearchConfig(**args)


def _get_default_global_search_config() -> GlobalSearchConfig:
    args: dict[str, Any] = {
        "temperature": defs.GLOBAL_SEARCH_LLM_TEMPERATURE,
        "top_p": defs.GLOBAL_SEARCH_LLM_TOP_P,
        "n": defs.GLOBAL_SEARCH_LLM_N,
        "max_tokens": defs.GLOBAL_SEARCH_MAX_TOKENS,
        "data_max_tokens": defs.GLOBAL_SEARCH_DATA_MAX_TOKENS,
        "map_max_tokens": defs.GLOBAL_SEARCH_MAP_MAX_TOKENS,
        "reduce_max_tokens": defs.GLOBAL_SEARCH_REDUCE_MAX_TOKENS,
        "concurrency": defs.GLOBAL_SEARCH_CONCURRENCY,
    }

    return GlobalSearchConfig(**args)


@pytest.fixture
def default_config() -> GraphRagConfig:
    args: dict[str, Any] = {
        "api_key": MOCK_API_KEY,
        "root_dir": str(Path.cwd()),
        "reporting": _get_default_reporting_config(),
        "storage": _get_default_storage_config(),
        "cache": _get_default_cache_config(),
        "input": _get_default_input_config(),
        "embed_graph": _get_default_embed_graph_config(),
        "embeddings": _get_default_text_embedding_config(),
        "chunks": _get_default_chunk_config(),
        "snapshots": _get_default_snapshot_config(),
        "entity_extraction": _get_default_entity_extraction_config(),
        "summarize_descriptions": _get_default_summarize_descriptions_config(),
        "community_reports": _get_default_community_reports_config(),
        "claim_extraction": _get_default_claim_extraction_config(),
        "cluster_graph": _get_default_cluster_graph_config(),
        "umap": _get_default_umap_config(),
        "local_search": _get_default_local_search_config(),
        "global_search": _get_default_global_search_config(),
        "encoding_model": defs.ENCODING_MODEL,
        "skip_workflows": [],
    }

    return GraphRagConfig(**args)
