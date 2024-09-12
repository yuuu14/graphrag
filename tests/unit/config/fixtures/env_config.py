from typing import Any
from unittest import mock

import pytest

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
    ReportingConfig,
    SnapshotsConfig,
    StorageConfig,
    SummarizeDescriptionsConfig,
    TextEmbeddingConfig,
)


@pytest.fixture
def all_env_vars() -> dict[str, str]:
    return {
        "GRAPHRAG_ASYNC_MODE": "asyncio",
        "GRAPHRAG_CACHE_STORAGE_ACCOUNT_BLOB_URL": "cache_account_blob_url",
        "GRAPHRAG_CACHE_BASE_DIR": "/some/cache/dir",
        "GRAPHRAG_CACHE_CONNECTION_STRING": "test_cs1",
        "GRAPHRAG_CACHE_CONTAINER_NAME": "test_cn1",
        "GRAPHRAG_CACHE_TYPE": "blob",
        # CHUNK -> CHUNKS
        # GRAPHRAG_CHUNK_BY_COLUMNS -> GRAPHRAG_CHUNKS_GROUP_BY_COLUMNS
        "GRAPHRAG_CHUNKS_GROUP_BY_COLUMNS": "a,b",
        "GRAPHRAG_CHUNKS_OVERLAP": "12",
        "GRAPHRAG_CHUNKS_SIZE": "500",
        "GRAPHRAG_CHUNKS_ENCODING_MODEL": "encoding-c",
        "GRAPHRAG_CLAIM_EXTRACTION_ENABLED": "True",
        "GRAPHRAG_CLAIM_EXTRACTION_DESCRIPTION": "test 123",
        "GRAPHRAG_CLAIM_EXTRACTION_MAX_GLEANINGS": "5000",
        "GRAPHRAG_CLAIM_EXTRACTION_PROMPT": "/some/prompt-a.txt",  # GRAPHRAG_CLAIM_EXTRACTION_PROMPT_FILE -> GRAPHRAG_CLAIM_EXTRACTION_PROMPT
        "GRAPHRAG_CLAIM_EXTRACTION_ENCODING_MODEL": "encoding_a",
        "GRAPHRAG_COMMUNITY_REPORTS_MAX_LENGTH": "23456",
        # GRAPHRAG_COMMUNITY_REPORTS_PROMPT_FILE -> GRAPHRAG_COMMUNITY_REPORTS_PROMPT
        "GRAPHRAG_COMMUNITY_REPORTS_PROMPT": "/some/prompt-b.txt",
        # EMBEDDING -> EMBEDDINGS
        "GRAPHRAG_EMBEDDINGS_BATCH_MAX_TOKENS": "17",
        "GRAPHRAG_EMBEDDINGS_BATCH_SIZE": "1000000",
        # GRAPHRAG_EMBEDDING_CONCURRENT_REQUESTS -> GRAPHRAG_EMBEDDINGS_LLM_CONCURRENT_REQUESTS
        "GRAPHRAG_EMBEDDINGS_LLM_CONCURRENT_REQUESTS": "12",
        # GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME -> GRAPHRAG_EMBEDDINGS_LLM_DEPLOYMENT_NAME
        "GRAPHRAG_EMBEDDINGS_LLM_DEPLOYMENT_NAME": "model-deployment-name",
        # GRAPHRAG_EMBEDDING_MAX_RETRIES -> GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRIES
        "GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRIES": "3",
        # GRAPHRAG_EMBEDDING_MAX_RETRY_WAIT -> GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRY_WAIT
        "GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRY_WAIT": "0.1123",
        # GRAPHRAG_EMBEDDING_MODEL -> GRAPHRAG_EMBEDDINGS_LLM_MODEL
        "GRAPHRAG_EMBEDDINGS_LLM_MODEL": "text-embeddingS-2",
        # GRAPHRAG_EMBEDDING_REQUESTS_PER_MINUTE -> GRAPHRAG_EMBEDDINGS_LLM_REQUESTS_PER_MINUTE
        "GRAPHRAG_EMBEDDINGS_LLM_REQUESTS_PER_MINUTE": "500",
        "GRAPHRAG_EMBEDDINGS_SKIP": "a1,b1,c1",
        # GRAPHRAG_EMBEDDING_SLEEP_ON_RATE_LIMIT_RECOMMENDATION -> GRAPHRAG_EMBEDDINGS_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION
        "GRAPHRAG_EMBEDDINGS_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION": "False",
        "GRAPHRAG_EMBEDDINGS_TARGET": "all",
        # GRAPHRAG_EMBEDDING_THREAD_COUNT -> GRAPHRAG_EMBEDDINGS_PARALLELIZATION_NUM_THREADS
        "GRAPHRAG_EMBEDDINGS_PARALLELIZATION_NUM_THREADS": "2345",
        # GRAPHRAG_EMBEDDING_THREAD_STAGGER -> GRAPHRAG_EMBEDDINGS_PARALLELIZATION_STAGGER
        "GRAPHRAG_EMBEDDINGS_PARALLELIZATION_STAGGER": "0.456",
        # GRAPHRAG_EMBEDDING_TOKENS_PER_MINUTE -> GRAPHRAG_EMBEDDINGS_LLM_TOKENS_PER_MINUTE
        "GRAPHRAG_EMBEDDINGS_LLM_TOKENS_PER_MINUTE": "7000",
        # GRAPHRAG_EMBEDDING_TYPE -> GRAPHRAG_EMBEDDINGS_LLM_TYPE
        "GRAPHRAG_EMBEDDINGS_LLM_TYPE": "azure_openai_embedding",
        "GRAPHRAG_ENCODING_MODEL": "test123",
        "GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES": "cat,dog,elephant",
        "GRAPHRAG_ENTITY_EXTRACTION_MAX_GLEANINGS": "112",
        # GRAPHRAG_ENTITY_EXTRACTION_PROMPT_FILE -> GRAPHRAG_ENTITY_EXTRACTION_PROMPT
        "GRAPHRAG_ENTITY_EXTRACTION_PROMPT": "/some/prompt-c.txt",
        "GRAPHRAG_ENTITY_EXTRACTION_ENCODING_MODEL": "encoding_b",
        "GRAPHRAG_INPUT_STORAGE_ACCOUNT_BLOB_URL": "input_account_blob_url",
        "GRAPHRAG_INPUT_BASE_DIR": "/some/input/dir",
        "GRAPHRAG_INPUT_CONNECTION_STRING": "input_cs",
        "GRAPHRAG_INPUT_CONTAINER_NAME": "input_cn",
        "GRAPHRAG_INPUT_DOCUMENT_ATTRIBUTE_COLUMNS": "test1,test2",
        "GRAPHRAG_INPUT_ENCODING": "utf-16",
        "GRAPHRAG_INPUT_FILE_PATTERN": ".*\\test\\.txt$$",
        "GRAPHRAG_INPUT_SOURCE_COLUMN": "test_source",
        "GRAPHRAG_INPUT_TYPE": "blob",
        "GRAPHRAG_INPUT_TEXT_COLUMN": "test_text",
        "GRAPHRAG_INPUT_TIMESTAMP_COLUMN": "test_timestamp",
        "GRAPHRAG_INPUT_TIMESTAMP_FORMAT": "test_format",
        "GRAPHRAG_INPUT_TITLE_COLUMN": "test_title",
        "GRAPHRAG_INPUT_FILE_TYPE": "text",
        # GRAPHRAG_API_BASE -> GRAPHRAG_LLM_API_BASE
        "GRAPHRAG_LLM_API_BASE": "http://some/base",
        # GRAPHRAG_API_KEY -> GRAPHRAG_LLM_API_KEY
        "GRAPHRAG_LLM_API_KEY": "test",
        # GRAPHRAG_API_ORGANIZATION -> GRAPHRAG_LLM_ORGANIZATION
        "GRAPHRAG_LLM_ORGANIZATION": "test_org",
        # GRAPHRAG_API_PROXY -> GRAPHRAG_LLM_PROXY
        "GRAPHRAG_LLM_PROXY": "http://some/proxy",
        # GRAPHRAG_API_VERSION -> GRAPHRAG_LLM_API_VERSION
        "GRAPHRAG_LLM_API_VERSION": "v1234",
        "GRAPHRAG_LLM_CONCURRENT_REQUESTS": "12",
        "GRAPHRAG_LLM_DEPLOYMENT_NAME": "model-deployment-name-x",
        "GRAPHRAG_LLM_MAX_RETRIES": "312",
        "GRAPHRAG_LLM_MAX_RETRY_WAIT": "0.1122",
        "GRAPHRAG_LLM_MAX_TOKENS": "15000",
        "GRAPHRAG_LLM_MODEL_SUPPORTS_JSON": "true",
        "GRAPHRAG_LLM_MODEL": "test-llm",
        "GRAPHRAG_LLM_N": "1",
        "GRAPHRAG_LLM_REQUEST_TIMEOUT": "12.7",
        "GRAPHRAG_LLM_REQUESTS_PER_MINUTE": "900",
        "GRAPHRAG_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION": "False",
        # GRAPHRAG_LLM_THREAD_COUNT -> GRAPHRAG_PARALLELIZATION_NUM_THREADS
        "GRAPHRAG_PARALLELIZATION_NUM_THREADS": "987",
        # GRAPHRAG_LLM_THREAD_STAGGER -> GRAPHRAG_PARALLELIZATION_STAGGER
        "GRAPHRAG_PARALLELIZATION_STAGGER": "0.123",
        "GRAPHRAG_LLM_TOKENS_PER_MINUTE": "8000",
        "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
        "GRAPHRAG_LLM_TEMPERATURE": "0.0",
        "GRAPHRAG_LLM_TOP_P": "1.0",
        # GRAPHRAG_MAX_CLUSTER_SIZE -> GRAPHRAG_CLUSTER_GRAPH_MAX_CLUSTER_SIZE
        "GRAPHRAG_CLUSTER_GRAPH_MAX_CLUSTER_SIZE": "123",
        # MOVE TO EMBED_GRAPH AND REMOVE NODE2VEC
        "GRAPHRAG_EMBED_GRAPH_ENABLED": "true",
        "GRAPHRAG_EMBED_GRAPH_ITERATIONS": "878787",
        "GRAPHRAG_EMBED_GRAPH_NUM_WALKS": "5000000",
        "GRAPHRAG_EMBED_GRAPH_RANDOM_SEED": "010101",
        "GRAPHRAG_EMBED_GRAPH_WALK_LENGTH": "555111",
        "GRAPHRAG_EMBED_GRAPH_WINDOW_SIZE": "12345",
        "GRAPHRAG_REPORTING_STORAGE_ACCOUNT_BLOB_URL": "reporting_account_blob_url",
        "GRAPHRAG_REPORTING_BASE_DIR": "/some/reporting/dir",
        "GRAPHRAG_REPORTING_CONNECTION_STRING": "test_cs2",
        "GRAPHRAG_REPORTING_CONTAINER_NAME": "test_cn2",
        "GRAPHRAG_REPORTING_TYPE": "blob",
        "GRAPHRAG_SKIP_WORKFLOWS": "a,b,c",
        # SNAPSHOT -> SNAPSHOTS
        "GRAPHRAG_SNAPSHOTS_GRAPHML": "true",
        "GRAPHRAG_SNAPSHOTS_RAW_ENTITIES": "true",
        "GRAPHRAG_SNAPSHOTS_TOP_LEVEL_NODES": "true",
        "GRAPHRAG_STORAGE_STORAGE_ACCOUNT_BLOB_URL": "storage_account_blob_url",
        "GRAPHRAG_STORAGE_BASE_DIR": "/some/storage/dir",
        "GRAPHRAG_STORAGE_CONNECTION_STRING": "test_cs",
        "GRAPHRAG_STORAGE_CONTAINER_NAME": "test_cn",
        "GRAPHRAG_STORAGE_TYPE": "blob",
        "GRAPHRAG_SUMMARIZE_DESCRIPTIONS_MAX_LENGTH": "12345",
        # GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT_FILE -> GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT
        "GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT": "/some/prompt-d.txt",
        "GRAPHRAG_UMAP_ENABLED": "true",
        "GRAPHRAG_LOCAL_SEARCH_TEXT_UNIT_PROP": "0.713",
        "GRAPHRAG_LOCAL_SEARCH_COMMUNITY_PROP": "0.1234",
        # GRAPHRAG_LOCAL_SEARCH_LLM_TEMPERATURE -> GRAPHRAG_LOCAL_SEARCH_TEMPERATURE
        "GRAPHRAG_LOCAL_SEARCH_TEMPERATURE": "0.1",
        # GRAPHRAG_LOCAL_SEARCH_LLM_TOP_P -> GRAPHRAG_LOCAL_SEARCH_TOP_P
        "GRAPHRAG_LOCAL_SEARCH_TOP_P": "0.9",
        # GRAPHRAG_LOCAL_SEARCH_LLM_N -> GRAPHRAG_LOCAL_SEARCH_N
        "GRAPHRAG_LOCAL_SEARCH_N": "2",
        "GRAPHRAG_LOCAL_SEARCH_TOP_K_RELATIONSHIPS": "15",
        "GRAPHRAG_LOCAL_SEARCH_TOP_K_ENTITIES": "14",
        "GRAPHRAG_LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS": "2",
        "GRAPHRAG_LOCAL_SEARCH_MAX_TOKENS": "142435",
        # GRAPHRAG_GLOBAL_SEARCH_LLM_TEMPERATURE -> GRAPHRAG_GLOBAL_SEARCH_TEMPERATURE
        "GRAPHRAG_GLOBAL_SEARCH_TEMPERATURE": "0.1",
        # GRAPHRAG_GLOBAL_SEARCH_LLM_TOP_P -> GRAPHRAG_GLOBAL_SEARCH_TOP_P
        "GRAPHRAG_GLOBAL_SEARCH_TOP_P": "0.9",
        # GRAPHRAG_GLOBAL_SEARCH_LLM_N -> GRAPHRAG_GLOBAL_SEARCH_N
        "GRAPHRAG_GLOBAL_SEARCH_N": "2",
        "GRAPHRAG_GLOBAL_SEARCH_MAX_TOKENS": "5123",
        "GRAPHRAG_GLOBAL_SEARCH_DATA_MAX_TOKENS": "123",
        "GRAPHRAG_GLOBAL_SEARCH_MAP_MAX_TOKENS": "4123",
        "GRAPHRAG_GLOBAL_SEARCH_CONCURRENCY": "7",
        "GRAPHRAG_GLOBAL_SEARCH_REDUCE_MAX_TOKENS": "15432",
    }


@pytest.fixture
def mock_env_vars(all_env_vars: dict[str, str]):  # noqa: PT004
    with mock.patch.dict("os.environ", all_env_vars, clear=True):
        yield


def _get_cache_config(all_env_vars: dict[str, str]) -> CacheConfig:
    args: dict[str, Any] = {
        "storage_account_blob_url": all_env_vars[
            "GRAPHRAG_CACHE_STORAGE_ACCOUNT_BLOB_URL"
        ],
        "base_dir": all_env_vars["GRAPHRAG_CACHE_BASE_DIR"],
        "connection_string": all_env_vars["GRAPHRAG_CACHE_CONNECTION_STRING"],
        "container_name": all_env_vars["GRAPHRAG_CACHE_CONTAINER_NAME"],
        "type": all_env_vars["GRAPHRAG_CACHE_TYPE"],
    }

    return CacheConfig(**args)


def _get_chunks_config(all_env_vars: dict[str, str]) -> ChunkingConfig:
    args = {
        "group_by_columns": all_env_vars["GRAPHRAG_CHUNKS_GROUP_BY_COLUMNS"].split(","),
        "overlap": int(all_env_vars["GRAPHRAG_CHUNKS_OVERLAP"]),
        "size": int(all_env_vars["GRAPHRAG_CHUNKS_SIZE"]),
        "encoding_model": all_env_vars["GRAPHRAG_CHUNKS_ENCODING_MODEL"],
    }

    return ChunkingConfig(**args)


def _get_claims_extraction_config(
    all_env_vars: dict[str, str],
) -> ClaimExtractionConfig:
    args = {
        "enabled": all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_ENABLED"].lower() == "true",
        "description": all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_DESCRIPTION"],
        "max_gleanings": int(all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_MAX_GLEANINGS"]),
        "prompt": all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_PROMPT"],
        "encoding_model": all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_ENCODING_MODEL"],
    }

    return ClaimExtractionConfig(**args)


def _get_community_reports_config(
    all_env_vars: dict[str, str],
) -> CommunityReportsConfig:
    args = {
        "max_length": int(all_env_vars["GRAPHRAG_COMMUNITY_REPORTS_MAX_LENGTH"]),
        "prompt": all_env_vars["GRAPHRAG_COMMUNITY_REPORTS_PROMPT"],
    }

    return CommunityReportsConfig(**args)


def _get_text_embeddings_config(all_env_vars: dict[str, str]) -> TextEmbeddingConfig:
    args = {
        "batch_max_tokens": int(all_env_vars["GRAPHRAG_EMBEDDINGS_BATCH_MAX_TOKENS"]),
        "batch_size": int(all_env_vars["GRAPHRAG_EMBEDDINGS_BATCH_SIZE"]),
        "llm": {
            "concurrent_requests": int(
                all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_CONCURRENT_REQUESTS"]
            ),
            "deployment_name": all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_DEPLOYMENT_NAME"],
            "max_retries": int(all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRIES"]),
            "max_retry_wait": float(
                all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRY_WAIT"]
            ),
            "model": all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_MODEL"],
            "requests_per_minute": int(
                all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_REQUESTS_PER_MINUTE"]
            ),
            "sleep_on_rate_limit_recommendation": all_env_vars[
                "GRAPHRAG_EMBEDDINGS_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION"
            ].lower()
            == "true",
            "type": all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_TYPE"],
            "tokens_per_minute": int(
                all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_TOKENS_PER_MINUTE"]
            ),
        },
        "skip": all_env_vars["GRAPHRAG_EMBEDDINGS_SKIP"].split(","),
        "target": all_env_vars["GRAPHRAG_EMBEDDINGS_TARGET"],
        "parallelization": {
            "num_threads": int(
                all_env_vars["GRAPHRAG_EMBEDDINGS_PARALLELIZATION_NUM_THREADS"]
            ),
            "stagger": float(
                all_env_vars["GRAPHRAG_EMBEDDINGS_PARALLELIZATION_STAGGER"]
            ),
        },
    }

    return TextEmbeddingConfig(**args)


def _get_entity_extraction_config(
    all_env_vars: dict[str, str],
) -> EntityExtractionConfig:
    args = {
        "entity_types": all_env_vars["GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES"].split(
            ","
        ),
        "max_gleanings": int(all_env_vars["GRAPHRAG_ENTITY_EXTRACTION_MAX_GLEANINGS"]),
        "prompt": all_env_vars["GRAPHRAG_ENTITY_EXTRACTION_PROMPT"],
        "encoding_model": all_env_vars["GRAPHRAG_ENTITY_EXTRACTION_ENCODING_MODEL"],
    }

    return EntityExtractionConfig(**args)


def _get_input_config(all_env_vars: dict[str, str]) -> InputConfig:
    args = {
        "storage_account_blob_url": all_env_vars[
            "GRAPHRAG_INPUT_STORAGE_ACCOUNT_BLOB_URL"
        ],
        "base_dir": all_env_vars["GRAPHRAG_INPUT_BASE_DIR"],
        "connection_string": all_env_vars["GRAPHRAG_INPUT_CONNECTION_STRING"],
        "container_name": all_env_vars["GRAPHRAG_INPUT_CONTAINER_NAME"],
        "document_attribute_columns": all_env_vars[
            "GRAPHRAG_INPUT_DOCUMENT_ATTRIBUTE_COLUMNS"
        ].split(","),
        "encoding": all_env_vars["GRAPHRAG_INPUT_ENCODING"],
        "file_pattern": all_env_vars["GRAPHRAG_INPUT_FILE_PATTERN"].replace("$$", "$"),
        "source_column": all_env_vars["GRAPHRAG_INPUT_SOURCE_COLUMN"],
        "type": all_env_vars["GRAPHRAG_INPUT_TYPE"],
        "text_column": all_env_vars["GRAPHRAG_INPUT_TEXT_COLUMN"],
        "timestamp_column": all_env_vars["GRAPHRAG_INPUT_TIMESTAMP_COLUMN"],
        "timestamp_format": all_env_vars["GRAPHRAG_INPUT_TIMESTAMP_FORMAT"],
        "title_column": all_env_vars["GRAPHRAG_INPUT_TITLE_COLUMN"],
        "file_type": all_env_vars["GRAPHRAG_INPUT_FILE_TYPE"],
    }

    return InputConfig(**args)


def _get_root_llm_parameters_config(all_env_vars: dict[str, str]) -> LLMParameters:
    args = {
        "api_base": all_env_vars["GRAPHRAG_LLM_API_BASE"],
        "api_key": all_env_vars["GRAPHRAG_LLM_API_KEY"],
        "organization": all_env_vars["GRAPHRAG_LLM_ORGANIZATION"],
        "proxy": all_env_vars["GRAPHRAG_LLM_PROXY"],
        "api_version": all_env_vars["GRAPHRAG_LLM_API_VERSION"],
        "concurrent_requests": int(all_env_vars["GRAPHRAG_LLM_CONCURRENT_REQUESTS"]),
        "deployment_name": all_env_vars["GRAPHRAG_LLM_DEPLOYMENT_NAME"],
        "max_retries": int(all_env_vars["GRAPHRAG_LLM_MAX_RETRIES"]),
        "max_retry_wait": float(all_env_vars["GRAPHRAG_LLM_MAX_RETRY_WAIT"]),
        "max_tokens": int(all_env_vars["GRAPHRAG_LLM_MAX_TOKENS"]),
        "model_supports_json": all_env_vars["GRAPHRAG_LLM_MODEL_SUPPORTS_JSON"],
        "model": all_env_vars["GRAPHRAG_LLM_MODEL"],
        "n": int(all_env_vars["GRAPHRAG_LLM_N"]),
        "request_timeout": float(all_env_vars["GRAPHRAG_LLM_REQUEST_TIMEOUT"]),
        "requests_per_minute": int(all_env_vars["GRAPHRAG_LLM_REQUESTS_PER_MINUTE"]),
        "sleep_on_rate_limit_recommendation": all_env_vars[
            "GRAPHRAG_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION"
        ],
        "tokens_per_minute": int(all_env_vars["GRAPHRAG_LLM_TOKENS_PER_MINUTE"]),
        "type": all_env_vars["GRAPHRAG_LLM_TYPE"],
        "temperature": float(all_env_vars["GRAPHRAG_LLM_TEMPERATURE"]),
        "top_p": float(all_env_vars["GRAPHRAG_LLM_TOP_P"]),
    }

    return LLMParameters(**args)


def _get_cluster_graph_config(all_env_vars: dict[str, str]) -> ClusterGraphConfig:
    args: dict[str, Any] = {
        "max_cluster_size": int(
            all_env_vars["GRAPHRAG_CLUSTER_GRAPH_MAX_CLUSTER_SIZE"]
        ),
    }

    return ClusterGraphConfig(**args)


def _get_embed_graph_config(all_env_vars: dict[str, str]) -> EmbedGraphConfig:
    args = {
        "enabled": all_env_vars["GRAPHRAG_EMBED_GRAPH_ENABLED"],
        "iterations": int(all_env_vars["GRAPHRAG_EMBED_GRAPH_ITERATIONS"]),
        "num_walks": int(all_env_vars["GRAPHRAG_EMBED_GRAPH_NUM_WALKS"]),
        "random_seed": int(all_env_vars["GRAPHRAG_EMBED_GRAPH_RANDOM_SEED"]),
        "walk_length": int(all_env_vars["GRAPHRAG_EMBED_GRAPH_WALK_LENGTH"]),
        "window_size": int(all_env_vars["GRAPHRAG_EMBED_GRAPH_WINDOW_SIZE"]),
    }

    return EmbedGraphConfig(**args)


def _get_reporting_config(all_env_vars: dict[str, str]) -> ReportingConfig:
    args = {
        "storage_account_blob_url": all_env_vars[
            "GRAPHRAG_REPORTING_STORAGE_ACCOUNT_BLOB_URL"
        ],
        "base_dir": all_env_vars["GRAPHRAG_REPORTING_BASE_DIR"],
        "connection_string": all_env_vars["GRAPHRAG_REPORTING_CONNECTION_STRING"],
        "container_name": all_env_vars["GRAPHRAG_REPORTING_CONTAINER_NAME"],
        "type": all_env_vars["GRAPHRAG_REPORTING_TYPE"],
    }

    return ReportingConfig(**args)  # type: ignore


def _get_snapshots_config(all_env_vars: dict[str, str]) -> SnapshotsConfig:
    args = {
        "graphml": all_env_vars["GRAPHRAG_SNAPSHOTS_GRAPHML"],
        "raw_entities": all_env_vars["GRAPHRAG_SNAPSHOTS_RAW_ENTITIES"],
        "top_level_nodes": all_env_vars["GRAPHRAG_SNAPSHOTS_TOP_LEVEL_NODES"],
    }

    return SnapshotsConfig(**args)  # type: ignore


def _get_storage_config(all_env_vars: dict[str, str]) -> StorageConfig:
    args = {
        "storage_account_blob_url": all_env_vars[
            "GRAPHRAG_STORAGE_STORAGE_ACCOUNT_BLOB_URL"
        ],
        "base_dir": all_env_vars["GRAPHRAG_STORAGE_BASE_DIR"],
        "connection_string": all_env_vars["GRAPHRAG_STORAGE_CONNECTION_STRING"],
        "container_name": all_env_vars["GRAPHRAG_STORAGE_CONTAINER_NAME"],
        "type": all_env_vars["GRAPHRAG_STORAGE_TYPE"],
    }

    return StorageConfig(**args)  # type: ignore


def _get_summarize_description_config(
    all_env_vars: dict[str, str],
) -> SummarizeDescriptionsConfig:
    args = {
        "max_length": int(all_env_vars["GRAPHRAG_SUMMARIZE_DESCRIPTIONS_MAX_LENGTH"]),
        "prompt": all_env_vars["GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT"],
    }

    return SummarizeDescriptionsConfig(**args)


def _get_local_search_config(all_env_vars: dict[str, str]) -> LocalSearchConfig:
    args = {
        "text_unit_prop": float(all_env_vars["GRAPHRAG_LOCAL_SEARCH_TEXT_UNIT_PROP"]),
        "community_prop": float(all_env_vars["GRAPHRAG_LOCAL_SEARCH_COMMUNITY_PROP"]),
        "temperature": float(all_env_vars["GRAPHRAG_LOCAL_SEARCH_TEMPERATURE"]),
        "top_p": float(all_env_vars["GRAPHRAG_LOCAL_SEARCH_TOP_P"]),
        "n": int(all_env_vars["GRAPHRAG_LOCAL_SEARCH_N"]),
        "top_k_relationships": int(
            all_env_vars["GRAPHRAG_LOCAL_SEARCH_TOP_K_RELATIONSHIPS"]
        ),
        "top_k_entities": int(all_env_vars["GRAPHRAG_LOCAL_SEARCH_TOP_K_ENTITIES"]),
        "conversation_history_max_turns": int(
            all_env_vars["GRAPHRAG_LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS"]
        ),
        "max_tokens": int(all_env_vars["GRAPHRAG_LOCAL_SEARCH_MAX_TOKENS"]),
    }

    return LocalSearchConfig(**args)


def _get_global_search_config(all_env_vars: dict[str, str]) -> GlobalSearchConfig:
    args = {
        "temperature": float(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_TEMPERATURE"]),
        "top_p": float(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_TOP_P"]),
        "n": int(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_N"]),
        "max_tokens": int(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_MAX_TOKENS"]),
        "data_max_tokens": int(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_DATA_MAX_TOKENS"]),
        "map_max_tokens": int(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_MAP_MAX_TOKENS"]),
        "concurrency": int(all_env_vars["GRAPHRAG_GLOBAL_SEARCH_CONCURRENCY"]),
        "reduce_max_tokens": int(
            all_env_vars["GRAPHRAG_GLOBAL_SEARCH_REDUCE_MAX_TOKENS"]
        ),
    }

    return GlobalSearchConfig(**args)


@pytest.fixture
def env_config(
    all_env_vars: dict[str, str],
) -> GraphRagConfig:
    args: dict[str, Any] = {
        "async_mode": all_env_vars["GRAPHRAG_ASYNC_MODE"],
        "encoding_model": all_env_vars["GRAPHRAG_ENCODING_MODEL"],
        "skip_workflows": all_env_vars["GRAPHRAG_SKIP_WORKFLOWS"].split(","),
        "cache": _get_cache_config(all_env_vars),
        "chunks": _get_chunks_config(all_env_vars),
        "claim_extraction": _get_claims_extraction_config(all_env_vars),
        "community_reports": _get_community_reports_config(all_env_vars),
        "embeddings": _get_text_embeddings_config(all_env_vars),
        "entity_extraction": _get_entity_extraction_config(all_env_vars),
        "input": _get_input_config(all_env_vars),
        "llm": _get_root_llm_parameters_config(all_env_vars),
        "cluster_graph": _get_cluster_graph_config(all_env_vars),
        "embed_graph": _get_embed_graph_config(all_env_vars),
        "reporting": _get_reporting_config(all_env_vars),
        "snapshots": _get_snapshots_config(all_env_vars),
        "storage": _get_storage_config(all_env_vars),
        "summarize_descriptions": _get_summarize_description_config(all_env_vars),
        "umap": {
            "enabled": all_env_vars["GRAPHRAG_UMAP_ENABLED"],
        },
        "local_search": _get_local_search_config(all_env_vars),
        "global_search": _get_global_search_config(all_env_vars),
    }

    return GraphRagConfig(**args)
