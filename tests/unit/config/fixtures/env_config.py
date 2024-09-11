from unittest import mock

import pytest

from graphrag.config2 import GraphRagConfig

from .default_config import default_config  # noqa


@pytest.fixture
def all_env_vars() -> dict[str, str]:
    return {
        "GRAPHRAG_API_BASE": "http://some/base",
        "GRAPHRAG_API_KEY": "test",
        "GRAPHRAG_API_ORGANIZATION": "test_org",
        "GRAPHRAG_API_PROXY": "http://some/proxy",
        "GRAPHRAG_API_VERSION": "v1234",
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
        "GRAPHRAG_CLAIM_EXTRACTION_PROMPT": "tests/unit/config/prompt-a.txt",  # GRAPHRAG_CLAIM_EXTRACTION_PROMPT_FILE -> GRAPHRAG_CLAIM_EXTRACTION_PROMPT
        "GRAPHRAG_CLAIM_EXTRACTION_ENCODING_MODEL": "encoding_a",
        "GRAPHRAG_COMMUNITY_REPORTS_MAX_LENGTH": "23456",
        # GRAPHRAG_COMMUNITY_REPORTS_PROMPT_FILE -> GRAPHRAG_COMMUNITY_REPORTS_PROMPT
        "GRAPHRAG_COMMUNITY_REPORTS_PROMPT": "tests/unit/config/prompt-b.txt",
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
        "GRAPHRAG_ENTITY_EXTRACTION_PROMPT": "tests/unit/config/prompt-c.txt",
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
        # "GRAPHRAG_LLM_CONCURRENT_REQUESTS": "12",
        # "GRAPHRAG_LLM_DEPLOYMENT_NAME": "model-deployment-name-x",
        # "GRAPHRAG_LLM_MAX_RETRIES": "312",
        # "GRAPHRAG_LLM_MAX_RETRY_WAIT": "0.1122",
        # "GRAPHRAG_LLM_MAX_TOKENS": "15000",
        # "GRAPHRAG_LLM_MODEL_SUPPORTS_JSON": "true",
        # "GRAPHRAG_LLM_MODEL": "test-llm",
        # "GRAPHRAG_LLM_N": "1",
        # "GRAPHRAG_LLM_REQUEST_TIMEOUT": "12.7",
        # "GRAPHRAG_LLM_REQUESTS_PER_MINUTE": "900",
        # "GRAPHRAG_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION": "False",
        # "GRAPHRAG_LLM_THREAD_COUNT": "987",
        # "GRAPHRAG_LLM_THREAD_STAGGER": "0.123",
        # "GRAPHRAG_LLM_TOKENS_PER_MINUTE": "8000",
        # "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
        # "GRAPHRAG_MAX_CLUSTER_SIZE": "123",
        # "GRAPHRAG_NODE2VEC_ENABLED": "true",
        # "GRAPHRAG_NODE2VEC_ITERATIONS": "878787",
        # "GRAPHRAG_NODE2VEC_NUM_WALKS": "5000000",
        # "GRAPHRAG_NODE2VEC_RANDOM_SEED": "010101",
        # "GRAPHRAG_NODE2VEC_WALK_LENGTH": "555111",
        # "GRAPHRAG_NODE2VEC_WINDOW_SIZE": "12345",
        # "GRAPHRAG_REPORTING_STORAGE_ACCOUNT_BLOB_URL": "reporting_account_blob_url",
        # "GRAPHRAG_REPORTING_BASE_DIR": "/some/reporting/dir",
        # "GRAPHRAG_REPORTING_CONNECTION_STRING": "test_cs2",
        # "GRAPHRAG_REPORTING_CONTAINER_NAME": "test_cn2",
        # "GRAPHRAG_REPORTING_TYPE": "blob",
        # "GRAPHRAG_SKIP_WORKFLOWS": "a,b,c",
        # "GRAPHRAG_SNAPSHOT_GRAPHML": "true",
        # "GRAPHRAG_SNAPSHOT_RAW_ENTITIES": "true",
        # "GRAPHRAG_SNAPSHOT_TOP_LEVEL_NODES": "true",
        # "GRAPHRAG_STORAGE_STORAGE_ACCOUNT_BLOB_URL": "storage_account_blob_url",
        # "GRAPHRAG_STORAGE_BASE_DIR": "/some/storage/dir",
        # "GRAPHRAG_STORAGE_CONNECTION_STRING": "test_cs",
        # "GRAPHRAG_STORAGE_CONTAINER_NAME": "test_cn",
        # "GRAPHRAG_STORAGE_TYPE": "blob",
        # "GRAPHRAG_SUMMARIZE_DESCRIPTIONS_MAX_LENGTH": "12345",
        # "GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT_FILE": "tests/unit/config/prompt-d.txt",
        # "GRAPHRAG_LLM_TEMPERATURE": "0.0",
        # "GRAPHRAG_LLM_TOP_P": "1.0",
        # "GRAPHRAG_UMAP_ENABLED": "true",
        # "GRAPHRAG_LOCAL_SEARCH_TEXT_UNIT_PROP": "0.713",
        # "GRAPHRAG_LOCAL_SEARCH_COMMUNITY_PROP": "0.1234",
        # "GRAPHRAG_LOCAL_SEARCH_LLM_TEMPERATURE": "0.1",
        # "GRAPHRAG_LOCAL_SEARCH_LLM_TOP_P": "0.9",
        # "GRAPHRAG_LOCAL_SEARCH_LLM_N": "2",
        # "GRAPHRAG_LOCAL_SEARCH_LLM_MAX_TOKENS": "12",
        # "GRAPHRAG_LOCAL_SEARCH_TOP_K_RELATIONSHIPS": "15",
        # "GRAPHRAG_LOCAL_SEARCH_TOP_K_ENTITIES": "14",
        # "GRAPHRAG_LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS": "2",
        # "GRAPHRAG_LOCAL_SEARCH_MAX_TOKENS": "142435",
        # "GRAPHRAG_GLOBAL_SEARCH_LLM_TEMPERATURE": "0.1",
        # "GRAPHRAG_GLOBAL_SEARCH_LLM_TOP_P": "0.9",
        # "GRAPHRAG_GLOBAL_SEARCH_LLM_N": "2",
        # "GRAPHRAG_GLOBAL_SEARCH_MAX_TOKENS": "5123",
        # "GRAPHRAG_GLOBAL_SEARCH_DATA_MAX_TOKENS": "123",
        # "GRAPHRAG_GLOBAL_SEARCH_MAP_MAX_TOKENS": "4123",
        # "GRAPHRAG_GLOBAL_SEARCH_CONCURRENCY": "7",
        # "GRAPHRAG_GLOBAL_SEARCH_REDUCE_MAX_TOKENS": "15432",
    }


@pytest.fixture
def mock_env_vars(all_env_vars: dict[str, str]):  # noqa: PT004
    with mock.patch.dict("os.environ", all_env_vars, clear=True):
        yield


@pytest.fixture
def env_config(
    all_env_vars: dict[str, str],
    default_config: GraphRagConfig,  # noqa: F811
) -> GraphRagConfig:
    # Global
    default_config.api_base = all_env_vars["GRAPHRAG_API_BASE"]
    default_config.api_key = all_env_vars["GRAPHRAG_API_KEY"]
    default_config.api_organization = all_env_vars["GRAPHRAG_API_ORGANIZATION"]
    default_config.api_proxy = all_env_vars["GRAPHRAG_API_PROXY"]
    default_config.api_version = all_env_vars["GRAPHRAG_API_VERSION"]
    default_config.async_mode = all_env_vars["GRAPHRAG_ASYNC_MODE"]  # type: ignore
    default_config.encoding_model = all_env_vars["GRAPHRAG_ENCODING_MODEL"]

    # Cache
    default_config.cache.storage_account_blob_url = all_env_vars[
        "GRAPHRAG_CACHE_STORAGE_ACCOUNT_BLOB_URL"
    ]
    default_config.cache.base_dir = all_env_vars["GRAPHRAG_CACHE_BASE_DIR"]
    default_config.cache.connection_string = all_env_vars[
        "GRAPHRAG_CACHE_CONNECTION_STRING"
    ]
    default_config.cache.container_name = all_env_vars["GRAPHRAG_CACHE_CONTAINER_NAME"]
    default_config.cache.type = all_env_vars["GRAPHRAG_CACHE_TYPE"]  # type: ignore

    # Chunks
    default_config.chunks.group_by_columns = all_env_vars[
        "GRAPHRAG_CHUNKS_GROUP_BY_COLUMNS"
    ].split(",")
    default_config.chunks.overlap = int(all_env_vars["GRAPHRAG_CHUNKS_OVERLAP"])
    default_config.chunks.size = int(all_env_vars["GRAPHRAG_CHUNKS_SIZE"])
    default_config.chunks.encoding_model = all_env_vars[
        "GRAPHRAG_CHUNKS_ENCODING_MODEL"
    ]

    # Claim Extraction
    default_config.claim_extraction.enabled = (
        all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_ENABLED"].lower() == "true"
    )
    default_config.claim_extraction.description = all_env_vars[
        "GRAPHRAG_CLAIM_EXTRACTION_DESCRIPTION"
    ]
    default_config.claim_extraction.max_gleanings = int(
        all_env_vars["GRAPHRAG_CLAIM_EXTRACTION_MAX_GLEANINGS"]
    )
    default_config.claim_extraction.prompt = all_env_vars[
        "GRAPHRAG_CLAIM_EXTRACTION_PROMPT"
    ]
    default_config.claim_extraction.encoding_model = all_env_vars[
        "GRAPHRAG_CLAIM_EXTRACTION_ENCODING_MODEL"
    ]

    # Community Reports
    default_config.community_reports.max_length = int(
        all_env_vars["GRAPHRAG_COMMUNITY_REPORTS_MAX_LENGTH"]
    )
    default_config.community_reports.prompt = all_env_vars[
        "GRAPHRAG_COMMUNITY_REPORTS_PROMPT"
    ]

    # Embedding
    default_config.embeddings.batch_max_tokens = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_BATCH_MAX_TOKENS"]
    )
    default_config.embeddings.batch_size = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_BATCH_SIZE"]
    )
    default_config.embeddings.llm.concurrent_requests = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_CONCURRENT_REQUESTS"]
    )
    default_config.embeddings.llm.deployment_name = all_env_vars[
        "GRAPHRAG_EMBEDDINGS_LLM_DEPLOYMENT_NAME"
    ]
    default_config.embeddings.llm.max_retries = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRIES"]
    )
    default_config.embeddings.llm.max_retry_wait = float(
        all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_MAX_RETRY_WAIT"]
    )
    default_config.embeddings.llm.model = all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_MODEL"]
    default_config.embeddings.llm.requests_per_minute = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_REQUESTS_PER_MINUTE"]
    )
    default_config.embeddings.skip = all_env_vars["GRAPHRAG_EMBEDDINGS_SKIP"].split(",")
    default_config.embeddings.llm.sleep_on_rate_limit_recommendation = (
        all_env_vars[
            "GRAPHRAG_EMBEDDINGS_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION"
        ].lower()
        == "true"
    )
    default_config.embeddings.target = all_env_vars["GRAPHRAG_EMBEDDINGS_TARGET"]  # type: ignore
    default_config.embeddings.parallelization.num_threads = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_PARALLELIZATION_NUM_THREADS"]
    )
    default_config.embeddings.parallelization.stagger = float(
        all_env_vars["GRAPHRAG_EMBEDDINGS_PARALLELIZATION_STAGGER"]
    )
    default_config.embeddings.llm.tokens_per_minute = int(
        all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_TOKENS_PER_MINUTE"]
    )
    default_config.embeddings.llm.type = all_env_vars["GRAPHRAG_EMBEDDINGS_LLM_TYPE"]  # type: ignore

    # Entity Extraction
    default_config.entity_extraction.entity_types = all_env_vars[
        "GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES"
    ].split(",")
    default_config.entity_extraction.max_gleanings = int(
        all_env_vars["GRAPHRAG_ENTITY_EXTRACTION_MAX_GLEANINGS"]
    )
    default_config.entity_extraction.prompt = all_env_vars[
        "GRAPHRAG_ENTITY_EXTRACTION_PROMPT"
    ]
    default_config.entity_extraction.encoding_model = all_env_vars[
        "GRAPHRAG_ENTITY_EXTRACTION_ENCODING_MODEL"
    ]

    # Input
    default_config.input.storage_account_blob_url = all_env_vars[
        "GRAPHRAG_INPUT_STORAGE_ACCOUNT_BLOB_URL"
    ]
    default_config.input.base_dir = all_env_vars["GRAPHRAG_INPUT_BASE_DIR"]
    default_config.input.connection_string = all_env_vars[
        "GRAPHRAG_INPUT_CONNECTION_STRING"
    ]
    default_config.input.container_name = all_env_vars["GRAPHRAG_INPUT_CONTAINER_NAME"]
    default_config.input.document_attribute_columns = all_env_vars[
        "GRAPHRAG_INPUT_DOCUMENT_ATTRIBUTE_COLUMNS"
    ].split(",")
    default_config.input.encoding = all_env_vars["GRAPHRAG_INPUT_ENCODING"]
    default_config.input.file_pattern = all_env_vars[
        "GRAPHRAG_INPUT_FILE_PATTERN"
    ].replace("$$", "$")
    default_config.input.source_column = all_env_vars["GRAPHRAG_INPUT_SOURCE_COLUMN"]
    default_config.input.type = all_env_vars["GRAPHRAG_INPUT_TYPE"]  # type: ignore
    default_config.input.text_column = all_env_vars["GRAPHRAG_INPUT_TEXT_COLUMN"]
    default_config.input.timestamp_column = all_env_vars[
        "GRAPHRAG_INPUT_TIMESTAMP_COLUMN"
    ]
    default_config.input.timestamp_format = all_env_vars[
        "GRAPHRAG_INPUT_TIMESTAMP_FORMAT"
    ]
    default_config.input.title_column = all_env_vars["GRAPHRAG_INPUT_TITLE_COLUMN"]
    default_config.input.file_type = all_env_vars["GRAPHRAG_INPUT_FILE_TYPE"]  # type: ignore

    return default_config
