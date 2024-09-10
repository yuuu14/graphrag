from graphrag.config2 import GraphRagConfig, LLMParameters, ParallelizationParameters


def _assert_llm_parameters(actual: LLMParameters, expected: LLMParameters):
    assert actual.api_key == expected.api_key
    assert actual.type == expected.type
    assert actual.model == expected.model
    assert actual.max_tokens == expected.max_tokens
    assert actual.temperature == expected.temperature
    assert actual.top_p == expected.top_p
    assert actual.n == expected.n
    assert actual.request_timeout == expected.request_timeout
    assert actual.api_base == expected.api_base
    assert actual.api_version == expected.api_version
    assert actual.organization == expected.organization
    assert actual.proxy == expected.proxy
    assert actual.cognitive_services_endpoint == expected.cognitive_services_endpoint
    assert actual.deployment_name == expected.deployment_name
    assert actual.model_supports_json == expected.model_supports_json
    assert actual.tokens_per_minute == expected.tokens_per_minute
    assert actual.requests_per_minute == expected.requests_per_minute
    assert actual.max_retries == expected.max_retries
    assert actual.max_retry_wait == expected.max_retry_wait
    assert (
        actual.sleep_on_rate_limit_recommendation
        == expected.sleep_on_rate_limit_recommendation
    )
    assert actual.concurrent_requests == expected.concurrent_requests


def _assert_parallelization_parameters(
    actual: ParallelizationParameters, expected: ParallelizationParameters
):
    assert actual.stagger == expected.stagger
    assert actual.num_threads == expected.num_threads


def assert_configs(actual: GraphRagConfig, expected: GraphRagConfig):
    assert actual.root_dir == expected.root_dir

    # Reporting
    assert actual.reporting.type == expected.reporting.type
    assert actual.reporting.base_dir == expected.reporting.base_dir
    assert actual.reporting.connection_string == expected.reporting.connection_string
    assert actual.reporting.container_name == expected.reporting.container_name
    assert (
        actual.reporting.storage_account_blob_url
        == expected.reporting.storage_account_blob_url
    )

    # Storage
    assert actual.storage.type == expected.storage.type
    assert actual.storage.base_dir == expected.storage.base_dir
    assert actual.storage.connection_string == expected.storage.connection_string
    assert actual.storage.container_name == expected.storage.container_name
    assert (
        actual.storage.storage_account_blob_url
        == expected.storage.storage_account_blob_url
    )

    # Cache
    assert actual.cache.type == expected.cache.type
    assert actual.cache.base_dir == expected.cache.base_dir
    assert actual.cache.connection_string == expected.cache.connection_string
    assert actual.cache.container_name == expected.cache.container_name
    assert (
        actual.cache.storage_account_blob_url == expected.cache.storage_account_blob_url
    )

    # Input
    assert actual.input.type == expected.input.type
    assert actual.input.file_type == expected.input.file_type
    assert actual.input.base_dir == expected.input.base_dir
    assert actual.input.connection_string == expected.input.connection_string
    assert (
        actual.input.storage_account_blob_url == expected.input.storage_account_blob_url
    )
    assert actual.input.container_name == expected.input.container_name
    assert actual.input.encoding == expected.input.encoding
    assert actual.input.file_pattern == expected.input.file_pattern
    assert actual.input.file_filter == expected.input.file_filter
    assert actual.input.source_column == expected.input.source_column
    assert actual.input.timestamp_column == expected.input.timestamp_column
    assert actual.input.timestamp_format == expected.input.timestamp_format
    assert actual.input.text_column == expected.input.text_column
    assert actual.input.title_column == expected.input.title_column
    assert (
        actual.input.document_attribute_columns
        == expected.input.document_attribute_columns
    )

    # Embed Graph
    assert actual.embed_graph.enabled == expected.embed_graph.enabled
    assert actual.embed_graph.num_walks == expected.embed_graph.num_walks
    assert actual.embed_graph.walk_length == expected.embed_graph.walk_length
    assert actual.embed_graph.window_size == expected.embed_graph.window_size
    assert actual.embed_graph.iterations == expected.embed_graph.iterations
    assert actual.embed_graph.random_seed == expected.embed_graph.random_seed
    assert actual.embed_graph.strategy == expected.embed_graph.strategy

    # Embeddings
    _assert_llm_parameters(actual.embeddings.llm, expected.embeddings.llm)
    _assert_parallelization_parameters(
        actual.embeddings.parallelization, expected.embeddings.parallelization
    )
    assert actual.embeddings.async_mode == expected.embeddings.async_mode
    assert actual.embeddings.batch_size == expected.embeddings.batch_size
    assert actual.embeddings.batch_max_tokens == expected.embeddings.batch_max_tokens
    assert actual.embeddings.target == expected.embeddings.target
    assert actual.embeddings.skip == expected.embeddings.skip
    assert actual.embeddings.vector_store == expected.embeddings.vector_store
    assert actual.embeddings.strategy == expected.embeddings.strategy

    # chunk config
    assert actual.chunks.size == expected.chunks.size
    assert actual.chunks.overlap == expected.chunks.overlap
    assert actual.chunks.group_by_columns == expected.chunks.group_by_columns
    assert actual.chunks.strategy == expected.chunks.strategy
    assert actual.chunks.encoding_model == expected.chunks.encoding_model

    # Snapshots
    # assert actual.snapshots.
