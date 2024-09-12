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
    # Global
    assert actual.root_dir == expected.root_dir
    assert actual.async_mode == expected.async_mode
    assert actual.encoding_model == expected.encoding_model
    assert actual.skip_workflows == expected.skip_workflows

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
    assert actual.snapshots.graphml == expected.snapshots.graphml
    assert actual.snapshots.raw_entities == expected.snapshots.raw_entities
    assert actual.snapshots.top_level_nodes == expected.snapshots.top_level_nodes

    # Entity Extraction
    _assert_llm_parameters(actual.entity_extraction.llm, expected.entity_extraction.llm)
    _assert_parallelization_parameters(
        actual.entity_extraction.parallelization,
        expected.entity_extraction.parallelization,
    )
    assert actual.entity_extraction.async_mode == expected.entity_extraction.async_mode
    assert actual.entity_extraction.prompt == expected.entity_extraction.prompt
    assert (
        actual.entity_extraction.entity_types == expected.entity_extraction.entity_types
    )
    assert (
        actual.entity_extraction.max_gleanings
        == expected.entity_extraction.max_gleanings
    )
    assert actual.entity_extraction.strategy == expected.entity_extraction.strategy
    assert (
        actual.entity_extraction.encoding_model
        == expected.entity_extraction.encoding_model
    )

    # Summarize Descriptions
    _assert_llm_parameters(
        actual.summarize_descriptions.llm, expected.summarize_descriptions.llm
    )
    _assert_parallelization_parameters(
        actual.summarize_descriptions.parallelization,
        expected.summarize_descriptions.parallelization,
    )
    assert (
        actual.summarize_descriptions.async_mode
        == expected.summarize_descriptions.async_mode
    )
    assert (
        actual.summarize_descriptions.prompt == expected.summarize_descriptions.prompt
    )
    assert (
        actual.summarize_descriptions.max_length
        == expected.summarize_descriptions.max_length
    )
    assert (
        actual.summarize_descriptions.strategy
        == expected.summarize_descriptions.strategy
    )

    # Community Reports
    _assert_llm_parameters(actual.community_reports.llm, expected.community_reports.llm)
    _assert_parallelization_parameters(
        actual.community_reports.parallelization,
        expected.community_reports.parallelization,
    )
    assert actual.community_reports.async_mode == expected.community_reports.async_mode
    assert actual.community_reports.prompt == expected.community_reports.prompt
    assert actual.community_reports.max_length == expected.community_reports.max_length
    assert (
        actual.community_reports.max_input_length
        == expected.community_reports.max_input_length
    )
    assert actual.community_reports.strategy == expected.community_reports.strategy

    # Claim Extraction
    _assert_llm_parameters(actual.claim_extraction.llm, expected.claim_extraction.llm)
    _assert_parallelization_parameters(
        actual.claim_extraction.parallelization,
        expected.claim_extraction.parallelization,
    )
    assert actual.claim_extraction.async_mode == expected.claim_extraction.async_mode
    assert actual.claim_extraction.enabled == expected.claim_extraction.enabled
    assert actual.claim_extraction.prompt == expected.claim_extraction.prompt
    assert actual.claim_extraction.description == expected.claim_extraction.description
    assert (
        actual.claim_extraction.max_gleanings == expected.claim_extraction.max_gleanings
    )
    assert actual.claim_extraction.strategy == expected.claim_extraction.strategy
    assert (
        actual.claim_extraction.encoding_model
        == expected.claim_extraction.encoding_model
    )

    # Cluster Graph
    assert (
        actual.cluster_graph.max_cluster_size == expected.cluster_graph.max_cluster_size
    )
    assert actual.cluster_graph.strategy == expected.cluster_graph.strategy

    # UMAP
    assert actual.umap.enabled == expected.umap.enabled

    # Local Search
    assert actual.local_search.text_unit_prop == expected.local_search.text_unit_prop
    assert actual.local_search.community_prop == expected.local_search.community_prop
    assert (
        actual.local_search.conversation_history_max_turns
        == expected.local_search.conversation_history_max_turns
    )
    assert actual.local_search.top_k_entities == expected.local_search.top_k_entities
    assert (
        actual.local_search.top_k_relationships
        == expected.local_search.top_k_relationships
    )
    assert actual.local_search.temperature == expected.local_search.temperature
    assert actual.local_search.top_p == expected.local_search.top_p
    assert actual.local_search.n == expected.local_search.n
    assert actual.local_search.max_tokens == expected.local_search.max_tokens
    assert actual.local_search.llm_max_tokens == expected.local_search.llm_max_tokens

    # Global Search
    assert actual.global_search.temperature == expected.global_search.temperature
    assert actual.global_search.top_p == expected.global_search.top_p
    assert actual.global_search.n == expected.global_search.n
    assert actual.global_search.max_tokens == expected.global_search.max_tokens
    assert (
        actual.global_search.data_max_tokens == expected.global_search.data_max_tokens
    )
    assert actual.global_search.map_max_tokens == expected.global_search.map_max_tokens
    assert (
        actual.global_search.reduce_max_tokens
        == expected.global_search.reduce_max_tokens
    )
    assert actual.global_search.concurrency == expected.global_search.concurrency
