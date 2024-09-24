# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from graphrag.config import create_pipeline_config
from graphrag.index import (
    run_pipeline,
    run_pipeline_with_config,
)


def test_exported_functions():
    assert callable(create_pipeline_config)
    assert callable(run_pipeline_with_config)
    assert callable(run_pipeline)
