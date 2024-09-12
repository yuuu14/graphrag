# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the default configuration.

Test loading configs without using config files (settings.yaml, .env)
"""

from pathlib import Path

import pytest

from graphrag.config2 import GraphRagConfig, load_config
from tests.unit.config.assert_configs import assert_configs

from .fixtures import (  # noqa pytest fixtures
    all_env_vars,
    default_config,
    env_config,
    mock_env_vars,
)

DIRNAME = Path(__file__).parent
MOCK_API_KEY_OVERRIDE = "ABCDEFGHIJ1234"


def test_default_config(default_config: GraphRagConfig):  # noqa: F811
    config = load_config()
    assert_configs(config, default_config)


@pytest.mark.usefixtures("mock_env_vars")
def test_env_config(env_config: GraphRagConfig):  # noqa: F811
    config = load_config()
    assert_configs(config, env_config)


def test_overrides_config(default_config: GraphRagConfig):  # noqa: F811
    default_config.llm.api_key = MOCK_API_KEY_OVERRIDE
    config = load_config(overrides={"llm": {"api_key": MOCK_API_KEY_OVERRIDE}})
    assert_configs(config, default_config)
