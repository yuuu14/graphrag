import pytest

from graphrag.config2 import GraphRagConfig, load_config
from tests.unit.config.assert_configs import assert_configs

from .fixtures import (  # noqa pytest fixture
    all_env_vars,
    default_config,
    env_config,
    mock_env_vars,
)


def assert_default(value: bool):
    assert value


@pytest.mark.usefixtures("mock_env_vars")
def test_env_config(env_config: GraphRagConfig):  # noqa: F811
    print(env_config.model_dump_json(indent=2))
    config = load_config()
    print(config.model_dump_json(indent=2))
    assert_configs(config, env_config)
