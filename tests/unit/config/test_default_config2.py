from graphrag.config2 import GraphRagConfig, load_config
from tests.unit.config.assert_configs import assert_configs

from .fixtures import MOCK_API_KEY, default_config  # noqa


def test_default_config(default_config: GraphRagConfig):  # noqa: F811
    print(default_config.model_dump_json(indent=2))
    config = load_config(overrides={"api_key": MOCK_API_KEY})
    print(config.model_dump_json(indent=2))
    assert_configs(config, default_config)
