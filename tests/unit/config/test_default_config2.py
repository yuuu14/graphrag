from graphrag.config2 import load_config
from tests.unit.config.assert_configs import assert_configs

from .fixtures import default_config  # noqa


def assert_default(value: bool):
    assert value


def test_default_config(default_config):
    print(default_config.model_dump_json(indent=2))
    config = load_config()
    print(config.model_dump_json(indent=2))
    assert_configs(config, default_config)
