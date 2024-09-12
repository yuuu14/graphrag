# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the default configuration.

Test loading configs without using config files (settings.yaml, .env)
"""

from pathlib import Path

from graphrag.config2 import GraphRagConfig, load_config
from tests.unit.config.assert_configs import assert_configs

from .fixtures import all_env_vars, env_config, settings_config  # noqa pytest fixtures

DIRNAME = Path(__file__).parent


def _write_env_file(path: Path, env_vars: dict[str, str]) -> None:
    with path.open("w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")


def test_with_dotenv_file(env_config: GraphRagConfig, all_env_vars: dict[str, str]):  # noqa: F811
    root_dir = (DIRNAME / "fixtures" / "dotenv_config").resolve()
    dotenv_path = (root_dir / ".env").resolve()
    _write_env_file(dotenv_path, all_env_vars)
    env_config.root_dir = str(root_dir)
    config = load_config(root_dir)
    assert_configs(config, env_config)


def test_with_root_dir(settings_config: GraphRagConfig):  # noqa: F811
    root_dir = (DIRNAME / "fixtures" / "settings_config").resolve()
    config = load_config(root_dir)
    assert_configs(config, settings_config)
