from pathlib import Path

import pytest
import yaml

from graphrag.config2 import GraphRagConfig

DIRNAME = Path(__file__).parent


@pytest.fixture
def settings_config() -> GraphRagConfig:
    settings_path = (DIRNAME / "settings_config" / "settings.yaml").resolve()
    with settings_path.open("rb") as file:
        return GraphRagConfig(
            **yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict")),
        )
