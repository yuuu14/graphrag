# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Default method for loading config."""

from pathlib import Path
from typing import Any

from essex_config.config import load_config as lc
from essex_config.sources import ArgSource, EnvSource, FileSource, Source

from .defaults import CONFIG_FILENAME, SUPPORTED_CONFIG_EXTENSIONS
from .models.graph_rag_config import GraphRagConfig


def search_for_config_in_root_dir(root: str | Path) -> Path | None:
    """Resolve the config path from the given root directory.

    Parameters
    ----------
    root : str | Path
        The path to the root directory containing the config file.
        Searches for a default config file (settings.{yaml,yml,json}).

    Returns
    -------
    Path | None
        returns a Path if there is a config in the root directory
        Otherwise returns None.
    """
    supported_configs = [
        f"{CONFIG_FILENAME}.{ext}" for ext in SUPPORTED_CONFIG_EXTENSIONS
    ]
    root = Path(root).resolve()

    if not root.is_dir():
        msg = f"Invalid config path: {root} is not a directory"
        raise FileNotFoundError(msg)

    for file in supported_configs:
        if (root / file).is_file():
            return root / file

    return None


def _search_for_env_file(path: str | Path) -> Path | None:
    """Search for an environment file in the given directory.

    Parameters
    ----------
    path : str | Path
        The path to the directory to search for the environment file.

    Returns
    -------
    Path | None
        The path to the environment file if found, otherwise None.
    """
    path = Path(path).resolve() / ".env"
    return path if path.is_file() else None


def load_config(
    directory_or_file_path: str | Path | None = None, overrides: dict[str, Any] = {}
) -> GraphRagConfig:
    """Load configuration from a file or create a default configuration.

    If a config file is not found the default configuration is created.

    Parameters
    ----------
    directory_or_file_path : str | Path | None
        The directory or file path to the config file.
        If filepath, load the configuration from the file.
        If directory, search for a supported config file in the directory.
        If None, create a default configuration.

    Returns
    -------
    GraphRagConfig
        The configuration object.
    """
    sources: list[Source] = [ArgSource(**overrides)]
    if directory_or_file_path:
        path = Path(directory_or_file_path)
        if path.is_dir():
            config_path = search_for_config_in_root_dir(path)
            if config_path:
                sources.append(FileSource(config_path, required=True))
        else:
            sources.append(FileSource(path, required=True))
        env_file = _search_for_env_file(path if path.is_dir() else path.parent)
        if env_file:
            sources.append(EnvSource(env_file, required=True, prefix="GRAPHRAG"))
    else:
        sources.append(EnvSource(prefix="GRAPHRAG"))
    return lc(GraphRagConfig, sources=sources, parse_env_values=True)
