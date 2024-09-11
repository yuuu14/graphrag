# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Common field parsers."""

from typing import Any


def parse_string_list(x: Any, _: Any) -> Any:
    """Parse a string list."""
    if isinstance(x, str):
        return x.split(",")
    return x
