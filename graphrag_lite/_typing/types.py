from typing import Any

from pydantic import BaseModel
from collections.abc import Callable

from devtools import pformat
from pydantic import BaseModel, Field


class SubscriptableBaseModel(BaseModel):
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        if key in self.model_fields_set:
            return True
        if key in self.model_fields:
            return self.model_fields[key].default is not None
        return False



class Config(BaseModel):
    """Contains default configurations for GraphRAGLite."""

    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json(indent=4)

