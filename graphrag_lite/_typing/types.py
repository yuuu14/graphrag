from typing import Any

from pydantic import BaseModel


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
