# Copyright (c) 2025 @ SUPCON.
"""自定义的ErrorType."""


class TooManyRetriesError(ValueError):
    def __init__(self, max_retries: int):
        msg = f"Max Retries larger than 5: {max_retries}."
        super().__init__(msg)