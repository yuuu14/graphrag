

from dataclasses import dataclass, field


@dataclass
class ProgressHolder:
    progress_id: str
    
    progress: float = field(default_factory=float)
    """ processed / total """

    processed: int = field(default_factory=int)

    total: int = field(default_factory=int)