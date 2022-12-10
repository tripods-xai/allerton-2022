from typing import Any, Dict, Protocol


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Any]
