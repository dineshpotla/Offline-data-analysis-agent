import json
import os
from typing import Any, Dict, List


class JSONLMemory:
    """Append-only JSONL memory store."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None

    def append(self, item: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")

    def read_last(self, n: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]
