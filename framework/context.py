from __future__ import annotations

from typing import Any, Dict, Iterable, List


class TrainContext:
    def __init__(self, **initial):
        self.storage: Dict[str, Any] = {}
        for key, value in initial.items():
            self.set(key, value)

    def set(self, path: str, value: Any):
        if not isinstance(path, str) or not path:
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        parts = path.split(".")
        cur = self.storage
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value

    def get(self, path: str, default: Any = None, *, required: bool = True):
        if not isinstance(path, str) or not path:
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        parts = path.split(".")
        cur: Any = self.storage
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                if required:
                    available = "\n  ".join(self.keys())
                    raise KeyError(f"missing ctx key '{path}'. Available keys:\n  {available}")
                return default
        return cur

    def has(self, path: str) -> bool:
        try:
            self.get(path)
            return True
        except KeyError:
            return False

    def update_dict(self, prefix: str, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError(f"update_dict expects dict, got {type(data)}")
        for key, value in data.items():
            self.set(f"{prefix}.{key}", value)

    def keys(self) -> List[str]:
        out: List[str] = []

        def visit(prefix: str, value: Any):
            if isinstance(value, dict):
                for k, v in value.items():
                    visit(f"{prefix}.{k}" if prefix else k, v)
            else:
                out.append(prefix)

        visit("", self.storage)
        return out

    def as_dict(self) -> Dict[str, Any]:
        return self.storage
