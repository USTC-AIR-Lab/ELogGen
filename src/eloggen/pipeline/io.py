"""Small JSON/YAML-compatible file helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_structured_file(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"{path} is not valid JSON and PyYAML is not available. "
            "Use JSON-compatible YAML for dependency-free loading."
        ) from exc
    return yaml.safe_load(text)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

