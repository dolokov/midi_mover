"""Minimal YAML fallback parser used when PyYAML is unavailable."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def load_simple_yaml(raw_text: str, config_path: Path) -> dict[str, Any]:
    """Parse a constrained YAML subset for environments without PyYAML."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(
                f"Failed to parse YAML config at {config_path}: line {line_number} uses unsupported indentation."
            )
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError(
                f"Failed to parse YAML config at {config_path}: block list syntax is unsupported by the built-in YAML fallback on line {line_number}."
            )
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(
                f"Failed to parse YAML config at {config_path}: expected 'key: value' on line {line_number}."
            )
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(
                f"Failed to parse YAML config at {config_path}: empty key on line {line_number}."
            )
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
