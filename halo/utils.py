from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

PLACEHOLDER_PATTERN = re.compile(r"{{\s*([^}]+?)\s*}}")
MISSING = object()


def render_template(
    template: str,
    context: Mapping[str, Any],
    *,
    on_missing: Callable[[str, str], None] | None = None,
) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        value = lookup_path(context, key, default=MISSING)
        if value is MISSING:
            if on_missing:
                on_missing(key, template)
            return f"{{{{ {key} }}}}"
        return str(value)

    return PLACEHOLDER_PATTERN.sub(replace, template)


def maybe_parse_json(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def lookup_path(obj: Any, path: str, *, default: Any = MISSING) -> Any:
    """Traverse dotted/indexed path within nested mappings/sequences."""
    current: Any = obj
    for part in filter(None, (segment.strip() for segment in path.split("."))):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                idx = int(part)
            except ValueError:
                return default
            if idx < 0 or idx >= len(current):
                return default
            current = current[idx]
            continue
        return default
    return current
