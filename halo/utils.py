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


def as_bool(value: Any) -> bool:
    """Interpret a YAML flag; quoted strings such as ``"false"`` count as false."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# Keys an HTTP operator may use to declare its latency, with their scale to seconds.
HTTP_LATENCY_KEYS = (
    ("sleep_s", 1.0),
    ("sleep_ms", 0.001),
    ("latency_s", 1.0),
    ("latency_ms", 0.001),
    ("timeout_s", 1.0),
    ("timeout_ms", 0.001),
)

# On a node with a ``url``, these declare the latency of the real call (the request is
# issued and the call padded to a sampled latency), the sleep keys inject latency alone
# (the url is not contacted), and ``timeout_s``/``timeout_ms`` set the request timeout.
HTTP_PADDED_LATENCY_KEYS = (("latency_s", 1.0), ("latency_ms", 0.001))
HTTP_SLEEP_KEYS = ("sleep_s", "sleep_ms")


def http_latency_seconds(
    raw: Mapping[str, Any],
    render: Callable[[str], str] | None = None,
    keys: Sequence[tuple[str, float]] = HTTP_LATENCY_KEYS,
) -> float | None:
    """The first parseable declared latency in ``raw`` among ``keys`` (seconds, >= 0), or None."""
    for key, scale in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, str):
            value = (render(value) if render else value).strip()
        try:
            return max(0.0, float(value) * scale)
        except (TypeError, ValueError):
            continue
    return None


def http_cost(node: Any, profiled: Mapping[str, float] | None) -> float:
    """Planning cost of an HTTP operator: its profiled latency, else the declared one."""
    if profiled and profiled.get(node.id) is not None:
        try:
            return max(0.0, float(profiled[node.id]))
        except (TypeError, ValueError):
            pass
    raw = node.raw if isinstance(node.raw, Mapping) else {}
    seconds = http_latency_seconds(raw)
    return 0.0 if seconds is None else seconds
