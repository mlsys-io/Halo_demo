from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, Mapping

from .models import Node

ProcessorFn = Callable[[Mapping[str, Any], Mapping[str, Any], Node], Dict[str, Any]]

_PROCESSOR_REGISTRY: Dict[str, ProcessorFn] = {}


def register_processor(name: str, func: ProcessorFn) -> None:
    if not name:
        raise ValueError("Processor name must be non-empty.")
    _PROCESSOR_REGISTRY[name] = func


def run_processor_node(node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
    processor_name = node.raw.get("processor")
    if not processor_name:
        raise RuntimeError(f"Processor node '{node.id}' missing 'processor' field.")
    processor = _PROCESSOR_REGISTRY.get(processor_name)
    if processor is None:
        raise RuntimeError(f"Unknown processor '{processor_name}' on node '{node.id}'.")
    config = node.raw.get("config") or {}
    if not isinstance(config, Mapping):
        config = {}
    outputs = processor(context, config, node)
    if outputs is None:
        return {}
    if isinstance(outputs, Mapping):
        return dict(outputs)
    if node.outputs:
        return {node.outputs[0]: outputs}
    return {}


def _strip_quotes(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _normalize_enum(value: Any, options: list[Any]) -> Any:
    if not isinstance(value, str):
        return value
    normalized = re.sub(r"\s+", " ", value.strip()).upper()
    normalized_no_space = normalized.replace(" ", "")
    for opt in options:
        opt_text = str(opt)
        opt_norm = re.sub(r"\s+", " ", opt_text.strip()).upper()
        if normalized == opt_norm:
            return opt
    for opt in options:
        opt_text = str(opt)
        opt_norm = re.sub(r"\s+", " ", opt_text.strip()).upper()
        if normalized_no_space == opt_norm.replace(" ", ""):
            return opt
    for opt in options:
        opt_text = str(opt)
        opt_norm = re.sub(r"\s+", " ", opt_text.strip()).upper()
        if opt_norm.startswith(normalized):
            return opt
    return normalized


def _normalize_date(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().replace("/", "-")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        return text


def _normalize_numeric(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return value
    try:
        return float(text)
    except ValueError:
        return value


def _apply_value(value: Any, func: Callable[[Any], Any]) -> Any:
    if isinstance(value, list):
        return [func(item) for item in value]
    return func(value)


def _compile_pattern(pattern: Any) -> re.Pattern[str] | None:
    if isinstance(pattern, str):
        return re.compile(pattern)
    if isinstance(pattern, Mapping):
        raw = pattern.get("pattern") or pattern.get("regex")
        if not raw:
            return None
        flags = 0
        flag_spec = pattern.get("flags")
        if isinstance(flag_spec, str):
            flag_spec = [flag_spec]
        if isinstance(flag_spec, list):
            for flag in flag_spec:
                if not isinstance(flag, str):
                    continue
                if flag.lower() == "i":
                    flags |= re.IGNORECASE
        return re.compile(str(raw), flags=flags)
    return None


def _extract_match_value(match: re.Match[str]) -> Any:
    groups = match.groups()
    if not groups:
        return match.group(0)
    if len(groups) == 1:
        return groups[0]
    return tuple(groups)


def regex_param_extractor(
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    node: Node,
) -> Dict[str, Any]:
    input_key = node.inputs[0] if node.inputs else "user_query"
    raw_query = context.get(input_key, "")
    if raw_query is None:
        raw_query = ""
    if not isinstance(raw_query, str):
        raw_query = str(raw_query)

    enums = config.get("enums") or {}
    regex_rules = config.get("regex") or {}
    defaults = config.get("defaults") or {}
    normalize_cfg = config.get("normalize") or {}
    allow_multi = set(config.get("allow_multi") or [])

    extracted: Dict[str, Any] = {}

    if isinstance(regex_rules, Mapping):
        for field, patterns in regex_rules.items():
            if not patterns:
                continue
            if not isinstance(patterns, list):
                patterns = [patterns]
            matches: list[Any] = []
            for pattern in patterns:
                compiled = _compile_pattern(pattern)
                if compiled is None:
                    continue
                match = compiled.search(raw_query)
                if not match:
                    continue
                value = _extract_match_value(match)
                if isinstance(value, tuple) and len(value) > 1:
                    if field.endswith("2"):
                        base = field[:-1]
                        if base and base not in extracted:
                            extracted[base] = value[0]
                        if field not in extracted:
                            extracted[field] = value[1]
                        matches.append(value[1])
                        break
                    value = value[0]
                matches.append(value)
                if field not in allow_multi:
                    break
            if matches:
                if field in allow_multi:
                    extracted.setdefault(field, []).extend(matches)
                else:
                    extracted.setdefault(field, matches[0])

    params: Dict[str, Any] = {}
    if isinstance(defaults, Mapping):
        params.update(defaults)
    params.update(extracted)

    uppercase_fields = set(normalize_cfg.get("uppercase_fields") or [])
    date_fields = set(normalize_cfg.get("date_fields") or [])
    numeric_fields = set(normalize_cfg.get("numeric_fields") or [])
    strip_quotes = bool(normalize_cfg.get("strip_quotes"))

    for key, value in list(params.items()):
        if strip_quotes:
            value = _apply_value(value, _strip_quotes)
        if key in uppercase_fields:
            value = _apply_value(value, lambda v: v.upper() if isinstance(v, str) else v)
        if key in enums and isinstance(enums, Mapping):
            options = enums.get(key) or []
            if isinstance(options, list):
                value = _apply_value(value, lambda v: _normalize_enum(v, options))
        if key in date_fields:
            value = _apply_value(value, _normalize_date)
        if key in numeric_fields:
            value = _apply_value(value, _normalize_numeric)
        params[key] = value

    output_key = node.outputs[0] if node.outputs else "params"
    return {output_key: params}


register_processor("regex_param_extractor", regex_param_extractor)
