"""
parser.py
Build a DAG of Operator objects from a YAML config (ops/* only).

Config schema (no legacy node keys):
- ops:
    <op_id>:
        model: <str>
        input_ops: [<op_id>, ...]     # optional
        output_ops: [<op_id>, ...]    # optional
        prompt: <str>                 # optional
        temperature: <float>          # default 0.7
        top_p: <float>                # default 0.9
        max_tokens: <int>             # default 256
        max_batch_size: <int|inf>     # default torch.inf
        dtype: <"bfloat16"|"float16"|...>    # default "bfloat16"
        quantization: <any>           # optional
        lora_config: <dict|None>      # optional
        max_model_len: <int|None>     # optional
        min_tokens: <int>             # default 0
        use_chat_template: <bool>     # default False
        keep_cache: <bool>            # optional; overrides inference
- start_ops: [<op_id>, ...]
- end_ops:   [<op_id>, ...]

During build we also:
- infer keep_cache if any downstream op shares the same model (unless keep_cache is explicitly set)
- initialize runtime fields: data_parallel, is_duplicate, duplicate_info, main_op, benchmark, max_distance
- compute max_distance to any end-op (with cycle detection)
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import yaml
import torch

from halo.components import Operator, ModelConfig


# ---------------- Public API ---------------- #

def load_config(config_path: str) -> dict:
    """Load YAML config as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_ops_from_config(config: dict) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
    """
    Build Operator graph from a config dict (ops/* only) and compute max_distance per op.

    Returns
    -------
    ops        : dict[str, Operator]
    start_ops  : list[Operator]
    end_ops    : list[Operator]
    models     : set[str]
    """
    # --- Required top-level keys ---
    conf_ops = config.get("ops")
    if not isinstance(conf_ops, dict) or not conf_ops:
        raise ValueError("Config must contain a non-empty 'ops' mapping.")
    start_keys = config.get("start_ops")
    end_keys = config.get("end_ops")
    if not isinstance(start_keys, list) or not start_keys:
        raise ValueError("'start_ops' must be a non-empty list of op ids.")
    if not isinstance(end_keys, list) or not end_keys:
        raise ValueError("'end_ops' must be a non-empty list of op ids.")

    # --- Create op objects ---
    ops: Dict[str, Operator] = {oid: Operator(id=oid) for oid in conf_ops.keys()}

    # --- Validate references & required fields ---
    for oid, spec in conf_ops.items():
        if "model" not in spec:
            raise ValueError(f"Op '{oid}' is missing required field 'model'.")
        in_ids = spec.get("input_ops", []) or []
        out_ids = spec.get("output_ops", []) or []
        for rid in in_ids + out_ids:
            if rid not in ops:
                raise ValueError(f"Op '{oid}' references unknown op '{rid}' in inputs/outputs.")

    # --- Link edges, attach ModelConfig, init runtime fields ---
    models: Set[str] = set()
    for oid, op in ops.items():
        spec = conf_ops[oid]
        input_ids = spec.get("input_ops", []) or []
        output_ids = spec.get("output_ops", []) or []

        op.input_ops = [ops[k] for k in input_ids]
        op.output_ops = [ops[k] for k in output_ids]

        model = spec["model"]
        models.add(model)

        # keep_cache: explicit override takes precedence; otherwise infer by same-model downstream
        explicit_keep = spec.get("keep_cache", None)
        inferred_keep = any(conf_ops[oid2]["model"] == model for oid2 in output_ids)
        op.keep_cache = bool(explicit_keep) if explicit_keep is not None else inferred_keep

        op.model_config = ModelConfig(
            model_name=model,
            system_prompt=spec.get("prompt"),
            temperature=spec.get("temperature", 0.7),
            top_p=spec.get("top_p", 0.9),
            max_tokens=spec.get("max_tokens", 256),
            max_batch_size=spec.get("max_batch_size", torch.inf),
            dtype=spec.get("dtype", "bfloat16"),
            quantization=spec.get("quantization", None),
            lora_config=spec.get("lora_config", None),
            max_model_len=spec.get("max_model_len", None),
            min_tokens=spec.get("min_tokens", 0),
            use_chat_template=spec.get("use_chat_template", True),
        )

        # runtime/scheduler fields
        op.data_parallel = False
        op.is_duplicate = False
        op.duplicate_info = None   # or [dup_index, total_dup]
        op.main_op = None          # when duplicate, refers to the main op
        op.max_distance = -1

    # --- Resolve start/end ops ---
    for k in start_keys + end_keys:
        if k not in ops:
            raise ValueError(f"Unknown op id '{k}' referenced in start_ops/end_ops.")
    start_ops = [ops[k] for k in start_keys]
    end_ops = [ops[k] for k in end_keys]

    # --- Compute longest distance to any end-op (with cycle detection) ---
    _compute_max_distances(ops, end_ops)

    return ops, start_ops, end_ops, models


def build_from_path(config_path: str) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
    """Convenience wrapper: load + build in one call."""
    return build_ops_from_config(load_config(config_path))


# ---------------- Internals ---------------- #

def _compute_max_distances(ops: Dict[str, Operator], end_ops: List[Operator]) -> None:
    """
    For each op, compute the longest distance (in edges) to ANY end-op.
    If an op cannot reach any end-op, its distance is -1.
    Detect cycles and raise ValueError if found.
    """
    end_set = set(end_ops)
    memo: Dict[Operator, int] = {}
    visiting: Set[Operator] = set()  # for cycle detection

    def dfs(op: Operator) -> int:
        if op in memo:
            return memo[op]
        if op in visiting:
            raise ValueError("Cycle detected in the op graph.")
        visiting.add(op)

        if op in end_set:
            memo[op] = 0
            visiting.remove(op)
            return 0

        best = -1
        for child in op.output_ops:
            d = dfs(child)
            if d != -1:  # reachable to an end-op
                best = max(best, d + 1)

        memo[op] = best
        visiting.remove(op)
        return best

    # Run for all ops and write back
    for op in ops.values():
        op.max_distance = dfs(op)
