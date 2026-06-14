"""Single source of truth for the DP payload sent into the Rust backend.

Before this file existed, the payload dict was built inline inside
``DPSolver._build_rust_input`` and the Rust ``parse_input`` in
``halo_dev/optimizers/dp_core_rs/src/lib.rs`` listed the same keys by string.
Every new cost-model knob required four parallel edits (config, solver,
``_build_rust_input``, ``parse_input``) and silent drift was common.

The dataclass below names every payload field exactly once. The Python
side calls ``.to_dict()``; the Rust side reads the matching string keys.
Each field has an inline note pointing at the Rust struct member it
corresponds to so cross-language reviews stay shallow.

When you add a field:
1. Add it here with the right type and a short note.
2. Add a key extract in ``parse_input`` in
   ``halo_dev/optimizers/dp_core_rs/src/lib.rs``.
3. Wire the value through ``DPSolver._build_rust_input``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List

from ..models import QueryPlanChoice


# Schema version: bump when payload shape changes incompatibly. The Rust
# backend does not yet enforce this — it's here so we can fail fast in
# Python if someone forgets to rebuild the .so after a payload edit.
PAYLOAD_SCHEMA_VERSION = 2


@dataclass
class RustPayload:
    """Schema for the dict passed into ``_dp_core_rs.solve``.

    Field names map 1:1 to keys checked by Rust's ``parse_input``.
    """

    # --- nodes / graph topology -------------------------------------------
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    """One dict per scheduled node:
       ``{id, is_gpu, model_id, queries: [{name, plans: [PlanEntry...]}]}``.
    """
    parents_mask: List[int] = field(default_factory=list)
    """Bitmask of parents per node (all kinds)."""
    gpu_parents_mask: List[int] = field(default_factory=list)
    """Bitmask of GPU-only parents per node, used for batch enumeration."""
    gpu_node_indices: List[int] = field(default_factory=list)
    db_node_indices: List[int] = field(default_factory=list)

    # --- workers ----------------------------------------------------------
    worker_ids: List[str] = field(default_factory=list)
    worker_kinds: List[str] = field(default_factory=list)
    """Parallel to worker_ids; each entry is 'gpu' or 'cpu'."""
    gpu_worker_indices: List[int] = field(default_factory=list)
    cpu_worker_indices: List[int] = field(default_factory=list)
    node_worker_options: List[List[int]] = field(default_factory=list)
    """Per node, the set of worker indices allowed to run that node."""

    # --- cost lookup tables (pre-computed in Python) -----------------------
    exec_costs: List[List[float]] = field(default_factory=list)
    """exec_costs[node][worker] = base exec cost in seconds."""
    model_init_costs: List[List[float]] = field(default_factory=list)
    """model_init_costs[node][model_offset] where slot 0 == ``none_id``."""
    llm_bonus: List[List[float]] = field(default_factory=list)
    """llm_bonus[node][last_node_offset] — cache-hit multiplier for LLM cost."""
    node_min_cost: List[float] = field(default_factory=list)
    """Pre-computed lower bound for branch-and-bound."""
    epoch_penalties: List[float] = field(default_factory=list)
    """epoch_penalties[i] = epoch_penalty_fn(i) for i in 0..|nodes|."""
    cpu_dep_counts: List[float] = field(default_factory=list)
    """|DB ancestors| per GPU node; used when cpu_cost_mode == 'naive'."""

    # --- state encoding ---------------------------------------------------
    initial_worker_states: List[Dict[str, int]] = field(default_factory=list)
    window_size: int = 2
    none_id: int = -1
    raw_cost_scale: float = 3.65e-6

    # --- cost-model weights (must mirror DPSolverConfig) -------------------
    cpu_load_cost_weight: float = 1.0
    cpu_load_early_weight: float = 2.0
    gpu_cost_max_weight: float = 0.9
    gpu_cost_sum_weight: float = 0.1
    gpu_depth_cost_weight: float = 1.0

    # --- toggles ----------------------------------------------------------
    disable_epoch_batch_cost: bool = False
    disable_cpu_load_cost: bool = False
    cpu_cost_mode: str = "default"
    enable_batch_shape_pruning: bool = True
    enable_lower_bound_pruning: bool = True
    enable_worker_symmetry: bool = True
    gpu_batch_slack: int = 1
    lower_bound_cost_factor: float = 0.2

    # --- Python callbacks the Rust backend has to keep crossing the GIL for
    cache_multiplier_fn: Callable[..., float] | None = None
    """Called as ``cache_multiplier_fn(window_signatures, plan_choice)``."""
    id_to_signature: List[Any] = field(default_factory=list)
    """Index-keyed table of ``QuerySignature`` objects; Rust passes the
    relevant slice into ``cache_multiplier_fn`` so the function signature
    stays unchanged."""

    schema_version: int = PAYLOAD_SCHEMA_VERSION

    # ---- factories ------------------------------------------------------

    @staticmethod
    def plan_entry(choice: QueryPlanChoice, base_cost: float, sig_id: int) -> Dict[str, Any]:
        """Build a single plan dict used inside ``nodes[i].queries[j].plans``."""
        return {
            "plan_id": choice.plan_id,
            "base_cost": float(base_cost),
            "sig_id": int(sig_id),
            "choice_obj": choice,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the dict shape expected by ``_dp_core_rs.solve``.

        We intentionally avoid ``dataclasses.asdict``: it deep-copies every
        leaf, which on the payload's nested lists (``exec_costs``,
        ``model_init_costs``, ...) is dominated by the deep copy and ends
        up being many times slower than the Rust solver itself on small
        graphs. The fields are simple, named, and known here, so a flat
        attribute-to-dict projection is both faster and clearer.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


__all__ = ["RustPayload", "PAYLOAD_SCHEMA_VERSION"]
