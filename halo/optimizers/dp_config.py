"""Configuration object for ``DPSolver``.

The solver historically took 25+ keyword arguments and read another 10
environment variables inside ``__init__``. That coupling made it hard to:

  * see what tunable knobs exist
  * keep the Rust backend's input schema in sync (each new weight had to be
    wired through ``DPSolver``, ``GraphOptimizer``, ``_build_rust_input`` and
    ``parse_input`` in ``lib.rs``)
  * write unit-style tests that override only one knob

``DPSolverConfig`` is a frozen dataclass that captures every cost-model
weight, every pruning toggle, and every observability flag in one place.
The solver's constructor still accepts the legacy individual kwargs for
back-compat, but converts them into a ``DPSolverConfig`` immediately and
relies on it from there on.

Env-variable handling is centralized in ``DPSolverConfig.from_env`` so it
can be reasoned about as a single layer instead of being inlined into 10
spots in ``DPSolver.__init__``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Mapping


def _read_float_env(name: str, default: float | None = None) -> float | None:
    """Parse a non-empty env var as float; return default on missing/invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


@dataclass(frozen=True)
class DPSolverConfig:
    """Static configuration for ``DPSolver``.

    Every field has a single source of truth: this dataclass. ``DPSolver``
    reads ``self.config.<field>`` rather than carrying duplicate ``self.<field>``
    attributes. ``RustPayload.from_config`` projects this dataclass onto the
    payload schema for the Rust backend, so adding a new knob is a 3-step
    change (field here, projection in ``RustPayload``, parse in ``lib.rs``).
    """

    # --- window / dedup -----------------------------------------------
    window_size: int = 2

    # --- cost-model weights -------------------------------------------
    cpu_load_cost_weight: float = 1.0
    cpu_load_early_weight: float = 2.0
    """Boost CPU-load weight for early epochs: weight *= 1 + early/(1+epoch)."""
    gpu_cost_max_weight: float = 0.9
    """Weight on max-over-GPUs cost (makespan-like)."""
    gpu_cost_sum_weight: float = 0.1
    """Weight on sum-over-GPUs cost (throughput-like)."""
    gpu_depth_cost_weight: float = 1.0
    """Penalty per unit of GPU-subgraph depth selected in an epoch."""
    epoch_penalty_weight: float = 1.0
    """Scalar multiplier applied to the GraphOptimizer's default
    ``epoch_penalty_fn``. ``DPSolver`` itself uses the function directly;
    this is included so payloads round-trip.
    """
    switch_penalty_weight: float = 0.1
    """RESERVED. Not currently consumed by ``_solve`` — kept for paper
    parity (model-switch penalty) so future cost-model changes can land
    without renaming the field.
    """

    # --- toggles -------------------------------------------------------
    disable_epoch_batch_cost: bool = False
    disable_cpu_load_cost: bool = False
    cpu_cost_mode: str = "default"
    """Either ``"default"`` (per-query plan cost summed) or ``"naive"``
    (count of DB ancestors per GPU node). ``"naive"`` is a baseline."""

    # --- pruning -------------------------------------------------------
    enable_batch_shape_pruning: bool = True
    gpu_batch_slack: int = 1
    enable_lower_bound_pruning: bool = True
    enable_worker_symmetry: bool = True
    lower_bound_cost_factor: float = 0.2

    # --- observability -------------------------------------------------
    debug_log: bool = True
    debug_every: int = 100000

    # --- backend selection --------------------------------------------
    prefer_rust: bool = False

    # Map of env-var name -> field name for ``from_env``. Listing them here
    # makes the env contract explicit and discoverable.
    _ENV_MAP: ClassVar[Mapping[str, str]] = {
        "HALO_DP_CPU_LOAD_COST_WEIGHT": "cpu_load_cost_weight",
        "HALO_DP_CPU_EARLY_WEIGHT": "cpu_load_early_weight",
        "HALO_DP_GPU_COST_MAX_WEIGHT": "gpu_cost_max_weight",
        "HALO_DP_GPU_COST_SUM_WEIGHT": "gpu_cost_sum_weight",
        "HALO_DP_GPU_DEPTH_WEIGHT": "gpu_depth_cost_weight",
        "HALO_DP_SWITCH_PENALTY_WEIGHT": "switch_penalty_weight",
        "HALO_DP_CPU_COST_MODE": "cpu_cost_mode",
        "HALO_DP_USE_RUST": "prefer_rust",
    }

    # --- factory helpers ----------------------------------------------

    @classmethod
    def from_env(cls, base: "DPSolverConfig | None" = None) -> "DPSolverConfig":
        """Overlay any ``HALO_DP_*`` env vars on top of ``base`` (or defaults)."""
        cfg = base or cls()
        overrides: dict[str, Any] = {}
        for env_name, field_name in cls._ENV_MAP.items():
            current = getattr(cfg, field_name)
            if isinstance(current, bool):
                overrides[field_name] = _read_bool_env(env_name, current)
            elif isinstance(current, int) and not isinstance(current, bool):
                v = _read_int_env(env_name, current)
                if v is not None:
                    overrides[field_name] = v
            elif isinstance(current, float):
                v = _read_float_env(env_name, current)
                if v is not None:
                    overrides[field_name] = v
            elif isinstance(current, str):
                raw = os.getenv(env_name)
                if raw is not None and raw.strip():
                    overrides[field_name] = raw.strip()
            # Other types are not env-overridable.
        return replace(cfg, **overrides) if overrides else cfg

    def normalized(self) -> "DPSolverConfig":
        """Return a config with the GPU max/sum weights re-normalized to sum to 1.

        The DP solver historically required ``max_w + sum_w == 1`` so that
        the GPU cost is a convex combination. Callers used to pass either
        weight and rely on the constructor to normalize; we preserve that
        behavior by exposing it as a method.
        """
        max_w = max(0.0, float(self.gpu_cost_max_weight))
        sum_w = max(0.0, float(self.gpu_cost_sum_weight))
        total = max_w + sum_w
        if total <= 0:
            max_w, sum_w = 0.9, 0.1
        else:
            max_w /= total
            sum_w /= total
        new_cpu_cost_mode = (self.cpu_cost_mode or "default").strip().lower()
        if new_cpu_cost_mode not in ("default", "naive"):
            raise ValueError(
                f"Unsupported cpu_cost_mode '{self.cpu_cost_mode}'. "
                f"Expected 'default' or 'naive'."
            )
        return replace(
            self,
            gpu_cost_max_weight=max_w,
            gpu_cost_sum_weight=sum_w,
            cpu_load_cost_weight=max(0.0, float(self.cpu_load_cost_weight)),
            cpu_load_early_weight=max(0.0, float(self.cpu_load_early_weight)),
            gpu_depth_cost_weight=max(0.0, float(self.gpu_depth_cost_weight)),
            switch_penalty_weight=max(0.0, float(self.switch_penalty_weight)),
            window_size=max(1, int(self.window_size)),
            gpu_batch_slack=max(0, int(self.gpu_batch_slack)),
            lower_bound_cost_factor=max(0.0, float(self.lower_bound_cost_factor)),
            debug_every=max(1, int(self.debug_every)),
            cpu_cost_mode=new_cpu_cost_mode,
        )

    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))


__all__ = ["DPSolverConfig"]
