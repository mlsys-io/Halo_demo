from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ..models import ExecutionPlan, ExecutionTask, GraphSpec, Worker
from .topo_utils import default_worker_filter, filtered_dependencies, topological_order


def build_rr_topo_plan(
    graph: GraphSpec,
    workers: Mapping[str, Worker],
    *,
    dependencies: Mapping[str, Sequence[str]] | None = None,
    schedulable_ids: Sequence[str] | None = None,
    node_worker_options: Mapping[str, Sequence[str]] | None = None,
    plan_choices: Mapping[tuple[str, str], Sequence] | None = None,
    selected_query_plans: Mapping[tuple[str, str], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionPlan:
    """Round-robin, topology-only scheduler."""
    worker_ids = tuple(sorted(workers))
    if not worker_ids:
        raise ValueError("RR-Topo requires at least one worker.")
    gpu_workers = tuple(sorted(wid for wid, worker in workers.items() if worker.kind == "gpu"))
    cpu_workers = tuple(sorted(wid for wid, worker in workers.items() if worker.kind != "gpu"))

    schedulable = tuple(
        sorted(
            schedulable_ids
            if schedulable_ids is not None
            else (nid for nid, node in graph.nodes.items() if node.type != "input")
        )
    )
    filtered_deps = filtered_dependencies(graph, schedulable, dependencies)
    order = topological_order(filtered_deps, schedulable)

    options: Dict[str, Tuple[str, ...]] = {}
    if node_worker_options is not None:
        for node_id in schedulable:
            opts = tuple(node_worker_options.get(node_id, ()))
            if not opts:
                raise RuntimeError(f"No eligible workers for node '{node_id}'")
            options[node_id] = opts
    else:
        for node_id in schedulable:
            node = graph.nodes[node_id]
            eligible = tuple(wid for wid, worker in workers.items() if default_worker_filter(node, worker))
            if not eligible:
                raise RuntimeError(f"No eligible workers for node '{node_id}'")
            options[node_id] = eligible

    tasks: list[ExecutionTask] = []
    gpu_idx = 0
    cpu_idx = 0
    for node_id in order:
        eligible = options[node_id]
        node = graph.nodes[node_id]
        is_gpu = node.engine == "vllm"
        if is_gpu:
            pool = gpu_workers or tuple(sorted(eligible))
            pool_idx = gpu_idx
            gpu_idx += 1
        else:
            pool = cpu_workers or tuple(sorted(eligible))
            pool_idx = cpu_idx
            cpu_idx += 1

        start = pool_idx % len(pool)
        chosen = None
        for offset in range(len(pool)):
            cand = pool[(start + offset) % len(pool)]
            if cand in eligible:
                chosen = cand
                break
        if chosen is None:
            chosen = eligible[pool_idx % len(eligible)]

        epoch = pool_idx // len(pool)
        tasks.append(
            ExecutionTask(
                node_id=node_id,
                worker_id=chosen,
                dependencies=filtered_deps.get(node_id, ()),
                epoch=epoch,
            )
        )

    return ExecutionPlan(
        workers=dict(workers),
        tasks=tuple(tasks),
        query_plans=plan_choices or {},
        selected_query_plans=selected_query_plans or {},
        metadata=dict(metadata or {}),
    )
