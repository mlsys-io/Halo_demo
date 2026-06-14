from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from ..models import ExecutionPlan, ExecutionTask, GraphSpec, Worker, Node
from .topo_utils import default_worker_filter, filtered_dependencies, topological_order


def build_greedy_cost_plan(
    graph: GraphSpec,
    workers: Mapping[str, Worker],
    *,
    dependencies: Mapping[str, Sequence[str]] | None = None,
    schedulable_ids: Sequence[str] | None = None,
    node_worker_options: Mapping[str, Sequence[str]] | None = None,
    plan_choices: Mapping[tuple[str, str], Sequence] | None = None,
    selected_query_plans: Mapping[tuple[str, str], Any] | None = None,
    score_fn: Callable[[Node, Worker, int, str | None], float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionPlan:
    """Greedy baseline: topo order, pick lowest score(worker) per node."""
    worker_ids = tuple(sorted(workers))
    if not worker_ids:
        raise ValueError("Greedy scheduler requires at least one worker.")
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

    load: Dict[str, int] = {wid: 0 for wid in worker_ids}
    last_model: Dict[str, str | None] = {wid: None for wid in worker_ids}
    tasks: list[ExecutionTask] = []
    scorer = score_fn or (lambda _node, _worker, l, _last: float(l))

    for idx, node_id in enumerate(order):
        node = graph.nodes[node_id]
        best_worker = None
        best_score = None
        for wid in options[node_id]:
            worker = workers[wid]
            score_value = scorer(node, worker, load.get(wid, 0), last_model.get(wid))
            score = (score_value, load.get(wid, 0), wid)
            if best_score is None or score < best_score:
                best_score = score
                best_worker = wid
        if best_worker is None:
            raise RuntimeError(f"Unable to place node '{node_id}' on any worker.")
        load[best_worker] = load.get(best_worker, 0) + 1
        if node.engine == "vllm":
            last_model[best_worker] = node.model or last_model.get(best_worker)
        epoch = idx // max(1, len(worker_ids))
        tasks.append(
            ExecutionTask(
                node_id=node_id,
                worker_id=best_worker,
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
