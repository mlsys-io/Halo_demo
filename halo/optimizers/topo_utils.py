from __future__ import annotations

import heapq
from typing import Dict, Mapping, Sequence

from ..models import GraphSpec, Node, Worker, build_dependency_list


def default_worker_filter(node: Node, worker: Worker) -> bool:
    """Basic compatibility: LLM nodes -> GPU, DB/HTTP/noop -> CPU."""
    if node.engine == "vllm":
        return worker.kind == "gpu"
    if node.engine in ("db", "http") or node.type == "db_query":
        return worker.kind == "cpu"
    return worker.kind == "cpu"


def filtered_dependencies(
    graph: GraphSpec,
    schedulable: Sequence[str],
    dependencies: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, tuple[str, ...]]:
    raw = dependencies or build_dependency_list(graph.edges)
    sched_set = set(schedulable)
    return {
        node_id: tuple(dep for dep in raw.get(node_id, []) if dep in sched_set)
        for node_id in schedulable
    }


def topological_order(
    dependencies: Mapping[str, Sequence[str]],
    nodes: Sequence[str] | None = None,
) -> list[str]:
    """Deterministic Kahn topo sort over the provided dependency mapping."""
    node_set = set(nodes) if nodes is not None else set(dependencies)
    for node_id, parents in dependencies.items():
        if nodes is None or node_id in node_set:
            for parent in parents:
                if nodes is None or parent in node_set:
                    node_set.add(parent)

    indegree: Dict[str, int] = {node_id: 0 for node_id in node_set}
    children: Dict[str, list[str]] = {node_id: [] for node_id in node_set}
    for node_id in node_set:
        for parent in dependencies.get(node_id, ()):
            if parent not in node_set:
                continue
            indegree[node_id] += 1
            children.setdefault(parent, []).append(node_id)

    ready = [node_id for node_id, deg in indegree.items() if deg == 0]
    heapq.heapify(ready)

    order: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for child in children.get(node_id, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    if len(order) != len(node_set):
        missing = node_set.difference(order)
        raise ValueError(f"Cycle detected or missing nodes when building topo order: {sorted(missing)}")
    return order
