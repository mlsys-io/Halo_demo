from typing import List, Dict
from collections import deque
from halo.components import Operator, Query


def _topo_order(all_ops: List[Operator]) -> List[Operator]:
    indeg = {op: len(op.input_ops) for op in all_ops}
    q = deque([op for op, d in indeg.items() if d == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in u.output_ops:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != len(all_ops):
        raise ValueError("Graph is not a DAG.")
    return topo


def schedule_by_levels(
    device_cnt: int,
    all_ops: List[Operator],
    queries: List[Query],
) -> List[List[Dict]]:
    """Level-based RR: assign ops within the same topo level round-robin; each op uses ALL queries."""
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]
    all_ids = [q.id for q in queries]
    topo = _topo_order(all_ops)

    level = {}
    for op in topo:
        level[op] = 0 if not op.input_ops else max(level[p] for p in op.input_ops) + 1

    d = 0
    for lv in sorted(set(level.values())):
        layer_ops = [op for op in topo if level[op] == lv]
        for op in layer_ops:
            workflows[d % device_cnt].append({"command": "execute", "params": (op, list(all_ids))})
            d += 1
    return workflows
