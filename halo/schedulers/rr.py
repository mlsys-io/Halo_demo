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


def schedule_rr(
    device_cnt: int,
    all_ops: List[Operator],
    queries: List[Query],
) -> List[List[Dict]]:
    """Round-Robin per op; each op handles ALL queries."""
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]
    all_ids = [q.id for q in queries]
    topo = _topo_order(all_ops)

    d = 0
    for op in topo:
        workflows[d].append({"command": "execute", "params": (op, list(all_ids))})
        d = (d + 1) % device_cnt
    return workflows
