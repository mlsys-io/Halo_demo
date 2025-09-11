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


def schedule_colocate_parents(
    device_cnt: int,
    all_ops: List[Operator],
    queries: List[Query],
) -> List[List[Dict]]:
    """
    Prefer assigning an op to any parent's device (locality). Each op uses ALL queries.
    """
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]
    all_ids = [q.id for q in queries]
    topo = _topo_order(all_ops)

    op2dev: Dict[Operator, int] = {}
    rr = 0
    for op in topo:
        chosen = None
        for p in op.input_ops:
            if p in op2dev:
                chosen = op2dev[p]
                break
        if chosen is None:
            chosen = rr
            rr = (rr + 1) % device_cnt
        op2dev[op] = chosen
        workflows[chosen].append({"command": "execute", "params": (op, list(all_ids))})
    return workflows
