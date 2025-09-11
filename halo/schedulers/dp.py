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


def schedule_dp(
    device_cnt: int,
    all_ops: List[Operator],
    queries: List[Query],
) -> List[List[Dict]]:
    """
    Sticky data-parallel with fixed shards:
      - Split query ids into D shards.
      - For each op in topo order, submit (op, shard_d) to device d.
    """
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]

    topo = _topo_order(all_ops)
    D = max(1, device_cnt)
    all_ids = [q.id for q in queries]

    shards = []
    N = len(all_ids)
    for d in range(device_cnt):
        start = (N * d) // D
        end = (N * (d + 1)) // D
        shard = all_ids[start:end]
        shards.append(shard)

    for op in topo:
        for d, shard in enumerate(shards):
            if not shard:
                continue
            workflows[d].append({"command": "execute", "params": (op, list(shard))})
    return workflows
