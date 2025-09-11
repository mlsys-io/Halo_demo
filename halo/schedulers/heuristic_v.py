from typing import List, Dict
from halo.components import Operator, Query


def _compute_longest_distances(all_ops: List[Operator], end_ops: List[Operator]) -> None:
    """Annotate each op with .max_distance to nearest end op."""
    end_set = set(end_ops)
    memo = {}

    def dfs(op):
        if op in memo:
            return memo[op]
        if op in end_set:
            memo[op] = 0
            return 0
        best = -1
        for child in op.output_ops:
            d = dfs(child)
            if d != -1:
                best = max(best, d + 1)
        memo[op] = best
        return best

    for op in all_ops:
        op.max_distance = dfs(op)


def _partition_query_ids(op: Operator, queries: List[Query], return_full: bool = False):
    """Partition queries among duplicates; otherwise return all query ids."""
    if getattr(op, "data_parallel", False):
        dup_index, total_dup = op.duplicate_info[0], op.duplicate_info[1] + 1
        if total_dup > 0:
            total = len(queries)
            per = total // total_dup
            start = per * dup_index
            end = per * (dup_index + 1) if dup_index < total_dup else total
            queries = queries[start:end]
    if return_full:
        return queries
    return [x.id for x in queries]


def schedule_heuristic(
    device_cnt: int,
    start_ops: List[Operator],
    end_ops: List[Operator],
    all_ops: List[Operator],
    queries: List[Query],
    dp_threshold: int = 2,
) -> List[List[Dict]]:
    """
    Heuristic scheduler:
      - Grow frontier respecting dependencies.
      - If ops > devices: pick by longest distance to end (desc).
      - If ops < devices: try to resume paused; else DP duplicate selected ops.
      - Prefer assigning to parent's device (cache locality).
      - Insert no-op cache cmds here (vLLMWorker 简化版无需 dump/get/send/resume)。
      - Append 'execute' tasks.
    """
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    _compute_longest_distances(all_ops, end_ops)

    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]
    requests_cnt = len(queries)
    last_ops: List[Operator] = []
    paused_ops: List[Operator] = []
    device_history: Dict[Operator, int] = {}

    while True:
        # 1) Frontier
        if not last_ops:
            new_ops = list(start_ops)
        else:
            new_ops = []
            for op in last_ops:
                new_ops.extend(op.output_ops)
            new_ops = list(set(new_ops))

        # 2) Dependencies ready?
        valid = []
        for op in new_ops:
            if all(inp in device_history for inp in op.input_ops):
                valid.append(op)
                if op in paused_ops:
                    paused_ops.remove(op)
            else:
                if op not in paused_ops:
                    paused_ops.append(op)
        new_ops = valid

        # Try resume
        if not new_ops:
            resumed = []
            for op in paused_ops[:]:
                if all(inp in device_history for inp in op.input_ops):
                    resumed.append(op)
                    paused_ops.remove(op)
            new_ops.extend(resumed)

        if not new_ops:
            break

        # 3) Fit to device count
        if len(new_ops) > device_cnt:
            new_ops.sort(key=lambda x: getattr(x, "max_distance", 0), reverse=True)
            extra = new_ops[device_cnt:]
            new_ops = new_ops[:device_cnt]
            # extra -> keep paused;（简化：不做 cache 指令）
            for op in extra:
                if op not in paused_ops:
                    paused_ops.append(op)

        elif len(new_ops) < device_cnt:
            # try resume more
            for op in paused_ops[:]:
                if all(inp in device_history for inp in op.input_ops):
                    new_ops.append(op)
                    paused_ops.remove(op)
                    if len(new_ops) == device_cnt:
                        break

            # still fewer -> DP duplicate
            if len(new_ops) < device_cnt and requests_cnt > dp_threshold and len(new_ops) > 0:
                required = device_cnt - len(new_ops)
                duplicates: List[Operator] = []
                for i in range(required):
                    original = new_ops[i % len(new_ops)]
                    if original.duplicate_info is None:
                        original.data_parallel = True
                        original.is_duplicate = False
                        original.duplicate_info = [0, 0]

                    dup = Operator(id=original.id)
                    dup.input_ops = original.input_ops
                    dup.output_ops = []
                    dup.prompt = getattr(original, "prompt", None)
                    dup.model_config = original.model_config
                    dup.keep_cache = original.keep_cache
                    dup.data_parallel = True
                    dup.is_duplicate = True
                    dup.main_op = original
                    dup.duplicate_info = [0, 0]

                    count = original.duplicate_info[1] + 1
                    original.duplicate_info = [0, count]
                    for x in duplicates:
                        if getattr(x, "main_op", None) == original:
                            x.duplicate_info[1] = count
                    dup.duplicate_info = [count, count]
                    duplicates.append(dup)

                new_ops.extend(duplicates)

        # 4) Device assignment (prefer one of parent's device)
        available = set(range(device_cnt))
        for op in new_ops:
            assigned = False
            for inp in op.input_ops:
                if inp in last_ops:
                    d = device_history[inp]
                    if d in available:
                        device_history[op] = d
                        available.remove(d)
                        assigned = True
                        break
            if not assigned:
                device_history[op] = available.pop()

        # 5) Emit execute tasks
        for op in new_ops:
            req_ids = _partition_query_ids(op, queries)
            workflows[device_history[op]].append({"command": "execute", "params": (op, req_ids)})

        last_ops = new_ops

    return workflows
