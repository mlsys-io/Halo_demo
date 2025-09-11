from typing import List, Dict
from halo.components import Operator, Query


def schedule_workflows(
    device_cnt: int,
    start_ops: List[Operator],
    end_ops: List[Operator],
    all_ops: List[Operator],
    queries: List[Query],
    req_id_map: Dict[int, Query],
    dp_threshold: int,
):
    """
    Build per-device workflows for the DAG of ops.

    Strategy:
      1) Start from start_ops (or successors of last round).
      2) Only schedule ops whose all input_ops already have a device.
      3) If more ops than devices: pick by `max_distance` (desc).
      4) If fewer ops than devices: resume paused ops; if still short and there
         are many queries, duplicate ops for data parallelism.
      5) Prefer assigning an op to the device of its input for cache locality.
      6) Insert cache sync commands across devices when necessary.
      7) Append 'execute' for each op; manage cache complete/merge for end/DP cases.
    """
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]
    requests_cnt = len(queries)

    def _partition_query_ids(op: Operator, q: List[Query], return_full: bool = False):
        """Partition queries among duplicates; otherwise return all query ids."""
        if getattr(op, "data_parallel", False):
            dup_index, total_dup = op.duplicate_info[0], op.duplicate_info[1] + 1
            if total_dup > 0:
                total = len(q)
                per = total // total_dup
                start = per * dup_index
                end = per * (dup_index + 1) if dup_index < total_dup else total
                q = q[start:end]
        if return_full:
            return q
        return [x.id for x in q]

    end_set = set(end_ops)
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

        # 2) Dependencies
        valid_ops = []
        for op in new_ops:
            if all(inp in device_history for inp in op.input_ops):
                valid_ops.append(op)
                if op in paused_ops:
                    paused_ops.remove(op)
            else:
                if op not in paused_ops:
                    for inp in op.input_ops:
                        if inp in last_ops:
                            workflows[device_history[inp]].append({"command": "dump_cache", "params": ()})
                    paused_ops.append(op)
        new_ops = valid_ops

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
            new_ops.sort(key=lambda x: x.max_distance, reverse=True)
            extra_ops = new_ops[device_cnt:]
            new_ops = new_ops[:device_cnt]
            for op in extra_ops:
                for inp in op.input_ops:
                    if inp in last_ops:
                        workflows[device_history[inp]].append({"command": "dump_cache", "params": ()})
                if op not in paused_ops:
                    paused_ops.append(op)

        elif len(new_ops) < device_cnt:
            for op in paused_ops[:]:
                if all(inp in device_history for inp in op.input_ops):
                    new_ops.append(op)
                    paused_ops.remove(op)
                    if len(new_ops) == device_cnt:
                        break

            if len(new_ops) < device_cnt and requests_cnt > dp_threshold and len(new_ops) > 0:
                required = device_cnt - len(new_ops)
                duplicates: List[Operator] = []
                for i in range(required):
                    original = new_ops[i % len(new_ops)]
                    if original.duplicate_info is None:
                        original.data_parallel = True
                        original.is_duplicate = False
                        original.duplicate_info = [0, 0]

                    duplicate = Operator(id=original.id)
                    duplicate.input_ops = original.input_ops
                    duplicate.output_ops = []
                    duplicate.prompt = getattr(original, "prompt", None)
                    duplicate.model_config = original.model_config
                    duplicate.keep_cache = original.keep_cache
                    duplicate.data_parallel = True
                    duplicate.is_duplicate = True
                    duplicate.main_op = original
                    duplicate.duplicate_info = [0, 0]

                    count = original.duplicate_info[1] + 1
                    original.duplicate_info = [0, count]

                    for op in duplicates:
                        if getattr(op, "main_op", None) == original:
                            op.duplicate_info[1] = count

                    duplicate.duplicate_info = [count, count]
                    duplicates.append(duplicate)

                new_ops.extend(duplicates)

        # 4) Device assignment (prefer input's device)
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

            # 5) Cache sync across devices or resume
            for inp in op.input_ops:
                if inp in last_ops:
                    src_dev = device_history[inp]
                    if src_dev != device_history[op]:
                        workflows[device_history[op]].append({"command": "get_cache", "params": ()})
                        req_ids = _partition_query_ids(op, queries)
                        workflows[src_dev].append({"command": "send_cache", "params": (device_history[op], req_ids)})
                else:
                    req_ids = _partition_query_ids(op, queries)
                    workflows[device_history[op]].append({"command": "resume_cache", "params": (inp.id, req_ids)})

        # 6) Execute + cache management
        for op in new_ops:
            req_ids = _partition_query_ids(op, queries)
            workflows[device_history[op]].append({"command": "execute", "params": (op, req_ids)})

            if op in end_ops or getattr(op, "main_op", None) in end_ops:
                workflows[device_history[op]].append({"command": "complete", "params": ()})
            elif getattr(op, "data_parallel", False):
                if getattr(op, "is_duplicate", False):
                    main_dev = device_history.get(op.main_op, None)
                    if main_dev is not None:
                        workflows[device_history[op]].append({"command": "send_cache", "params": (main_dev, req_ids)})
                else:
                    for _ in range(op.duplicate_info[1]):
                        workflows[device_history[op]].append({"command": "get_cache", "params": ()})

        last_ops = new_ops

    return workflows
