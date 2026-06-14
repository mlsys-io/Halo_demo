from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import pulp

from ..models import (
    ExecutionPlan,
    ExecutionTask,
    GraphSpec,
    QueryPlanChoice,
    Worker,
    build_dependency_list,
)
from .topo_utils import default_worker_filter, filtered_dependencies, topological_order


@dataclass(frozen=True, slots=True)
class MilpSolveStats:
    status: str
    objective_makespan: float | None
    solve_time_s: float
    num_nodes: int
    num_workers: int
    num_binary_vars: int
    num_continuous_vars: int
    num_constraints: int
    time_limit_s: float | None
    gap_rel: float | None


def _plan_base_cost(choice: QueryPlanChoice, raw_cost_scale: float) -> float:
    if choice.raw_cost is not None:
        return max(0.05, raw_cost_scale * float(choice.raw_cost))
    if choice.cost is not None:
        return float(choice.cost)
    return 1.0


def _prune_plan_choices(
    plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
    *,
    raw_cost_scale: float,
    plan_top_k: int | None,
    merge_plan_footprints: bool,
) -> Dict[tuple[str, str], Tuple[QueryPlanChoice, ...]]:
    if not plan_choices:
        return {}
    pruned: Dict[tuple[str, str], Tuple[QueryPlanChoice, ...]] = {}
    for key, choices in plan_choices.items():
        seq = list(choices) if choices else []
        if merge_plan_footprints:
            best_by_fp: Dict[Tuple[Tuple[str, int], ...], QueryPlanChoice] = {}
            for ch in seq:
                fp_key = tuple(sorted((ch.footprints or {}).items()))
                base = _plan_base_cost(ch, raw_cost_scale)
                prev = best_by_fp.get(fp_key)
                if prev is None or _plan_base_cost(prev, raw_cost_scale) > base:
                    best_by_fp[fp_key] = ch
            seq = list(best_by_fp.values())
        seq.sort(key=lambda ch: _plan_base_cost(ch, raw_cost_scale))
        if plan_top_k is not None:
            seq = seq[: max(1, int(plan_top_k))]
        pruned[key] = tuple(seq)
    return pruned


def _footprint_overlap(fp_a: Mapping[str, int] | None, fp_b: Mapping[str, int] | None) -> float:
    if not fp_a or not fp_b:
        return 0.0
    overlap = 0.0
    for name, weight in fp_a.items():
        other = fp_b.get(name)
        if other is None:
            continue
        overlap += min(int(weight), int(other))
    return float(overlap)


def _batch_gpu_depths(
    gpu_nodes: Sequence[str],
    deps: Mapping[str, Sequence[str]],
) -> Dict[str, int]:
    if not gpu_nodes:
        return {}
    gpu_set = set(gpu_nodes)
    indegree: Dict[str, int] = {nid: 0 for nid in gpu_nodes}
    children: Dict[str, list[str]] = {nid: [] for nid in gpu_nodes}
    for nid in gpu_nodes:
        for parent in deps.get(nid, ()):
            if parent not in gpu_set:
                continue
            indegree[nid] += 1
            children.setdefault(parent, []).append(nid)
    ready = [nid for nid, deg in indegree.items() if deg == 0]
    ready.sort()
    depths: Dict[str, int] = {nid: 0 for nid in ready}
    while ready:
        nid = ready.pop(0)
        base_depth = depths.get(nid, 0)
        for child in children.get(nid, []):
            depths[child] = max(depths.get(child, 0), base_depth + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    return depths


def _order_cpu_nodes_by_parent_depth(
    cpu_nodes: Sequence[str],
    gpu_depths: Mapping[str, int],
    graph: GraphSpec,
) -> list[str]:
    if not cpu_nodes or not gpu_depths:
        return list(cpu_nodes)
    def key(node_id: str) -> tuple[int, str]:
        node = graph.nodes.get(node_id)
        parent = None
        if node is not None and isinstance(node.raw, dict):
            parent = node.raw.get("parent")
        return (gpu_depths.get(parent, 0) if parent else 0, node_id)
    return sorted(cpu_nodes, key=key)


def _auto_cpu_batch(
    pending: set[str],
    done: set[str],
    gpu_nodes: Sequence[str],
    *,
    deps: Mapping[str, Sequence[str]],
    topo_order: Sequence[str],
    graph: GraphSpec,
) -> list[str]:
    if not pending:
        return []
    batch_nodes = set(done) | set(gpu_nodes)
    allowed_parents = set(gpu_nodes)
    selected: list[str] = []
    progressed = True
    while progressed:
        progressed = False
        for nid in topo_order:
            if nid not in pending:
                continue
            node = graph.nodes.get(nid)
            parent = None
            if node is not None and isinstance(node.raw, dict):
                parent = node.raw.get("parent")
            if parent is not None and parent not in allowed_parents:
                continue
            parents = deps.get(nid, ())
            if any(parent_id not in batch_nodes for parent_id in parents):
                continue
            selected.append(nid)
            pending.remove(nid)
            batch_nodes.add(nid)
            progressed = True
    return selected


def build_continuous_milp_plan(
    graph: GraphSpec,
    workers: Mapping[str, Worker],
    *,
    dependencies: Mapping[str, Sequence[str]] | None = None,
    schedulable_ids: Sequence[str] | None = None,
    node_worker_options: Mapping[str, Sequence[str]] | None = None,
    plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None = None,
    selected_query_plans: Mapping[tuple[str, str], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    exec_cost_fn=None,
    model_init_cost_fn=None,
    llm_cache_bonus_fn=None,
    raw_cost_scale: float = 3.65e-6,
    window_size: int = 2,
    cpu_load_cost_weight: float | None = None,
    cpu_load_early_weight: float | None = None,
    gpu_cost_max_weight: float | None = None,
    gpu_cost_sum_weight: float | None = None,
    plan_top_k: int | None = None,
    merge_plan_footprints: bool = True,
) -> tuple[ExecutionPlan, MilpSolveStats]:
    """Continuous-time MILP oracle scheduler for small graphs.

    - Decision variables: worker assignment, per-worker sequences, per-node start times.
    - Objective: minimize makespan (continuous time).
    - Cost model:
        * LLM duration matches DP: exec_cost * llm_cache_bonus + model_init_cost (sequence-dependent),
          plus per-node CPU load cost from its associated CPU nodes.
        * CPU nodes are auto-filled per epoch after GPU scheduling to align with DP's CPU handling.
    """
    max_nodes_env = os.getenv("HALO_MILP_MAX_NODES", "30")
    try:
        max_nodes = max(1, int(max_nodes_env))
    except ValueError:
        max_nodes = 30

    node_ids_all = tuple(
        sorted(
            schedulable_ids
            if schedulable_ids is not None
            else (nid for nid, node in graph.nodes.items() if node.type != "input")
        )
    )
    llm_node_ids = tuple(nid for nid in node_ids_all if graph.nodes[nid].engine == "vllm")
    db_node_ids_all = tuple(nid for nid in node_ids_all if graph.nodes[nid].engine != "vllm")
    # Align with DP: optimize only GPU/LLM nodes when available, then auto-fill DB nodes per epoch.
    node_ids = llm_node_ids if llm_node_ids else node_ids_all
    if len(node_ids) > max_nodes:
        raise RuntimeError(f"MILP scheduler supports up to {max_nodes} nodes; got {len(node_ids)}.")

    if window_size > 2:
        raise RuntimeError(f"MILP scheduler currently supports last_query_window <= 2; got {window_size}.")

    worker_ids = tuple(sorted(workers))
    if not worker_ids:
        raise ValueError("MILP scheduler requires at least one worker.")

    deps = filtered_dependencies(graph, node_ids, dependencies)
    full_deps = dependencies or build_dependency_list(graph.edges)

    options: Dict[str, Tuple[str, ...]] = {}
    if node_worker_options is not None:
        for node_id in node_ids:
            opts = tuple(node_worker_options.get(node_id, ()))
            if not opts:
                raise RuntimeError(f"No eligible workers for node '{node_id}'")
            options[node_id] = opts
    else:
        for node_id in node_ids:
            node = graph.nodes[node_id]
            eligible = tuple(wid for wid, worker in workers.items() if default_worker_filter(node, worker))
            if not eligible:
                raise RuntimeError(f"No eligible workers for node '{node_id}'")
            options[node_id] = eligible

    exec_cost_fn = exec_cost_fn or (lambda _node, _worker: 1.0)
    model_init_cost_fn = model_init_cost_fn or (lambda _node, _last_model: 0.0)
    llm_cache_bonus_fn = llm_cache_bonus_fn or (lambda _node, _last_node, _parents: 1.0)

    db_nodes = tuple(nid for nid in node_ids if graph.nodes[nid].engine != "vllm")

    fallback_choice = QueryPlanChoice(
        plan_id="default",
        description="fallback",
        cost=1.0,
        raw_cost=None,
        explain_json=None,
        samples=tuple(),
        footprints={},
    )
    no_query_choice = QueryPlanChoice(
        plan_id="no_query",
        description="no_query",
        cost=0.0,
        raw_cost=None,
        explain_json=None,
        samples=tuple(),
        footprints={},
    )

    if cpu_load_cost_weight is None:
        env_weight = os.getenv("HALO_DP_CPU_LOAD_COST_WEIGHT", "").strip()
        if env_weight:
            try:
                cpu_load_cost_weight = float(env_weight)
            except ValueError:
                cpu_load_cost_weight = 1.0
        else:
            cpu_load_cost_weight = 1.0
    cpu_load_cost_weight = max(0.0, float(cpu_load_cost_weight))

    max_weight = gpu_cost_max_weight
    sum_weight = gpu_cost_sum_weight
    if max_weight is None and sum_weight is None:
        env_max = (os.getenv("HALO_DP_GPU_COST_MAX_WEIGHT") or "").strip()
        env_sum = (os.getenv("HALO_DP_GPU_COST_SUM_WEIGHT") or "").strip()
        if env_max:
            try:
                max_weight = float(env_max)
            except ValueError:
                max_weight = None
        if env_sum:
            try:
                sum_weight = float(env_sum)
            except ValueError:
                sum_weight = None
    if max_weight is None and sum_weight is None:
        max_weight = 0.9
        sum_weight = 0.1
    if max_weight is None:
        max_weight = 0.0
    if sum_weight is None:
        sum_weight = 0.0
    total = float(max_weight) + float(sum_weight)
    if total <= 0:
        max_weight = 0.9
        sum_weight = 0.1
    else:
        max_weight = float(max_weight) / total
        sum_weight = float(sum_weight) / total
    gpu_cost_max_weight = max(0.0, float(max_weight))
    gpu_cost_sum_weight = max(0.0, float(sum_weight))

    if cpu_load_early_weight is None:
        env_weight = os.getenv("HALO_DP_CPU_EARLY_WEIGHT", "").strip()
        if env_weight:
            try:
                cpu_load_early_weight = float(env_weight)
            except ValueError:
                cpu_load_early_weight = 1.0
        else:
            cpu_load_early_weight = 2.0
    cpu_load_early_weight = max(0.0, float(cpu_load_early_weight))

    llm_cpu_costs: Dict[str, float] = {}
    if llm_node_ids and db_node_ids_all and plan_choices is not None:
        parent_costs: Dict[str, float] = {}
        for node_id in db_node_ids_all:
            node = graph.nodes[node_id]
            parent = None
            if isinstance(node.raw, dict):
                parent = node.raw.get("parent")
            if not parent:
                continue
            min_cost = 0.0
            queries = list(getattr(node, "db_queries", []) or [])
            for query in queries:
                choices = plan_choices.get((node_id, query.name), ())
                if not choices:
                    choices = (fallback_choice,)
                best = min(_plan_base_cost(choice, raw_cost_scale) for choice in choices)
                min_cost += best
            parent_costs[parent] = parent_costs.get(parent, 0.0) + min_cost
        for node_id in llm_node_ids:
            llm_cpu_costs[node_id] = cpu_load_cost_weight * parent_costs.get(node_id, 0.0)

    pruned_plan_choices = _prune_plan_choices(
        plan_choices,
        raw_cost_scale=raw_cost_scale,
        plan_top_k=plan_top_k,
        merge_plan_footprints=merge_plan_footprints,
    )

    db_query_name: Dict[str, str] = {}
    db_plan_options: Dict[str, Tuple[QueryPlanChoice, ...]] = {}
    for node_id in db_nodes:
        node = graph.nodes[node_id]
        if not node.db_queries:
            query_name = "no_query"
            db_query_name[node_id] = query_name
            db_plan_options[node_id] = (no_query_choice,)
            continue
        query_name = node.db_queries[0].name
        db_query_name[node_id] = query_name
        key = (node_id, query_name)
        choices = pruned_plan_choices.get(key)
        if not choices:
            choices = (fallback_choice,)
        db_plan_options[node_id] = tuple(choices)

    plan_base: Dict[tuple[str, int], float] = {}
    plan_fp: Dict[tuple[str, int], Mapping[str, int]] = {}
    for node_id, choices in db_plan_options.items():
        for idx, choice in enumerate(choices):
            plan_base[(node_id, idx)] = _plan_base_cost(choice, raw_cost_scale)
            plan_fp[(node_id, idx)] = choice.footprints or {}

    max_pair_overlap = 0.0
    for a in db_nodes:
        for ai in range(len(db_plan_options[a])):
            for b in db_nodes:
                for bi in range(len(db_plan_options[b])):
                    max_pair_overlap = max(max_pair_overlap, _footprint_overlap(plan_fp[(a, ai)], plan_fp[(b, bi)]))
    big_overlap = max(1.0, max_pair_overlap * max(1, window_size))

    # --- MILP model ---------------------------------------------------------

    model = pulp.LpProblem("halo_continuous_milp_scheduler", pulp.LpMinimize)

    # Assignment vars x[node, worker] \in {0,1}
    x: Dict[tuple[str, str], pulp.LpVariable] = {}
    for node_id in node_ids:
        for wid in options[node_id]:
            x[(node_id, wid)] = pulp.LpVariable(f"x__{node_id}__{wid}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        model += pulp.lpSum(x[(node_id, wid)] for wid in options[node_id]) == 1, f"assign__{node_id}"

    # Start / completion times
    start: Dict[str, pulp.LpVariable] = {
        node_id: pulp.LpVariable(f"s__{node_id}", lowBound=0, cat=pulp.LpContinuous) for node_id in node_ids
    }
    completion: Dict[str, pulp.LpVariable] = {
        node_id: pulp.LpVariable(f"c__{node_id}", lowBound=0, cat=pulp.LpContinuous) for node_id in node_ids
    }
    makespan = pulp.LpVariable("makespan", lowBound=0, cat=pulp.LpContinuous)

    # Per-worker sequencing via immediate predecessor arcs pred[i,j,w]
    pred: Dict[tuple[str, str, str], pulp.LpVariable] = {}
    start_dummy: Dict[str, str] = {}
    end_dummy: Dict[str, str] = {}
    eligible_nodes_by_worker: Dict[str, Tuple[str, ...]] = {}
    u_vars: Dict[tuple[str, str], pulp.LpVariable] = {}
    worker_rank_limit: Dict[str, int] = {}
    for wid in worker_ids:
        eligible = tuple(nid for nid in node_ids if wid in options[nid])
        eligible_nodes_by_worker[wid] = eligible
        s_id = f"__START__{wid}"
        e_id = f"__END__{wid}"
        start_dummy[wid] = s_id
        end_dummy[wid] = e_id

        # Start -> (task or end)
        for j in list(eligible) + [e_id]:
            pred[(s_id, j, wid)] = pulp.LpVariable(f"pred__{wid}__{s_id}__{j}", 0, 1, pulp.LpBinary)
        # (task or start) -> end
        for i in list(eligible) + [s_id]:
            if (i, e_id, wid) not in pred:
                pred[(i, e_id, wid)] = pulp.LpVariable(f"pred__{wid}__{i}__{e_id}", 0, 1, pulp.LpBinary)
        # task -> task arcs
        for i in eligible:
            for j in eligible:
                if i == j:
                    continue
                pred[(i, j, wid)] = pulp.LpVariable(f"pred__{wid}__{i}__{j}", 0, 1, pulp.LpBinary)

        # Flow constraints
        model += (
            pulp.lpSum(pred[(s_id, j, wid)] for j in list(eligible) + [e_id]) == 1
        ), f"start_out__{wid}"
        model += (
            pulp.lpSum(pred[(i, e_id, wid)] for i in list(eligible) + [s_id]) == 1
        ), f"end_in__{wid}"

        for node_id in eligible:
            incoming = []
            for i in list(eligible) + [s_id]:
                if i == node_id:
                    continue
                incoming.append(pred[(i, node_id, wid)])
            outgoing = []
            for j in list(eligible) + [e_id]:
                if j == node_id:
                    continue
                outgoing.append(pred[(node_id, j, wid)])
            model += (pulp.lpSum(incoming) == x[(node_id, wid)]), f"in_deg__{wid}__{node_id}"
            model += (pulp.lpSum(outgoing) == x[(node_id, wid)]), f"out_deg__{wid}__{node_id}"

        # MTZ subtour elimination (optional but keeps model well-posed even with zero durations)
        k = max(1, len(eligible))
        worker_rank_limit[wid] = k
        u: Dict[str, pulp.LpVariable] = {}
        for node_id in eligible:
            u[node_id] = pulp.LpVariable(f"u__{wid}__{node_id}", lowBound=0, upBound=k, cat=pulp.LpInteger)
            u_vars[(wid, node_id)] = u[node_id]
            model += u[node_id] <= k * x[(node_id, wid)], f"mtz_ub__{wid}__{node_id}"
            model += u[node_id] >= x[(node_id, wid)], f"mtz_lb__{wid}__{node_id}"
        for i in eligible:
            for j in eligible:
                if i == j:
                    continue
                model += u[i] - u[j] + k * pred[(i, j, wid)] <= k - 1, f"mtz__{wid}__{i}__{j}"

    # --- DB plan selection & cache multiplier (DP-equivalent, window_size <= 2) ---

    # z[node, plan_idx] \in {0,1}
    z: Dict[tuple[str, int], pulp.LpVariable] = {}
    overlap: Dict[tuple[str, int], pulp.LpVariable] = {}
    mult: Dict[tuple[str, int], pulp.LpVariable] = {}

    for node_id in db_nodes:
        plan_count = len(db_plan_options[node_id])
        for pidx in range(plan_count):
            z[(node_id, pidx)] = pulp.LpVariable(f"z__{node_id}__p{pidx}", 0, 1, pulp.LpBinary)
            overlap[(node_id, pidx)] = pulp.LpVariable(f"ov__{node_id}__p{pidx}", lowBound=0, cat=pulp.LpContinuous)
            mult[(node_id, pidx)] = pulp.LpVariable(f"m__{node_id}__p{pidx}", lowBound=0, upBound=1, cat=pulp.LpContinuous)

        model += (
            pulp.lpSum(z[(node_id, pidx)] for pidx in range(plan_count)) == 1
        ), f"plan_pick__{node_id}"

        # Multiplier constraints are gated by z so non-chosen plans do not constrain.
        for pidx in range(plan_count):
            model += overlap[(node_id, pidx)] <= big_overlap * z[(node_id, pidx)], f"ov_gate__{node_id}__p{pidx}"
            model += mult[(node_id, pidx)] <= z[(node_id, pidx)], f"m_ub__{node_id}__p{pidx}"
            model += mult[(node_id, pidx)] >= 0.5 * z[(node_id, pidx)], f"m_lb__{node_id}__p{pidx}"
            # m >= 1 - 0.01 * overlap, only when selected
            model += (
                mult[(node_id, pidx)]
                >= 1.0 - 0.01 * overlap[(node_id, pidx)] - 1.0 * (1.0 - z[(node_id, pidx)])
            ), f"m_cache__{node_id}__p{pidx}"

    # Pair variables for immediate predecessor (i -> j on worker wid).
    pair: Dict[tuple[str, str, str, int, int], pulp.LpVariable] = {}
    # Triple variables for second predecessor (k -> i -> j) when window_size == 2.
    triple: Dict[tuple[str, str, str, str, int, int], pulp.LpVariable] = {}

    # Only CPU workers contribute to DB window.
    cpu_workers = tuple(wid for wid in worker_ids if workers[wid].kind != "gpu")

    for wid in cpu_workers:
        eligible = [nid for nid in eligible_nodes_by_worker[wid] if nid in db_nodes]
        if not eligible:
            continue
        # Immediate predecessor overlap
        for i in eligible:
            for j in eligible:
                if i == j:
                    continue
                pij = pred.get((i, j, wid))
                if pij is None:
                    continue
                for pi in range(len(db_plan_options[i])):
                    for pj in range(len(db_plan_options[j])):
                        var = pulp.LpVariable(f"pair__{wid}__{i}__{j}__p{pi}__p{pj}", 0, 1, pulp.LpBinary)
                        pair[(i, j, wid, pi, pj)] = var
                        model += var <= pij, f"pair_le_pred__{wid}__{i}__{j}__{pi}__{pj}"
                        model += var <= z[(i, pi)], f"pair_le_zi__{wid}__{i}__{j}__{pi}__{pj}"
                        model += var <= z[(j, pj)], f"pair_le_zj__{wid}__{i}__{j}__{pi}__{pj}"
                        model += (
                            var >= pij + z[(i, pi)] + z[(j, pj)] - 2
                        ), f"pair_ge__{wid}__{i}__{j}__{pi}__{pj}"

        if window_size == 2:
            # Second predecessor overlap: k -> i -> j
            for k_node in eligible:
                for i in eligible:
                    if k_node == i:
                        continue
                    pki = pred.get((k_node, i, wid))
                    if pki is None:
                        continue
                    for j in eligible:
                        if j in (k_node, i):
                            continue
                        pij = pred.get((i, j, wid))
                        if pij is None:
                            continue
                        for pk in range(len(db_plan_options[k_node])):
                            for pj in range(len(db_plan_options[j])):
                                var = pulp.LpVariable(
                                    f"tri__{wid}__{k_node}__{i}__{j}__p{pk}__p{pj}", 0, 1, pulp.LpBinary
                                )
                                triple[(k_node, i, j, wid, pk, pj)] = var
                                model += var <= pki, f"tri_le_pki__{wid}__{k_node}__{i}__{j}__{pk}__{pj}"
                                model += var <= pij, f"tri_le_pij__{wid}__{k_node}__{i}__{j}__{pk}__{pj}"
                                model += var <= z[(k_node, pk)], f"tri_le_zk__{wid}__{k_node}__{i}__{j}__{pk}__{pj}"
                                model += var <= z[(j, pj)], f"tri_le_zj__{wid}__{k_node}__{i}__{j}__{pk}__{pj}"
                                model += (
                                    var >= pki + pij + z[(k_node, pk)] + z[(j, pj)] - 3
                                ), f"tri_ge__{wid}__{k_node}__{i}__{j}__{pk}__{pj}"

    # Tie overlap variables to chosen predecessor chain & plans.
    for node_id in db_nodes:
        for pj in range(len(db_plan_options[node_id])):
            expr_parts = []
            # immediate predecessor
            for wid in cpu_workers:
                eligible = [nid for nid in eligible_nodes_by_worker[wid] if nid in db_nodes]
                if not eligible:
                    continue
                for i in eligible:
                    if i == node_id:
                        continue
                    for pi in range(len(db_plan_options[i])):
                        var = pair.get((i, node_id, wid, pi, pj))
                        if var is None:
                            continue
                        ov = _footprint_overlap(plan_fp[(i, pi)], plan_fp[(node_id, pj)])
                        if ov:
                            expr_parts.append(float(ov) * var)
            # second predecessor
            if window_size == 2:
                for wid in cpu_workers:
                    eligible = [nid for nid in eligible_nodes_by_worker[wid] if nid in db_nodes]
                    if not eligible:
                        continue
                    for k_node in eligible:
                        if k_node == node_id:
                            continue
                        for i in eligible:
                            if i in (k_node, node_id):
                                continue
                            for pk in range(len(db_plan_options[k_node])):
                                var = triple.get((k_node, i, node_id, wid, pk, pj))
                                if var is None:
                                    continue
                                ov = _footprint_overlap(plan_fp[(k_node, pk)], plan_fp[(node_id, pj)])
                                if ov:
                                    expr_parts.append(float(ov) * var)
            expr = pulp.lpSum(expr_parts) if expr_parts else 0.0
            model += overlap[(node_id, pj)] == expr, f"ov_def__{node_id}__p{pj}"

    # Define completion times using predecessor-dependent durations
    for node_id in node_ids:
        node = graph.nodes[node_id]
        if node.engine == "vllm":
            dur_terms = []
            db_cost = float(llm_cpu_costs.get(node_id, 0.0))
            for wid in options[node_id]:
                s_id = start_dummy[wid]
                # predecessor is either start dummy or another eligible LLM node on the same worker
                # Start dummy -> node
                pvar = pred.get((s_id, node_id, wid))
                if pvar is not None:
                    bonus = float(llm_cache_bonus_fn(node, None, deps.get(node_id, ())))
                    cost = (
                        float(exec_cost_fn(node, workers[wid])) * bonus
                        + float(model_init_cost_fn(node, None))
                        + db_cost
                    )
                    dur_terms.append(cost * pvar)
                for prev in eligible_nodes_by_worker[wid]:
                    if prev == node_id:
                        continue
                    if graph.nodes[prev].engine != "vllm":
                        continue
                    pvar = pred.get((prev, node_id, wid))
                    if pvar is None:
                        continue
                    last_model = graph.nodes[prev].model
                    bonus = float(llm_cache_bonus_fn(node, prev, deps.get(node_id, ())))
                    cost = (
                        float(exec_cost_fn(node, workers[wid])) * bonus
                        + float(model_init_cost_fn(node, last_model))
                        + db_cost
                    )
                    dur_terms.append(cost * pvar)
            dur_expr = pulp.lpSum(dur_terms) if dur_terms else 0.0
            model += completion[node_id] == start[node_id] + dur_expr, f"dur_llm__{node_id}"
        else:
            # DB node: duration = exec_cost (input overhead) + sum_p base_cost[p] * multiplier[p]
            plan_count = len(db_plan_options.get(node_id, ()))
            exec_expr = pulp.lpSum(
                float(exec_cost_fn(node, workers[wid])) * x[(node_id, wid)]
                for wid in options[node_id]
            )
            if plan_count <= 0:
                dur_expr = exec_expr
            else:
                plan_expr = pulp.lpSum(
                    plan_base[(node_id, pidx)] * mult[(node_id, pidx)] for pidx in range(plan_count)
                )
                dur_expr = exec_expr + plan_expr
            model += completion[node_id] == start[node_id] + dur_expr, f"dur_db__{node_id}"

    # Precedence constraints (DAG): start-to-start (S-S).
    for node_id in node_ids:
        for parent in deps.get(node_id, ()):
            if parent not in completion:
                continue
            model += start[node_id] >= start[parent], f"dep_ss__{parent}__{node_id}"

    # Disjunctive sequencing on each worker (immediate predecessor implies finish-before-start)
    # Big-M should upper bound the makespan so non-selected arcs are truly inactive.
    big_m = 0.0
    for node_id in node_ids:
        node = graph.nodes[node_id]
        if node.engine == "vllm":
            # Upper bound on per-node runtime: no cache bonus + worst-case model switch cost.
            wid0 = options[node_id][0]
            base = float(exec_cost_fn(node, workers[wid0]))
            init_none = float(model_init_cost_fn(node, None))
            init_switch = float(model_init_cost_fn(node, "__other_model__"))
            db_cost = float(llm_cpu_costs.get(node_id, 0.0))
            big_m += base + max(init_none, init_switch) + db_cost + 1.0
        else:
            plan_count = len(db_plan_options.get(node_id, ()))
            exec_ub = 0.0
            try:
                exec_ub = max(float(exec_cost_fn(node, workers[wid])) for wid in options[node_id])
            except Exception:
                exec_ub = 0.0
            if plan_count:
                big_m += exec_ub + max(plan_base[(node_id, pidx)] for pidx in range(plan_count)) + 1.0
            else:
                big_m += exec_ub + 1.0
    big_m = max(100.0, big_m)

    for wid in worker_ids:
        eligible = eligible_nodes_by_worker[wid]
        for i in eligible:
            for j in eligible:
                if i == j:
                    continue
                pij = pred.get((i, j, wid))
                if pij is None:
                    continue
                model += start[j] >= completion[i] - big_m * (1.0 - pij), f"seq__{wid}__{i}__{j}"

    for node_id in node_ids:
        model += makespan >= completion[node_id], f"mk__{node_id}"

    # Objective: primary makespan, optional secondary tie-breakers to better align with DP-style
    # "start early / parallelize" behavior and deterministic worker preferences.
    def _env_nonneg_float(name: str, default: float = 0.0) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return float(default)
        try:
            return max(0.0, float(raw))
        except ValueError:
            return float(default)

    sum_start_weight = _env_nonneg_float("HALO_MILP_OBJ_SUM_START_WEIGHT", 0.01)
    worker_tiebreak_weight = _env_nonneg_float("HALO_MILP_OBJ_WORKER_TIEBREAK_WEIGHT", 1e-5)

    worker_rank = {wid: float(idx) for idx, wid in enumerate(worker_ids)}
    assign_pref = pulp.lpSum(
        worker_rank[wid] * x[(node_id, wid)]
        for node_id in node_ids
        for wid in options[node_id]
    )
    start_pref = pulp.lpSum(start[node_id] for node_id in node_ids)
    llm_duration_sum = pulp.lpSum(
        completion[node_id] - start[node_id] for node_id in node_ids
    )
    early_pref_terms: list[pulp.LpAffineExpression] = []
    if cpu_load_early_weight > 0 and llm_cpu_costs:
        for node_id, db_cost in llm_cpu_costs.items():
            if db_cost <= 0:
                continue
            for wid in options.get(node_id, ()):
                rank_limit = worker_rank_limit.get(wid)
                u_var = u_vars.get((wid, node_id))
                if rank_limit is None or u_var is None:
                    continue
                coeff = (cpu_load_early_weight * db_cost) / float(rank_limit + 1)
                early_pref_terms.append(
                    coeff * ((rank_limit + 1) * x[(node_id, wid)] - u_var)
                )
    early_pref = pulp.lpSum(early_pref_terms) if early_pref_terms else 0.0

    model += (
        gpu_cost_max_weight * makespan
        + gpu_cost_sum_weight * llm_duration_sum
        + sum_start_weight * start_pref
        + worker_tiebreak_weight * assign_pref
        + early_pref
    )

    # --- Solve --------------------------------------------------------------

    time_limit_s: float | None = None
    gap_rel: float | None = None
    env_time = os.getenv("HALO_MILP_TIME_LIMIT")
    if env_time:
        try:
            time_limit_s = float(env_time)
        except ValueError:
            time_limit_s = None
    env_gap = os.getenv("HALO_MILP_GAP_REL")
    if env_gap:
        try:
            gap_rel = float(env_gap)
        except ValueError:
            gap_rel = None

    msg = os.getenv("HALO_MILP_MSG", "0").lower() not in ("", "0", "false", "no")
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s, gapRel=gap_rel)
    t0 = time.perf_counter()
    model.solve(solver)
    solve_time_s = time.perf_counter() - t0

    status = pulp.LpStatus.get(model.status, str(model.status))
    objective = pulp.value(makespan)
    objective_val = float(objective) if objective is not None else None

    # Extract solution into ExecutionPlan.
    assignment: Dict[str, str] = {}
    for node_id in node_ids:
        chosen = None
        best = -1.0
        for wid in options[node_id]:
            val = float(pulp.value(x[(node_id, wid)]) or 0.0)
            if val > best:
                best = val
                chosen = wid
        if chosen is None:
            raise RuntimeError(f"MILP produced no assignment for node '{node_id}' (status={status}).")
        assignment[node_id] = chosen

    start_times: Dict[str, float] = {node_id: float(pulp.value(start[node_id]) or 0.0) for node_id in node_ids}

    # Derive per-worker sequences from pred variables (to preserve intended ordering),
    # then derive epoch from relative (continuous) start_time buckets so each epoch
    # can contain multiple nodes across workers.
    worker_sequences: Dict[str, list[str]] = {}
    sequence_index: Dict[str, int] = {}
    for wid in worker_ids:
        s_id = f"__START__{wid}"
        e_id = f"__END__{wid}"
        # Collect successors for arcs with value ~ 1.
        succ: Dict[str, str] = {}
        for (i, j, w), var in pred.items():
            if w != wid:
                continue
            val = float(pulp.value(var) or 0.0)
            if val >= 0.5:
                succ[i] = j
        seq: list[str] = []
        cur = s_id
        guard = 0
        while guard <= len(node_ids) + 2:
            guard += 1
            nxt = succ.get(cur)
            if nxt is None or nxt == e_id:
                break
            seq.append(nxt)
            cur = nxt
        worker_sequences[wid] = seq
        for idx, nid in enumerate(seq):
            sequence_index[nid] = idx

    completion_times: Dict[str, float] = {
        node_id: float(pulp.value(completion[node_id]) or 0.0) for node_id in node_ids
    }
    durations: list[float] = []
    for nid in node_ids:
        dur = completion_times.get(nid, 0.0) - start_times.get(nid, 0.0)
        if dur > 0:
            durations.append(dur)
    durations.sort()
    median_dur = durations[len(durations) // 2] if durations else 1.0
    makespan_val = (
        objective_val if objective_val is not None and objective_val > 0 else max(completion_times.values(), default=1.0)
    )

    tasks: list[ExecutionTask] = []
    used_epochs: list[int] = []
    epoch_quantum: float | None = None

    if llm_node_ids:
        epoch_by_node: Dict[str, int] = {
            nid: int(sequence_index.get(nid, 0)) for nid in node_ids
        }
        max_epoch = max(epoch_by_node.values(), default=0)
        used_epochs = list(range(max_epoch + 1))
        plan_order = {nid: idx for idx, nid in enumerate(node_ids)}
        cpu_workers = [wid for wid in worker_ids if workers[wid].kind != "gpu"]
        if not cpu_workers:
            cpu_workers = list(worker_ids)
        db_topo_order = (
            topological_order(full_deps, nodes=db_node_ids_all) if db_node_ids_all else []
        )
        pending_db = set(db_node_ids_all)
        done: set[str] = set()

        for epoch in used_epochs:
            epoch_gpu = [nid for nid in node_ids if epoch_by_node.get(nid, 0) == epoch]
            gpu_depths = _batch_gpu_depths(epoch_gpu, deps)
            if gpu_depths:
                epoch_gpu = sorted(
                    epoch_gpu,
                    key=lambda nid: (gpu_depths.get(nid, 0), plan_order.get(nid, 0), nid),
                )
            else:
                epoch_gpu = sorted(epoch_gpu, key=lambda nid: (plan_order.get(nid, 0), nid))
            for nid in epoch_gpu:
                tasks.append(
                    ExecutionTask(
                        node_id=nid,
                        worker_id=assignment[nid],
                        dependencies=full_deps.get(nid, ()),
                        epoch=epoch,
                    )
                )
            cpu_nodes = _auto_cpu_batch(
                pending_db,
                done,
                epoch_gpu,
                deps=full_deps,
                topo_order=db_topo_order,
                graph=graph,
            )
            cpu_nodes = _order_cpu_nodes_by_parent_depth(cpu_nodes, gpu_depths, graph)
            for idx, nid in enumerate(cpu_nodes):
                wid = cpu_workers[idx % len(cpu_workers)] if cpu_workers else worker_ids[0]
                tasks.append(
                    ExecutionTask(
                        node_id=nid,
                        worker_id=wid,
                        dependencies=full_deps.get(nid, ()),
                        epoch=epoch,
                    )
                )
            done.update(epoch_gpu)
            done.update(cpu_nodes)

        extra_epoch = (used_epochs[-1] + 1) if used_epochs else 0
        while pending_db:
            cpu_nodes = _auto_cpu_batch(
                pending_db,
                done,
                [],
                deps=full_deps,
                topo_order=db_topo_order,
                graph=graph,
            )
            if not cpu_nodes:
                cpu_nodes = sorted(pending_db)
                pending_db.clear()
            cpu_nodes = _order_cpu_nodes_by_parent_depth(cpu_nodes, {}, graph)
            for idx, nid in enumerate(cpu_nodes):
                wid = cpu_workers[idx % len(cpu_workers)] if cpu_workers else worker_ids[0]
                tasks.append(
                    ExecutionTask(
                        node_id=nid,
                        worker_id=wid,
                        dependencies=full_deps.get(nid, ()),
                        epoch=extra_epoch,
                    )
                )
            done.update(cpu_nodes)
            used_epochs.append(extra_epoch)
            extra_epoch += 1
    else:
        epoch_quantum_env = os.getenv("HALO_MILP_EPOCH_QUANTUM", "").strip()
        if epoch_quantum_env:
            try:
                epoch_quantum = max(1e-6, float(epoch_quantum_env))
            except ValueError:
                epoch_quantum = max(1e-3, 0.25 * median_dur)
        else:
            epoch_quantum = max(1e-3, 0.25 * median_dur)
            # If durations are degenerate, fall back to a fraction of makespan.
            if epoch_quantum <= 0:
                epoch_quantum = max(1e-3, makespan_val / max(1, len(worker_ids) * 4))

        epoch_by_node = {}
        for nid in node_ids:
            t = start_times.get(nid, 0.0)
            epoch_by_node[nid] = int(max(0.0, t) // float(epoch_quantum))

        used_epochs = sorted(set(epoch_by_node.values()))
        epoch_remap = {bucket: idx for idx, bucket in enumerate(used_epochs)}
        for nid in node_ids:
            epoch_by_node[nid] = epoch_remap.get(epoch_by_node[nid], 0)

        tasks = [
            ExecutionTask(
                node_id=nid,
                worker_id=assignment[nid],
                dependencies=full_deps.get(nid, ()),
                epoch=epoch_by_node.get(nid, 0),
            )
            for nid in node_ids
        ]
        # Order tasks so that within the same epoch, we respect MILP's relative timing and
        # per-worker sequence to preserve model/query-plan state effects.
        tasks.sort(
            key=lambda t: (
                int(t.epoch),
                float(start_times.get(t.node_id, 0.0)),
                sequence_index.get(t.node_id, 1_000_000),
                str(t.worker_id),
                t.node_id,
            )
        )

    chosen_plans: Dict[tuple[str, str], QueryPlanChoice] = dict(selected_query_plans or {})
    for node_id in db_nodes:
        query_name = db_query_name[node_id]
        choices = db_plan_options[node_id]
        best_idx = None
        best_val = -1.0
        for pidx in range(len(choices)):
            val = float(pulp.value(z[(node_id, pidx)]) or 0.0)
            if val > best_val:
                best_val = val
                best_idx = pidx
        if best_idx is None:
            best_idx = 0
        chosen_plans[(node_id, query_name)] = choices[best_idx]
    if plan_choices:
        for node_id in db_node_ids_all:
            node = graph.nodes[node_id]
            for query in getattr(node, "db_queries", []) or []:
                key = (node_id, query.name)
                if key in chosen_plans:
                    continue
                choices = plan_choices.get(key, ())
                if not choices:
                    choices = (fallback_choice,)
                best = min(choices, key=lambda ch: _plan_base_cost(ch, raw_cost_scale))
                chosen_plans[key] = best

    # Stats
    num_binary = sum(1 for v in model.variables() if v.cat == pulp.LpBinary)
    num_integer = sum(1 for v in model.variables() if v.cat == pulp.LpInteger)
    num_cont = sum(1 for v in model.variables() if v.cat == pulp.LpContinuous)
    num_binary_vars = num_binary + num_integer
    num_cont_vars = num_cont
    stats = MilpSolveStats(
        status=status,
        objective_makespan=objective_val,
        solve_time_s=float(solve_time_s),
        num_nodes=len(node_ids),
        num_workers=len(worker_ids),
        num_binary_vars=int(num_binary_vars),
        num_continuous_vars=int(num_cont_vars),
        num_constraints=int(len(model.constraints)),
        time_limit_s=time_limit_s,
        gap_rel=gap_rel,
    )

    plan_meta = dict(metadata or {})
    node_times: Dict[str, Dict[str, float]] = {}
    for node_id in node_ids:
        start_val = float(start_times.get(node_id, 0.0))
        end_val = float(completion_times.get(node_id, 0.0))
        node_times[node_id] = {
            "start_s": start_val,
            "end_s": end_val,
            "dur_s": max(0.0, end_val - start_val),
        }
    plan_meta["milp"] = {
        "status": stats.status,
        "objective_makespan": stats.objective_makespan,
        "objective_weights": {
            "gpu_cost_max": float(gpu_cost_max_weight),
            "gpu_cost_sum": float(gpu_cost_sum_weight),
            "sum_start": float(sum_start_weight),
            "worker_tiebreak": float(worker_tiebreak_weight),
            "cpu_early_weight": float(cpu_load_early_weight),
        },
        "solve_time_s": stats.solve_time_s,
        "num_nodes": stats.num_nodes,
        "num_workers": stats.num_workers,
        "num_binary_vars": stats.num_binary_vars,
        "num_continuous_vars": stats.num_continuous_vars,
        "num_constraints": stats.num_constraints,
        "time_limit_s": stats.time_limit_s,
        "gap_rel": stats.gap_rel,
        "epoch_quantum_s": epoch_quantum,
        "epoch_buckets": len(used_epochs),
        "epoch_mode": "sequence_index" if llm_node_ids else "time_bucket",
        "node_times_s": node_times,
    }

    return (
        ExecutionPlan(
            workers=dict(workers),
            tasks=tuple(tasks),
            query_plans=plan_choices or {},
            selected_query_plans=chosen_plans,
            metadata=plan_meta,
        ),
        stats,
    )
