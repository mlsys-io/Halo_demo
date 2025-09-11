from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import math

# Project-local types
from halo.components import Operator, Query


# ----------------------------- Tunables ----------------------------- #
# Simple latency model (relative units). You can calibrate these numbers with
# offline profiling if you have real measurements.
PER_TOKEN_COST = 1.0          # base cost unit per generated token
MODEL_SWITCH_PENALTY = 300.0  # penalty for switching model on the same device (e.g., load/re-init)
SAME_MODEL_BONUS = -80.0      # small bonus for keeping the same model consecutively
MIN_STAGE_REPL_DEMAND = 0.6   # when ready ops < devices, replicate only if op demand >= this fraction of max


# ----------------------------- Helpers ------------------------------ #

import re

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?")

def _model_size_billion(model_name: str) -> float:
    """
    Parse model size (in billions of parameters) from model_name.
    Examples:
      "meta-llama/Llama-3.1-8B-Instruct" -> 8.0
      "Llama-3-70B" -> 70.0
      "mistral-7b" -> 7.0
    Returns 1.0 if parsing fails.
    """
    m = _SIZE_RE.search(model_name or "")
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return 1.0  # conservative fallback

def _out_degree(op: Operator) -> int:
    """Out-degree used to rank ready operators."""
    return len(getattr(op, "output_ops", []) or [])

def _op_demand(op: Operator, num_queries: int) -> float:
    """
    Coarse demand proxy that scales with:
      - number of queries,
      - per-op generation length,
      - model size in billions ("xB" in the model name).
    """
    mc = op.model_config
    gen_tokens = int(getattr(mc, "max_tokens", 256) or 256)
    size_b = _model_size_billion(getattr(mc, "model_name", ""))
    return num_queries * gen_tokens * PER_TOKEN_COST * size_b


def _estimate_device_inc_cost(
    last_op: Optional[Operator],
    new_op: Operator,
    num_queries: int,
) -> float:
    """
    Incremental cost to append `new_op` to a device that last ran `last_op`.
    Includes:
      - compute demand for the op
      - model-switch penalty / same-model bonus
    """
    cost = _op_demand(new_op, num_queries)

    if last_op is None:
        return cost  # first task on this device

    last_model = last_op.model_config.model_name
    new_model = new_op.model_config.model_name
    if last_model != new_model:
        cost += MODEL_SWITCH_PENALTY
    else:
        cost += SAME_MODEL_BONUS
    return cost


def _ready_ops(all_ops: List[Operator], assigned: Set[Operator]) -> List[Operator]:
    """Return ops whose all inputs have been assigned."""
    ready = []
    assigned_set = set(assigned)
    for op in all_ops:
        if op in assigned_set:
            continue
        parents = getattr(op, "input_ops", []) or []
        if all(p in assigned_set for p in parents):
            ready.append(op)
    return ready


def _select_ready_subset(
    ready: List[Operator],
    device_cnt: int,
) -> List[Operator]:
    """
    Select at most device_cnt ready ops for the current beam expansion step.
    Rank by:
      - out-degree (desc)
      - max_distance (desc; longer remaining tail first)
    """
    ready_sorted = sorted(
        ready,
        key=lambda x: (_out_degree(x), getattr(x, "max_distance", 0)),
        reverse=True,
    )
    return ready_sorted[: device_cnt]


def _shard_ids(ids: List[int], parts: int) -> List[List[int]]:
    """Split a list of ids into 'parts' shards (nearly equal)."""
    n = len(ids)
    shards = []
    for i in range(parts):
        start = (n * i) // parts
        end = (n * (i + 1)) // parts
        shards.append(ids[start:end])
    return shards


# ------------------------------ State ------------------------------- #

@dataclass(frozen=True)
class Placement:
    """
    A single stage placement for one device: which op to run, and
    (rep_idx, rep_total) if this op is replicated across multiple devices this stage.
    """
    op: Operator
    rep_idx: int = 0
    rep_total: int = 1


@dataclass
class BeamState:
    """
    A partial assignment across devices.

    Attributes
    ----------
    seqs : per-device list of stage placements (the time-ordered execution list).
    assigned : set of ops that are already assigned (covered at least once).
    cost_accum : per-device accumulated cost (for incremental max makespan).
    """
    seqs: List[List[Placement]] = field(default_factory=list)
    assigned: Set[Operator] = field(default_factory=set)
    cost_accum: List[float] = field(default_factory=list)
    # cached global cost for pruning (computed by scorer)
    score: float = 0.0


# --------------------------- Beam Scheduler ------------------------- #

def schedule_search(
    device_cnt: int,
    start_ops: List[Operator],
    end_ops: List[Operator],
    all_ops: List[Operator],
    queries: List[Query],
    beam_width: int = 4,
) -> List[List[Dict[str, Any]]]:
    """
    Beam-search scheduler (search-based). Produces workflows compatible with the
    rest of your system:

        workflows: List[ per-device List[ {"command": "execute", "params": (op, query_ids)} ] ]

    Strategy
    --------
    - Maintain a beam of partial device schedules (BeamState).
    - At each step, find ready ops whose parents are already assigned.
    - Select up to |D| ready ops (ranked by out-degree & remaining tail).
    - If fewer than |D|, replicate the heaviest ready op to fill all devices (query sharding).
    - For each candidate mapping (device -> op), compute incremental cost and a lookahead LB.
    - Keep the top-`beam_width` by score. Repeat until all ops are covered.

    Notes
    -----
    - Replication only happens *within a stage* to ensure every device has work in that step.
    - Query IDs are sharded consistently per replicated op; otherwise, the op receives all IDs.
    """
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    all_ids = [q.id for q in queries]
    num_queries = len(all_ids)

    # Initialize beam with empty schedules
    init = BeamState(
        seqs=[[] for _ in range(device_cnt)],
        assigned=set(),
        cost_accum=[0.0] * device_cnt,
        score=0.0,
    )
    beam: List[BeamState] = [init]

    # Expand until all operators are assigned at least once
    target_count = len(all_ops)
    iters = 0
    while True:
        # Check completeness
        done_states = [s for s in beam if len(s.assigned) == target_count]
        if done_states:
            best = min(done_states, key=lambda s: s.score)
            return _beamstate_to_workflows(best, all_ids)

        # Expand beam
        next_beam: List[BeamState] = []
        for state in beam:
            ready = _ready_ops(all_ops, state.assigned)
            if not ready:
                # No ready ops but not done -> dead end
                continue

            # Choose up to D ready ops for this stage
            chosen = _select_ready_subset(ready, device_cnt)

            # Decide whether to replicate (fill idle devices)
            placements_per_device: List[Placement] = []
            if len(chosen) < device_cnt:
                # Replicate the heaviest chosen op if beneficial
                # Heaviness by demand proxy
                heaviest = max(chosen, key=lambda op: _op_demand(op, num_queries)) if chosen else None
                # If no chosen (shouldn't happen because ready non-empty), skip
                to_fill = device_cnt - len(chosen)
                if heaviest is not None:
                    repl_total = to_fill + 1
                    # Only replicate if demand is high enough
                    demand = _op_demand(heaviest, num_queries)
                    max_demand = max(_op_demand(op, num_queries) for op in ready)
                    if max_demand > 0 and demand / max_demand >= MIN_STAGE_REPL_DEMAND:
                        # Build per-device stage: first assign originals, then replicas
                        base = [Placement(op=o, rep_idx=0, rep_total=1) for o in chosen]
                        reps = [Placement(op=heaviest, rep_idx=i, rep_total=repl_total) for i in range(repl_total)]
                        placements_per_device = base + reps[1:]  # keep one replica spot for the device that also runs original
                        placements_per_device = placements_per_device[:device_cnt]
                    else:
                        # Fallback: just duplicate the first chosen op to fill
                        base = [Placement(op=o, rep_idx=0, rep_total=1) for o in chosen]
                        fill = [Placement(op=chosen[0], rep_idx=0, rep_total=1) for _ in range(to_fill)]
                        placements_per_device = (base + fill)[:device_cnt]
                else:
                    # Defensive fallback: replicate first ready
                    heaviest = ready[0]
                    repl_total = device_cnt
                    placements_per_device = [
                        Placement(op=heaviest, rep_idx=i, rep_total=repl_total) for i in range(repl_total)
                    ]
            else:
                # Exactly one ready op per device for this stage (no replication)
                placements_per_device = [Placement(op=o, rep_idx=0, rep_total=1) for o in chosen[:device_cnt]]

            # Generate a *few* candidate permutations to avoid huge branching
            proposals = _generate_device_proposals(state, placements_per_device, max_variants=min(beam_width, 6))

            # Score and push
            for prop in proposals:
                new_state = _apply_stage(state, prop, num_queries)
                new_state.score = _score_state(new_state, all_ops, num_queries, device_cnt)
                next_beam.append(new_state)

        # Prune beam
        if not next_beam:
            # If we get here, scheduling is impossible under current constraints.
            # Return a degenerate RR as a last resort to keep the system running.
            return _fallback_rr(all_ops, all_ids, device_cnt)

        # Keep best K
        next_beam.sort(key=lambda s: s.score)
        beam = next_beam[:beam_width]
        iters += 1


# ------------------------- Beam Internals --------------------------- #

def _generate_device_proposals(
    state: BeamState,
    placements: List[Placement],
    max_variants: int,
) -> List[List[Placement]]:
    """
    Generate a small set of device->placement orderings for this stage.
    Start from a greedy order (prefer same-model continuation per device),
    then add a few rotated/neighborhood variants to diversify the beam.
    """
    # Greedy: for each device, pick the placement that minimizes its incremental cost
    remaining = placements[:]
    greedy: List[Placement] = []
    for dev in range(len(state.seqs)):
        last = state.seqs[dev][-1].op if state.seqs[dev] else None
        best_idx, best_cost = 0, math.inf
        for idx, plc in enumerate(remaining):
            c = _estimate_device_inc_cost(last, plc.op, num_queries=1)  # normalized; absolute scale unimportant
            if c < best_cost:
                best_idx, best_cost = idx, c
        greedy.append(remaining.pop(best_idx))
        if not remaining and dev < len(state.seqs) - 1:
            # If we exhausted placements early, reuse the last choice (defensive)
            greedy += [greedy[-1]] * (len(state.seqs) - 1 - dev)
            break

    proposals = [greedy]

    # Neighborhood variants: rotate a couple of positions to explore alternatives
    for i in range(1, max_variants):
        variant = greedy[:]
        a = i % len(variant)
        b = (i + 1) % len(variant)
        variant[a], variant[b] = variant[b], variant[a]
        proposals.append(variant)

    # Deduplicate exact duplicates
    uniq = []
    seen = set()
    for p in proposals:
        key = tuple((plc.op.id, plc.rep_idx, plc.rep_total) for plc in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _apply_stage(state: BeamState, proposal: List[Placement], num_queries: int) -> BeamState:
    """
    Return a new state with the proposed placements appended as the next stage.
    Update per-device accumulated cost and the set of assigned ops.
    """
    new = BeamState(
        seqs=[lst[:] for lst in state.seqs],
        assigned=set(state.assigned),
        cost_accum=list(state.cost_accum),
        score=0.0,
    )

    # Compute how many devices run each op in this stage (for query sharding later)
    per_op_repl: Dict[str, int] = {}
    for plc in proposal:
        per_op_repl[plc.op.id] = per_op_repl.get(plc.op.id, 0) + 1

    # Append placements and costs
    for dev, plc in enumerate(proposal):
        last = new.seqs[dev][-1].op if new.seqs[dev] else None

        # Normalize replication info for consistent sharding later
        rep_total = per_op_repl[plc.op.id]
        rep_idx = sum(1 for p in proposal[:dev] if p.op.id == plc.op.id)
        norm_plc = Placement(op=plc.op, rep_idx=rep_idx, rep_total=rep_total)

        new.seqs[dev].append(norm_plc)
        inc = _estimate_device_inc_cost(last, plc.op, num_queries=num_queries)
        new.cost_accum[dev] += inc
        new.assigned.add(plc.op)

    return new


def _score_state(state: BeamState, all_ops: List[Operator], num_queries: int, device_cnt: int) -> float:
    """
    Score = current makespan (max per-device accumulated) + lower bound for remaining work.
    The LB is a crude optimistic estimate: remaining total demand divided evenly across devices.
    """
    current = max(state.cost_accum) if state.cost_accum else 0.0

    remaining_ops = [op for op in all_ops if op not in state.assigned]
    remaining_demand = sum(_op_demand(op, num_queries) for op in remaining_ops)
    optimistic_lb = remaining_demand / max(1, device_cnt)

    return current + optimistic_lb


# ---------------------- Workflows Serialization --------------------- #

def _beamstate_to_workflows(state: BeamState, all_ids: List[int]) -> List[List[Dict[str, Any]]]:
    """
    Convert a completed BeamState into the workflows structure expected by the runtime:
      List[ per-device List[ {"command": "execute", "params": (op, query_ids)} ] ]
    Replicated ops within a stage receive sharded query IDs; standalone ops receive all IDs.
    """
    device_cnt = len(state.seqs)
    workflows: List[List[Dict[str, Any]]] = [[] for _ in range(device_cnt)]

    for dev in range(device_cnt):
        for plc in state.seqs[dev]:
            if plc.rep_total > 1:
                shards = _shard_ids(all_ids, plc.rep_total)
                qids = shards[plc.rep_idx]
            else:
                qids = all_ids

            workflows[dev].append({
                "command": "execute",
                "params": (plc.op, list(qids)),
            })

    return workflows


# --------------------------- Fallback RR ---------------------------- #

def _fallback_rr(all_ops: List[Operator], all_ids: List[int], device_cnt: int) -> List[List[Dict[str, Any]]]:
    """
    Safety net: if the search fails to produce a schedule, use a simple round-robin plan.
    """
    workflows: List[List[Dict[str, Any]]] = [[] for _ in range(device_cnt)]
    dev = 0
    for op in _topo_order(all_ops):
        workflows[dev].append({"command": "execute", "params": (op, list(all_ids))})
        dev = (dev + 1) % device_cnt
    return workflows


def _topo_order(all_ops: List[Operator]) -> List[Operator]:
    """Kahn's algorithm; assumes the graph is a DAG."""
    indeg = {op: len(getattr(op, "input_ops", []) or []) for op in all_ops}
    q = [op for op, d in indeg.items() if d == 0]
    out: List[Operator] = []
    while q:
        u = q.pop(0)
        out.append(u)
        for v in getattr(u, "output_ops", []) or []:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(all_ops):
        raise ValueError("Graph is not a DAG.")
    return out

if __name__ == "__main__":
    """
    Smoke test:
      - Read a workflow DAG from a YAML template (ops/start_ops/end_ops).
      - Create N dummy queries.
      - Run beam-search scheduler and pretty-print per-device workflows.

    Example:
      PYTHONPATH=. python schedulers/search.py \
        --config templates/your_workflow.yaml \
        --num-queries 16 \
        --devices 2 \
        --beam-width 4
    """
    import argparse
    from typing import Optional
    from halo.parser import build_from_path  # root-level parser.py as we defined earlier

    # Try importing project-local Query; if absent, define a tiny stub.
    try:
        from halo.components import Query  # type: ignore
        has_project_query = True
    except Exception:
        has_project_query = False

        class Query:  # minimal stub
            def __init__(self, id: int, prompt: str):
                self.id = id
                self.prompt = prompt

    parser_cli = argparse.ArgumentParser(description="Beam-search scheduler smoke test")
    parser_cli.add_argument("--config", type=str, required=True, help="Path to YAML template")
    parser_cli.add_argument("--num-queries", type=int, default=8, help="Number of dummy queries")
    parser_cli.add_argument("--devices", type=int, default=2, help="Number of devices (workers)")
    parser_cli.add_argument("--beam-width", type=int, default=4, help="Beam width")
    args = parser_cli.parse_args()

    # 1) Build graph from template
    ops_dict, start_ops, end_ops, _models = build_from_path(args.config)
    all_ops = list(ops_dict.values())

    # 2) Create dummy queries (IDs only are used by the scheduler)
    queries = [Query(i, f"Q{i}: dummy prompt") for i in range(args.num_queries)]

    # 3) Run the scheduler
    workflows = schedule_search(
        device_cnt=args.devices,
        start_ops=start_ops,
        end_ops=end_ops,
        all_ops=all_ops,
        queries=queries,
        beam_width=args.beam_width,
    )

    # 4) Pretty-print the result
    def _pp_workflows(wfs):
        print("\n=== Beam-Search Schedule (per device) ===")
        for dev, plan in enumerate(wfs):
            print(f"\nDevice {dev}:")
            for step, item in enumerate(plan):
                op, qids = item["params"][0], item["params"][1]
                op_id = getattr(op, "id", str(op))
                mdl = getattr(getattr(op, "model_config", None), "model_name", "unknown")
                size = _model_size_billion(mdl)
                print(f"  [{step:02d}] execute {op_id:>4s} | model={mdl} (~{size}B) | queries={len(qids)} -> {qids}")

    _pp_workflows(workflows)
