from __future__ import annotations

import os
import time
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from .optimizers.dp import DPSolver, QuerySignature, WorkerState
from .optimizers import (
    build_greedy_cost_plan,
    build_min_switch_plan,
    build_continuous_milp_plan,
    build_random_topo_plan,
    build_rr_topo_plan,
)
from .models import (
    ExecutionPlan,
    ExecutionTask,
    GraphSpec,
    Node,
    QueryPlanChoice,
    Worker,
    build_dependency_list,
)
from .query_planner import QueryPlanEvaluator, plan_evaluator_from_env
from .profiler import GraphProfiler, GraphProfile


def _detect_gpu_count() -> int:
    """根据环境变量或 torch 自动侦测 GPU 数量。"""
    env_override = os.getenv("HALO_NUM_GPUS")
    if env_override is not None:
        try:
            return max(int(env_override), 0)
        except ValueError:
            pass
    try:
        import torch
        cnt = torch.cuda.device_count()
        print("Detected GPU count:", cnt)
        if cnt > 0:
            return cnt
    except Exception:
        return 0


@dataclass(slots=True)
class WorkerPoolConfig:
    num_gpus: int = 1
    num_cpus: int = 1

    def build_workers(self) -> Dict[str, Worker]:
        """构造 GPU + CPU worker，CPU/input 节点在主进程内处理。"""
        workers: Dict[str, Worker] = {}
        for idx in range(max(0, self.num_cpus)):
            worker_id = f"cpu-{idx}"
            workers[worker_id] = Worker(
                id=worker_id,
                kind="cpu",
                device="cpu",
                capacity=1.0,
            )
        for idx in range(self.num_gpus):
            worker_id = f"gpu-{idx}"
            workers[worker_id] = Worker(
                id=worker_id,
                kind="gpu",
                device=f"cuda:{idx}",
                capacity=1.0,
            )
        return workers


class GraphOptimizer:
    """单阶段 DP：同时优化节点顺序、worker 绑定和 query 顺序。"""

    def __init__(
        self,
        num_gpus: int | None = None,
        num_cpu_workers: int | None = None,
        plan_evaluator: QueryPlanEvaluator | None = None,
        *,
        plan_mode: str = "profiled",
        scheduler_mode: str = "dp",
        last_query_window: int = 2,
        epoch_penalty_weight: float = 1.0,
        dp_cpu_load_cost_weight: float | None = None,
        dp_cpu_cost_mode: str | None = 'default',
        dp_switch_penalty_weight: float | None = None,
        dp_gpu_cost_max_weight: float | None = None,
        dp_gpu_cost_sum_weight: float | None = None,
        dp_disable_epoch_batch_cost: bool = False,
        dp_disable_cpu_load_cost: bool | None = False,
        enable_post_refine: bool = False,
    ):
        computed = num_gpus if num_gpus is not None else _detect_gpu_count()
        cpu_workers = (1 if num_cpu_workers is None else num_cpu_workers)
        self.pool = WorkerPoolConfig(num_gpus=max(1, computed), num_cpus=max(1, cpu_workers))
        self.plan_mode = plan_mode
        self.scheduler_mode = self._normalize_scheduler_mode(scheduler_mode)
        self._plan_evaluator = plan_evaluator or plan_evaluator_from_env()
        self._profiler = GraphProfiler(plan_evaluator=self._plan_evaluator)
        self._http_profile_latency: Dict[str, float] = {}
        self._http_profile_samples: Dict[str, int] = {}
        self.last_query_window = max(1, last_query_window)
        env_db_weight = os.getenv("HALO_DP_CPU_LOAD_COST_WEIGHT")
        if env_db_weight is not None and env_db_weight.strip() != "":
            try:
                self.dp_cpu_load_cost_weight = max(0.0, float(env_db_weight))
            except ValueError:
                self.dp_cpu_load_cost_weight = 1.0
        elif dp_cpu_load_cost_weight is not None:
            self.dp_cpu_load_cost_weight = max(0.0, float(dp_cpu_load_cost_weight))
        else:
            self.dp_cpu_load_cost_weight = 1.0
        if dp_cpu_cost_mode is not None:
            normalized = dp_cpu_cost_mode.strip().lower()
            if normalized not in ("default", "naive"):
                raise ValueError(
                    f"Unsupported dp_cpu_cost_mode '{dp_cpu_cost_mode}'. Expected 'default' or 'naive'."
                )
            self.dp_cpu_cost_mode = normalized
        else:
            self.dp_cpu_cost_mode = None
        env_gpu_cost_max = (os.getenv("HALO_DP_GPU_COST_MAX_WEIGHT") or "").strip()
        env_gpu_cost_sum = (os.getenv("HALO_DP_GPU_COST_SUM_WEIGHT") or "").strip()
        max_weight = dp_gpu_cost_max_weight
        sum_weight = dp_gpu_cost_sum_weight
        if max_weight is None and sum_weight is None:
            if env_gpu_cost_max:
                try:
                    max_weight = float(env_gpu_cost_max)
                except ValueError:
                    max_weight = None
            if env_gpu_cost_sum:
                try:
                    sum_weight = float(env_gpu_cost_sum)
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
        self.dp_gpu_cost_max_weight = max(0.0, float(max_weight))
        self.dp_gpu_cost_sum_weight = max(0.0, float(sum_weight))
        self.dp_disable_epoch_batch_cost = bool(dp_disable_epoch_batch_cost)
        self.dp_disable_cpu_load_cost = bool(dp_disable_cpu_load_cost)
        env_switch_weight = os.getenv("HALO_DP_SWITCH_PENALTY_WEIGHT")
        if env_switch_weight is not None and env_switch_weight.strip() != "":
            try:
                self.dp_switch_penalty_weight = max(0.0, float(env_switch_weight))
            except ValueError:
                self.dp_switch_penalty_weight = 0.1
        elif dp_switch_penalty_weight is not None:
            self.dp_switch_penalty_weight = max(0.0, float(dp_switch_penalty_weight))
        else:
            self.dp_switch_penalty_weight = 0.1
        self.enable_post_refine = bool(enable_post_refine)
        
        # Cost-model knobs roughly calibrated from telemetry (20B model stats).
        self._model_size_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:b|bn|billion)", re.IGNORECASE)
        self._default_model_size_b = 20.0
        # Allow quick tuning via env without code edits.
        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        self._model_init_sec_per_b = _env_float("HALO_COST_MODEL_INIT_SEC_PER_B", 0.75)
        self._llm_base_sec_per_b = _env_float("HALO_COST_LLM_BASE_SEC_PER_B", 0.25)
        self._llm_input_sec = _env_float("HALO_COST_LLM_INPUT_SEC", 0.15)
        self._db_input_sec = _env_float("HALO_COST_DB_INPUT_SEC", 0.05)
        self._raw_cost_scale = 3.65e-6
        self._epoch_penalty_weight = float(epoch_penalty_weight)
        # cost model 中“输入 query 数量”（一个 batch/run 内的 user queries 个数）。
        # 默认按 1；可在 build_plan 时通过 input_query_count 覆盖（runner 里通常传 --sample-count）。
        self._input_query_count = 1
        self._validate_modes()

    def _normalize_scheduler_mode(self, mode: str) -> str:
        normalized = (mode or "").lower()
        aliases = {
            "random": "random_topo",
            "rand": "random_topo",
        }
        return aliases.get(normalized, normalized)

    def _validate_modes(self) -> None:
        allowed_plan = {"profiled", "baseline", "default"}
        if self.plan_mode not in allowed_plan:
            raise ValueError(f"Unsupported plan_mode '{self.plan_mode}'. Expected one of {allowed_plan}.")
        allowed_scheduler = {"auto", "dp", "rr_topo", "random_topo", "model_first", "greedy", "minswitch", "milp", "opwise"}
        if self.scheduler_mode not in allowed_scheduler:
            raise ValueError(
                f"Unsupported scheduler_mode '{self.scheduler_mode}'. Expected one of {allowed_scheduler}."
            )

    def _resolve_auto_scheduler_mode(self, input_query_count: int) -> str:
        if self.scheduler_mode != "auto":
            return self.scheduler_mode
        return "greedy" if input_query_count < 512 else "dp"

    def build_plan(
        self,
        graph: GraphSpec,
        sample_contexts: Sequence[Mapping[str, Any]] | None = None,
        *,
        input_query_count: int | None = None,
    ) -> ExecutionPlan:
        profile_start = time.perf_counter()
        profile = self._profile_graph(graph, sample_contexts)
        plan_eval_duration = time.perf_counter() - profile_start
        plan_choices = profile.plan_choices
        self._http_profile_latency = dict(profile.http_latencies_s)
        self._http_profile_samples = dict(profile.http_samples)

        # 规划阶段的 cost 需要一个“输入 query 数量”标尺：
        # - runner remember: --sample-count 是本次 run 处理的 query 数量（建议用它）
        # - 若不提供，则退化用 sample_contexts 数量（plan profiling 的 sample 数）
        inferred = len(sample_contexts) if sample_contexts else 1
        if input_query_count is None:
            self._input_query_count = max(1, int(inferred))
        else:
            self._input_query_count = max(1, int(input_query_count))

        requested_scheduler = self.scheduler_mode
        resolved_scheduler = self._resolve_auto_scheduler_mode(self._input_query_count)
        if requested_scheduler == "auto":
            self.scheduler_mode = resolved_scheduler

        # When a graph mixes DB + HTTP nodes, ensure we have at least 2 CPU
        # workers so HTTP sleeps cannot stall DB dispatch on the same thread.
        has_http = any(n.engine == "http" for n in graph.nodes.values())
        has_db = any(n.engine == "db" or n.type == "db_query" for n in graph.nodes.values())
        if has_http and has_db and self.pool.num_cpus < 2:
            self.pool.num_cpus = 2
        workers = self.pool.build_workers()
        schedulable_ids = self._schedulable_node_ids(graph)
        dependencies = self._filtered_dependencies(graph, schedulable_ids)
        node_worker_options = self._node_worker_options(graph, workers, schedulable_ids)

        def _finalize_plan(plan: ExecutionPlan) -> ExecutionPlan:
            if requested_scheduler == "auto":
                plan.metadata["scheduler_mode_requested"] = requested_scheduler
                plan.metadata["scheduler_mode_resolved"] = resolved_scheduler
                self.scheduler_mode = requested_scheduler
            self._attach_profile_metadata(plan)
            return plan

        if self.scheduler_mode == "rr_topo":
            return _finalize_plan(self._build_rr_topo_execution_plan(
                graph=graph,
                workers=workers,
                dependencies=dependencies,
                plan_choices=plan_choices,
                schedulable_ids=schedulable_ids,
                plan_eval_duration=plan_eval_duration,
                node_worker_options=node_worker_options,
            ))
        if self.scheduler_mode == "random_topo":
            return _finalize_plan(self._build_random_topo_execution_plan(
                graph=graph,
                workers=workers,
                dependencies=dependencies,
                plan_choices=plan_choices,
                schedulable_ids=schedulable_ids,
                plan_eval_duration=plan_eval_duration,
                node_worker_options=node_worker_options,
            ))
        if self.scheduler_mode == "milp":
            return _finalize_plan(self._build_milp_execution_plan(
                graph=graph,
                workers=workers,
                dependencies=dependencies,
                plan_choices=plan_choices,
                schedulable_ids=schedulable_ids,
                plan_eval_duration=plan_eval_duration,
                node_worker_options=node_worker_options,
            ))
        if self.scheduler_mode == "greedy":
            return _finalize_plan(self._build_greedy_execution_plan(
                graph=graph,
                workers=workers,
                dependencies=dependencies,
                plan_choices=plan_choices,
                schedulable_ids=schedulable_ids,
                plan_eval_duration=plan_eval_duration,
                node_worker_options=node_worker_options,
            ))
        if self.scheduler_mode == "minswitch":
            return _finalize_plan(self._build_min_switch_execution_plan(
                graph=graph,
                workers=workers,
                dependencies=dependencies,
                plan_choices=plan_choices,
                schedulable_ids=schedulable_ids,
                plan_eval_duration=plan_eval_duration,
                node_worker_options=node_worker_options,
            ))
        if self.scheduler_mode == "opwise":
            return _finalize_plan(self._build_data_parallel_execution_plan(
                graph=graph,
                workers=workers,
                dependencies=dependencies,
                schedulable_ids=schedulable_ids,
                plan_choices=plan_choices,
                plan_eval_duration=plan_eval_duration,
                node_worker_options=node_worker_options,
            ))

        worker_ids = tuple(sorted(workers))
        gpu_worker_ids = tuple(sorted([wid for wid, w in workers.items() if w.kind == "gpu"]))
        cpu_worker_ids = tuple(sorted([wid for wid, w in workers.items() if w.kind != "gpu"]))

        # 动态规划调度器
        schedule_start = time.perf_counter()
        # 初始化 worker 状态（cache/model 窗口）
        initial_worker_states = tuple(
            WorkerState(
                worker_idx=idx,
                last_model_id=-1,
                last_node_id=-1,
            )
            for idx, _ in enumerate(worker_ids)
        )

        solver = DPSolver(
            graph=graph,
            dependencies=dependencies,
            workers=workers,
            worker_ids=worker_ids,
            plan_choices=plan_choices,
            window_size=self.last_query_window,
            exec_cost_fn=self._exec_cost,
            cache_multiplier_fn=self._cache_multiplier,
            model_init_cost_fn=self._model_init_cost,
            llm_cache_bonus_fn=self._llm_cache_bonus,
            node_ids=schedulable_ids,
            epoch_penalty_fn=self._epoch_penalty,
            cpu_load_cost_weight=self.dp_cpu_load_cost_weight,
            cpu_cost_mode=self.dp_cpu_cost_mode,
            switch_penalty_weight=self.dp_switch_penalty_weight,
            gpu_cost_max_weight=self.dp_gpu_cost_max_weight,
            gpu_cost_sum_weight=self.dp_gpu_cost_sum_weight,
            disable_epoch_batch_cost=self.dp_disable_epoch_batch_cost,
            disable_cpu_load_cost=self.dp_disable_cpu_load_cost,
            node_worker_options=node_worker_options,
            gpu_worker_ids=gpu_worker_ids,
            cpu_worker_ids=cpu_worker_ids,
            http_latency_s=self._http_profile_latency,
            debug_log=False,
        )
        _, schedule, selected_plans = solver.solve(initial_worker_states)

        tasks: List[ExecutionTask] = []
        for epoch, worker_id, node_id in schedule:
            deps = tuple(dependencies.get(node_id, []))
            tasks.append(
                ExecutionTask(
                    node_id=node_id,
                    worker_id=worker_id,
                    dependencies=deps,
                    epoch=epoch,
                )
            )

        schedule_duration = time.perf_counter() - schedule_start
        metadata = {
            "optimize_duration_s": plan_eval_duration + schedule_duration,
            "planning_breakdown_s": {
                "plan_evaluation": plan_eval_duration,
                "scheduling": schedule_duration,
            },
            "scheduler_mode": self.scheduler_mode,
            "dp_stats": {
                "states_explored": solver._solve_calls,
                "memo_entries": len(solver._memo),
                "memo_hits": solver._memo_hits,
                "best_cost": solver._global_best_cost,
            },
            "dp_cpu_load_cost_weight": self.dp_cpu_load_cost_weight,
            "dp_cpu_cost_mode": solver.cpu_cost_mode,
            "dp_switch_penalty_weight": self.dp_switch_penalty_weight,
            "dp_gpu_cost_max_weight": self.dp_gpu_cost_max_weight,
            "dp_gpu_cost_sum_weight": self.dp_gpu_cost_sum_weight,
            "dp_disable_epoch_batch_cost": self.dp_disable_epoch_batch_cost,
            "dp_disable_cpu_load_cost": self.dp_disable_cpu_load_cost,
            "input_query_count": self._input_query_count,
        }

        plan = ExecutionPlan(
            workers=workers,
            tasks=tuple(tasks),
            query_plans=plan_choices,
            selected_query_plans=selected_plans,
            metadata=metadata,
        )
        return _finalize_plan(self._apply_post_refinements(plan, graph))

    def _build_rr_topo_execution_plan(
        self,
        *,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        dependencies: Mapping[str, Sequence[str]],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
        schedulable_ids: Sequence[str],
        plan_eval_duration: float,
        node_worker_options: Mapping[str, Sequence[str]],
    ) -> ExecutionPlan:
        schedule_start = time.perf_counter()
        selected_plans = self._select_query_plan_defaults(plan_choices)
        plan = build_rr_topo_plan(
            graph=graph,
            workers=workers,
            dependencies=dependencies,
            schedulable_ids=schedulable_ids,
            node_worker_options=node_worker_options,
            plan_choices=plan_choices,
            selected_query_plans=selected_plans,
        )
        schedule_duration = time.perf_counter() - schedule_start
        plan.metadata.update(
            {
                "optimize_duration_s": plan_eval_duration + schedule_duration,
                "planning_breakdown_s": {
                    "plan_evaluation": plan_eval_duration,
                    "scheduling": schedule_duration,
                },
                "scheduler_mode": self.scheduler_mode,
            }
        )
        return self._apply_post_refinements(plan, graph)

    def _build_random_topo_execution_plan(
        self,
        *,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        dependencies: Mapping[str, Sequence[str]],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
        schedulable_ids: Sequence[str],
        plan_eval_duration: float,
        node_worker_options: Mapping[str, Sequence[str]],
    ) -> ExecutionPlan:
        schedule_start = time.perf_counter()
        selected_plans = self._select_query_plan_defaults(plan_choices)
        seed_env = os.getenv("HALO_SCHED_SEED")
        seed: int | None = None
        if seed_env is not None and seed_env != "":
            try:
                seed = int(seed_env)
            except ValueError:
                seed = None

        plan = build_random_topo_plan(
            graph=graph,
            workers=workers,
            dependencies=dependencies,
            schedulable_ids=schedulable_ids,
            node_worker_options=node_worker_options,
            plan_choices=plan_choices,
            selected_query_plans=selected_plans,
            seed=seed,
        )
        schedule_duration = time.perf_counter() - schedule_start
        plan.metadata.update(
            {
                "optimize_duration_s": plan_eval_duration + schedule_duration,
                "planning_breakdown_s": {
                    "plan_evaluation": plan_eval_duration,
                    "scheduling": schedule_duration,
                },
                "scheduler_mode": self.scheduler_mode,
            }
        )
        return self._apply_post_refinements(plan, graph)

    def _build_milp_execution_plan(
        self,
        *,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        dependencies: Mapping[str, Sequence[str]],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
        schedulable_ids: Sequence[str],
        plan_eval_duration: float,
        node_worker_options: Mapping[str, Sequence[str]],
    ) -> ExecutionPlan:
        schedule_start = time.perf_counter()
        plan, _stats = build_continuous_milp_plan(
            graph=graph,
            workers=workers,
            dependencies=dependencies,
            schedulable_ids=schedulable_ids,
            node_worker_options=node_worker_options,
            plan_choices=plan_choices,
            exec_cost_fn=self._exec_cost,
            model_init_cost_fn=self._model_init_cost,
            llm_cache_bonus_fn=self._llm_cache_bonus,
            raw_cost_scale=self._raw_cost_scale,
            window_size=self.last_query_window,
            cpu_load_cost_weight=self.dp_cpu_load_cost_weight,
            gpu_cost_max_weight=self.dp_gpu_cost_max_weight,
            gpu_cost_sum_weight=self.dp_gpu_cost_sum_weight,
        )
        schedule_duration = time.perf_counter() - schedule_start
        plan.metadata.update(
            {
                "optimize_duration_s": plan_eval_duration + schedule_duration,
                "planning_breakdown_s": {
                    "plan_evaluation": plan_eval_duration,
                    "scheduling": schedule_duration,
                },
                "scheduler_mode": self.scheduler_mode,
            }
        )
        return self._apply_post_refinements(plan, graph)

    def _build_greedy_execution_plan(
        self,
        *,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        dependencies: Mapping[str, Sequence[str]],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
        schedulable_ids: Sequence[str],
        plan_eval_duration: float,
        node_worker_options: Mapping[str, Sequence[str]],
    ) -> ExecutionPlan:
        schedule_start = time.perf_counter()
        selected_plans = self._select_query_plan_defaults(plan_choices)
        plan = build_greedy_cost_plan(
            graph=graph,
            workers=workers,
            dependencies=dependencies,
            schedulable_ids=schedulable_ids,
            node_worker_options=node_worker_options,
            plan_choices=plan_choices,
            selected_query_plans=selected_plans,
            score_fn=self._greedy_score,
        )
        schedule_duration = time.perf_counter() - schedule_start
        plan.metadata.update(
            {
                "optimize_duration_s": plan_eval_duration + schedule_duration,
                "planning_breakdown_s": {
                    "plan_evaluation": plan_eval_duration,
                    "scheduling": schedule_duration,
                },
                "scheduler_mode": self.scheduler_mode,
            }
        )
        return self._apply_post_refinements(plan, graph)

    def _build_min_switch_execution_plan(
        self,
        *,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        dependencies: Mapping[str, Sequence[str]],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
        schedulable_ids: Sequence[str],
        plan_eval_duration: float,
        node_worker_options: Mapping[str, Sequence[str]],
    ) -> ExecutionPlan:
        schedule_start = time.perf_counter()
        selected_plans = self._select_query_plan_defaults(plan_choices)
        plan = build_min_switch_plan(
            graph=graph,
            workers=workers,
            dependencies=dependencies,
            schedulable_ids=schedulable_ids,
            node_worker_options=node_worker_options,
            plan_choices=plan_choices,
            selected_query_plans=selected_plans,
        )
        schedule_duration = time.perf_counter() - schedule_start
        plan.metadata.update(
            {
                "optimize_duration_s": plan_eval_duration + schedule_duration,
                "planning_breakdown_s": {
                    "plan_evaluation": plan_eval_duration,
                    "scheduling": schedule_duration,
                },
                "scheduler_mode": self.scheduler_mode,
            }
        )
        return self._apply_post_refinements(plan, graph)

    def _build_data_parallel_execution_plan(
        self,
        *,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        dependencies: Mapping[str, Sequence[str]],
        schedulable_ids: Sequence[str],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
        plan_eval_duration: float,
        node_worker_options: Mapping[str, Sequence[str]],
    ) -> ExecutionPlan:
        """One logical node per epoch; LLM 节点以数据并行方式同时跑在所有 GPU worker 上。"""
        schedule_start = time.perf_counter()
        selected_plans = self._select_query_plan_defaults(plan_choices)

        gpu_worker_ids = [wid for wid, w in workers.items() if w.kind == "gpu"]
        cpu_worker_ids = [wid for wid, w in workers.items() if w.kind != "gpu"]
        if not cpu_worker_ids:
            raise RuntimeError("Data-parallel scheduler requires at least one CPU worker for DB nodes.")

        order = list(schedulable_ids)
        # 简单拓扑排序，若失败则使用原顺序。
        try:
            indeg = {nid: 0 for nid in order}
            for nid in order:
                for dep in dependencies.get(nid, ()):
                    indeg[nid] += 1
            queue = [nid for nid, deg in indeg.items() if deg == 0]
            topo: list[str] = []
            while queue:
                nid = queue.pop()
                topo.append(nid)
                for child in order:
                    if nid in dependencies.get(child, ()):
                        indeg[child] -= 1
                        if indeg[child] == 0:
                            queue.append(child)
            if len(topo) == len(order):
                order = topo
        except Exception:
            pass

        tasks: list[ExecutionTask] = []
        epoch = 0
        for node_id in order:
            node = graph.nodes[node_id]
            deps = tuple(dependencies.get(node_id, ()))
            if node.engine == "vllm":
                if not gpu_worker_ids:
                    raise RuntimeError(f"No GPU workers available for LLM node '{node_id}'.")
                allowed = tuple(node_worker_options.get(node_id, gpu_worker_ids))
                worker_assignment = tuple(wid for wid in gpu_worker_ids if wid in allowed)
                if not worker_assignment:
                    raise RuntimeError(f"No eligible GPU workers for LLM node '{node_id}'.")
            else:
                allowed = tuple(node_worker_options.get(node_id, cpu_worker_ids))
                worker_assignment = next((wid for wid in cpu_worker_ids if wid in allowed), None)
                if worker_assignment is None:
                    raise RuntimeError(f"No eligible CPU workers for node '{node_id}'.")
            tasks.append(
                ExecutionTask(
                    node_id=node_id,
                    worker_id=worker_assignment,
                    dependencies=deps,
                    epoch=epoch,
                )
            )
            epoch += 1

        schedule_duration = time.perf_counter() - schedule_start
        metadata = {
            "optimize_duration_s": plan_eval_duration + schedule_duration,
            "planning_breakdown_s": {
                "plan_evaluation": plan_eval_duration,
                "scheduling": schedule_duration,
            },
            "scheduler_mode": self.scheduler_mode,
        }
        plan = ExecutionPlan(
            workers=dict(workers),
            tasks=tuple(tasks),
            query_plans=plan_choices or {},
            selected_query_plans=selected_plans,
            metadata=metadata,
        )
        return self._apply_post_refinements(plan, graph)

    def _exec_cost(self, node: Node, _worker: Worker) -> float:
        """执行成本（不含模型切换）。

        - vLLM: 仅依赖 model size + 输入 query 数量（通常等于 runner 的 --sample-count）
        - DB: 仅依赖输入 query 数量（estimate cost 由 query plan choice 单独提供）
        - HTTP: 使用 profiler 结果（或 fallback 到配置的 sleep/latency）
        """
        if node.engine == "vllm":
            size_b = self._model_size_b(node)
            input_factor = self._llm_input_sec * max(1, int(getattr(self, "_input_query_count", 1)))
            return self._llm_base_sec_per_b * size_b + input_factor
        if node.engine == "http":
            return self._http_exec_cost(node)
        return self._db_input_sec * max(1, int(getattr(self, "_input_query_count", 1)))

    def _greedy_score(self, node: Node, worker: Worker, load: int, last_model: str | None) -> float:
        """Greedy scheduler综合分：执行成本 + 切换成本 + 负载成本。"""
        exec_cost = self._exec_cost(node, worker)
        switch_cost = self._model_init_cost(node, last_model)
        load_cost = 2.0 * max(0, load)
        return exec_cost + switch_cost + load_cost

    def _http_exec_cost(self, node: Node) -> float:
        if self._http_profile_latency:
            profiled = self._http_profile_latency.get(node.id)
            if profiled is not None:
                try:
                    return max(0.0, float(profiled))
                except (TypeError, ValueError):
                    pass
        raw = node.raw if isinstance(node.raw, dict) else {}
        for key, scale in (
            ("sleep_s", 1.0),
            ("sleep_ms", 0.001),
            ("latency_s", 1.0),
            ("latency_ms", 0.001),
            ("timeout_s", 1.0),
            ("timeout_ms", 0.001),
        ):
            if key not in raw:
                continue
            value = raw.get(key)
            if isinstance(value, str):
                value = value.strip()
            try:
                seconds = float(value) * scale
            except (TypeError, ValueError):
                continue
            return max(0.0, seconds)
        return 0.0

    def _profile_graph(
        self,
        graph: GraphSpec,
        contexts: Sequence[Mapping[str, Any]] | None,
    ) -> GraphProfile:
        include_sql = self.plan_mode != "baseline" and self._plan_evaluator is not None
        default_only = self.plan_mode == "default"
        return self._profiler.profile_graph(
            graph,
            contexts,
            default_plan_only=default_only,
            include_sql=include_sql,
            include_http=True,
        )

    def _attach_profile_metadata(self, plan: ExecutionPlan) -> None:
        if self._http_profile_latency:
            plan.metadata.setdefault("http_profile_latency_s", dict(self._http_profile_latency))
        if self._http_profile_samples:
            plan.metadata.setdefault("http_profile_samples", dict(self._http_profile_samples))

    def _select_query_plan_defaults(
        self,
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]] | None,
    ) -> Dict[tuple[str, str], QueryPlanChoice]:
        """Select the cheapest available plan per query for baseline schedulers."""
        selected: Dict[tuple[str, str], QueryPlanChoice] = {}
        if not plan_choices:
            return selected
        for key, choices in plan_choices.items():
            if not choices:
                continue
            best = min(choices, key=self._plan_choice_score)
            selected[key] = best
        return selected

    def _apply_post_refinements(self, plan: ExecutionPlan, graph: GraphSpec) -> ExecutionPlan:
        """Apply lightweight post-processing tweaks after building a plan."""
        if not self.enable_post_refine:
            return plan
        refined = self._refine_shard_last_epoch_idle_workers(plan, graph)
        return refined

    def _refine_shard_last_epoch_idle_workers(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
    ) -> ExecutionPlan:
        """If the final epoch has idle GPU workers, shard the last LLM node across them."""
        if not plan.tasks:
            return plan
        max_epoch = max(task.epoch for task in plan.tasks)
        last_epoch_indices = [idx for idx, task in enumerate(plan.tasks) if task.epoch == max_epoch]
        if not last_epoch_indices:
            return plan

        # Choose the last LLM task in the final epoch (ignore trailing DB tasks).
        target_idx = None
        target_task = None
        for idx in reversed(last_epoch_indices):
            candidate = plan.tasks[idx]
            node = graph.nodes.get(candidate.node_id)
            if node is not None and node.engine == "vllm":
                target_idx = idx
                target_task = candidate
                break
        if target_idx is None or target_task is None:
            return plan

        node = graph.nodes.get(target_task.node_id)
        if node is None:
            return plan

        # Workers already used in the final epoch.
        used_workers: Set[str] = set()
        for idx in last_epoch_indices:
            task = plan.tasks[idx]
            wid_seq = task.worker_id if isinstance(task.worker_id, (list, tuple)) else (task.worker_id,)
            for wid in wid_seq:
                if wid:
                    used_workers.add(wid)

        idle_gpu_workers = [
            wid for wid, worker in plan.workers.items()
            if worker.kind == "gpu" and wid not in used_workers
        ]
        if not idle_gpu_workers:
            return plan

        current_assignment = target_task.worker_id if isinstance(target_task.worker_id, (list, tuple)) else (target_task.worker_id,)
        current_assignment = tuple(w for w in current_assignment if w)
        if not current_assignment:
            return plan
        # Only shard GPU-assigned LLM nodes.
        if any(plan.workers.get(wid) is None or plan.workers[wid].kind != "gpu" for wid in current_assignment):
            return plan

        new_assignment = tuple(dict.fromkeys(list(current_assignment) + sorted(idle_gpu_workers)))
        if len(new_assignment) == len(current_assignment):
            return plan

        tasks = list(plan.tasks)
        tasks[target_idx] = ExecutionTask(
            node_id=target_task.node_id,
            worker_id=new_assignment,
            dependencies=target_task.dependencies,
            epoch=target_task.epoch,
        )

        metadata = dict(plan.metadata)
        refinements = dict(metadata.get("refinements") or {})
        refinements["shard_last_epoch_idle_gpu"] = {
            "target_node": target_task.node_id,
            "epoch": target_task.epoch,
            "workers": new_assignment,
            "added_workers": [w for w in new_assignment if w not in current_assignment],
        }
        metadata["refinements"] = refinements

        return ExecutionPlan(
            workers=plan.workers,
            tasks=tuple(tasks),
            query_plans=plan.query_plans,
            selected_query_plans=plan.selected_query_plans,
            metadata=metadata,
        )

    def _plan_choice_score(self, choice: QueryPlanChoice) -> tuple[float, float, str]:
        # Prefer explicit cost, fall back to raw_cost, then plan_id for stability.
        cost = choice.cost if choice.cost is not None else float("inf")
        raw = choice.raw_cost if choice.raw_cost is not None else float("inf")
        return (float(cost), float(raw), choice.plan_id)

    # 基于 footprint 的粗略 cache 折扣
    def _cache_multiplier(self, window: Sequence[QuerySignature], choice: QueryPlanChoice) -> float:
        base_fp = choice.footprints or {}
        if not base_fp or not window:
            return 1.0
        overlap = 0.0
        for sig in window:
            fp = dict(sig.footprints)
            for name, weight in fp.items():
                if name in base_fp:
                    overlap += min(weight, base_fp[name])
        if overlap <= 0:
            return 1.0
        return max(0.5, 1.0 - 0.01 * overlap)

    # 模型初始化成本（仅 vLLM）
    def _model_init_cost(self, node: Node, last_model: str | None) -> float:
        if node.engine != "vllm":
            return 0.0
        node_model = node.model or ""
        size_b = self._model_size_b(node)
        if last_model is None:
            return (self._model_init_sec_per_b * size_b) / 2.0
        if last_model == node_model:
            return 0.0
        return self._model_init_sec_per_b * size_b

    # 父子节点同 worker 时的 cache 利好
    def _llm_cache_bonus(self, node: Node, last_node: str | None, parents: Sequence[str]) -> float:
        if not parents or not last_node:
            return 1.0
        if last_node in parents:
            return 0.9
        return 1.0

    def _epoch_penalty(self, epoch: int) -> float:
        """随 epoch 轻微增长的惩罚，用于鼓励早完成。"""
        return self._epoch_penalty_weight * (1.0 + 0.1 * max(0, epoch))

    def _schedulable_node_ids(self, graph: GraphSpec) -> Tuple[str, ...]:
        return tuple(
            sorted(node_id for node_id, node in graph.nodes.items() if self._should_schedule(node))
        )

    def _should_schedule(self, node: Node) -> bool:
        return node.type != "input"

    def _node_worker_options(
        self,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
        node_ids: Sequence[str],
    ) -> Dict[str, Tuple[str, ...]]:
        cpu_worker_ids = sorted(wid for wid, w in workers.items() if w.kind == "cpu")
        # Affinity routing: when at least 2 CPU workers are available and the
        # graph mixes DB + HTTP nodes, dedicate cpu-1 to HTTP so that an
        # ``time.sleep`` in HTTP cannot block DB dispatch on cpu-0. Processor /
        # noop nodes share the DB-affinity worker (cpu-0).
        has_http = any(n.engine == "http" for n in graph.nodes.values())
        has_db = any(n.engine == "db" or n.type == "db_query" for n in graph.nodes.values())
        split_affinity = has_http and has_db and len(cpu_worker_ids) >= 2
        db_workers = (cpu_worker_ids[0],) if cpu_worker_ids else tuple()
        http_workers = (cpu_worker_ids[1],) if split_affinity else tuple(cpu_worker_ids)

        options: Dict[str, Tuple[str, ...]] = {}
        for node_id in node_ids:
            node = graph.nodes[node_id]
            if split_affinity and node.engine == "http":
                eligible: Tuple[str, ...] = http_workers
            elif split_affinity and (node.engine == "db" or node.type == "db_query"):
                eligible = db_workers
            else:
                eligible = tuple(
                    wid for wid, worker in workers.items() if self._worker_can_run(node, worker)
                )
            if not eligible:
                raise RuntimeError(
                    f"No compatible workers for node '{node_id}' (engine={node.engine}, type={node.type})."
                )
            options[node_id] = tuple(sorted(eligible))
        return options

    def _worker_can_run(self, node: Node, worker: Worker) -> bool:
        if node.engine == "vllm":
            return worker.kind == "gpu"
        if node.engine in ("db", "http") or node.type == "db_query":
            return worker.kind == "cpu"
        # 默认认为非 LLM 节点跑在 CPU
        return worker.kind == "cpu"

    def _model_size_b(self, node: Node) -> float:
        """Best-effort parse of model size (in billions of params)."""
        if not node.model:
            return self._default_model_size_b
        match = self._model_size_pattern.search(node.model)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        # Fall back to heuristic based on known families.
        lowered = node.model.lower()
        if "7b" in lowered:
            return 7.0
        if "13b" in lowered:
            return 13.0
        if "20b" in lowered or "20-billion" in lowered:
            return 20.0
        if "32b" in lowered:
            return 32.0
        return self._default_model_size_b

    def _filtered_dependencies(
        self,
        graph: GraphSpec,
        node_ids: Sequence[str] | None = None,
    ) -> Dict[str, Tuple[str, ...]]:
        raw_dependencies = build_dependency_list(graph.edges)
        schedulable_ids = set(node_ids or self._schedulable_node_ids(graph))
        return {
            node_id: tuple(
                dep for dep in raw_dependencies.get(node_id, []) if dep in schedulable_ids
            )
            for node_id in schedulable_ids
        }
