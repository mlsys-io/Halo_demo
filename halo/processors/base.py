from __future__ import annotations

import abc
import os
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence, Set

from .. import metrics
from ..db import (
    DatabaseExecutor,
    DefaultDatabaseExecutor,
    make_peer_postgres_executor,
)
from ..models import ExecutionPlan, GraphSpec, Node
from ..executor import DBNodeExecutor, HTTPNodeExecutor, ProcessorNodeExecutor, VLLMNodeExecutor
from ..engines import make_vllm_provider


_PROGRESS_ENGINES = {"vllm", "db", "http"}


def is_progress_node(node: Node | None) -> bool:
    return bool(node and node.engine in _PROGRESS_ENGINES)


def count_progress_nodes(plan: ExecutionPlan, graph: GraphSpec) -> int:
    count = 0
    for task in plan.tasks:
        node = graph.nodes.get(task.node_id)
        if is_progress_node(node):
            count += 1
    return count


class BaseGraphProcessor(abc.ABC):
    """Abstract base class for graph processors."""

    def __init__(
        self,
        *,
        engine_kwargs: Dict[str, Any] | None = None,
        db_connect_kwargs: Dict[str, Any] | None = None,
        db_pool_size: int | None = None,
        db_concurrency: int | None = None,
        enable_result_cache: bool | None = None,
        enable_prepared_statements: bool | None = None,
        prepare_threshold: int | None = None,
        enable_db_explain_analyze: bool | None = None,
        db_explain_mode: str | None = None,
        db_explain_sample_rate: float | None = None,
        http_concurrency: int | None = None,
        http_default_sleep_s: float | None = None,
        enable_tensor_parallel: bool = False,
        tensor_parallel_size: int | None = None,
    ) -> None:
        self.engine_kwargs = dict(engine_kwargs or {})

        if db_connect_kwargs:
            self.db_executor_factory = lambda: make_peer_postgres_executor(
                **db_connect_kwargs,
                pool_size=db_pool_size,
            )
        else:
            self.db_executor_factory = DefaultDatabaseExecutor
        
        self.db_executor = self.db_executor_factory()
        resolved_result_cache = True if enable_result_cache is None else enable_result_cache
        resolved_prepare = (
            True if enable_prepared_statements is None else enable_prepared_statements
        )
        self.db_node_executor = DBNodeExecutor(
            db_executor=self.db_executor,
            db_concurrency=max(1, db_concurrency or 1),
            enable_result_cache=resolved_result_cache,
            enable_prepared_statements=resolved_prepare,
            prepare_threshold=prepare_threshold,
            enable_db_explain_analyze=self._resolve_db_explain_flag(enable_db_explain_analyze),
            db_explain_mode=self._resolve_db_explain_mode(db_explain_mode),
            db_explain_sample_rate=self._resolve_db_explain_sample_rate(db_explain_sample_rate),
        )
        resolved_http_concurrency = max(1, int(http_concurrency or (db_concurrency or 1)))
        resolved_http_sleep = 0.0 if http_default_sleep_s is None else float(http_default_sleep_s)
        self.http_executor = HTTPNodeExecutor(
            http_concurrency=resolved_http_concurrency,
            default_sleep_s=resolved_http_sleep,
        )

        tp_enabled = bool(enable_tensor_parallel or (tensor_parallel_size is not None and tensor_parallel_size > 1))
        tp_size = tensor_parallel_size if tp_enabled else None
        if tp_enabled and (tp_size is None or tp_size <= 0):
            tp_size = self._detect_gpu_count()

        vllm_kwargs = dict(self.engine_kwargs)
        vllm_kwargs["allow_tensor_parallel"] = tp_enabled
        if tp_enabled and tp_size is not None:
            vllm_kwargs.setdefault("tensor_parallel_size", tp_size)

        self.engine_provider = make_vllm_provider(**vllm_kwargs)
        self.vllm_executor = VLLMNodeExecutor(engine_provider=self.engine_provider)
        self.processor_executor = ProcessorNodeExecutor()
        self.current_model: str | None = None

    def _resolve_db_explain_flag(self, override: bool | None) -> bool:
        if override is not None:
            return bool(override)
        value = (os.getenv("HALO_DB_EXPLAIN_ANALYZE") or "").strip().lower()
        return value not in ("", "0", "false", "no")

    def _resolve_db_explain_mode(self, override: str | None) -> str:
        if override:
            return override
        return (os.getenv("HALO_DB_EXPLAIN_MODE") or "wrap").strip()

    def _resolve_db_explain_sample_rate(self, override: float | None) -> float:
        if override is not None:
            return float(override)
        raw = (os.getenv("HALO_DB_EXPLAIN_SAMPLE_RATE") or "").strip()
        if not raw:
            return 1.0
        try:
            return float(raw)
        except ValueError:
            return 1.0

    @abc.abstractmethod
    def run(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        initial_inputs: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def run_batch(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        initial_inputs_list: Sequence[MutableMapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _compute_topo_order(self, plan: ExecutionPlan) -> List[str]:
        dependencies: Dict[str, Set[str]] = {task.node_id: set(task.dependencies) for task in plan.tasks}
        dependents: Dict[str, List[str]] = {}
        for task in plan.tasks:
            for dep in task.dependencies:
                dependents.setdefault(dep, []).append(task.node_id)

        plan_order: Dict[str, int] = {task.node_id: idx for idx, task in enumerate(plan.tasks)}
        ready: List[str] = sorted([nid for nid, deps in dependencies.items() if not deps], key=lambda nid: plan_order.get(nid, 0))
        order: List[str] = []

        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for child in dependents.get(node_id, []):
                deps = dependencies.get(child)
                if deps is None:
                    continue
                deps.discard(node_id)
                if not deps:
                    ready.append(child)
            ready.sort(key=lambda nid: plan_order.get(nid, 0))

        if len(order) != len(dependencies):
            missing = set(dependencies) - set(order)
            raise RuntimeError(f"Graph execution incomplete; missing or cyclic nodes: {missing}")
        return order

    def _detect_gpu_count(self) -> int:
        try:
            import torch

            count = torch.cuda.device_count()
            if count > 0:
                return count
        except Exception:
            pass
        return 1
        
    def _record_worker_metrics(self, stats: Mapping[str, Any] | None) -> None:
        if not stats:
            return
        db_calls = int(stats.get("db_calls") or 0)
        db_time = float(stats.get("db_time") or 0.0)
        if db_calls > 0 and db_time >= 0:
            metrics.record_query_execution_time(db_time, count=db_calls)
        api_calls = int(stats.get("api_calls") or stats.get("http_calls") or 0)
        api_time = float(stats.get("api_time") or stats.get("http_time") or 0.0)
        if api_calls > 0 and api_time >= 0:
            metrics.record_api_execution_time(api_time, count=api_calls)
        llm_calls = int(stats.get("llm_calls") or 0)
        llm_time = float(stats.get("llm_time") or 0.0)
        llm_prompts = int(stats.get("llm_prompts") or 0)
        if llm_calls > 0 and llm_time >= 0:
            metrics.record_llm_execution_time(
                llm_time,
                call_count=llm_calls,
                prompt_count=llm_prompts,
            )
        model_inits = int(stats.get("model_init_calls") or 0)
        model_init_time = float(stats.get("model_init_time") or 0.0)
        if model_inits > 0 and model_init_time >= 0:
            metrics.record_model_init_time(model_init_time, count=model_inits)

    def _record_node_metrics(
        self,
        node: Node,
        *,
        total_time: float,
        stats: Mapping[str, Any] | None,
        count: int,
    ) -> None:
        if count <= 0:
            return
        stats_map = stats or {}
        prepare_time = float(stats_map.get("model_init_time") or 0.0)
        execute_time = (
            float(stats_map.get("db_time") or 0.0)
            + float(stats_map.get("api_time") or stats_map.get("http_time") or 0.0)
            + float(stats_map.get("llm_time") or 0.0)
        )
        if execute_time <= 0:
            execute_time = max(0.0, total_time)
        metrics.record_node_metrics(
            node.id,
            total_time=total_time,
            prepare_time=prepare_time,
            execute_time=execute_time,
            count=count,
            engine=node.engine,
            model=node.model,
        )
