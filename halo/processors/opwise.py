from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Sequence

from .. import metrics
from ..models import ExecutionPlan, GraphSpec, Node
from ..monitoring import ProgressMonitor, start_progress_monitor, start_system_monitor
from ..worker import ResultMessage, TaskMessage, worker_process_loop
from .base import BaseGraphProcessor, count_progress_nodes, is_progress_node


@dataclass
class _DPWorkerPool:
    worker_ids: List[str]
    task_queues: Dict[str, Any]
    result_queue: "mp.Queue[ResultMessage]"
    processes: Dict[str, mp.Process]
    current_model: str | None = None


class OpwiseGraphProcessor(BaseGraphProcessor):
    """Operator-wise processor: single GPU/CPU worker, topo-ordered, no parallelism."""

    def __init__(
        self,
        *,
        max_batch_size: int | None = None,
        parallel_mode: str | None = None,
        parallel_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_batch_size = self._normalize_max_batch_size(max_batch_size)
        self.parallel_mode = (parallel_mode or "tensor").strip().lower()
        if self.parallel_mode not in ("tensor", "dp"):
            raise ValueError(f"Unsupported opwise parallel mode: {self.parallel_mode}")
        self.parallel_size = parallel_size

    def run(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        initial_inputs: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        results = self.run_batch(plan, graph, [initial_inputs])
        return results[0] if results else {}

    def run_batch(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        initial_inputs_list: Sequence[MutableMapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not initial_inputs_list:
            return []

        monitor = start_system_monitor(graph.name)
        progress_monitor = start_progress_monitor(
            graph.name,
            total_units=len(initial_inputs_list) * count_progress_nodes(plan, graph),
        )
        dp_pool: _DPWorkerPool | None = None
        try:
            if self.parallel_mode == "dp":
                dp_pool = self._start_dp_workers()
            topo_order = self._compute_topo_order(plan)
            contexts: List[Dict[str, Any]] = [dict(inputs) for inputs in initial_inputs_list]

            for node_id in topo_order:
                node = graph.nodes.get(node_id)
                if node is None:
                    raise RuntimeError(f"Node '{node_id}' not found in graph.")

                start_time = time.perf_counter()
                if node.type == "input":
                    outputs_list = []
                    for ctx in contexts:
                        outputs = {}
                        for name in node.outputs:
                            if name not in ctx:
                                raise RuntimeError(
                                    f"Input node '{node.id}' missing initial value for output '{name}'."
                                )
                            outputs[name] = ctx[name]
                        outputs_list.append(outputs)
                    stats = {}
                elif node.type == "processor":
                    outputs_list = self.processor_executor.execute_batch(node, contexts)
                    stats = self.processor_executor.consume_stats()
                elif node.engine == "vllm":
                    print(f"Executing VLLM node {node.id} with batch size {len(contexts)}")
                    node_model = node.model or ""
                    if dp_pool is not None:
                        self._configure_dp_model(dp_pool, node_model)
                        outputs_list, stats = self._execute_vllm_data_parallel(
                            node,
                            contexts,
                            dp_pool,
                            progress_monitor=progress_monitor,
                        )
                    else:
                        if node_model and self.current_model != node_model:
                            # Switch model: clear cache and update tracker.
                            # The engine provider will lazy-load the new model on next use.
                            self.engine_provider.clear_cache()
                            self.current_model = node_model
                        outputs_list, stats = self._execute_vllm_with_limit(
                            node,
                            contexts,
                            progress_monitor=progress_monitor,
                        )
                elif node.engine == "db":
                    print(f"Executing DB node {node.id} with batch size {len(contexts)}")
                    outputs_list = self.db_node_executor.execute_batch(node, contexts)
                    stats = self.db_node_executor.consume_stats()
                elif node.engine == "http":
                    print(f"Executing HTTP node {node.id} with batch size {len(contexts)}")
                    outputs_list = self.http_executor.execute_batch(node, contexts)
                    stats = self.http_executor.consume_stats()
                elif node.engine == "noop":
                    outputs_list = [{name: ctx.get(name) for name in node.outputs} for ctx in contexts]
                    stats = {}
                else:
                    raise RuntimeError(f"Unsupported node: engine={node.engine}, type={node.type}")
                total_time = time.perf_counter() - start_time

                self._record_worker_metrics(stats)
                self._record_node_metrics(node, total_time=total_time, stats=stats, count=len(contexts))
                if progress_monitor and is_progress_node(node) and node.engine != "vllm":
                    progress_monitor.record(len(contexts))

                for i, outputs in enumerate(outputs_list):
                    if outputs:
                        contexts[i].update(outputs)

            return contexts
        finally:
            if dp_pool is not None:
                self._stop_dp_workers(dp_pool)
            if progress_monitor:
                progress_monitor.stop()
            if monitor:
                monitor.stop()

    def _normalize_max_batch_size(self, max_batch_size: int | None) -> int | None:
        if max_batch_size is None:
            return None
        try:
            value = int(max_batch_size)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _execute_vllm_with_limit(
        self,
        node: Node,
        contexts: Sequence[MutableMapping[str, Any]],
        *,
        progress_monitor: ProgressMonitor | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.max_batch_size is None or len(contexts) <= self.max_batch_size:
            outputs = self.vllm_executor.execute_batch(node, contexts)
            stats = self.vllm_executor.consume_stats()
            if progress_monitor:
                progress_monitor.record(len(contexts))
            return outputs, stats

        aggregated_outputs: List[Dict[str, Any]] = []
        aggregated_stats: Dict[str, Any] = {}
        for start in range(0, len(contexts), self.max_batch_size):
            # Limit each vLLM invocation to keep memory usage predictable.
            chunk_contexts = contexts[start : start + self.max_batch_size]
            chunk_outputs = self.vllm_executor.execute_batch(node, chunk_contexts)
            aggregated_outputs.extend(chunk_outputs)
            chunk_stats = self.vllm_executor.consume_stats()
            self._merge_stats(aggregated_stats, chunk_stats)
            if progress_monitor:
                progress_monitor.record(len(chunk_contexts))
        return aggregated_outputs, aggregated_stats

    def _merge_stats(self, total: Dict[str, Any], delta: Dict[str, Any]) -> None:
        if not delta:
            return
        for key, value in delta.items():
            if not isinstance(value, (int, float)):
                continue
            existing = total.get(key, 0)
            if not isinstance(existing, (int, float)):
                existing = 0
            total[key] = existing + value

    def _start_dp_workers(self) -> _DPWorkerPool:
        devices = self._resolve_dp_devices(self.parallel_size)
        if not devices:
            raise RuntimeError("No GPU devices available for opwise data-parallel mode.")
        result_queue: "mp.Queue[ResultMessage]" = mp.Queue()
        task_queues: Dict[str, Any] = {}
        processes: Dict[str, mp.Process] = {}
        engine_kwargs = dict(self.engine_kwargs)
        engine_kwargs["allow_tensor_parallel"] = False
        engine_kwargs.pop("tensor_parallel_size", None)

        for idx, device in enumerate(devices):
            worker_id = f"opwise-gpu-{idx}"
            task_q: Any = mp.SimpleQueue()
            proc = mp.Process(
                target=worker_process_loop,
                args=(
                    worker_id,
                    device,
                    task_q,
                    result_queue,
                    engine_kwargs,
                    {},
                ),
            )
            proc.start()
            task_queues[worker_id] = task_q
            processes[worker_id] = proc

        return _DPWorkerPool(
            worker_ids=list(task_queues.keys()),
            task_queues=task_queues,
            result_queue=result_queue,
            processes=processes,
        )

    def _stop_dp_workers(self, pool: _DPWorkerPool) -> None:
        for worker_id, q in pool.task_queues.items():
            try:
                q.put(TaskMessage(node_id="__STOP__", node=None, is_stop=True))
            except Exception:
                pass
        for worker_id, proc in pool.processes.items():
            try:
                proc.join(timeout=5.0)
            except Exception:
                pass

    def _configure_dp_model(self, pool: _DPWorkerPool, model: str | None) -> None:
        model_name = model or ""
        if model_name and pool.current_model == model_name:
            return
        for worker_id in pool.worker_ids:
            pool.task_queues[worker_id].put(
                TaskMessage(
                    node_id="__CONFIG__",
                    node=None,
                    config={"epoch": 0, "model": model_name},
                )
            )
        pool.current_model = model_name

    def _execute_vllm_data_parallel(
        self,
        node: Node,
        contexts: Sequence[MutableMapping[str, Any]],
        pool: _DPWorkerPool,
        *,
        progress_monitor: ProgressMonitor | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.max_batch_size is None or len(contexts) <= self.max_batch_size:
            return self._execute_vllm_data_parallel_once(
                node,
                contexts,
                pool,
                progress_monitor=progress_monitor,
            )

        aggregated_outputs: List[Dict[str, Any]] = []
        aggregated_stats: Dict[str, Any] = {}
        for start in range(0, len(contexts), self.max_batch_size):
            chunk = contexts[start : start + self.max_batch_size]
            chunk_outputs, chunk_stats = self._execute_vllm_data_parallel_once(
                node,
                chunk,
                pool,
                progress_monitor=progress_monitor,
            )
            aggregated_outputs.extend(chunk_outputs)
            self._merge_stats(aggregated_stats, chunk_stats)
        return aggregated_outputs, aggregated_stats

    def _execute_vllm_data_parallel_once(
        self,
        node: Node,
        contexts: Sequence[MutableMapping[str, Any]],
        pool: _DPWorkerPool,
        *,
        progress_monitor: ProgressMonitor | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        worker_ids = pool.worker_ids
        if not worker_ids:
            outputs = self.vllm_executor.execute_batch(node, contexts)
            stats = self.vllm_executor.consume_stats()
            if progress_monitor:
                progress_monitor.record(len(contexts))
            return outputs, stats

        shards: Dict[str, List[int]] = {wid: [] for wid in worker_ids}
        for idx in range(len(contexts)):
            wid = worker_ids[idx % len(worker_ids)]
            shards[wid].append(idx)

        dispatched = 0
        for wid, idxs in shards.items():
            if not idxs:
                continue
            batch = [contexts[i] for i in idxs]
            pool.task_queues[wid].put(
                TaskMessage(
                    node_id=node.id,
                    node=node,
                    context_slices=batch,
                    context_indices=idxs,
                )
            )
            dispatched += 1

        outputs_list: List[Dict[str, Any] | None] = [None] * len(contexts)
        aggregated_stats: Dict[str, Any] = {}
        for _ in range(dispatched):
            result = pool.result_queue.get()
            if result.error is not None:
                raise RuntimeError(f"Node {node.id} failed in DP worker: {result.error}")
            if result.stats:
                busy_time = float(result.stats.get("node_total_time") or 0.0)
                if busy_time >= 0:
                    metrics.record_worker_busy_time(result.worker_id, busy_time, kind="gpu")
                self._merge_stats(aggregated_stats, result.stats)
            for idx, output in zip(result.context_indices, result.outputs or []):
                outputs_list[idx] = output
            if progress_monitor:
                progress_monitor.record(len(result.context_indices))

        final_outputs = [output or {} for output in outputs_list]
        return final_outputs, aggregated_stats

    def _resolve_dp_devices(self, parallel_size: int | None) -> List[str]:
        visible_ids = self._parse_visible_gpu_ids()
        if not visible_ids:
            count = self._detect_gpu_count()
            visible_ids = list(range(count))
        if not visible_ids:
            return []
        size = None
        if parallel_size is not None and parallel_size > 0:
            size = parallel_size
        if size is None:
            size = len(visible_ids)
        size = max(1, min(size, len(visible_ids)))
        return [f"cuda:{idx}" for idx in visible_ids[:size]]

    def _parse_visible_gpu_ids(self) -> List[int]:
        raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if not raw:
            return []
        ids: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                return []
        return ids
