from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from collections import defaultdict, deque
import queue
import heapq
from typing import Any, Callable, Deque, Dict, List, Mapping, MutableMapping, Sequence, Set

from .. import metrics
from ..db import (
    DatabaseExecutor,
    DefaultDatabaseExecutor,
    make_peer_postgres_executor,
)
from ..models import ExecutionPlan, ExecutionTask, GraphSpec
from ..monitoring import ProgressMonitor, start_progress_monitor, start_system_monitor
from ..worker import (
    ResultMessage,
    TaskMessage,
    cpu_worker_loop,
    worker_process_loop,
)
from .base import count_progress_nodes, is_progress_node

LOGGER = logging.getLogger(__name__)
_RED = "\033[31m"
_RESET = "\033[0m"

_BRIDGE_STOP = object()


def _gpu_result_bridge(
    src: "mp.Queue[ResultMessage]",
    dst: "queue.SimpleQueue[ResultMessage]",
) -> None:
    """Forward GPU-worker results from ``mp.Queue`` to the in-process queue."""
    while True:
        try:
            msg = src.get()
        except (EOFError, OSError):
            return
        if msg is _BRIDGE_STOP:
            return
        dst.put(msg)


class MultiProcessGraphProcessor:
    """多进程版本的 Graph Processor。

    - 主进程负责：
        * 维护全局 context
        * 跟踪依赖完成情况 (dependencies / dependents)
        * 按 ExecutionPlan.worker_id 将任务派发到各个 worker 进程
    - 每个 worker 进程负责：
        * GPU worker 持有各自的 EngineProvider/vLLM cache
        * CPU worker 持有 DB 执行器
    """

    def __init__(
        self,
        engine_kwargs: Dict[str, Any] | None = None,
        db_connect_kwargs: Dict[str, Any] | None = None,
        db_pool_size: int | None = None,
        max_batch_size: int | None = None,
        executor_kwargs: Dict[str, Any] | None = None,
        debug_log: bool = False,
        debug_every: int = 50,
        persistent_workers: bool = False,
        enforce_epoch_barrier: bool = False,
    ):
        self.engine_kwargs = engine_kwargs or {}
        if max_batch_size is None:
            self.max_batch_size = 32
        else:
            self.max_batch_size = int(max_batch_size)
            if self.max_batch_size <= 0:
                self.max_batch_size = None
        self.executor_kwargs = executor_kwargs or {}
        self.executor_kwargs.setdefault("db_concurrency", 32)
        # Size the PG connection pool to at least ``db_concurrency`` so the
        # ThreadPoolExecutor inside DBNodeExecutor doesn't fall back to
        # creating one-off connections (TCP + auth handshake per call).
        if db_connect_kwargs:
            effective_pool = db_pool_size
            db_conc = int(self.executor_kwargs.get("db_concurrency") or 1)
            if effective_pool is None or effective_pool < db_conc:
                effective_pool = max(db_conc, effective_pool or 0)
            self.db_executor_factory = lambda pool=effective_pool, kw=db_connect_kwargs: (
                make_peer_postgres_executor(**kw, pool_size=pool)
            )
        else:
            self.db_executor_factory = DefaultDatabaseExecutor
        if "enable_db_explain_analyze" not in self.executor_kwargs:
            flag = (os.getenv("HALO_DB_EXPLAIN_ANALYZE") or "").strip().lower()
            self.executor_kwargs["enable_db_explain_analyze"] = flag not in ("", "0", "false", "no")
        self.executor_kwargs.setdefault("db_explain_mode", (os.getenv("HALO_DB_EXPLAIN_MODE") or "wrap").strip())
        if "db_explain_sample_rate" not in self.executor_kwargs:
            raw = (os.getenv("HALO_DB_EXPLAIN_SAMPLE_RATE") or "").strip()
            if raw:
                try:
                    self.executor_kwargs["db_explain_sample_rate"] = float(raw)
                except ValueError:
                    self.executor_kwargs["db_explain_sample_rate"] = 1.0
            else:
                self.executor_kwargs["db_explain_sample_rate"] = 1.0
        self.debug_log = bool(debug_log)
        self.debug_every = max(1, int(debug_every))
        self._persistent_workers = bool(persistent_workers)
        self._worker_state: tuple[Dict[str, Any], Dict[str, Any], "mp.Queue[ResultMessage]", "mp.Queue[ResultMessage]"] | None = None
        self._worker_plan_id: int | None = None
        self._worker_lock = threading.Lock()
        self.enforce_epoch_barrier = bool(enforce_epoch_barrier)

    # ---- Public API ---------------------------------------------------------

    def run(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        initial_inputs: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        """执行给定的 ExecutionPlan，并返回最终 context。"""
        final_contexts = self.run_batch(plan, graph, [initial_inputs])
        return final_contexts[0] if final_contexts else {}

    def run_batch(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        initial_inputs_list: Sequence[MutableMapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """批量执行 ExecutionPlan，返回每个 query 的最终 context。"""
        if not initial_inputs_list:
            return []

        contexts: List[Dict[str, Any]] = [dict(inputs) for inputs in initial_inputs_list]
        worker_state: tuple[Dict[str, Any], Dict[str, Any], "queue.SimpleQueue[ResultMessage]", "mp.Queue[ResultMessage]"]
        if self._persistent_workers:
            with self._worker_lock:
                if self._worker_state is None or self._worker_plan_id != id(plan):
                    if self._worker_state is not None:
                        workers, task_queues, _, gpu_q = self._worker_state
                        self._stop_workers(workers, task_queues, gpu_q)
                    self._worker_state = self._start_workers(plan)
                    self._worker_plan_id = id(plan)
                worker_state = self._worker_state
        else:
            worker_state = self._start_workers(plan)

        workers, task_queues, result_queue, gpu_result_queue = worker_state
        monitor = start_system_monitor(graph.name)
        progress_monitor = start_progress_monitor(
            graph.name,
            total_units=len(contexts) * count_progress_nodes(plan, graph),
        )
        try:
            self._execute_plan(
                plan,
                graph,
                contexts,
                task_queues,
                result_queue,
                progress_monitor=progress_monitor,
            )
        finally:
            if progress_monitor:
                progress_monitor.stop()
            if monitor:
                monitor.stop()
            if not self._persistent_workers:
                self._stop_workers(workers, task_queues, gpu_result_queue)

        return contexts

    def close(self) -> None:
        """Stop persistent workers and release resources."""
        with self._worker_lock:
            if self._worker_state is None:
                return
            workers, task_queues, _, gpu_q = self._worker_state
            self._stop_workers(workers, task_queues, gpu_q)
            self._worker_state = None
            self._worker_plan_id = None

    # ---- Worker lifecycle management ---------------------------------------

    def _start_workers(
        self,
        plan: ExecutionPlan,
    ) -> tuple[Dict[str, Any], Dict[str, Any], "queue.SimpleQueue[ResultMessage]", "mp.Queue[ResultMessage]"]:
        """Start workers and return (handles, task_queues, unified_result_q, gpu_result_q).

        The unified result queue is an in-process ``queue.SimpleQueue``. CPU worker
        threads write to it directly (no pickle). A small bridge thread drains the
        ``mp.Queue`` that GPU worker processes write to, forwarding each message
        into the unified queue. The scheduler can therefore wait on a single
        in-process queue without polling timeouts.
        """
        worker_handles: Dict[str, Any] = {}
        task_queues: Dict[str, Any] = {}
        unified_result_queue: "queue.SimpleQueue[ResultMessage]" = queue.SimpleQueue()
        gpu_result_queue: "mp.Queue[ResultMessage]" = mp.Queue()
        cpu_exec_kwargs = {
            k: v
            for k, v in self.executor_kwargs.items()
            if k in (
                "db_concurrency",
                "enable_result_cache",
                "enable_prepared_statements",
                "prepare_threshold",
                "enable_db_explain_analyze",
                "db_explain_mode",
                "db_explain_sample_rate",
            )
        }

        has_gpu = any(w.kind == "gpu" for w in plan.workers.values())

        for worker_id, worker in plan.workers.items():
            if worker.kind == "gpu":
                task_q: Any = mp.SimpleQueue()
            else:
                task_q = queue.SimpleQueue()
            if worker.kind == "gpu":
                handle = mp.Process(
                    target=worker_process_loop,
                    args=(
                        worker_id,
                        worker.device,
                        task_q,
                        gpu_result_queue,
                        self.engine_kwargs,
                        {},
                    ),
                )
                handle.start()
                LOGGER.info("Started GPU worker %s (device=%s, pid=%s)", worker_id, worker.device, handle.pid)
            else:
                handle = threading.Thread(
                    target=cpu_worker_loop,
                    args=(worker_id, task_q, unified_result_queue, self.db_executor_factory, cpu_exec_kwargs),
                    daemon=True,
                )
                handle.start()
                LOGGER.info("Started CPU worker %s (thread)", worker_id)

            worker_handles[worker_id] = handle
            task_queues[worker_id] = task_q

        if has_gpu:
            bridge = threading.Thread(
                target=_gpu_result_bridge,
                args=(gpu_result_queue, unified_result_queue),
                daemon=True,
                name="halo-gpu-bridge",
            )
            bridge.start()
            worker_handles["__bridge__"] = bridge

        return worker_handles, task_queues, unified_result_queue, gpu_result_queue

    def _stop_workers(
        self,
        workers: Dict[str, Any],
        task_queues: Dict[str, Any],
        gpu_result_queue: "mp.Queue[ResultMessage] | None" = None,
    ) -> None:
        for worker_id, q in task_queues.items():
            try:
                q.put(TaskMessage(node_id="__STOP__", node=None, is_stop=True))
            except Exception:
                LOGGER.warning("Failed to send stop signal to worker %s", worker_id)

        bridge = workers.pop("__bridge__", None)
        for worker_id, handle in workers.items():
            try:
                if isinstance(handle, mp.Process):
                    handle.join(timeout=5.0)
                    if handle.is_alive():
                        LOGGER.warning("Worker %s (pid=%s) did not exit in time", worker_id, handle.pid)
                else:
                    handle.join(timeout=5.0)
            except Exception:
                LOGGER.exception("Failed to join worker %s", worker_id)
        if bridge is not None and gpu_result_queue is not None:
            try:
                gpu_result_queue.put(_BRIDGE_STOP)
            except Exception:
                pass
            try:
                bridge.join(timeout=5.0)
            except Exception:
                pass

    # ---- DAG scheduling logic ----------------------------------------------

    def _build_dependency_graph(
        self,
        plan: ExecutionPlan,
    ) -> tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
        """从 ExecutionPlan 构建 dependencies / dependents 映射。"""
        dependencies: Dict[str, Set[str]] = {
            t.node_id: set(t.dependencies) for t in plan.tasks
        }
        dependents: Dict[str, List[str]] = {}
        for t in plan.tasks:
            for dep in t.dependencies:
                dependents.setdefault(dep, []).append(t.node_id)
        return dependencies, dependents

    def _execute_plan(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        contexts: List[Dict[str, Any]],
        task_queues: Dict[str, Any],
        result_queue: "queue.SimpleQueue[ResultMessage]",
        *,
        progress_monitor: ProgressMonitor | None = None,
    ) -> None:
        if not contexts:
            return

        dependencies, dependents = self._build_dependency_graph(plan)
        # 全局依赖驱动调度
        task_map: Dict[str, ExecutionTask] = {task.node_id: task for task in plan.tasks}
        # 保留优化器给出的原始顺序，用作同 epoch 的稳定 tie-breaker。
        plan_order: Dict[str, int] = {task.node_id: idx for idx, task in enumerate(plan.tasks)}
        worker_by_id = plan.workers
        node_worker: Dict[str, Any] = {task.node_id: task.worker_id for task in plan.tasks}
        node_specs = graph.nodes
        all_nodes = set(task_map.keys())
        batch_size = len(contexts)
        total_work = len(plan.tasks) * batch_size
        db_total_units = 0
        llm_total_units = 0
        for task in plan.tasks:
            node = node_specs.get(task.node_id)
            if node and node.engine == "vllm":
                llm_total_units += batch_size
            else:
                db_total_units += batch_size
        db_done_units = 0
        llm_done_units = 0
        completed_units = 0

        pending_dependencies: List[Dict[str, Set[str]]] = [
            {node_id: set(dependencies.get(node_id, [])) for node_id in all_nodes} for _ in range(batch_size)
        ]
        ready: Dict[str, Deque[int]] = {node_id: deque() for node_id in all_nodes}
        # Track which node ids currently hold a non-empty ready queue so the
        # outer loop doesn't have to scan task_map every iteration.
        ready_nonempty: Set[str] = set()
        for node_id, deps in dependencies.items():
            if not deps:
                for idx in range(batch_size):
                    ready[node_id].append(idx)
                if ready[node_id]:
                    ready_nonempty.add(node_id)

        completed: List[Set[str]] = [set() for _ in range(batch_size)]
        remaining_counts: Dict[str, int] = {node_id: batch_size for node_id in all_nodes}
        in_flight: Dict[str, Set[int]] = defaultdict(set)
        last_model_by_worker: Dict[str, str | None] = {
            wid: None for wid, w in plan.workers.items() if w.kind == "gpu"
        }
        worker_lock_model: Dict[str, str | None] = {
            wid: None for wid, w in plan.workers.items() if w.kind == "gpu"
        }
        worker_inflight_counts: Dict[str, int] = defaultdict(int)
        # Track which worker is processing which (node_id, context_idx) so we can release locks per model.
        inflight_worker: Dict[tuple[str, int], str] = {}
        started_nodes: Set[str] = set()
        finished_nodes: Set[str] = set()
        gpu_preload_plan: Dict[str, List[str]] = {}
        for task in plan.tasks:
            node = node_specs.get(task.node_id)
            if not node or node.engine != "vllm":
                continue
            assignment = task.worker_id
            wid_seq = tuple(assignment) if isinstance(assignment, (list, tuple)) else (assignment,)
            for wid in wid_seq:
                if wid is None:
                    continue
                gpu_preload_plan.setdefault(wid, []).append(task.node_id)
        for wid, nodes in gpu_preload_plan.items():
            nodes.sort(key=lambda nid: plan_order.get(nid, 0))
        gpu_preload_cursor: Dict[str, int] = {wid: 0 for wid in gpu_preload_plan}

        def log_node_start(node_id: str) -> None:
            if not self.debug_log:
                return
            if node_id in started_nodes:
                return
            started_nodes.add(node_id)
            node = node_specs.get(node_id)
            kind = "llm" if node and node.engine == "vllm" else "db"
            LOGGER.info(
                "%s[Processor] start node=%s [%s] progress db=%d/%d llm=%d/%d total=%d/%d%s",
                _RED,
                node_id,
                kind,
                db_done_units,
                db_total_units,
                llm_done_units,
                llm_total_units,
                db_done_units + llm_done_units,
                total_work,
                _RESET,
            )

        def log_node_done(node_id: str) -> None:
            if not self.debug_log:
                return
            if node_id in finished_nodes:
                return
            finished_nodes.add(node_id)
            node = node_specs.get(node_id)
            kind = "llm" if node and node.engine == "vllm" else "db"
            LOGGER.info(
                "%s[Processor] done node=%s [%s] progress db=%d/%d llm=%d/%d total=%d/%d%s",
                _RED,
                node_id,
                kind,
                db_done_units,
                db_total_units,
                llm_done_units,
                llm_total_units,
                db_done_units + llm_done_units,
                total_work,
                _RESET,
            )

        # Precompute (worker_id, model) -> list of vLLM node_ids. The check
        # ``worker_has_ready_same_model`` is on the hot dispatch path, so doing
        # this lookup by index is O(matching vLLM nodes) instead of O(all
        # ready nodes) every call.
        vllm_nodes_by_wm: Dict[tuple[str, str], List[str]] = defaultdict(list)
        for nid, node in node_specs.items():
            if not node or node.engine != "vllm":
                continue
            assignment = node_worker.get(nid)
            wid_seq = (
                tuple(assignment) if isinstance(assignment, (list, tuple)) else (assignment,)
            )
            model = node.model or ""
            for wid in wid_seq:
                if wid is None:
                    continue
                vllm_nodes_by_wm[(wid, model)].append(nid)

        def worker_has_ready_same_model(model_name: str | None, worker_id: str) -> bool:
            if not model_name:
                return False
            for nid in vllm_nodes_by_wm.get((worker_id, model_name), ()):
                if ready.get(nid):
                    return True
            return False

        while task_map:
            min_epoch_incomplete = None
            if self.enforce_epoch_barrier and task_map:
                min_epoch_incomplete = min(task.epoch for task in task_map.values())
            # Drain any node ids that no longer have a pending ready batch
            # (they may have been emptied since the last loop iteration).
            ready_nodes = [nid for nid in ready_nonempty if nid in task_map and ready.get(nid)]
            ready_nonempty.intersection_update(ready_nodes)
            if min_epoch_incomplete is not None:
                ready_nodes = [nid for nid in ready_nodes if task_map[nid].epoch == min_epoch_incomplete]
            if ready_nodes:
                ready_heap = [(task_map[nid].epoch, plan_order.get(nid, 0), nid) for nid in ready_nodes]
                heapq.heapify(ready_heap)
                min_ready_epoch = ready_heap[0][0]
            else:
                ready_heap = []
                min_ready_epoch = None

            # Idle GPU workers pre-load the next planned LLM model after finishing the current one.
            for wid, worker in worker_by_id.items():
                if worker.kind != "gpu":
                    continue
                if worker_inflight_counts.get(wid, 0) > 0:
                    continue
                if worker_lock_model.get(wid):
                    continue
                plan_nodes = gpu_preload_plan.get(wid, [])
                if not plan_nodes:
                    continue
                cursor = gpu_preload_cursor.get(wid, 0)
                while cursor < len(plan_nodes) and plan_nodes[cursor] not in task_map:
                    cursor += 1
                gpu_preload_cursor[wid] = cursor
                if cursor >= len(plan_nodes):
                    continue
                preload_node = plan_nodes[cursor]
                if min_epoch_incomplete is not None and task_map[preload_node].epoch != min_epoch_incomplete:
                    continue
                desired_model = (node_specs[preload_node].model or "") if node_specs.get(preload_node) else ""
                curr_model = last_model_by_worker.get(wid)
                if not desired_model or curr_model == desired_model:
                    continue
                cfg_msg = TaskMessage(
                    node_id="__CONFIG__",
                    node=None,
                    config={"epoch": task_map[preload_node].epoch, "model": desired_model},
                )
                task_queues[wid].put(cfg_msg)
                last_model_by_worker[wid] = desired_model

            dispatched = False

            while ready_heap:
                _, _, node_id = heapq.heappop(ready_heap)
                ready_queue = ready.get(node_id)
                if not ready_queue:
                    continue
                if min_epoch_incomplete is not None and task_map[node_id].epoch != min_epoch_incomplete:
                    continue

                node = node_specs[node_id]
                worker_assignment = node_worker.get(node_id)
                worker_ids_seq = (
                    tuple(worker_assignment)
                    if isinstance(worker_assignment, (list, tuple))
                    else (worker_assignment,)
                )
                assigned_worker_id = worker_ids_seq[0] if worker_ids_seq else None
                worker = worker_by_id.get(assigned_worker_id) if assigned_worker_id else None
                worker_idle = bool(
                    assigned_worker_id and worker_inflight_counts.get(assigned_worker_id, 0) <= 0
                )
                # 鼓励低 epoch 先跑：如果当前节点 epoch 大于已就绪的最小 epoch，且 CPU worker 闲置，则跳过
                if (
                    min_ready_epoch is not None
                    and task_map[node_id].epoch > min_ready_epoch
                    and worker
                    and worker.kind != "gpu"
                    and worker_idle
                ):
                    continue
                batch_indices: List[int] = []
                while ready_queue and (
                    self.max_batch_size is None or len(batch_indices) < self.max_batch_size
                ):
                    idx = ready_queue.popleft()
                    if idx in in_flight[node_id] or node_id in completed[idx]:
                        continue
                    batch_indices.append(idx)
                if not ready_queue:
                    ready_nonempty.discard(node_id)

                if not batch_indices:
                    continue

                if node.type == "input":
                    start_time = time.perf_counter()
                    log_node_start(node_id)
                    for idx in batch_indices:
                        # Input nodes just validate and passthrough existing context entries.
                        outputs: Dict[str, Any] = {}
                        for name in node.outputs:
                            if name not in contexts[idx]:
                                raise RuntimeError(
                                    f"Input node '{node.id}' missing initial value for output '{name}'."
                                )
                            outputs[name] = contexts[idx][name]
                        contexts[idx].update(outputs)
                        completed[idx].add(node_id)
                        remaining_counts[node_id] -= 1
                        completed_units += 1
                        self._unlock_epoch_dependents(
                            finished_node_id=node_id,
                            context_idx=idx,
                            epoch_dependents=dependents,
                            pending_dependencies=pending_dependencies,
                            completed=completed,
                            ready=ready,
                            ready_nonempty=ready_nonempty,
                        )
                    if remaining_counts[node_id] <= 0:
                        log_node_done(node_id)
                        task_map.pop(node_id, None)
                        ready.pop(node_id, None)
                        remaining_counts.pop(node_id, None)
                    total_time = time.perf_counter() - start_time
                    self._record_node_metrics(
                        node_id=node_id,
                        node=node,
                        stats=None,
                        count=len(batch_indices),
                        total_time=total_time,
                    )
                    dispatched = True
                    continue

                task = task_map[node_id]
                def slice_context(idx: int) -> Dict[str, Any]:
                    ctx = contexts[idx]
                    needed = set(node.inputs)
                    # Keep search_keyword if present even if not declared input (defensive for mappings)
                    if "search_keyword" in ctx:
                        needed.add("search_keyword")
                    return {key: ctx.get(key) for key in needed if key in ctx}

                context_batch = [slice_context(idx) for idx in batch_indices]

                # GPU worker 模型级串行：同一 GPU 上不同模型需等待在飞批次完成后再切换
                desired_model = node.model or "" if node.engine == "vllm" else None
                if worker_ids_seq and all(worker_by_id.get(wid) and worker_by_id[wid].kind == "gpu" for wid in worker_ids_seq):
                    model_conflict = False
                    for wid in worker_ids_seq:
                        locked_model = worker_lock_model.get(wid)
                        if locked_model is None:
                            continue
                        if desired_model == locked_model:
                            continue
                        if worker_inflight_counts[wid] > 0:
                            model_conflict = True
                            break
                        if worker_has_ready_same_model(locked_model, wid):
                            model_conflict = True
                            break
                    if model_conflict:
                        # Restore contexts back to the ready queue so they are not dropped.
                        for idx in reversed(batch_indices):
                            ready_queue.appendleft(idx)
                        if ready_queue:
                            ready_nonempty.add(node_id)
                        continue

                if node.engine == "vllm":
                    for wid in worker_ids_seq:
                        curr_model = last_model_by_worker.get(wid)
                        if curr_model != desired_model:
                            cfg_msg = TaskMessage(
                                node_id="__CONFIG__",
                                node=None,
                                config={"epoch": task.epoch, "model": desired_model},
                            )
                            task_queues[wid].put(cfg_msg)
                            last_model_by_worker[wid] = desired_model

                if len(worker_ids_seq) > 1 and node.engine == "vllm":
                    shards: Dict[str, List[int]] = {wid: [] for wid in worker_ids_seq}
                    # Load-aware sharding: prefer GPUs with fewer in-flight batches (capacity-adjusted).
                    load_est: Dict[str, float] = {}
                    for wid in worker_ids_seq:
                        worker = worker_by_id.get(wid)
                        capacity = worker.capacity if worker and worker.capacity else 1.0
                        capacity = capacity if capacity > 0 else 1.0
                        load_est[wid] = worker_inflight_counts.get(wid, 0) / capacity
                    for idx in batch_indices:
                        wid = min(worker_ids_seq, key=lambda w: load_est.get(w, 0.0))
                        shards[wid].append(idx)
                        load_est[wid] += 1.0  # optimistic reservation to balance within this dispatch
                    for wid, idxs in shards.items():
                        if not idxs:
                            continue
                        batch = [slice_context(i) for i in idxs]
                        msg = TaskMessage(
                            node_id=node_id,
                            node=node,
                            context_slices=batch,
                            context_indices=idxs,
                        )
                        task_queues[wid].put(msg)
                        log_node_start(node_id)
                    for wid in worker_ids_seq:
                        idxs = shards.get(wid, [])
                        if not idxs:
                            continue
                        if worker_by_id.get(wid) and worker_by_id[wid].kind == "gpu":
                            worker_lock_model[wid] = desired_model
                            worker_inflight_counts[wid] += len(idxs)
                            for ctx_idx in idxs:
                                inflight_worker[(node_id, ctx_idx)] = wid
                    for idx in batch_indices:
                        in_flight[node_id].add(idx)
                    dispatched = True
                else:
                    wid = worker_ids_seq[0]
                    msg = TaskMessage(
                        node_id=node_id,
                        node=node,
                        context_slices=context_batch,
                        context_indices=batch_indices,
                    )
                    task_queues[wid].put(msg)
                    log_node_start(node_id)
                    worker_inflight_counts[wid] += len(batch_indices)
                    if worker and worker.kind == "gpu":
                        worker_lock_model[wid] = desired_model
                        for idx in batch_indices:
                            inflight_worker[(node_id, idx)] = wid
                    for idx in batch_indices:
                        in_flight[node_id].add(idx)
                        if worker is None or worker.kind != "gpu":
                            inflight_worker[(node_id, idx)] = wid
                    dispatched = True

            if dispatched:
                continue

            if not any(in_flight.values()):
                # Try to rebuild ready queues in case of lost wakeups.
                requeued = 0
                for ctx_idx in range(batch_size):
                    for nid, deps in pending_dependencies[ctx_idx].items():
                        if deps:
                            continue
                        if nid in completed[ctx_idx]:
                            continue
                        if ctx_idx in in_flight[nid]:
                            continue
                        ready.setdefault(nid, deque()).append(ctx_idx)
                        ready_nonempty.add(nid)
                        requeued += 1
                if requeued:
                    LOGGER.warning(
                        "%s[Processor] rebuilt ready queues (%d contexts) progress db=%d/%d llm=%d/%d total=%d/%d%s",
                        _RED,
                        requeued,
                        db_done_units,
                        db_total_units,
                        llm_done_units,
                        llm_total_units,
                        db_done_units + llm_done_units,
                        total_work,
                        _RESET,
                    )
                    continue

                if self.debug_log:
                    stuck_nodes = sorted(nid for nid, cnt in remaining_counts.items() if cnt > 0)
                    pending_ctx0 = {
                        nid: sorted(pending_dependencies[0].get(nid, set()))
                        for nid in stuck_nodes
                    } if pending_dependencies else {}
                    LOGGER.error(
                        "%s[Processor] stalled: remaining_nodes=%s pending_deps_ctx0=%s "
                        "progress db=%d/%d llm=%d/%d total=%d/%d%s",
                        _RED,
                        stuck_nodes,
                        pending_ctx0,
                        db_done_units,
                        db_total_units,
                        llm_done_units,
                        llm_total_units,
                        db_done_units + llm_done_units,
                        total_work,
                        _RESET,
                    )
                break

            try:
                # Unified in-process queue: CPU threads put directly, a bridge
                # thread forwards GPU mp.Queue messages here. Block until a
                # result arrives — no polling timeout.
                result = result_queue.get()
            except queue.Empty:
                continue
            self._record_worker_metrics(result.stats)
            node_id = result.node_id
            worker_id = result.worker_id
            if worker_id is None and result.context_indices:
                worker_id = inflight_worker.get((node_id, result.context_indices[0]))
            busy_time = float((result.stats or {}).get("node_total_time") or 0.0)
            if worker_id and busy_time >= 0:
                worker = worker_by_id.get(worker_id)
                worker_kind = worker.kind if worker else None
                metrics.record_worker_busy_time(worker_id, busy_time, kind=worker_kind)
            if node_id not in task_map:
                continue

            for idx in result.context_indices:
                in_flight[node_id].discard(idx)

            if result.error is not None:
                raise RuntimeError(
                    f"Node {node_id} failed for contexts {result.context_indices or 'unknown'}: "
                    f"{result.error}"
                )
            node = node_specs.get(node_id)
            if progress_monitor and is_progress_node(node):
                progress_monitor.record(len(result.context_indices))
            self._record_node_metrics(
                node_id=node_id,
                node=node,
                stats=result.stats,
                count=len(result.context_indices),
                total_time=float((result.stats or {}).get("node_total_time") or 0.0),
            )

            wid_entry = node_worker.get(node_id)
            wid_seq = wid_entry if isinstance(wid_entry, (list, tuple)) else (wid_entry,)

            outputs_batch = result.outputs or [{} for _ in result.context_indices]
            if len(outputs_batch) != len(result.context_indices):
                raise RuntimeError(
                    f"Node {node_id} produced {len(outputs_batch)} outputs "
                    f"for {len(result.context_indices)} contexts."
                )

            for idx, outputs in zip(result.context_indices, outputs_batch):
                if outputs:
                    contexts[idx].update(outputs)
                completed[idx].add(node_id)
                remaining_counts[node_id] -= 1
                completed_units += 1
                node = node_specs.get(node_id)
                if node and node.engine == "vllm":
                    llm_done_units += 1
                else:
                    db_done_units += 1
                wid_for_idx = inflight_worker.pop((node_id, idx), None)
                if wid_for_idx is not None:
                    worker_inflight_counts[wid_for_idx] = max(0, worker_inflight_counts[wid_for_idx] - 1)
                    if worker_inflight_counts[wid_for_idx] == 0:
                        locked_model = worker_lock_model.get(wid_for_idx)
                        if locked_model is not None and not worker_has_ready_same_model(locked_model, wid_for_idx):
                            worker_lock_model[wid_for_idx] = None
                self._unlock_epoch_dependents(
                    finished_node_id=node_id,
                    context_idx=idx,
                    epoch_dependents=dependents,
                    pending_dependencies=pending_dependencies,
                    completed=completed,
                    ready=ready,
                    ready_nonempty=ready_nonempty,
                )

            if remaining_counts.get(node_id, 0) <= 0:
                log_node_done(node_id)
                task_map.pop(node_id, None)
                ready.pop(node_id, None)
                remaining_counts.pop(node_id, None)

    def _unlock_epoch_dependents(
        self,
        *,
        finished_node_id: str,
        context_idx: int,
        epoch_dependents: Dict[str, List[str]],
        pending_dependencies: List[Dict[str, Set[str]]],
        completed: List[Set[str]],
        ready: Dict[str, Deque[int]],
        ready_nonempty: Set[str] | None = None,
    ) -> None:
        for child in epoch_dependents.get(finished_node_id, []):
            deps = pending_dependencies[context_idx][child]
            deps.discard(finished_node_id)
            if not deps and child not in completed[context_idx]:
                ready.setdefault(child, deque()).append(context_idx)
                if ready_nonempty is not None:
                    ready_nonempty.add(child)

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
        *,
        node_id: str,
        node: Any,
        stats: Mapping[str, Any] | None,
        count: int,
        total_time: float,
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
        if total_time <= 0:
            total_time = execute_time if execute_time > 0 else 0.0
        if execute_time <= 0:
            execute_time = max(0.0, total_time)
        engine = getattr(node, "engine", None) if node is not None else None
        model = getattr(node, "model", None) if node is not None else None
        metrics.record_node_metrics(
            node_id,
            total_time=total_time,
            prepare_time=prepare_time,
            execute_time=execute_time,
            count=count,
            engine=engine,
            model=model,
        )
