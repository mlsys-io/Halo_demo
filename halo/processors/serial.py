from __future__ import annotations

import time
from typing import Any, Dict, List, MutableMapping, Sequence, Set

from ..models import ExecutionPlan, ExecutionTask, GraphSpec
from ..monitoring import ProgressMonitor, start_progress_monitor, start_system_monitor
from .base import BaseGraphProcessor, count_progress_nodes, is_progress_node

class SerialGraphProcessor(BaseGraphProcessor):
    """串行执行：逐个 context 顺序执行 DAG（无多进程/批处理）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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
        try:
            contexts = []
            total_inputs = len(initial_inputs_list)
            for i, inputs in enumerate(initial_inputs_list):
                print(f"[91mProcessing input {i + 1}/{total_inputs}[0m")
                ctx = dict(inputs)
                final_ctx = self._run_single(plan, graph, ctx, progress_monitor)
                contexts.append(final_ctx)
            return contexts
        finally:
            if progress_monitor:
                progress_monitor.stop()
            if monitor:
                monitor.stop()

    def _run_single(
        self,
        plan: ExecutionPlan,
        graph: GraphSpec,
        context: Dict[str, Any],
        progress_monitor: ProgressMonitor | None,
    ) -> Dict[str, Any]:
        tasks_by_id: Dict[str, ExecutionTask] = {task.node_id: task for task in plan.tasks}
        plan_order: Dict[str, int] = {task.node_id: idx for idx, task in enumerate(plan.tasks)}
        dependencies: Dict[str, Set[str]] = {
            task.node_id: set(task.dependencies) for task in plan.tasks
        }
        dependents: Dict[str, List[str]] = {}
        for task in plan.tasks:
            for dep in task.dependencies:
                dependents.setdefault(dep, []).append(task.node_id)

        ready: List[str] = sorted(
            [node_id for node_id, deps in dependencies.items() if not deps],
            key=lambda nid: plan_order.get(nid, 0),
        )
        completed: Set[str] = set()

        while ready:
            node_id = ready.pop(0)
            node = graph.nodes.get(node_id)
            if node is None:
                raise RuntimeError(f"Node '{node_id}' not found in graph.")

            start_time = time.perf_counter()
            if node.type == "input":
                outputs: Dict[str, Any] = {}
                for name in node.outputs:
                    if name not in context:
                        raise RuntimeError(f"Input node '{node.id}' missing initial value for output '{name}'.")
                    outputs[name] = context[name]
                stats: Dict[str, Any] = {}
            elif node.type == "processor":
                outputs = self.processor_executor.execute(node, context)
                stats = self.processor_executor.consume_stats()
            elif node.engine == "vllm":
                node_model = node.model or ""
                if node_model and self.current_model != node_model:
                    # Switch model: clear cache and update tracker.
                    # The engine provider will lazy-load the new model on next use.
                    self.engine_provider.clear_cache()
                    self.current_model = node_model

                outputs = self.vllm_executor.execute(node, context)
                stats = self.vllm_executor.consume_stats()
            elif node.engine == "db":
                outputs = self.db_node_executor.execute(node, context)
                stats = self.db_node_executor.consume_stats()
            elif node.engine == "http":
                outputs = self.http_executor.execute(node, context)
                stats = self.http_executor.consume_stats()
            elif node.engine == "noop":
                outputs = {name: context.get(name) for name in node.outputs}
                stats = {}
            else:
                raise RuntimeError(f"Unsupported node: engine={node.engine}, type={node.type}")
            total_time = time.perf_counter() - start_time
            self._record_worker_metrics(stats)
            self._record_node_metrics(node, total_time=total_time, stats=stats, count=1)

            if outputs:
                context.update(outputs)
            completed.add(node_id)
            if progress_monitor and is_progress_node(node):
                progress_monitor.record(1)

            for child in dependents.get(node_id, []):
                deps = dependencies[child]
                deps.discard(node_id)
                if not deps and child not in completed and child not in ready:
                    ready.append(child)
            ready.sort(key=lambda nid: plan_order.get(nid, 0))

        if len(completed) != len(tasks_by_id):
            missing = set(tasks_by_id) - completed
            raise RuntimeError(f"Graph execution incomplete; missing nodes: {missing}")

        return context
