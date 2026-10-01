"""Online serving: mini-batch execution with re-optimization at template boundaries."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .models import ExecutionPlan, GraphSpec


class StreamingSession:
    """Serve a stream of queries over one or more workload templates.

    Arriving queries are buffered; ``flush`` segments the buffer into mini-batches
    at structurally stable boundaries, i.e. runs of consecutive queries that share
    a template (optionally capped at ``max_batch`` queries). The optimizer runs for
    a segment whose template has no plan yet; plans are cached per template, so a
    single-template stream is planned once and re-optimization happens exactly
    when the template mix changes.
    """

    def __init__(self, optimizer: Any, processor: Any, *, max_batch: int | None = None) -> None:
        self.optimizer = optimizer
        self.processor = processor
        self.max_batch = max_batch if max_batch is None else max(1, int(max_batch))
        self._plans: Dict[Tuple[str, Tuple[str, ...]], ExecutionPlan] = {}
        self._buffer: List[Tuple[int, GraphSpec, Dict[str, Any]]] = []
        self._next_ticket = 0
        self.optimizer_runs = 0

    def submit(self, graph: GraphSpec, context: Mapping[str, Any]) -> int:
        """Queue one query (a template instance) and return its ticket."""
        ticket = self._next_ticket
        self._next_ticket += 1
        self._buffer.append((ticket, graph, dict(context)))
        return ticket

    def flush(self) -> Dict[int, Dict[str, Any]]:
        """Execute everything buffered so far; returns final contexts by ticket."""
        results: Dict[int, Dict[str, Any]] = {}
        for graph, items in self._segments():
            contexts = [ctx for _, ctx in items]
            plan = self._plan_for(graph, contexts)
            outputs = self.processor.run_batch(plan, graph, contexts)
            for (ticket, _), out in zip(items, outputs):
                results[ticket] = out
        self._buffer.clear()
        return results

    @staticmethod
    def _template_key(graph: GraphSpec) -> Tuple[str, Tuple[str, ...]]:
        return (graph.name, tuple(sorted(graph.nodes)))

    def _segments(self) -> List[Tuple[GraphSpec, List[Tuple[int, Dict[str, Any]]]]]:
        segments: List[Tuple[GraphSpec, List[Tuple[int, Dict[str, Any]]]]] = []
        for ticket, graph, ctx in self._buffer:
            same_template = segments and self._template_key(segments[-1][0]) == self._template_key(graph)
            if same_template and (self.max_batch is None or len(segments[-1][1]) < self.max_batch):
                segments[-1][1].append((ticket, ctx))
            else:
                segments.append((graph, [(ticket, ctx)]))
        return segments

    def _plan_for(self, graph: GraphSpec, contexts: List[Dict[str, Any]]) -> ExecutionPlan:
        key = self._template_key(graph)
        plan = self._plans.get(key)
        if plan is None:
            plan = self.optimizer.build_plan(
                graph, sample_contexts=contexts[:1] or [{}], input_query_count=len(contexts)
            )
            self._plans[key] = plan
            self.optimizer_runs += 1
        return plan
