"""Online serving: mini-batch execution with re-optimization at every mini-batch boundary."""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, List, Mapping, Tuple

from .models import ExecutionPlan, GraphSpec


class StreamingSession:
    """Serve a stream of queries over one or more workload templates.

    Arriving queries are buffered; ``flush`` segments the buffer into mini-batches
    at structurally stable boundaries, i.e. runs of consecutive queries that share
    a template (optionally capped at ``max_batch`` queries). The optimizer re-runs
    at every mini-batch boundary, since each mini-batch binds a new set of queries.
    Templates are identified by their structure (nodes and edges), not just their
    name.

    If a segment fails, ``flush`` re-raises after keeping the results of the
    segments that finished (returned by the next ``flush``) and re-queuing the
    segments it did not reach; the failed segment's queries are dropped and
    listed in the exception's ``failed_tickets``.
    """

    def __init__(self, optimizer: Any, processor: Any, *, max_batch: int | None = None) -> None:
        self.optimizer = optimizer
        self.processor = processor
        self.max_batch = max_batch if max_batch is None else max(1, int(max_batch))
        self._buffer: List[Tuple[int, GraphSpec, Dict[str, Any]]] = []
        self._finished: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._next_ticket = 0
        self.optimizer_runs = 0

    def submit(self, graph: GraphSpec, context: Mapping[str, Any]) -> int:
        """Queue one query (a template instance) and return its ticket."""
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._buffer.append((ticket, graph, dict(context)))
        return ticket

    def flush(self) -> Dict[int, Dict[str, Any]]:
        """Execute everything buffered so far; returns final contexts by ticket."""
        with self._lock:
            buffered, self._buffer = self._buffer, []
        segments = self._segments(buffered)
        for pos, (graph, items) in enumerate(segments):
            contexts = [ctx for _, ctx in items]
            try:
                plan = self._plan_for(graph, contexts)
                outputs = self.processor.run_batch(plan, graph, contexts)
            except Exception as exc:
                retry = [(t, g, c) for g, seg in segments[pos + 1:] for t, c in seg]
                with self._lock:
                    self._buffer[:0] = retry
                exc.failed_tickets = [ticket for ticket, _ in items]  # type: ignore[attr-defined]
                raise
            for (ticket, _), out in zip(items, outputs):
                self._finished[ticket] = out
        results, self._finished = self._finished, {}
        return results

    @staticmethod
    def _template_key(graph: GraphSpec) -> str:
        structure = repr((graph.name, sorted(graph.nodes.items()), graph.edges))
        return hashlib.sha1(structure.encode("utf-8")).hexdigest()

    def _segments(
        self, buffered: List[Tuple[int, GraphSpec, Dict[str, Any]]]
    ) -> List[Tuple[GraphSpec, List[Tuple[int, Dict[str, Any]]]]]:
        segments: List[Tuple[GraphSpec, List[Tuple[int, Dict[str, Any]]]]] = []
        for ticket, graph, ctx in buffered:
            same_template = segments and self._template_key(segments[-1][0]) == self._template_key(graph)
            if same_template and (self.max_batch is None or len(segments[-1][1]) < self.max_batch):
                segments[-1][1].append((ticket, ctx))
            else:
                segments.append((graph, [(ticket, ctx)]))
        return segments

    def _plan_for(self, graph: GraphSpec, contexts: List[Dict[str, Any]]) -> ExecutionPlan:
        plan = self.optimizer.build_plan(
            graph, sample_contexts=contexts[:1] or [{}], input_query_count=len(contexts)
        )
        self.optimizer_runs += 1
        return plan
