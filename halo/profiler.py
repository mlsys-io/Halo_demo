from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .executor import HTTPNodeExecutor
from .models import GraphSpec, QueryPlanChoice
from .node_processors import run_processor_node
from .query_planner import QueryPlanEvaluator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphProfile:
    plan_choices: Dict[tuple[str, str], Sequence[QueryPlanChoice]] = field(default_factory=dict)
    http_latencies_s: Dict[str, float] = field(default_factory=dict)
    http_samples: Dict[str, int] = field(default_factory=dict)
    processor_latencies_s: Dict[str, float] = field(default_factory=dict)
    processor_samples: Dict[str, int] = field(default_factory=dict)


class GraphProfiler:
    """Profiles SQL plans (EXPLAIN), HTTP nodes, and local-function nodes (timed sample invocations)."""

    def __init__(
        self,
        plan_evaluator: QueryPlanEvaluator | None = None,
        *,
        http_executor: HTTPNodeExecutor | None = None,
        http_concurrency: int = 32,
        http_default_sleep_s: float = 0.0,
    ) -> None:
        self._plan_evaluator = plan_evaluator
        self._http_executor = http_executor
        self._http_concurrency = max(1, int(http_concurrency))
        self._http_default_sleep_s = max(0.0, float(http_default_sleep_s))

    def profile_graph(
        self,
        graph: GraphSpec,
        contexts: Sequence[Mapping[str, Any]] | None = None,
        *,
        default_plan_only: bool = False,
        include_sql: bool = True,
        include_http: bool = True,
        include_processor: bool = True,
    ) -> GraphProfile:
        plan_choices: Dict[tuple[str, str], Sequence[QueryPlanChoice]] = {}
        if include_sql and self._plan_evaluator is not None:
            plan_choices = self._plan_evaluator.evaluate_graph(
                graph, contexts or (), default_plan_only=default_plan_only
            )

        http_latencies, http_samples = ({}, {})
        if include_http:
            http_latencies, http_samples = self._profile_http_nodes(graph, contexts)

        processor_latencies, processor_samples = ({}, {})
        if include_processor:
            processor_latencies, processor_samples = self._profile_processor_nodes(graph, contexts)

        return GraphProfile(
            plan_choices=plan_choices,
            http_latencies_s=http_latencies,
            http_samples=http_samples,
            processor_latencies_s=processor_latencies,
            processor_samples=processor_samples,
        )

    def _profile_http_nodes(
        self,
        graph: GraphSpec,
        contexts: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[Dict[str, float], Dict[str, int]]:
        http_nodes = [node for node in graph.nodes.values() if node.engine == "http"]
        if not http_nodes:
            return {}, {}

        context_list = [dict(ctx) for ctx in contexts] if contexts else [{}]
        executor = self._ensure_http_executor()
        results: Dict[str, float] = {}
        samples: Dict[str, int] = {}

        for node in http_nodes:
            try:
                executor.clear_caches()  # measure real calls, not cached coalesced results
                executor.execute_batch(node, context_list)
                stats = executor.consume_stats()
                calls = int(stats.get("api_calls") or stats.get("http_calls") or stats.get("db_calls") or 0)
                total = float(stats.get("api_time") or stats.get("http_time") or stats.get("db_time") or 0.0)
                if calls <= 0:
                    continue
                avg_latency = total / calls
                results[node.id] = max(0.0, avg_latency)
                samples[node.id] = max(1, calls)
            except Exception:
                LOGGER.exception("Failed to profile HTTP node %s", node.id)

        return results, samples

    def _profile_processor_nodes(
        self,
        graph: GraphSpec,
        contexts: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[Dict[str, float], Dict[str, int]]:
        processor_nodes = [node for node in graph.nodes.values() if node.type == "processor"]
        if not processor_nodes:
            return {}, {}

        context_list = list(contexts) if contexts else [{}]
        results: Dict[str, float] = {}
        samples: Dict[str, int] = {}

        for node in processor_nodes:
            try:
                start = time.perf_counter()
                for ctx in context_list:
                    run_processor_node(node, dict(ctx))
                elapsed = time.perf_counter() - start
            except Exception:
                # Inputs produced by upstream operators are not available before planning.
                LOGGER.debug("Could not profile processor node %s", node.id, exc_info=True)
                continue
            results[node.id] = max(0.0, elapsed / len(context_list))
            samples[node.id] = len(context_list)

        return results, samples

    def _ensure_http_executor(self) -> HTTPNodeExecutor:
        if self._http_executor is None:
            self._http_executor = HTTPNodeExecutor(
                http_concurrency=self._http_concurrency,
                default_sleep_s=self._http_default_sleep_s,
            )
        return self._http_executor
