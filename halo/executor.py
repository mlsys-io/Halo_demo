from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import threading
from datetime import datetime
import re
import time
from typing import Any, Dict, List, Mapping, Sequence
from decimal import Decimal
from collections import OrderedDict

from . import metrics
from .db import DatabaseExecutor, MissingQueryInputs, PostgresDatabaseExecutor, PostgresPlanExplainer, resolve_query_parameters
from .engines import EngineProvider
from .models import DBQuery, Node
from .utils import MISSING, lookup_path, maybe_parse_json, render_template
from .node_processors import run_processor_node


@dataclass
class ExecutionStats:
    db_time: float = 0.0
    db_calls: int = 0
    api_time: float = 0.0
    api_calls: int = 0
    llm_time: float = 0.0
    llm_calls: int = 0
    llm_prompts: int = 0
    model_init_time: float = 0.0
    model_init_calls: int = 0

    def clear(self) -> None:
        self.db_time = 0.0
        self.db_calls = 0
        self.api_time = 0.0
        self.api_calls = 0
        self.llm_time = 0.0
        self.llm_calls = 0
        self.llm_prompts = 0
        self.model_init_time = 0.0
        self.model_init_calls = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_time": self.db_time,
            "db_calls": self.db_calls,
            "api_time": self.api_time,
            "api_calls": self.api_calls,
            "llm_time": self.llm_time,
            "llm_calls": self.llm_calls,
            "llm_prompts": self.llm_prompts,
            "model_init_time": self.model_init_time,
            "model_init_calls": self.model_init_calls,
        }


@dataclass(slots=True)
class _BaseExecutor:
    _stats: ExecutionStats = field(init=False, repr=False)
    _stats_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._stats = ExecutionStats()
        self._stats_lock = threading.Lock()

    def _begin_execution(self) -> None:
        with self._stats_lock:
            self._stats.clear()

    def consume_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            payload = self._stats.to_dict()
            self._stats.clear()
            return payload

    def _record_db_time(self, elapsed: float, count: int = 1) -> None:
        if elapsed < 0:
            elapsed = 0.0
        with self._stats_lock:
            self._stats.db_time += elapsed
            self._stats.db_calls += count

    def _record_api_time(self, elapsed: float, count: int = 1) -> None:
        if elapsed < 0:
            elapsed = 0.0
        with self._stats_lock:
            self._stats.api_time += elapsed
            self._stats.api_calls += count

    def _record_llm_time(self, elapsed: float, prompt_count: int = 0, call_count: int = 1) -> None:
        if elapsed < 0:
            elapsed = 0.0
        with self._stats_lock:
            self._stats.llm_time += elapsed
            self._stats.llm_calls += call_count
            self._stats.llm_prompts += prompt_count

    def _record_model_init_time(self, elapsed: float, count: int = 1) -> None:
        if elapsed < 0:
            elapsed = 0.0
        with self._stats_lock:
            self._stats.model_init_time += elapsed
            self._stats.model_init_calls += count


@dataclass(slots=True)
class DBNodeExecutor(_BaseExecutor):
    """Only handles DB nodes."""

    db_executor: DatabaseExecutor
    db_concurrency: int = 32
    result_cache_size: int = 1024
    enable_result_cache: bool = True
    enable_prepared_statements: bool = True
    prepare_threshold: int | None = 1
    enable_db_explain_analyze: bool = False
    db_explain_mode: str = "wrap"  # "wrap" (run + explain) or "replace" (explain only)
    db_explain_sample_rate: float = 1.0
    _thread_pool: ThreadPoolExecutor | None = field(init=False, repr=False, default=None)
    _result_cache: "OrderedDict[tuple[str, tuple], Mapping[str, Any]]" = field(init=False, repr=False)
    _result_cache_lock: threading.Lock = field(init=False, repr=False)
    _plan_explainer: PostgresPlanExplainer | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        _BaseExecutor.__post_init__(self)
        if self.db_concurrency > 1:
            # Reuse a small thread pool across queries to avoid per-call creation overhead.
            self._thread_pool = ThreadPoolExecutor(max_workers=self.db_concurrency)
        self.enable_result_cache = bool(self.enable_result_cache)
        if not self.enable_result_cache:
            self.result_cache_size = 0
        self.result_cache_size = max(0, int(self.result_cache_size))
        self._result_cache = OrderedDict()
        self._result_cache_lock = threading.Lock()
        self.db_explain_sample_rate = float(self.db_explain_sample_rate)
        if self.db_explain_sample_rate < 0:
            self.db_explain_sample_rate = 0.0
        if self.db_explain_sample_rate > 1:
            self.db_explain_sample_rate = 1.0
        self.db_explain_mode = (self.db_explain_mode or "wrap").lower()
        self._apply_prepare_settings(self.db_executor)
        if self.enable_db_explain_analyze:
            self._plan_explainer = self._maybe_build_plan_explainer()
            if self._plan_explainer is not None:
                self._apply_prepare_settings(self._plan_explainer)

    def execute(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        self._begin_execution()
        if node.engine != "db":
            raise RuntimeError(
                f"DBNodeExecutor only supports engine='db' nodes (got engine={node.engine}, type={node.type})"
            )
        return self._execute_db(node, context)

    def execute_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._begin_execution()
        if not contexts:
            return []
        if node.engine != "db":
            raise RuntimeError(
                f"DBNodeExecutor only supports engine='db' nodes (got engine={node.engine}, type={node.type})"
            )
        return self._execute_db_batch(node, contexts)

    def _execute_input(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for name in node.outputs:
            if name not in context:
                raise RuntimeError(
                    f"Input node '{node.id}' missing initial value for output '{name}'."
                )
            outputs[name] = context[name]
        return outputs

    def _execute_passthrough(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        return {name: context.get(name) for name in node.outputs}

    def _execute_db(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        results = self._execute_db_batch(node, [context])
        return results[0] if results else {}

    def _execute_db_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        queries = list(node.db_queries)
        if not queries:
            return [{} for _ in contexts]
        db_results = self._run_db_queries_for_batch(node, contexts, queries)
        outputs: List[Dict[str, Any]] = []
        for results in db_results:
            ctx_outputs: Dict[str, Any] = {}
            for query, result in zip(queries, results):
                key = result.get("query", query.name)
                ctx_outputs[key] = result
                ctx_outputs.update(self._extract_result_mappings(query, result))
            outputs.append(ctx_outputs)
        return outputs

    def _run_db_queries_for_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
        queries: Sequence[DBQuery],
    ) -> List[List[Mapping[str, Any]]]:
        num_contexts = len(contexts)
        if num_contexts == 0:
            return []
        if not queries:
            return [[] for _ in range(num_contexts)]
        if not self.enable_result_cache:
            return self._run_db_queries_no_cache(node, contexts, queries)

        results_per_context: List[List[Mapping[str, Any]]] = [[] for _ in range(num_contexts)]
        batch_result_cache: Dict[tuple[str, str], Mapping[str, Any]] = {}

        for query in queries:
            per_ctx_params: List[tuple[Dict[str, Any], bool]] = []
            signature_owner: Dict[str, int] = {}

            for idx, context in enumerate(contexts):
                params, missing = resolve_query_parameters(query, context, node_id=node.id)
                has_missing = bool(missing)
                per_ctx_params.append((params, has_missing))
                if has_missing:
                    continue
                signature = self._serialize_parameters(params)
                if signature not in signature_owner:
                    signature_owner[signature] = idx

            to_run: List[tuple[tuple[str, str], int, Dict[str, Any]]] = []
            for signature, ctx_idx in signature_owner.items():
                cache_key = (query.name, signature)
                cached = batch_result_cache.get(cache_key) or self._get_cached_result(cache_key)
                if cached is not None:
                    continue
                to_run.append((cache_key, ctx_idx, per_ctx_params[ctx_idx][0]))

            if to_run:
                pool = self._thread_pool
                if pool is not None and len(to_run) > 1:
                    future_map = {
                        pool.submit(
                            self._run_query_with_fallback, query, contexts[ctx_idx], node.id, params
                        ): cache_key
                        for cache_key, ctx_idx, params in to_run
                    }
                    for future in as_completed(future_map):
                        cache_key = future_map[future]
                        result = future.result()
                        batch_result_cache[cache_key] = result
                else:
                    for cache_key, ctx_idx, params in to_run:
                        context = contexts[ctx_idx]
                        result = self._run_query_with_fallback(query, context, node.id, params)
                        batch_result_cache[cache_key] = result

            for idx, (params, has_missing) in enumerate(per_ctx_params):
                if has_missing:
                    results_per_context[idx].append(self._make_missing_query_result(query, params))
                    continue
                signature = self._serialize_parameters(params)
                cache_key = (query.name, signature)
                result = batch_result_cache.get(cache_key) or self._get_cached_result(cache_key)
                if result is None:
                    context = contexts[idx]
                    result = self._run_db_query(query, context, node_id=node.id)
                    batch_result_cache[cache_key] = result
                results_per_context[idx].append(result)

            # Flush this query's batch results into the global LRU (bounded).
            for cache_key, result in batch_result_cache.items():
                self._cache_result(cache_key, result)

        return results_per_context

    def _run_db_queries_no_cache(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
        queries: Sequence[DBQuery],
    ) -> List[List[Mapping[str, Any]]]:
        num_contexts = len(contexts)
        results_per_context: List[List[Mapping[str, Any]]] = [[] for _ in range(num_contexts)]
        pool = self._thread_pool

        for query in queries:
            per_ctx_params: List[tuple[Dict[str, Any], bool]] = []
            to_run: List[tuple[int, Dict[str, Any]]] = []
            for idx, context in enumerate(contexts):
                params, missing = resolve_query_parameters(query, context, node_id=node.id)
                has_missing = bool(missing)
                per_ctx_params.append((params, has_missing))
                if not has_missing:
                    to_run.append((idx, params))

            results_by_idx: Dict[int, Mapping[str, Any]] = {}
            if to_run:
                if pool is not None and len(to_run) > 1:
                    future_map = {
                        pool.submit(
                            self._run_query_with_fallback, query, contexts[idx], node.id, params
                        ): idx
                        for idx, params in to_run
                    }
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        results_by_idx[idx] = future.result()
                else:
                    for idx, params in to_run:
                        results_by_idx[idx] = self._run_query_with_fallback(
                            query, contexts[idx], node.id, params
                        )

            for idx, (params, has_missing) in enumerate(per_ctx_params):
                if has_missing:
                    results_per_context[idx].append(self._make_missing_query_result(query, params))
                    continue
                result = results_by_idx.get(idx)
                if result is None:
                    result = self._run_db_query(query, contexts[idx], node_id=node.id)
                results_per_context[idx].append(result)

        return results_per_context

    def _cache_result(self, cache_key: tuple[str, str], result: Mapping[str, Any]) -> None:
        """Store a result in the instance-level LRU cache."""
        if not self.enable_result_cache or self.result_cache_size <= 0:
            return
        with self._result_cache_lock:
            self._result_cache[cache_key] = result
            self._result_cache.move_to_end(cache_key)
            while len(self._result_cache) > self.result_cache_size:
                self._result_cache.popitem(last=False)

    def _get_cached_result(self, cache_key: tuple[str, str]) -> Mapping[str, Any] | None:
        if not self.enable_result_cache or self.result_cache_size <= 0:
            return None
        with self._result_cache_lock:
            result = self._result_cache.get(cache_key)
            if result is None:
                return None
            self._result_cache.move_to_end(cache_key)
            return result

    def _serialize_parameters(self, params: Mapping[str, Any]) -> tuple:
        """Build a hashable signature for ``params`` without going through JSON.

        We only need the signature for batch-internal dedup / LRU lookup; equality
        on the tuple is enough. Falls back to ``repr`` only for non-hashable
        values (e.g. dict / list), which is much rarer than the typical scalar
        parameter set.
        """
        out: list = []
        for key in sorted(params):
            value = params[key]
            try:
                hash(value)
            except TypeError:
                value = repr(value)
            out.append((key, value))
        return tuple(out)

    def _make_missing_query_result(
        self,
        query: DBQuery,
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "query": query.name,
            "sql": query.sql,
            "parameters": params,
            "rows": [],
            "rowcount": 0,
            "note": "Skipped due to missing query inputs.",
        }

    def _extract_result_mappings(
        self,
        query: DBQuery,
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        mapped: Dict[str, Any] = {}
        rows = result.get("rows", [])
        for key, path in query.result_mappings.items():
            value = lookup_path(rows, path, default=MISSING)
            if value is MISSING:
                continue
            mapped[key] = value
        return mapped

    def _run_db_query(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        *,
        node_id: str,
    ) -> Mapping[str, Any]:
        start_time = time.perf_counter()
        result: Mapping[str, Any]
        try:
            if self.enable_db_explain_analyze and self.db_explain_mode == "replace":
                result = self._run_db_query_explain_only(query, context, node_id=node_id)
            else:
                result = self.db_executor.run(query, context, node_id=node_id)
        finally:
            self._record_db_time(time.perf_counter() - start_time)

        if self.enable_db_explain_analyze and self.db_explain_mode == "wrap":
            self._maybe_record_explain(query, node_id=node_id, result=result)
        return result

    def _maybe_build_plan_explainer(self) -> PostgresPlanExplainer | None:
        if not isinstance(self.db_executor, PostgresDatabaseExecutor):
            return None
        dsn = getattr(self.db_executor, "dsn", None)
        connect_kwargs = getattr(self.db_executor, "connect_kwargs", None)
        try:
            return PostgresPlanExplainer(
                dsn=dsn,
                connect_kwargs=dict(connect_kwargs) if connect_kwargs else None,
                pool_size=1,
                analyze=True,
            )
        except Exception:
            return None

    def _apply_prepare_settings(self, executor: DatabaseExecutor) -> None:
        if not isinstance(executor, PostgresDatabaseExecutor):
            return
        if not self.enable_prepared_statements:
            executor.prepare_threshold = 0
            return
        if self.prepare_threshold is not None:
            executor.prepare_threshold = self.prepare_threshold

    def _extract_planned_total_cost(self, plan_json: Any) -> float | None:
        try:
            if isinstance(plan_json, list) and plan_json:
                top = plan_json[0]
                if isinstance(top, Mapping):
                    plan = top.get("Plan")
                    if isinstance(plan, Mapping):
                        cost = plan.get("Total Cost")
                        if cost is not None:
                            return float(cost)
        except Exception:
            return None
        return None

    def _maybe_record_explain(self, query: DBQuery, *, node_id: str, result: Mapping[str, Any]) -> None:
        explainer = self._plan_explainer
        if explainer is None:
            return
        if self.db_explain_sample_rate <= 0:
            return
        if self.db_explain_sample_rate < 1.0:
            import random
            if random.random() > self.db_explain_sample_rate:
                return
        params = result.get("parameters")
        if not isinstance(params, Mapping):
            return
        try:
            plan_json, plan_metric = explainer.explain(query, params, settings=None)
        except Exception:
            return
        planned_cost = self._extract_planned_total_cost(plan_json)
        query_key = f"{node_id}:{query.name}"
        metrics.record_db_explain(
            query_key,
            runtime_ms=plan_metric.runtime_ms,
            planned_total_cost=planned_cost,
            shared_hit_blocks=plan_metric.shared_hit_blocks,
            shared_read_blocks=plan_metric.shared_read_blocks,
            local_hit_blocks=plan_metric.local_hit_blocks,
            local_read_blocks=plan_metric.local_read_blocks,
            relations=dict(plan_metric.relations) if plan_metric.relations else None,
            indexes=dict(plan_metric.indexes) if plan_metric.indexes else None,
        )

    def _run_db_query_explain_only(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        *,
        node_id: str,
    ) -> Mapping[str, Any]:
        explainer = self._plan_explainer
        if explainer is None:
            # Fallback to normal execution if explain is unavailable.
            return self.db_executor.run(query, context, node_id=node_id)
        params, missing = resolve_query_parameters(query, context, node_id=node_id)
        if missing:
            raise MissingQueryInputs(query.name, missing)
        plan_json, plan_metric = explainer.explain(query, params, settings=None)
        planned_cost = self._extract_planned_total_cost(plan_json)
        query_key = f"{node_id}:{query.name}"
        metrics.record_db_explain(
            query_key,
            runtime_ms=plan_metric.runtime_ms,
            planned_total_cost=planned_cost,
            shared_hit_blocks=plan_metric.shared_hit_blocks,
            shared_read_blocks=plan_metric.shared_read_blocks,
            local_hit_blocks=plan_metric.local_hit_blocks,
            local_read_blocks=plan_metric.local_read_blocks,
            relations=dict(plan_metric.relations) if plan_metric.relations else None,
            indexes=dict(plan_metric.indexes) if plan_metric.indexes else None,
        )
        # Return a minimal DB-like result so downstream prompt building still works.
        return {
            "query": query.name,
            "sql": query.sql,
            "parameters": params,
            "rows": [],
            "rowcount": 0,
            "note": "EXPLAIN ANALYZE (BUFFERS) mode: query results not collected.",
            "explain": plan_json,
        }

    def _run_query_with_fallback(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        node_id: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            return self._run_db_query(query, context, node_id=node_id)
        except MissingQueryInputs:
            return self._make_missing_query_result(query, params)


@dataclass(slots=True)
class HTTPNodeExecutor(_BaseExecutor):
    """Simulates HTTP nodes by sleeping for a configured duration."""

    http_concurrency: int = 32
    default_sleep_s: float = 0.0
    _thread_pool: ThreadPoolExecutor | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        _BaseExecutor.__post_init__(self)
        self.http_concurrency = max(1, int(self.http_concurrency))
        self.default_sleep_s = max(0.0, float(self.default_sleep_s))
        if self.http_concurrency > 1:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.http_concurrency)

    def execute(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        self._begin_execution()
        if node.engine != "http":
            raise RuntimeError(f"HTTPNodeExecutor only supports engine='http' (got {node.engine})")
        return self._execute_http_once(node, context)

    def execute_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._begin_execution()
        if not contexts:
            return []
        if node.engine != "http":
            raise RuntimeError(f"HTTPNodeExecutor only supports engine='http' (got {node.engine})")
        if self._thread_pool is None or len(contexts) == 1:
            return [self._execute_http_once(node, ctx) for ctx in contexts]

        outputs: List[Dict[str, Any]] = [{} for _ in contexts]
        future_map = {}
        for idx, context in enumerate(contexts):
            future_map[self._thread_pool.submit(self._execute_http_once, node, context)] = idx
        for future in as_completed(future_map):
            idx = future_map[future]
            outputs[idx] = future.result()
        return outputs

    def _execute_http_once(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        mean_sleep_s = self._resolve_sleep_seconds(node, context)
        sleep_s = self._sample_sleep_seconds(mean_sleep_s)
        start_time = time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        self._record_api_time(time.perf_counter() - start_time, count=1)
        return self._build_outputs(node, sleep_s)

    def _resolve_sleep_seconds(self, node: Node, context: Mapping[str, Any]) -> float:
        raw = node.raw or {}
        keys = (
            ("sleep_s", 1.0),
            ("sleep_ms", 0.001),
            ("latency_s", 1.0),
            ("latency_ms", 0.001),
            ("timeout_s", 1.0),
            ("timeout_ms", 0.001),
        )
        for key, scale in keys:
            if key not in raw:
                continue
            value = raw.get(key)
            if isinstance(value, str):
                value = render_template(value, context).strip()
            try:
                seconds = float(value) * scale
            except (TypeError, ValueError):
                continue
            return max(0.0, seconds)
        return self.default_sleep_s

    def _sample_sleep_seconds(self, mean_sleep_s: float) -> float:
        if mean_sleep_s <= 0:
            return 0.0
        # Chi-square(df=k) where mean=k; use Gamma(k/2, 2).
        return max(0.0, random.gammavariate(mean_sleep_s / 2.0, 2.0))

    def _build_outputs(self, node: Node, sleep_s: float) -> Dict[str, Any]:
        if not node.outputs:
            return {}
        payload = {
            "status": "ok",
            "sleep_s": sleep_s,
            "sleep_ms": int(round(sleep_s * 1000.0)),
        }
        if all(name in payload for name in node.outputs):
            return {name: payload[name] for name in node.outputs}
        if len(node.outputs) == 1:
            return {node.outputs[0]: payload}
        return {name: dict(payload) for name in node.outputs}


@dataclass(slots=True)
class ProcessorNodeExecutor(_BaseExecutor):
    """Handles local processor nodes (non-LLM, non-DB)."""

    def execute(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        self._begin_execution()
        if node.type != "processor":
            raise RuntimeError(
                f"ProcessorNodeExecutor only supports type='processor' (got type={node.type})"
            )
        return run_processor_node(node, context)

    def execute_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._begin_execution()
        if node.type != "processor":
            raise RuntimeError(
                f"ProcessorNodeExecutor only supports type='processor' (got type={node.type})"
            )
        return [run_processor_node(node, ctx) for ctx in contexts]


@dataclass(slots=True)
class VLLMNodeExecutor(_BaseExecutor):
    """Only handles vLLM nodes."""

    engine_provider: EngineProvider

    def execute(self, node: Node, context: Mapping[str, Any]) -> Dict[str, Any]:
        self._begin_execution()
        if node.engine != "vllm":
            raise RuntimeError(f"Unsupported node for VLLM executor: engine={node.engine}")
        return self._execute_vllm_batch(node, [context])[0]

    def execute_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._begin_execution()
        if node.engine != "vllm":
            raise RuntimeError(f"Unsupported node for VLLM executor: engine={node.engine}")
        return self._execute_vllm_batch(node, contexts)

    def warmup_model(self, model: str) -> None:
        """Preload a model to overlap initialization before real tasks."""
        dummy = Node(
            id="__warmup__",
            type="inference",
            engine="vllm",
            model=model,
            inputs=(),
            outputs=(),
            db_queries=(),
        )
        self._begin_execution()
        try:
            engine = self.engine_provider.resolve(dummy, on_initialize=self._record_model_init_time)
        except Exception:
            # Warmup best-effort; failures will surface on real execution.
            return

    def _execute_vllm_batch(
        self,
        node: Node,
        contexts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        outputs_list: List[Dict[str, Any]] = []
        if not contexts:
            return outputs_list

        engine = self.engine_provider.resolve(
            node,
            on_initialize=self._record_model_init_time,
        )

        # Many context entries reference the *same* upstream DB result dict
        # (batch-internal dedup happens in DBNodeExecutor). Memoize the
        # ``json.dumps(prepared_result)`` output by object identity so we do
        # the prepare+encode work once per unique result, not once per context.
        serialize_cache: Dict[int, str] = {}
        prompts = [self._build_prompt(node, ctx, [], serialize_cache=serialize_cache) for ctx in contexts]
        start_time = time.perf_counter()
        try:
            responses = engine.generate_batch(prompts, label=node.id)
        finally:
            self._record_llm_time(time.perf_counter() - start_time, prompt_count=len(prompts))

        for response, context in zip(responses, contexts):
            outputs = self._extract_outputs(node, response, context)
            outputs_list.append(outputs)

        return outputs_list

    def _build_prompt(
        self,
        node: Node,
        context: Mapping[str, Any],
        db_results: List[Any],
        *,
        serialize_cache: Dict[int, str] | None = None,
    ) -> str:
        parts: List[str] = []
        if getattr(node, "system_prompt", None):
            parts.append(render_template(node.system_prompt.strip(), context))

        for input_name in node.inputs:
            if input_name in context:
                rendered = self._render_prompt_value(
                    context[input_name], serialize_cache=serialize_cache
                )
                parts.append(f"[Input::{input_name}]\n{rendered}")

        for result in db_results:
            label = result.get("query", "unknown") if isinstance(result, Mapping) else "unknown"
            serialized = self._serialize_db_result(result, serialize_cache=serialize_cache)
            parts.append(f"[DB::{label}]\n{serialized}")

        return "\n\n".join(parts)

    def _render_prompt_value(
        self,
        value: Any,
        *,
        serialize_cache: Dict[int, str] | None = None,
    ) -> str:
        if isinstance(value, Mapping) and "rows" in value:
            return self._serialize_db_result(value, serialize_cache=serialize_cache)
        try:
            if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
                return json.dumps(value, ensure_ascii=False)
        except TypeError:
            pass
        return str(value)

    def _serialize_db_result(
        self,
        result: Any,
        *,
        serialize_cache: Dict[int, str] | None = None,
    ) -> str:
        if not isinstance(result, Mapping):
            try:
                return json.dumps(result, ensure_ascii=False)
            except TypeError:
                return str(result)
        if serialize_cache is not None:
            key = id(result)
            cached = serialize_cache.get(key)
            if cached is not None:
                return cached
        prepared = self._prepare_result_for_prompt(result)
        encoded = json.dumps(prepared, ensure_ascii=False)
        if serialize_cache is not None:
            serialize_cache[id(result)] = encoded
        return encoded

    def _maybe_parse_structured_response(self, response: str) -> Any:
        candidates: List[str] = [response.strip()]
        block_pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", re.IGNORECASE)
        candidates.extend(block_pattern.findall(response))

        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(response[first_brace:last_brace + 1])

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            parsed = maybe_parse_json(candidate)
            if isinstance(parsed, dict):
                return parsed

        text = response.strip()
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            brace = text.find("{", idx)
            if brace == -1:
                break
            try:
                parsed, end = decoder.raw_decode(text, brace)
            except json.JSONDecodeError:
                idx = brace + 1
                continue
            if isinstance(parsed, dict):
                return parsed
            idx = end
        return response

    def _extract_outputs(
        self,
        node: Node,
        response: str,
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        parsed = self._maybe_parse_structured_response(response)
        outputs: Dict[str, Any] = {}

        if isinstance(parsed, dict):
            for key in node.outputs:
                if key in parsed:
                    outputs[key] = parsed[key]
            if not outputs and node.outputs:
                outputs[node.outputs[0]] = parsed
        else:
            if node.outputs:
                outputs[node.outputs[0]] = parsed

        value = outputs.get("search_keyword")
        if isinstance(value, str):
            first = value.split(",")[0].strip()
            if first:
                outputs["search_keyword"] = first

        return outputs

    def _sanitize_value(self, value: Any, max_chars: int) -> Any:
        if isinstance(value, str):
            if len(value) > max_chars:
                return value[:max_chars] + "...[truncated]"
            return value
        if isinstance(value, Decimal):
            try:
                return float(value)
            except (ValueError, OverflowError):
                return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {k: self._sanitize_value(v, max_chars) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize_value(item, max_chars) for item in value]
        return value

    def _prepare_result_for_prompt(self, result: Mapping[str, Any]) -> Mapping[str, Any]:
        max_rows = 5
        max_chars = 2000
        prepared: Dict[str, Any] = dict(result)
        rows = prepared.get("rows")
        if isinstance(rows, list):
            trimmed_rows: List[Dict[str, Any]] = []
            for row in rows[:max_rows]:
                if not isinstance(row, Mapping):
                    trimmed_rows.append(row)  # type: ignore[arg-type]
                    continue
                compact_row: Dict[str, Any] = {}
                for key, value in row.items():
                    compact_row[key] = self._sanitize_value(value, max_chars)
                trimmed_rows.append(compact_row)
            prepared["rows"] = trimmed_rows
            if len(rows) > max_rows:
                prepared["rows_truncated"] = f"{len(rows) - max_rows} rows omitted"
        return prepared


# Backward compatibility export
__all__ = [
    "ExecutionStats",
    "DBNodeExecutor",
    "HTTPNodeExecutor",
    "ProcessorNodeExecutor",
    "VLLMNodeExecutor",
]
