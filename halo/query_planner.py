from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence

from .db import PostgresPlanExplainer, resolve_query_parameters
from .models import DBQuery, GraphSpec, PlanMetric, QueryPlanChoice, QueryPlanOption
from .utils import MISSING, PLACEHOLDER_PATTERN, lookup_path


LOGGER = logging.getLogger(__name__)


# 默认 EXPLAIN 方案：覆盖常见的扫描/Join 偏好，便于 profile。
DEFAULT_QUERY_PLANS: Sequence[QueryPlanOption] = (
    QueryPlanOption(id="default", description="Postgres default"),
    QueryPlanOption(
        id="prefer_index",
        description="Disable sequential scans to prefer index scans",
        settings={"enable_seqscan": "off"},
    ),
    QueryPlanOption(
        id="prefer_seq",
        description="Disable index/bitmap scans to force seq scan",
        settings={"enable_indexscan": "off", "enable_bitmapscan": "off"},
    ),
    QueryPlanOption(
        id="prefer_nestloop",
        description="Disable hash/merge joins to bias toward nested loops",
        settings={"enable_hashjoin": "off", "enable_mergejoin": "off"},
    ),
)


class QueryPlanEvaluator:
    def __init__(
        self,
        explainer_connect_kwargs: Dict[str, Any] | None = None,
        explainer_pool_size: int | None = None,
        explainer_factory: Callable[[], PostgresPlanExplainer] | None = None,
        base_context: Mapping[str, Any] | None = None,
    ) -> None:
        self._explainers: List[PostgresPlanExplainer] = []

        if explainer_factory is not None:
            def factory() -> PostgresPlanExplainer:
                explainer = explainer_factory()
                self._explainers.append(explainer)
                return explainer
        else:
            def factory() -> PostgresPlanExplainer:
                explainer = PostgresPlanExplainer(
                    connect_kwargs=explainer_connect_kwargs,
                    pool_size=explainer_pool_size,
                )
                self._explainers.append(explainer)
                return explainer

        self._explainer_factory = factory
        default_base = {"search_keyword": "sample keyword"}
        self._base_context = dict(default_base)
        if base_context:
            self._base_context.update(base_context)

    def cleanup(self) -> None:
        """Best-effort cleanup of explainer connection pools."""
        for explainer in self._explainers:
            pool = getattr(explainer, "_pool", None)
            lifo = getattr(pool, "_pool", None)
            if lifo is None:
                continue
            while True:
                try:
                    conn = lifo.get_nowait()
                except Exception:
                    break
                try:
                    conn.close()
                except Exception:
                    pass
            try:
                lifo.queue.clear()
            except Exception:
                pass
        self._explainers.clear()

    def evaluate_graph(
        self,
        graph: GraphSpec,
        contexts: Sequence[Mapping[str, Any]] | None = None,
        *,
        default_plan_only: bool = False,
    ) -> Dict[tuple[str, str], List[QueryPlanChoice]]:
        """遍历 DAG 中所有 DBQuery，并收集候选执行计划。"""
        LOGGER.debug("Collecting EXPLAIN plans for graph %s", graph.name)
        evaluations: Dict[tuple[str, str], List[QueryPlanChoice]] = {}
        for node in graph.nodes.values():
            for query in node.db_queries:
                plans = self._plan_candidates(query, default_only=default_plan_only)
                LOGGER.debug(
                    "Planning %s/%s with %d candidates", node.id, query.name, len(plans)
                )
                params = self._resolve_params(query, contexts or (), node_id=node.id)
                plan_choices: List[QueryPlanChoice] = []
                for plan in plans:
                    choice = self._evaluate_plan(node.id, query, plan, params)
                    plan_choices.append(choice)
                plan_choices.sort(key=lambda c: c.cost if c.cost is not None else float("inf"))
                evaluations[(node.id, query.name)] = plan_choices
        return evaluations

    def _evaluate_plan(
        self,
        node_id: str,
        query: DBQuery,
        plan: QueryPlanOption,
        params: Mapping[str, Any] | None,
    ) -> QueryPlanChoice:
        if params is None:
            LOGGER.warning("Skip plan %s for %s/%s: missing parameters", plan.id, node_id, query.name)
            return QueryPlanChoice(
                plan_id=plan.id,
                description=plan.description,
                cost=None,
                raw_cost=None,
                explain_json=None,
                samples=tuple(),
                footprints={},
            )
        try:
            explainer = self._explainer_factory()
            explain_json, metric = explainer.explain(query, params, settings=plan.settings)
            raw_cost, cost = self._extract_costs(explain_json)
            footprint = self._footprints_from_metric(metric)
            return QueryPlanChoice(
                plan_id=plan.id,
                description=plan.description,
                cost=cost,
                raw_cost=raw_cost,
                explain_json=explain_json,
                samples=(metric,),
                footprints=footprint,
            )
        except Exception:
            LOGGER.exception("Failed to EXPLAIN plan %s for %s/%s", plan.id, node_id, query.name)
            return QueryPlanChoice(
                plan_id=plan.id,
                description=plan.description,
                cost=None,
                raw_cost=None,
                explain_json=None,
                samples=tuple(),
                footprints={},
            )

    def _resolve_params(
        self,
        query: DBQuery,
        contexts: Sequence[Mapping[str, Any]],
        *,
        node_id: str,
    ) -> Dict[str, Any] | None:
        """优先使用真实上下文；不足时自动填充 placeholder。"""
        placeholders = self._collect_placeholder_types(query)
        # 先尝试用户提供的上下文
        for ctx in contexts:
            augmented = self._augment_context(ctx, placeholders)
            try:
                params, missing = resolve_query_parameters(query, augmented, node_id=node_id)
            except Exception:
                continue
            if not missing:
                return params
        # 不足则用占位构造
        try:
            fallback_ctx = self._build_placeholder_context(placeholders)
            params, missing = resolve_query_parameters(query, fallback_ctx, node_id=node_id)
            if not missing:
                return params
        except Exception:
            return None
        return None

    def _footprints_from_metric(self, metric: PlanMetric) -> Dict[str, int]:
        acc: Dict[str, int] = {}
        for name, value in metric.relations.items():
            acc[name] = acc.get(name, 0) + value
        for name, value in metric.indexes.items():
            acc[name] = acc.get(name, 0) + value
        return acc

    def _extract_costs(self, explain_json: Any) -> tuple[float | None, float | None]:
        try:
            if isinstance(explain_json, list) and explain_json:
                plan = explain_json[0].get("Plan")
                if plan:
                    raw_cost = plan.get("Total Cost")
                    if raw_cost is not None:
                        raw_value = float(raw_cost)
                        return raw_value, self._normalize_cost(raw_value)
        except Exception:
            return None, None
        return None, None

    def _normalize_cost(self, raw_cost: float | None) -> float | None:
        """通过 log 缩放，使得 explain 的 cost 与执行 cost 处于同一量级。"""
        if raw_cost is None:
            return None
        if raw_cost <= 0:
            return 0.0
        return math.log1p(raw_cost) / 5.0

    def _collect_placeholder_types(self, query: DBQuery) -> Dict[str, str | None]:
        path_types: Dict[str, str | None] = {}
        for param_name, template in query.parameters.items():
            type_hint = query.param_types.get(param_name)
            for path in self._gather_placeholders(template):
                path_types.setdefault(path, type_hint)
        for required in query.required_inputs:
            path_types.setdefault(required, None)
        return path_types

    def _build_placeholder_context(self, placeholders: Mapping[str, str | None]) -> Dict[str, Any]:
        context: Dict[str, Any] = dict(self._base_context)
        for path, type_hint in placeholders.items():
            sample_value = self._sample_value_for_type(type_hint)
            self._ensure_context_path(context, path, sample_value)
        return context

    def _augment_context(
        self,
        ctx: Mapping[str, Any],
        placeholders: Mapping[str, str | None],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(self._base_context)
        merged.update(ctx)
        for path, type_hint in placeholders.items():
            if self._lookup_path(merged, path) is not MISSING:
                continue
            sample_value = self._sample_value_for_type(type_hint)
            self._ensure_context_path(merged, path, sample_value)
        return merged

    def _gather_placeholders(self, value: Any) -> List[str]:
        placeholders: List[str] = []
        if isinstance(value, str):
            for match in PLACEHOLDER_PATTERN.finditer(value):
                placeholders.append(match.group(1).strip())
        elif isinstance(value, Mapping):
            for item in value.values():
                placeholders.extend(self._gather_placeholders(item))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                placeholders.extend(self._gather_placeholders(item))
        return placeholders

    def _ensure_context_path(self, context: MutableMapping[str, Any], path: str, value: Any) -> None:
        segments = [segment.strip() for segment in path.split(".") if segment.strip()]
        if not segments:
            return
        current: MutableMapping[str, Any] = context
        for segment in segments[:-1]:
            next_value = current.get(segment)
            if not isinstance(next_value, MutableMapping):
                next_value = {}
                current[segment] = next_value
            current = next_value  # type: ignore[assignment]
        current.setdefault(segments[-1], value)

    def _lookup_path(self, context: Mapping[str, Any], path: str) -> Any:
        return lookup_path(context, path, default=MISSING)

    def _sample_value_for_type(self, type_hint: str | None) -> Any:
        if not type_hint:
            return "sample"
        hint = type_hint.lower()
        if hint == "date":
            return "1995-03-15"
        if hint in {"timestamp", "timestamptz"}:
            return "1995-03-15 00:00:00"
        if hint == "time":
            return "00:00:00"
        if hint in {"numeric", "decimal"}:
            return 0.05
        if hint in {"int", "integer"}:
            return 1
        if hint in {"float", "double"}:
            return 1.0
        if hint in {"bool", "boolean"}:
            return True
        if hint == "json":
            return {"sample": "value"}
        if hint.endswith("[]"):
            base = self._sample_value_for_type(hint[:-2])
            return [base, base]
        if hint in {"text", "string", "str"}:
            return "sample"
        return "sample"

    def _plan_candidates(self, query: DBQuery, default_only: bool = False) -> Sequence[QueryPlanOption]:
        """确保始终有可用的计划候选集。"""
        if default_only:
            if query.plans:
                for plan in query.plans:
                    if plan.id == "default":
                        return (plan,)
                return (query.plans[0],)
            return (DEFAULT_QUERY_PLANS[0],)
        if query.plans:
            return query.plans
        return DEFAULT_QUERY_PLANS


def plan_evaluator_from_env(env: Mapping[str, str] | None = None) -> QueryPlanEvaluator:
    """基于环境变量构造 Postgres plan evaluator。"""
    env_map = env or os.environ
    dsn = env_map.get("HALO_PLAN_PG_DSN") or env_map.get("HALO_PG_DSN")
    connect_kwargs = _plan_db_connect_kwargs(env_map)

    def factory() -> PostgresPlanExplainer:
        return PostgresPlanExplainer(dsn=dsn, connect_kwargs=dict(connect_kwargs))

    return QueryPlanEvaluator(explainer_factory=factory)


def _plan_db_connect_kwargs(env: Mapping[str, str]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}

    host = env.get("HALO_PLAN_PG_HOST") or env.get("HALO_PG_HOST") or "/var/run/postgresql"
    if host:
        kwargs["host"] = host

    port_value = env.get("HALO_PLAN_PG_PORT") or env.get("HALO_PG_PORT") or "4032"
    try:
        kwargs["port"] = int(port_value)
    except (TypeError, ValueError):
        pass

    dbname = env.get("HALO_PLAN_PG_DBNAME") or env.get("HALO_PG_DBNAME") or "finewiki"
    if dbname:
        kwargs["dbname"] = dbname

    user = env.get("HALO_PLAN_PG_USER") or env.get("HALO_PG_USER") or env.get("USER") or "postgres"
    if user:
        kwargs["user"] = user

    password = env.get("HALO_PLAN_PG_PASSWORD") or env.get("HALO_PG_PASSWORD")
    if password:
        kwargs["password"] = password

    return kwargs


__all__ = [
    "DEFAULT_QUERY_PLANS",
    "QueryPlanEvaluator",
    "plan_evaluator_from_env",
]
