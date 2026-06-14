from __future__ import annotations

import atexit
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _DBExplainAgg:
    calls: int = 0
    runtime_ms_sum: float = 0.0
    planned_total_cost_sum: float = 0.0
    planned_total_cost_count: int = 0
    planned_total_cost_min: float | None = None
    planned_total_cost_max: float | None = None
    shared_hit_blocks_sum: int = 0
    shared_read_blocks_sum: int = 0
    local_hit_blocks_sum: int = 0
    local_read_blocks_sum: int = 0

    def add(
        self,
        *,
        runtime_ms: float | None,
        planned_total_cost: float | None,
        shared_hit_blocks: int | None,
        shared_read_blocks: int | None,
        local_hit_blocks: int | None,
        local_read_blocks: int | None,
    ) -> None:
        self.calls += 1
        if runtime_ms is not None:
            self.runtime_ms_sum += float(runtime_ms)
        if planned_total_cost is not None:
            value = float(planned_total_cost)
            self.planned_total_cost_sum += value
            self.planned_total_cost_count += 1
            self.planned_total_cost_min = (
                value
                if self.planned_total_cost_min is None
                else min(self.planned_total_cost_min, value)
            )
            self.planned_total_cost_max = (
                value
                if self.planned_total_cost_max is None
                else max(self.planned_total_cost_max, value)
            )
        if shared_hit_blocks is not None:
            self.shared_hit_blocks_sum += int(shared_hit_blocks)
        if shared_read_blocks is not None:
            self.shared_read_blocks_sum += int(shared_read_blocks)
        if local_hit_blocks is not None:
            self.local_hit_blocks_sum += int(local_hit_blocks)
        if local_read_blocks is not None:
            self.local_read_blocks_sum += int(local_read_blocks)


@dataclass
class _NodeMetricsAgg:
    calls: int = 0
    total_time_sum: float = 0.0
    prepare_time_sum: float = 0.0
    execute_time_sum: float = 0.0

    def add(
        self,
        *,
        total_time: float,
        prepare_time: float,
        execute_time: float,
        count: int,
    ) -> None:
        self.calls += max(count, 0)
        self.total_time_sum += max(total_time, 0.0)
        self.prepare_time_sum += max(prepare_time, 0.0)
        self.execute_time_sum += max(execute_time, 0.0)


def _env_flag(name: str) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    return value not in ("", "0", "false", "no")


@dataclass
class MetricsTracker:
    run_name: Optional[str] = None
    run_start: Optional[float] = None
    run_end: Optional[float] = None
    total_queries: int = 0
    query_time_sum: float = 0.0
    query_count: int = 0
    db_exec_time_sum: float = 0.0
    db_exec_count: int = 0
    api_exec_time_sum: float = 0.0
    api_exec_count: int = 0
    llm_exec_time_sum: float = 0.0
    llm_exec_count: int = 0
    llm_prompt_count: int = 0
    model_init_time_sum: float = 0.0
    model_init_count: int = 0
    batch_start: Optional[float] = None
    worker_busy_time_by_id: Dict[str, float] = field(default_factory=dict)
    worker_meta_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    db_explain_calls: int = 0
    db_explain_by_query: Dict[str, _DBExplainAgg] = field(default_factory=dict)
    db_explain_relations: Dict[str, int] = field(default_factory=dict)
    db_explain_indexes: Dict[str, int] = field(default_factory=dict)
    node_metrics_enabled: bool = field(default_factory=lambda: _env_flag("HALO_METRICS_NODE_STATS"))
    node_metrics_by_node: Dict[str, _NodeMetricsAgg] = field(default_factory=dict)
    node_metadata: Dict[str, Dict[str, str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start_run(self, name: str, total_queries: int) -> None:
        with self.lock:
            self.run_name = name
            self.run_start = time.perf_counter()
            self.run_end = None
            self.total_queries = total_queries
            self.query_time_sum = 0.0
            self.query_count = 0
            self.db_exec_time_sum = 0.0
            self.db_exec_count = 0
            self.api_exec_time_sum = 0.0
            self.api_exec_count = 0
            self.llm_exec_time_sum = 0.0
            self.llm_exec_count = 0
            self.llm_prompt_count = 0
            self.model_init_time_sum = 0.0
            self.model_init_count = 0
            self.batch_start = None
            self.worker_busy_time_by_id.clear()
            self.worker_meta_by_id.clear()
            self.metadata.clear()
            self.db_explain_calls = 0
            self.db_explain_by_query.clear()
            self.db_explain_relations.clear()
            self.db_explain_indexes.clear()
            self.node_metrics_enabled = _env_flag("HALO_METRICS_NODE_STATS")
            self.node_metrics_by_node.clear()
            self.node_metadata.clear()

    def start_batch(self) -> None:
        with self.lock:
            self.batch_start = time.perf_counter()

    def end_batch(self) -> Optional[float]:
        with self.lock:
            if self.batch_start is None:
                return None
            duration = time.perf_counter() - self.batch_start
            self.batch_start = None
            return duration

    def record_query_latency(self, latency: float, count: int = 1) -> None:
        if latency < 0:
            return
        with self.lock:
            self.query_time_sum += latency * max(count, 1)
            self.query_count += max(count, 1)

    def record_query_execution_time(self, duration: float, count: int = 1) -> None:
        if duration < 0 or count <= 0:
            return
        with self.lock:
            self.db_exec_time_sum += duration
            self.db_exec_count += count

    def record_api_execution_time(self, duration: float, count: int = 1) -> None:
        if duration < 0 or count <= 0:
            return
        with self.lock:
            self.api_exec_time_sum += duration
            self.api_exec_count += count

    def record_llm_execution_time(
        self,
        duration: float,
        *,
        call_count: int = 1,
        prompt_count: int = 1,
    ) -> None:
        if duration < 0 or call_count <= 0:
            return
        prompts = max(prompt_count, 0)
        with self.lock:
            self.llm_exec_time_sum += duration
            self.llm_exec_count += call_count
            self.llm_prompt_count += prompts

    def record_model_init_time(self, duration: float, count: int = 1) -> None:
        if duration < 0 or count <= 0:
            return
        with self.lock:
            self.model_init_time_sum += duration
            self.model_init_count += count

    def record_worker_busy_time(
        self,
        worker_id: str | None,
        duration: float,
        *,
        kind: str | None = None,
    ) -> None:
        if not worker_id or duration < 0:
            return
        with self.lock:
            self.worker_busy_time_by_id[worker_id] = (
                self.worker_busy_time_by_id.get(worker_id, 0.0) + duration
            )
            if kind:
                meta = self.worker_meta_by_id.setdefault(worker_id, {})
                meta.setdefault("kind", kind)

    def add_metadata(self, key: str, value: Any) -> None:
        with self.lock:
            self.metadata[key] = value

    def record_db_explain(
        self,
        query_key: str,
        *,
        runtime_ms: float | None,
        planned_total_cost: float | None,
        shared_hit_blocks: int | None,
        shared_read_blocks: int | None,
        local_hit_blocks: int | None,
        local_read_blocks: int | None,
        relations: Dict[str, int] | None = None,
        indexes: Dict[str, int] | None = None,
    ) -> None:
        with self.lock:
            self.db_explain_calls += 1
            agg = self.db_explain_by_query.get(query_key)
            if agg is None:
                agg = _DBExplainAgg()
                self.db_explain_by_query[query_key] = agg
            agg.add(
                runtime_ms=runtime_ms,
                planned_total_cost=planned_total_cost,
                shared_hit_blocks=shared_hit_blocks,
                shared_read_blocks=shared_read_blocks,
                local_hit_blocks=local_hit_blocks,
                local_read_blocks=local_read_blocks,
            )
            if relations:
                for name, weight in relations.items():
                    self.db_explain_relations[name] = self.db_explain_relations.get(name, 0) + int(weight)
            if indexes:
                for name, weight in indexes.items():
                    self.db_explain_indexes[name] = self.db_explain_indexes.get(name, 0) + int(weight)

    def record_node_metrics(
        self,
        node_id: str,
        *,
        total_time: float,
        prepare_time: float,
        execute_time: float,
        count: int = 1,
        engine: str | None = None,
        model: str | None = None,
    ) -> None:
        if not node_id or count <= 0:
            return
        with self.lock:
            if not self.node_metrics_enabled:
                return
            agg = self.node_metrics_by_node.get(node_id)
            if agg is None:
                agg = _NodeMetricsAgg()
                self.node_metrics_by_node[node_id] = agg
            agg.add(
                total_time=total_time,
                prepare_time=prepare_time,
                execute_time=execute_time,
                count=count,
            )
            if engine or model:
                meta = self.node_metadata.get(node_id)
                if meta is None:
                    meta = {}
                    self.node_metadata[node_id] = meta
                if engine:
                    meta["engine"] = engine
                if model:
                    meta["model"] = model

    def end_run(self) -> None:
        with self.lock:
            if self.run_start is None:
                return
            self.run_end = time.perf_counter()

    def report(self) -> None:
        with self.lock:
            if self.run_start is None:
                return
            end_time = self.run_end or time.perf_counter()
            e2e = end_time - self.run_start
            avg_query_latency = (
                (self.query_time_sum / self.query_count) if self.query_count else None
            )
            throughput = (
                (self.total_queries / e2e) if e2e > 0 and self.total_queries else None
            )
            avg_db_exec = (
                (self.db_exec_time_sum / self.db_exec_count) if self.db_exec_count else None
            )
            avg_llm_exec = (
                (self.llm_exec_time_sum / self.llm_exec_count) if self.llm_exec_count else None
            )
            avg_llm_prompt = (
                (self.llm_exec_time_sum / self.llm_prompt_count)
                if self.llm_prompt_count
                else None
            )
            avg_model_init = (
                (self.model_init_time_sum / self.model_init_count)
                if self.model_init_count
                else None
            )
            explain_calls = int(self.db_explain_calls)
            explain_by_query = dict(self.db_explain_by_query)
            explain_relations = dict(self.db_explain_relations)
            explain_indexes = dict(self.db_explain_indexes)
            node_metrics_enabled = bool(self.node_metrics_enabled)
            node_metrics_by_node = dict(self.node_metrics_by_node)
            node_metadata = dict(self.node_metadata)

        print("\n===== Halo_dev Metrics =====")
        print(f"Run: {self.run_name or 'unknown'}")
        print(f"Total queries: {self.total_queries}")
        print(f"E2E latency: {e2e:.2f}s")
        if avg_query_latency is not None:
            print(f"Avg per-query latency: {avg_query_latency:.2f}s")
        if throughput is not None:
            print(f"Throughput: {throughput:.2f} query/s")
        if self.db_exec_count:
            db_avg_str = f"{avg_db_exec:.2f}s" if avg_db_exec is not None else "n/a"
            print(
                f"DB queries: count={self.db_exec_count} "
                f"total={self.db_exec_time_sum:.2f}s "
                f"avg={db_avg_str}"
            )
        if self.api_exec_count:
            api_avg = (
                (self.api_exec_time_sum / self.api_exec_count)
                if self.api_exec_count
                else None
            )
            api_avg_str = f"{api_avg:.2f}s" if api_avg is not None else "n/a"
            print(
                f"API: count={self.api_exec_count} "
                f"total={self.api_exec_time_sum:.2f}s "
                f"avg={api_avg_str}"
            )
        if self.llm_exec_count:
            llm_avg_call_str = f"{avg_llm_exec:.2f}s" if avg_llm_exec is not None else "n/a"
            llm_avg_prompt_str = (
                f"{avg_llm_prompt:.2f}s" if avg_llm_prompt is not None else "n/a"
            )
            print(
                f"LLM exec: calls={self.llm_exec_count} prompts={self.llm_prompt_count} "
                f"total={self.llm_exec_time_sum:.2f}s "
                f"avg_call={llm_avg_call_str} avg_prompt={llm_avg_prompt_str}"
            )
        init_avg_str = f"{avg_model_init:.2f}s" if avg_model_init is not None else "n/a"
        print(
            f"Model init: count={self.model_init_count} "
            f"total={self.model_init_time_sum:.2f}s avg={init_avg_str}"
        )
        if self.metadata:
            print("Metadata:")
            for key, value in self.metadata.items():
                print(f"  - {key}: {value}")

        if self.worker_busy_time_by_id:
            print("\n===== Worker Utilization =====")
            workers = sorted(
                self.worker_busy_time_by_id.items(), key=lambda item: item[1], reverse=True
            )
            for worker_id, busy in workers:
                meta = self.worker_meta_by_id.get(worker_id) or {}
                kind = meta.get("kind")
                util = (busy / e2e) if e2e > 0 else None
                util_str = f"{util * 100:.1f}%" if util is not None else "n/a"
                kind_str = kind if kind else "n/a"
                print(
                    f"  - {worker_id}: kind={kind_str} busy={busy:.2f}s util={util_str}"
                )
            total_busy = sum(busy for _, busy in workers)
            total_workers = len(workers)
            total_util = (
                (total_busy / (e2e * total_workers)) if e2e > 0 and total_workers else None
            )
            total_util_str = f"{total_util * 100:.1f}%" if total_util is not None else "n/a"
            print(
                f"Total: workers={total_workers} busy={total_busy:.2f}s util={total_util_str}"
            )
            kind_busy: Dict[str, float] = {}
            kind_counts: Dict[str, int] = {}
            for worker_id, busy in workers:
                kind = (self.worker_meta_by_id.get(worker_id) or {}).get("kind") or "unknown"
                kind_busy[kind] = kind_busy.get(kind, 0.0) + busy
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            for kind in sorted(kind_busy.keys()):
                count = kind_counts.get(kind, 0)
                util = (kind_busy[kind] / (e2e * count)) if e2e > 0 and count else None
                util_str = f"{util * 100:.1f}%" if util is not None else "n/a"
                print(
                    f"  kind={kind} workers={count} busy={kind_busy[kind]:.2f}s util={util_str}"
                )

        if node_metrics_enabled and node_metrics_by_node:
            print("\n===== Node Metrics =====")
            def _node_sort(item: tuple[str, _NodeMetricsAgg]) -> float:
                return item[1].total_time_sum

            for node_id, agg in sorted(node_metrics_by_node.items(), key=_node_sort, reverse=True):
                avg_latency = (agg.execute_time_sum / agg.calls) if agg.calls else None
                throughput = (agg.calls / agg.execute_time_sum) if agg.execute_time_sum > 0 else None
                avg_latency_str = f"{avg_latency:.2f}s" if avg_latency is not None else "n/a"
                throughput_str = f"{throughput:.2f} query/s" if throughput is not None else "n/a"
                meta = node_metadata.get(node_id) or {}
                info = []
                engine = meta.get("engine")
                model = meta.get("model")
                if engine:
                    info.append(f"engine={engine}")
                if model:
                    info.append(f"model={model}")
                info_str = f" ({' '.join(info)})" if info else ""
                print(
                    f"  - {node_id}: count={agg.calls} "
                    f"total={agg.total_time_sum:.2f}s "
                    f"prepare={agg.prepare_time_sum:.2f}s "
                    f"execute={agg.execute_time_sum:.2f}s "
                    f"avg_latency={avg_latency_str} "
                    f"throughput={throughput_str}{info_str}"
                )

        if explain_calls > 0:
            # Snapshot-style summary (kept outside lock from here on).
            print("\n===== DB EXPLAIN ANALYZE (BUFFERS) =====")
            print(f"Explained statements: {explain_calls}")
            unique = len(explain_by_query)
            print(f"Unique query keys: {unique}")

            total_hit = sum(agg.shared_hit_blocks_sum for agg in explain_by_query.values())
            total_read = sum(agg.shared_read_blocks_sum for agg in explain_by_query.values())
            denom = total_hit + total_read
            hit_rate = (total_hit / denom) if denom > 0 else None
            print(f"Shared buffers: hit={total_hit} read={total_read} hit_rate={(hit_rate if hit_rate is not None else 'n/a')}")

            total_cost_sum = sum(agg.planned_total_cost_sum for agg in explain_by_query.values())
            total_cost_count = sum(agg.planned_total_cost_count for agg in explain_by_query.values())
            total_cost_min = min(
                (agg.planned_total_cost_min for agg in explain_by_query.values() if agg.planned_total_cost_min is not None),
                default=None,
            )
            total_cost_max = max(
                (agg.planned_total_cost_max for agg in explain_by_query.values() if agg.planned_total_cost_max is not None),
                default=None,
            )
            if total_cost_count > 0:
                print(
                    "Planned total cost: "
                    f"avg={(total_cost_sum / total_cost_count):.2f} "
                    f"min={(f'{total_cost_min:.2f}' if total_cost_min is not None else 'n/a')} "
                    f"max={(f'{total_cost_max:.2f}' if total_cost_max is not None else 'n/a')}"
                )

            def avg_runtime(agg: _DBExplainAgg) -> float:
                return (agg.runtime_ms_sum / agg.calls) if agg.calls else 0.0
            def avg_cost(agg: _DBExplainAgg) -> float | None:
                if agg.planned_total_cost_count <= 0:
                    return None
                return agg.planned_total_cost_sum / agg.planned_total_cost_count

            top_n = 10
            slowest = sorted(explain_by_query.items(), key=lambda kv: avg_runtime(kv[1]), reverse=True)[:top_n]
            print(f"Top {min(top_n, len(slowest))} by avg runtime (ms):")
            for key, agg in slowest:
                avg_ms = avg_runtime(agg)
                avg_cost_value = avg_cost(agg)
                print(
                    f"  - {key}: calls={agg.calls} avg_ms={avg_ms:.2f} "
                    f"avg_cost={(f'{avg_cost_value:.2f}' if avg_cost_value is not None else 'n/a')} "
                    f"shared_hit={agg.shared_hit_blocks_sum} shared_read={agg.shared_read_blocks_sum}"
                )

            top_read = sorted(explain_by_query.items(), key=lambda kv: kv[1].shared_read_blocks_sum, reverse=True)[:top_n]
            print(f"Top {min(top_n, len(top_read))} by shared read blocks:")
            for key, agg in top_read:
                avg_ms = avg_runtime(agg)
                print(
                    f"  - {key}: calls={agg.calls} avg_ms={avg_ms:.2f} shared_read={agg.shared_read_blocks_sum}"
                )

            if explain_relations:
                top_rel = sorted(explain_relations.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
                print(f"Top {min(top_n, len(top_rel))} relations by footprint:")
                for name, weight in top_rel:
                    print(f"  - {name}: {weight}")
            if explain_indexes:
                top_idx = sorted(explain_indexes.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
                print(f"Top {min(top_n, len(top_idx))} indexes by footprint:")
                for name, weight in top_idx:
                    print(f"  - {name}: {weight}")
            print("======================================\n")

        print("============================\n")


_TRACKER = MetricsTracker()
atexit.register(_TRACKER.report)


def start_run(name: str, total_queries: int) -> None:
    _TRACKER.start_run(name, total_queries)


def start_batch() -> None:
    _TRACKER.start_batch()


def end_batch() -> Optional[float]:
    return _TRACKER.end_batch()


def record_query_latency(latency: float, count: int = 1) -> None:
    _TRACKER.record_query_latency(latency, count)


def record_query_execution_time(duration: float, count: int = 1) -> None:
    _TRACKER.record_query_execution_time(duration, count)


def record_api_execution_time(duration: float, count: int = 1) -> None:
    _TRACKER.record_api_execution_time(duration, count)


def record_http_execution_time(duration: float, count: int = 1) -> None:
    _TRACKER.record_api_execution_time(duration, count)


def record_llm_execution_time(
    duration: float,
    *,
    call_count: int = 1,
    prompt_count: int = 1,
) -> None:
    _TRACKER.record_llm_execution_time(duration, call_count=call_count, prompt_count=prompt_count)


def record_model_init_time(duration: float, count: int = 1) -> None:
    _TRACKER.record_model_init_time(duration, count=count)


def record_worker_busy_time(
    worker_id: str | None,
    duration: float,
    *,
    kind: str | None = None,
) -> None:
    _TRACKER.record_worker_busy_time(worker_id, duration, kind=kind)


def add_metadata(key: str, value: Any) -> None:
    _TRACKER.add_metadata(key, value)


def record_db_explain(
    query_key: str,
    *,
    runtime_ms: float | None,
    planned_total_cost: float | None,
    shared_hit_blocks: int | None,
    shared_read_blocks: int | None,
    local_hit_blocks: int | None,
    local_read_blocks: int | None,
    relations: Dict[str, int] | None = None,
    indexes: Dict[str, int] | None = None,
) -> None:
    _TRACKER.record_db_explain(
        query_key,
        runtime_ms=runtime_ms,
        planned_total_cost=planned_total_cost,
        shared_hit_blocks=shared_hit_blocks,
        shared_read_blocks=shared_read_blocks,
        local_hit_blocks=local_hit_blocks,
        local_read_blocks=local_read_blocks,
        relations=relations,
        indexes=indexes,
    )


def record_node_metrics(
    node_id: str,
    *,
    total_time: float,
    prepare_time: float,
    execute_time: float,
    count: int = 1,
    engine: str | None = None,
    model: str | None = None,
) -> None:
    _TRACKER.record_node_metrics(
        node_id,
        total_time=total_time,
        prepare_time=prepare_time,
        execute_time=execute_time,
        count=count,
        engine=engine,
        model=model,
    )


def end_run() -> None:
    _TRACKER.end_run()
