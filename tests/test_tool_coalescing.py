"""Tests for tool-operator coalescing (HTTP / processor) and class-aware CPU placement.

Run: python -m pytest tests/test_tool_coalescing.py -q   (no GPU or Postgres needed)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import halo.profiler as profiler  # noqa: E402
from halo import GraphOptimizer, GraphTemplateParser  # noqa: E402
from halo.executor import HTTPNodeExecutor, ProcessorNodeExecutor, sql_coalescible  # noqa: E402
from halo.models import Node  # noqa: E402

# Planning must not sleep in HTTP profiling.
profiler.GraphProfiler._profile_http_nodes = lambda self, graph, contexts: ({}, {})


def _http_node(coalesce: bool, **raw) -> Node:
    body = {"id": "fetch", "type": "http", "engine": "http", "sleep_ms": 1, "inputs": ["ticker"],
            "outputs": ["resp"], **raw}
    if coalesce:
        body["coalesce"] = True
    return Node(id="fetch", type="http", engine="http", model=None, inputs=("ticker",),
                outputs=("resp",), db_queries=(), raw=body)


def test_http_coalescing_executes_once_per_distinct_request():
    executor = HTTPNodeExecutor(http_concurrency=4)
    contexts = [{"ticker": t} for t in ("AAPL", "MSFT", "AAPL", "AAPL", "MSFT")]
    outputs = executor.execute_batch(_http_node(coalesce=True), contexts)
    stats = executor.consume_stats()
    assert stats["api_calls"] == 2  # two distinct requests, five logical calls
    assert len(outputs) == 5
    assert outputs[0] == outputs[2] == outputs[3]
    assert outputs[0] is not outputs[2]  # each query gets its own copy


def test_http_coalescing_is_opt_in():
    executor = HTTPNodeExecutor(http_concurrency=4)
    contexts = [{"ticker": "AAPL"}] * 3
    executor.execute_batch(_http_node(coalesce=False), contexts)
    assert executor.consume_stats()["api_calls"] == 3


def test_http_signature_includes_rendered_request_fields():
    executor = HTTPNodeExecutor(http_concurrency=1)
    node = _http_node(coalesce=True, url="https://example.org/q?region={{ region }}")
    contexts = [{"ticker": "AAPL", "region": r} for r in ("ASIA", "EUROPE", "ASIA")]
    executor.execute_batch(node, contexts)
    assert executor.consume_stats()["api_calls"] == 2


def test_processor_coalescing_runs_once_per_distinct_input(monkeypatch):
    import halo.executor as executor_mod

    calls = []

    def fake_run(node, context):
        calls.append(context["text"])
        return {"out": context["text"].upper()}

    monkeypatch.setattr(executor_mod, "run_processor_node", fake_run)
    node = Node(id="extract", type="processor", engine=None, model=None, inputs=("text",),
                outputs=("out",), db_queries=(), raw={"processor": "x", "coalesce": True})
    outputs = ProcessorNodeExecutor().execute_batch(node, [{"text": "a"}, {"text": "b"}, {"text": "a"}])
    assert calls == ["a", "b"]
    assert [o["out"] for o in outputs] == ["A", "B", "A"]


def test_sql_coalescing_eligibility():
    assert sql_coalescible("SELECT title FROM movies WHERE id = :id")
    assert sql_coalescible("WITH t AS (SELECT 1) SELECT * FROM t -- now() in a comment")
    assert sql_coalescible("SELECT * FROM notes WHERE body = 'insert into x'")
    assert not sql_coalescible("INSERT INTO answers (q) VALUES (:q)")
    assert not sql_coalescible("UPDATE stock SET qty = qty - 1 WHERE id = :id")
    assert not sql_coalescible("SELECT * FROM district WHERE id = :id FOR UPDATE")
    assert not sql_coalescible("SELECT random() AS r")
    assert not sql_coalescible("SELECT * FROM orders WHERE ts > now() - interval '1 day'")
    assert not sql_coalescible("WITH d AS (DELETE FROM q RETURNING *) SELECT * FROM d")


MIXED_TEMPLATE = textwrap.dedent(
    """
    graph:
      name: mixed_cpu_classes
      nodes:
        - id: user_input
          type: input
          outputs: [user_query]
        - id: lookup
          type: inference
          engine: vllm
          model: meta-llama/Llama-3.2-3B-Instruct
          system_prompt: "Answer."
          inputs: [user_query]
          outputs: [answer]
          db_queries:
            - name: q
              sql: "SELECT 1 WHERE :x = :x"
              parameters: {x: user_query}
        - id: news
          type: http
          engine: http
          sleep_s: 2
          inputs: [user_query]
          outputs: [news]
        - id: summarize
          type: inference
          engine: vllm
          model: meta-llama/Llama-3.1-8B-Instruct
          system_prompt: "Summarize."
          inputs: [answer, news]
          outputs: [summary]
      edges:
        - {from: user_input, to: lookup, mapping: {user_query: "{{ user_query }}"}}
        - {from: user_input, to: news, mapping: {user_query: "{{ user_query }}"}}
        - {from: lookup, to: summarize, mapping: {answer: "{{ answer }}"}}
        - {from: news, to: summarize, mapping: {news: "{{ news }}"}}
    """
)


def test_dp_places_http_and_sql_on_isolated_cpu_workers(tmp_path):
    path = tmp_path / "mixed.yaml"
    path.write_text(MIXED_TEMPLATE)
    graph = GraphTemplateParser(str(path)).parse()
    plan = GraphOptimizer(num_gpus=2, num_cpu_workers=2, scheduler_mode="dp", plan_mode="default").build_plan(
        graph, sample_contexts=[{"user_query": "q"}], input_query_count=8
    )
    cpu_tasks = [t for t in plan.tasks if plan.workers[t.worker_id].kind == "cpu"]
    http_workers = {t.worker_id for t in cpu_tasks if graph.nodes[t.node_id].engine == "http"}
    db_workers = {t.worker_id for t in cpu_tasks if graph.nodes[t.node_id].engine == "db"}
    assert http_workers and db_workers
    assert not (http_workers & db_workers)


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- embedded tool calls, streaming re-optimization, auto micro-batch size ----

EMBEDDED_TEMPLATE = textwrap.dedent(
    """
    graph:
      name: embedded_tools
      nodes:
        - id: user_input
          type: input
          outputs: [user_query]
        - id: analyst
          type: inference
          engine: vllm
          model: meta-llama/Llama-3.2-3B-Instruct
          system_prompt: "Analyze."
          inputs: [user_query]
          outputs: [report]
          tool_calls:
            - {name: news, kind: http, sleep_ms: 1, coalesce: true}
            - {name: extract, kind: processor, processor: regex_param_extractor, config: {}, post_llm: true}
      edges:
        - {from: user_input, to: analyst, mapping: {user_query: "{{ user_query }}"}}
    """
)


def test_parser_extracts_embedded_tool_calls(tmp_path):
    path = tmp_path / "embedded.yaml"
    path.write_text(EMBEDDED_TEMPLATE)
    graph = GraphTemplateParser(str(path)).parse()
    pre, post = graph.nodes["analyst__pre__news"], graph.nodes["analyst__post__extract"]
    assert pre.engine == "http" and pre.outputs == ("news",) and pre.raw.get("coalesce") is True
    assert post.type == "processor" and post.raw["processor"] == "regex_param_extractor"
    assert "news" in graph.nodes["analyst"].inputs and "tool_calls" not in graph.nodes["analyst"].raw
    pairs = {(e.source, e.target) for e in graph.edges}
    assert ("user_input", "analyst__pre__news") in pairs
    assert ("analyst__pre__news", "analyst") in pairs
    assert ("analyst", "analyst__post__extract") in pairs
    plan = GraphOptimizer(num_gpus=1, scheduler_mode="dp", plan_mode="default").build_plan(
        graph, sample_contexts=[{"user_query": "q"}], input_query_count=4
    )
    assert {t.node_id for t in plan.tasks} >= {"analyst", "analyst__pre__news", "analyst__post__extract"}


def test_streaming_session_replans_only_when_template_changes():
    from halo.models import GraphSpec
    from halo.streaming import StreamingSession

    class FakeOptimizer:
        def __init__(self):
            self.calls = []

        def build_plan(self, graph, sample_contexts, input_query_count):
            self.calls.append((graph.name, input_query_count))
            return f"plan:{graph.name}"

    class FakeProcessor:
        def __init__(self):
            self.batches = []

        def run_batch(self, plan, graph, contexts):
            self.batches.append((plan, len(contexts)))
            return [{"answer": ctx["q"]} for ctx in contexts]

    t1 = GraphSpec(name="t1", description="", nodes={}, edges=[])
    t2 = GraphSpec(name="t2", description="", nodes={}, edges=[])
    opt, proc = FakeOptimizer(), FakeProcessor()
    session = StreamingSession(opt, proc, max_batch=2)
    tickets = [session.submit(t, {"q": i}) for i, t in enumerate([t1, t1, t1, t2, t2, t1])]
    results = session.flush()
    assert [results[t]["answer"] for t in tickets] == [0, 1, 2, 3, 4, 5]
    # Mini-batches respect template boundaries and the size cap.
    assert proc.batches == [("plan:t1", 2), ("plan:t1", 1), ("plan:t2", 2), ("plan:t1", 1)]
    # The optimizer runs once per template; returning to t1 reuses its plan.
    assert [name for name, _ in opt.calls] == ["t1", "t2"] and session.optimizer_runs == 2


def test_auto_micro_batch_size_is_half_the_batch():
    from halo.processors.multi_process import MultiProcessGraphProcessor

    proc = MultiProcessGraphProcessor(max_batch_size="auto")
    for n, expected in ((256, 128), (1024, 512), (1, 1), (7, 4)):
        proc._resolve_batch_size(n)
        assert proc.max_batch_size == expected
    fixed = MultiProcessGraphProcessor(max_batch_size=64)
    fixed._resolve_batch_size(1024)
    assert fixed.max_batch_size == 64


def test_coalesced_results_are_reused_across_micro_batches():
    executor = HTTPNodeExecutor(http_concurrency=2)
    node = _http_node(coalesce=True)
    executor.execute_batch(node, [{"ticker": "AAPL"}, {"ticker": "MSFT"}])
    assert executor.consume_stats()["api_calls"] == 2
    executor.execute_batch(node, [{"ticker": "AAPL"}, {"ticker": "NVDA"}])
    assert executor.consume_stats()["api_calls"] == 1  # AAPL served from the result cache


def test_processor_nodes_are_profiled_into_dp_cost(monkeypatch):
    import halo.node_processors as node_processors
    from halo.optimizers.dp import DPSolver

    def slow_rule(context, config, node):
        import time

        time.sleep(0.01)
        return {"params": context.get("user_query")}

    monkeypatch.setitem(node_processors._PROCESSOR_REGISTRY, "slow_rule", slow_rule)
    node = Node(id="extract", type="processor", engine=None, model=None, inputs=("user_query",),
                outputs=("params",), db_queries=(), raw={"processor": "slow_rule"})
    graph = type("G", (), {"nodes": {"extract": node}})()
    profile = profiler.GraphProfiler().profile_graph(graph, [{"user_query": "a"}, {"user_query": "b"}],
                                                     include_sql=False, include_http=False)
    assert profile.processor_latencies_s["extract"] >= 0.01
    assert profile.processor_samples["extract"] == 2
    solver = DPSolver.__new__(DPSolver)
    solver._processor_latency_s = dict(profile.processor_latencies_s)
    assert solver._processor_cost(node) == profile.processor_latencies_s["extract"]
