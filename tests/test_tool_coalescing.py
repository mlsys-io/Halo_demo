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
_REAL_PROFILE_HTTP = profiler.GraphProfiler._profile_http_nodes
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


def test_streaming_session_replans_every_mini_batch():
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
    # The optimizer re-runs at every mini-batch boundary, sized to that mini-batch.
    assert opt.calls == [("t1", 2), ("t1", 1), ("t2", 2), ("t1", 1)] and session.optimizer_runs == 4


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
    assert solver._fixed_cpu_cost(node) == profile.processor_latencies_s["extract"]


# ---- review fixes: SQL safety, cache keys, parser inputs, streaming, profiling ----


class _StockDB:
    """Fake DB with one counter: UPDATE decrements it, SELECT reads it."""

    def __init__(self):
        self.qty = 9
        self.calls = []

    def run(self, query, context, *, node_id=None):
        self.calls.append(query.name)
        if query.sql.lstrip().upper().startswith("UPDATE"):
            self.qty -= 1
            return {"query": query.name, "rows": []}
        return {"query": query.name, "rows": [{"v": f"{query.name}:{self.qty}"}]}


def _db_node(node_id, *queries):
    from halo.models import DBQuery

    dbq = tuple(DBQuery(name=n, sql=sql, parameters={"id": "id"}) for n, sql in queries)
    return Node(id=node_id, type="db_query", engine="db", model=None, inputs=("id",), outputs=(),
                db_queries=dbq, raw={})


def test_sql_classifier_handles_literals_and_side_effects():
    assert not sql_coalescible("WITH c AS (SELECT * FROM s WHERE sep = '--') UPDATE t SET q = 1 RETURNING q")
    assert not sql_coalescible("SELECT COALESCE(note, '--'), nextval('seq') FROM t")
    assert not sql_coalescible("SELECT * INTO backup FROM t")
    assert not sql_coalescible("SELECT pg_advisory_lock(1)")
    assert sql_coalescible("SELECT * FROM t WHERE d < :now")
    assert sql_coalescible('SELECT "update" FROM t')
    assert sql_coalescible("(SELECT a FROM t) UNION (SELECT a FROM u)")


def test_write_invalidates_cached_reads():
    from halo.executor import DBNodeExecutor

    db = _StockDB()
    executor = DBNodeExecutor(db_executor=db, db_concurrency=1)
    node = _db_node("stock", ("dec", "UPDATE stock SET qty = qty - 1 WHERE id = :id"),
                    ("read", "SELECT qty FROM stock WHERE id = :id"))
    seen = [executor.execute_batch(node, [{"id": 7}])[0]["read"]["rows"][0]["v"] for _ in range(3)]
    assert seen == ["read:8", "read:7", "read:6"]


def test_db_cache_distinguishes_sql_with_same_query_name():
    from halo.executor import DBNodeExecutor

    executor = DBNodeExecutor(db_executor=_StockDB(), db_concurrency=1)
    a = executor.execute_batch(_db_node("movie", ("lookup", "SELECT title FROM movies WHERE id = :id")), [{"id": 1}])
    db = executor.db_executor
    executor.execute_batch(_db_node("actor", ("lookup", "SELECT name FROM actors WHERE id = :id")), [{"id": 1}])
    assert db.calls == ["lookup", "lookup"] and a


def test_coalescing_key_covers_operator_spec_and_nested_requests(monkeypatch):
    import halo.executor as executor_mod

    monkeypatch.setattr(executor_mod, "run_processor_node",
                        lambda node, ctx: {node.outputs[0]: node.raw["config"]["tag"] + ctx["x"]})
    executor = ProcessorNodeExecutor()

    def proc(tag, out):
        return Node(id="extract", type="processor", engine=None, model=None, inputs=("x",), outputs=(out,),
                    db_queries=(), raw={"processor": "p", "config": {"tag": tag}, "coalesce": True})

    assert executor.execute_batch(proc("A", "a"), [{"x": "1"}])[0] == {"a": "A1"}
    assert executor.execute_batch(proc("B", "b"), [{"x": "1"}])[0] == {"b": "B1"}  # not A's cached result

    http = HTTPNodeExecutor(http_concurrency=1)
    node = _http_node(coalesce=True, headers={"Authorization": "Bearer {{ token }}"})
    http.execute_batch(node, [{"ticker": "AAPL", "token": t} for t in ("alice", "bob")])
    assert http.consume_stats()["api_calls"] == 2


def test_coalesced_outputs_are_independent_copies(monkeypatch):
    import halo.executor as executor_mod

    monkeypatch.setattr(executor_mod, "run_processor_node", lambda node, ctx: {"out": {"v": ctx["x"]}})
    node = Node(id="p", type="processor", engine=None, model=None, inputs=("x",), outputs=("out",),
                db_queries=(), raw={"processor": "p", "coalesce": "true"})
    outs = ProcessorNodeExecutor().execute_batch(node, [{"x": 1}, {"x": 1}])
    assert outs[0]["out"] == outs[1]["out"] and outs[0]["out"] is not outs[1]["out"]
    off = Node(id="q", type="processor", engine=None, model=None, inputs=("x",), outputs=("out",),
               db_queries=(), raw={"processor": "p", "coalesce": "false"})
    from halo.executor import coalescing_enabled
    assert not coalescing_enabled(off)


TOOLS_WITH_DB_TEMPLATE = textwrap.dedent(
    """
    graph:
      name: tools_with_db
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
          db_queries:
            - {name: q, sql: "SELECT 1 WHERE :x = :x", parameters: {x: user_query}}
          tool_calls:
            - {name: news, kind: http, sleep_ms: 1}
            - {name: extract, kind: processor, processor: regex_param_extractor, config: {}, post_llm: true}
      edges:
        - {from: user_input, to: analyst, mapping: {user_query: "{{ user_query }}"}}
    """
)


def test_tool_calls_get_own_inputs_and_no_false_db_dependency(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(TOOLS_WITH_DB_TEMPLATE)
    graph = GraphTemplateParser(str(path)).parse()
    pre, post = graph.nodes["analyst__pre__news"], graph.nodes["analyst__post__extract"]
    assert pre.inputs == ("user_query",)
    assert post.inputs[0] == "report"  # processors reading their first input see the LLM output
    parents = {e.source for e in graph.edges if e.target == pre.id}
    assert parents == {"user_input"}
    assert set(graph.nodes["analyst"].inputs) >= {"user_query", "q", "news"}


def test_streaming_distinguishes_templates_with_same_name_and_ids():
    from halo.models import Edge, GraphSpec
    from halo.streaming import StreamingSession

    def graph(proc):
        nodes = {"t": Node(id="t", type="processor", engine=None, model=None, inputs=("x",), outputs=("y",),
                           db_queries=(), raw={"processor": proc})}
        return GraphSpec(name="unnamed", description="", nodes=nodes, edges=[])

    class Opt:
        def build_plan(self, g, **_):
            return g.nodes["t"].raw["processor"]

    class Proc:
        def run_batch(self, plan, g, contexts):
            return [{"plan": plan} for _ in contexts]

    session = StreamingSession(Opt(), Proc())
    session.submit(graph("upper"), {"x": "a"})
    session.submit(graph("reverse"), {"x": "b"})
    results = session.flush()
    assert [results[0]["plan"], results[1]["plan"]] == ["upper", "reverse"]
    assert session.optimizer_runs == 2


def test_streaming_failure_keeps_finished_and_requeues_rest():
    from halo.models import GraphSpec
    from halo.streaming import StreamingSession

    def graph(name):
        return GraphSpec(name=name, description="", nodes={}, edges=[])

    class Opt:
        def build_plan(self, g, **_):
            return g.name

    class Proc:
        runs = []

        def run_batch(self, plan, g, contexts):
            self.runs.append(plan)
            if plan == "bad":
                raise RuntimeError("boom")
            return [{"plan": plan} for _ in contexts]

    proc = Proc()
    session = StreamingSession(Opt(), proc)
    for name in ("ok1", "bad", "ok2"):
        session.submit(graph(name), {})
    try:
        session.flush()
        raise AssertionError("flush should re-raise")
    except RuntimeError as exc:
        assert exc.failed_tickets == [1]
    results = session.flush()
    assert sorted(results) == [0, 2] and proc.runs == ["ok1", "bad", "ok2"]  # no segment re-run


def test_profiling_coalesced_http_twice_keeps_real_latency():
    graph = type("G", (), {})()
    node = _http_node(coalesce=True, sleep_ms=50)
    graph.nodes = {"fetch": node}
    prof = profiler.GraphProfiler.__new__(profiler.GraphProfiler)
    prof._http_executor = HTTPNodeExecutor(http_concurrency=1)
    first, _ = _REAL_PROFILE_HTTP(prof, graph, [{"ticker": "AAPL"}])
    second, _ = _REAL_PROFILE_HTTP(prof, graph, [{"ticker": "AAPL"}])
    assert first["fetch"] > 0 and second["fetch"] > 0


def test_cpu_nodes_round_robin_without_cost_estimates(tmp_path):
    queries = "".join(
        f'        - {{name: q{i}, sql: "SELECT {i} WHERE :x = :x", parameters: {{x: user_query}}}}\n'
        for i in range(4)
    )
    path = tmp_path / "fan.yaml"
    path.write_text(
        "graph:\n"
        "  name: fan\n"
        "  nodes:\n"
        "    - {id: user_input, type: input, outputs: [user_query]}\n"
        "    - id: llm\n"
        "      type: inference\n"
        "      engine: vllm\n"
        "      model: meta-llama/Llama-3.2-3B-Instruct\n"
        "      inputs: [user_query]\n"
        "      outputs: [a]\n"
        "      db_queries:\n" + queries +
        "  edges:\n"
        "    - {from: user_input, to: llm, mapping: {user_query: '{{ user_query }}'}}\n"
    )
    graph = GraphTemplateParser(str(path)).parse()
    plan = GraphOptimizer(num_gpus=1, num_cpu_workers=4, scheduler_mode="dp", plan_mode="baseline").build_plan(
        graph, sample_contexts=[{"user_query": "q"}], input_query_count=8
    )
    cpu = {t.worker_id for t in plan.tasks if plan.workers[t.worker_id].kind == "cpu"}
    assert len(cpu) == 4


def test_http_node_issues_live_request_and_extracts_response(monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    monkeypatch.setenv("no_proxy", "*")
    seen = []

    class ChatCompletions(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body["messages"][0]["content"])
            data = json.dumps({"choices": [{"message": {"content": f"re: {seen[-1]}"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), ChatCompletions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        raw = {"url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "method": "POST",
               "json": {"model": "m", "messages": [{"role": "user", "content": "{{ q }}"}]},
               "response_path": "choices.0.message.content"}
        node = Node(id="api_llm", type="http", engine="http", model=None, inputs=("q",), outputs=("answer",),
                    db_queries=(), raw=raw)
        executor = HTTPNodeExecutor(http_concurrency=2)
        outputs = executor.execute_batch(node, [{"q": "a"}, {"q": "b"}])
        assert outputs == [{"answer": "re: a"}, {"answer": "re: b"}]  # an API-served LLM as an HTTP operator
        assert sorted(seen) == ["a", "b"] and executor.consume_stats()["api_calls"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_http_node_with_declared_latency_is_simulated():
    node = _http_node(coalesce=False, url="http://127.0.0.1:9/never-contacted")
    outputs = HTTPNodeExecutor(http_concurrency=1).execute_batch(node, [{"ticker": "AAPL"}])
    assert outputs[0]["resp"]["status"] == "ok"  # sleep_ms keeps latency injection


LOOP_TEMPLATE = textwrap.dedent(
    """
    graph:
      name: critic_loop
      nodes:
        - {id: user_input, type: input, outputs: [user_query]}
        - {id: writer, type: inference, engine: vllm, model: meta-llama/Llama-3.2-3B-Instruct,
           inputs: [user_query, critique], outputs: [draft]}
        - {id: critic, type: inference, engine: vllm, model: meta-llama/Llama-3.2-3B-Instruct,
           inputs: [draft], outputs: [critique]}
        - {id: editor, type: inference, engine: vllm, model: meta-llama/Llama-3.2-3B-Instruct,
           inputs: [draft], outputs: [final_answer]}
      edges:
        - {from: user_input, to: writer}
        - {from: writer, to: critic}
        - {from: critic, to: writer}
        - {from: critic, to: editor}
      loops:
        - {nodes: [writer, critic], max_iterations: 3}
    """
)


def test_parser_unrolls_bounded_loops(tmp_path):
    path = tmp_path / "loop.yaml"
    path.write_text(LOOP_TEMPLATE)
    graph = GraphTemplateParser(str(path)).parse()
    assert set(graph.nodes) == {"user_input", "writer", "critic", "writer__iter1", "critic__iter1",
                                "writer__iter2", "critic__iter2", "editor"}
    assert graph.nodes["critic__iter2"].raw["id"] == "critic__iter2"
    assert {(e.source, e.target) for e in graph.edges} == {
        ("user_input", "writer"), ("user_input", "writer__iter1"), ("user_input", "writer__iter2"),
        ("writer", "critic"), ("writer__iter1", "critic__iter1"), ("writer__iter2", "critic__iter2"),
        ("critic", "writer__iter1"), ("critic__iter1", "writer__iter2"), ("critic__iter2", "editor"),
    }
    plan = GraphOptimizer(num_gpus=1, scheduler_mode="dp", plan_mode="default").build_plan(
        graph, sample_contexts=[{"user_query": "q"}], input_query_count=4
    )
    assert {t.node_id for t in plan.tasks} >= set(graph.nodes) - {"user_input"}


def test_tool_backpressure_caps_inflight_queries_per_cpu_worker(tmp_path, monkeypatch):
    from halo.processors.multi_process import MultiProcessGraphProcessor

    monkeypatch.setenv("HALO_MONITOR_ENABLE", "0")
    path = tmp_path / "cpu_only.yaml"
    path.write_text(
        "graph:\n"
        "  name: cpu_only\n"
        "  nodes:\n"
        "    - {id: user_input, type: input, outputs: [user_query]}\n"
        "    - {id: extract, type: processor, processor: regex_param_extractor, config: {},"
        " inputs: [user_query], outputs: [params]}\n"
        "  edges:\n"
        "    - {from: user_input, to: extract, mapping: {user_query: '{{ user_query }}'}}\n"
    )
    graph = GraphTemplateParser(str(path)).parse()
    plan = GraphOptimizer(num_gpus=1, num_cpu_workers=1, scheduler_mode="dp", plan_mode="baseline").build_plan(
        graph, sample_contexts=[{"user_query": "q"}], input_query_count=10
    )
    sizes = []
    original = ProcessorNodeExecutor.execute_batch

    def recording(self, node, contexts):
        sizes.append(len(contexts))
        return original(self, node, contexts)

    monkeypatch.setattr(ProcessorNodeExecutor, "execute_batch", recording)
    contexts = [{"user_query": f"q{i}"} for i in range(10)]
    MultiProcessGraphProcessor(max_batch_size=None, max_tool_inflight=3).run_batch(plan, graph, contexts)
    assert sizes and max(sizes) <= 3 and sum(sizes) == 10
