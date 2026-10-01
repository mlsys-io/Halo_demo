"""SGLang engine support. Runs without a GPU or sglang: a fake ``sglang`` package is injected.

    python -m pytest tests/test_sglang_engine.py      (or: python tests/test_sglang_engine.py)
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import queue
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from halo import GraphOptimizer, GraphTemplateParser, MultiProcessGraphProcessor, SerialGraphProcessor  # noqa: E402
from halo import engines  # noqa: E402
from halo.engines import SGLangEngine, make_llm_provider, make_sglang_provider, make_vllm_provider  # noqa: E402
from halo.models import Node, is_llm_engine, llm_model_key  # noqa: E402
from halo.processors.base import is_progress_node  # noqa: E402
from halo.processors.opwise import OpwiseGraphProcessor, _DPWorkerPool  # noqa: E402
from halo.worker import TaskMessage, worker_process_loop  # noqa: E402

SCHEDULERS = ["dp", "greedy", "minswitch", "rr_topo", "random_topo", "milp", "opwise"]

MIXED_TEMPLATE = """
graph:
  name: mixed_engines
  nodes:
    - id: user_input
      type: input
      outputs: [user_query]
    - id: draft
      type: inference
      engine: vllm
      model: Qwen/Qwen3-0.6B
      system_prompt: "Draft an answer."
      inputs: [user_query]
      outputs: [draft]
    - id: refine
      type: inference
      engine: sglang
      model: Qwen/Qwen3-1.7B
      system_prompt: "Refine the draft."
      inputs: [user_query, draft]
      outputs: [final_answer]
  edges:
    - {from: user_input, to: draft, mapping: {user_query: "{{ user_query }}"}}
    - {from: draft, to: refine, mapping: {user_query: "{{ user_query }}", draft: "{{ draft }}"}}
"""

# Cross-process event log (inherited by forked GPU workers).
EVENT_LOG: list[Path] = []


def _log(event: str) -> None:
    if EVENT_LOG:
        with EVENT_LOG[0].open("a") as fh:
            fh.write(event + "\n")


# ---- fakes ------------------------------------------------------------------


@dataclasses.dataclass
class FakeServerArgs:
    """Subset of sglang.srt.server_args.ServerArgs."""

    model_path: str
    tp_size: int = 1
    mem_fraction_static: float | None = None
    context_length: int | None = None
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = False
    trust_remote_code: bool = False
    random_seed: int | None = None


class FakeSamplingParams:
    """Subset of sglang.srt.sampling.sampling_params.SamplingParams."""

    def __init__(self, max_new_tokens=128, stop=None, temperature=1.0, top_p=1.0, top_k=-1,
                 min_new_tokens=0, n=1, ignore_eos=False, sampling_seed=None):
        pass


class FakeSGLEngine:
    """Stands in for ``sglang.Engine``; validates kwargs the way SGLang does."""

    instances: list["FakeSGLEngine"] = []

    def __init__(self, **kwargs):
        if kwargs["model_path"] == "no-runtime":  # frontend-only sglang: lazy runtime import fails
            raise ModuleNotFoundError("No module named 'sgl_kernel'")
        FakeServerArgs(**kwargs)  # unknown engine args -> TypeError, as in SGLang
        self.kwargs = kwargs
        self.calls: list[tuple[list[str], dict]] = []
        self.shutdown_calls = 0
        self.child: subprocess.Popen | None = None
        if kwargs["model_path"] == "spawns-child":
            self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        FakeSGLEngine.instances.append(self)
        _log(f"sglang:init:{kwargs['model_path']}")

    def generate(self, prompt=None, sampling_params=None, **kwargs):
        assert isinstance(sampling_params, list) and len(sampling_params) == len(prompt)
        for params in sampling_params:
            FakeSamplingParams(**params)  # unknown sampling keys -> TypeError, as in SGLang
        self.calls.append((list(prompt), sampling_params))
        return [{"text": f"  sglang[{self.kwargs['model_path']}]:{p}  ", "meta_info": {}} for p in prompt]

    def shutdown(self):
        # Like SGLang: kill the subprocesses, do not wait for them.
        self.shutdown_calls += 1
        if self.child is not None:
            self.child.kill()
        _log(f"sglang:shutdown:{self.kwargs['model_path']}")


class FakeVLLMEngine:
    """Stands in for ``VLLMEngine`` (same constructor; no ``shutdown``, like the real one)."""

    instances: list["FakeVLLMEngine"] = []

    def __init__(self, model, *, allow_tensor_parallel=False, **kwargs):
        self.model = model
        self.kwargs = kwargs
        FakeVLLMEngine.instances.append(self)
        _log(f"vllm:init:{model}")

    def generate(self, prompt, *, label=None, **kwargs):
        return self.generate_batch([prompt], label=label, **kwargs)[0]

    def generate_batch(self, prompts, *, label=None, **kwargs):
        return [f"vllm[{self.model}]:{p}" for p in prompts]


@pytest.fixture
def fake_sglang(monkeypatch):
    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType("sglang.srt.sampling.sampling_params"),
    }
    for name in ("sglang", "sglang.srt", "sglang.srt.sampling"):
        modules[name].__path__ = []  # mark as packages
    modules["sglang"].Engine = FakeSGLEngine
    modules["sglang.srt.server_args"].ServerArgs = FakeServerArgs
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    FakeSGLEngine.instances.clear()
    FakeVLLMEngine.instances.clear()
    monkeypatch.setattr(engines, "VLLMEngine", FakeVLLMEngine)
    monkeypatch.setenv("HALO_MONITOR_ENABLE", "0")
    return FakeSGLEngine


def _parse(tmp_path: Path, text: str, name: str = "template.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return GraphTemplateParser(path).parse()


def _node(node_id: str, engine: str | None, model: str | None) -> Node:
    return Node(id=node_id, type="inference", engine=engine, model=model, inputs=("q",), outputs=("out",))


def _worker_ids(task) -> tuple:
    return tuple(task.worker_id) if isinstance(task.worker_id, (list, tuple)) else (task.worker_id,)


# ---- parsing / planning -----------------------------------------------------


def test_parser_accepts_sglang_node(tmp_path):
    graph = _parse(tmp_path, MIXED_TEMPLATE)
    refine = graph.nodes["refine"]
    assert (refine.engine, refine.model) == ("sglang", "Qwen/Qwen3-1.7B")
    assert refine.system_prompt == "Refine the draft."
    assert refine.inputs == ("user_query", "draft") and refine.outputs == ("final_answer",)
    assert is_llm_engine(refine.engine) and is_llm_engine("vllm")
    assert not any(is_llm_engine(e) for e in (None, "db", "http", "noop", "SGLang"))
    assert is_progress_node(refine)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_mixed_template_schedules_both_llm_nodes_on_gpus(tmp_path, monkeypatch, scheduler):
    monkeypatch.setenv("HALO_SCHED_SEED", "7")
    graph = _parse(tmp_path, MIXED_TEMPLATE)
    optimizer = GraphOptimizer(num_gpus=2, scheduler_mode=scheduler, plan_mode="default")
    plan = optimizer.build_plan(graph, sample_contexts=[{"user_query": "q"}])
    tasks = {task.node_id: task for task in plan.tasks}
    assert set(tasks) == {"draft", "refine"}
    for node_id in ("draft", "refine"):
        assert all(plan.workers[wid].kind == "gpu" for wid in _worker_ids(tasks[node_id]))
    assert tuple(tasks["refine"].dependencies) == ("draft",)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_sglang_nodes_are_planned_like_vllm_nodes(tmp_path, monkeypatch, scheduler):
    monkeypatch.setenv("HALO_SCHED_SEED", "7")
    source = (ROOT / "templates" / "example_dag.yaml").read_text()
    assert "engine: vllm" in source

    def plan_of(text: str, name: str):
        graph = _parse(tmp_path, text, name)
        optimizer = GraphOptimizer(num_gpus=2, scheduler_mode=scheduler, plan_mode="default")
        plan = optimizer.build_plan(graph, sample_contexts=[{"user_query": "q"}], input_query_count=64)
        tasks = [(t.node_id, t.worker_id, tuple(t.dependencies), t.epoch) for t in plan.tasks]
        return tasks, plan.metadata.get("dp_stats")

    assert plan_of(source.replace("engine: vllm", "engine: sglang"), "sglang.yaml") == plan_of(source, "vllm.yaml")


def test_same_checkpoint_on_vllm_and_sglang_is_a_model_switch():
    vllm_node = _node("a", "vllm", "Qwen/Qwen3-0.6B")
    sglang_node = _node("b", "sglang", "Qwen/Qwen3-0.6B")
    assert llm_model_key(vllm_node) == "Qwen/Qwen3-0.6B"  # vLLM identity unchanged
    assert llm_model_key(sglang_node) == "sglang:Qwen/Qwen3-0.6B"
    assert llm_model_key(_node("c", "sglang", None)) is None
    optimizer = GraphOptimizer(num_gpus=1, plan_mode="default")
    assert optimizer._model_init_cost(sglang_node, llm_model_key(sglang_node)) == 0.0
    assert optimizer._model_init_cost(vllm_node, llm_model_key(vllm_node)) == 0.0
    assert optimizer._model_init_cost(sglang_node, llm_model_key(vllm_node)) > 0.0
    assert optimizer._model_init_cost(sglang_node, None) == optimizer._model_init_cost(vllm_node, None)


# ---- SGLangEngine -----------------------------------------------------------


def test_sglang_engine_default_sampling_matches_vllm(fake_sglang):
    engine = SGLangEngine("m")
    assert engine.generate_batch(["x"]) == ["sglang[m]:x"]
    fake = fake_sglang.instances[-1]
    assert fake.kwargs == {"model_path": "m"}
    assert fake.calls[-1][1] == [{"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 1024}]


def test_sglang_engine_maps_vllm_kwargs_and_generates(fake_sglang, caplog):
    engine = SGLangEngine(
        "Qwen/Qwen3-0.6B",
        tensor_parallel_size=2,  # dropped: TP not allowed (single GPU per worker)
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        swap_space=4,  # vLLM-only -> dropped
        sampling_params={"max_tokens": 64, "temperature": 0.0, "logprobs": 5, "n": 3},
    )
    fake = fake_sglang.instances[-1]
    assert fake.kwargs == {
        "model_path": "Qwen/Qwen3-0.6B",
        "mem_fraction_static": 0.7,
        "context_length": 4096,
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "trust_remote_code": True,
    }
    assert engine.generate_batch(["a", "b"], label="node") == ["sglang[Qwen/Qwen3-0.6B]:a", "sglang[Qwen/Qwen3-0.6B]:b"]
    expected = {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 64}
    prompts, sampling = fake.calls[-1]
    assert prompts == ["a", "b"] and sampling == [expected, expected]
    sampling[0]["max_new_tokens"] = 1  # SGLang may edit params in place (auto-truncation) ...
    assert sampling[1] == expected  # ... so each prompt gets its own dict
    assert engine.generate_batch(["e"]) and fake.calls[-1][1] == [expected]  # defaults untouched
    assert engine.generate("c") == "sglang[Qwen/Qwen3-0.6B]:c"
    assert engine.generate_batch([]) == []
    # Per-call override: vLLM names mapped, merged over the engine defaults.
    engine.generate_batch(["d"], sampling_params={"max_tokens": 8, "stop": ["\n"], "seed": 1})
    assert fake.calls[-1][1] == [
        {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 8, "stop": ["\n"], "sampling_seed": 1}
    ]
    assert "swap_space" in caplog.text and "logprobs" in caplog.text


def test_sglang_engine_keeps_tp_when_allowed(fake_sglang):
    SGLangEngine("m", allow_tensor_parallel=True, tensor_parallel_size=2, seed=3)
    assert fake_sglang.instances[-1].kwargs == {"model_path": "m", "tp_size": 2, "random_seed": 3}


def test_sglang_engine_requires_sglang(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang", None)  # `import sglang` -> ImportError
    with pytest.raises(RuntimeError, match="SGLang is required"):
        SGLangEngine("m")


def test_sglang_engine_reports_missing_runtime(fake_sglang):
    with pytest.raises(RuntimeError, match=r"install `sglang\[all\]`"):
        SGLangEngine("no-runtime")


def test_sglang_shutdown_waits_for_engine_subprocesses(fake_sglang, monkeypatch):
    psutil = pytest.importorskip("psutil")
    unregistered = []
    monkeypatch.setattr(engines.atexit, "unregister", unregistered.append)
    engine = SGLangEngine("spawns-child")
    fake = fake_sglang.instances[-1]
    pid = fake.child.pid
    assert [proc.pid for proc in engine._procs] == [pid]
    engine.shutdown()
    assert fake.shutdown_calls == 1
    # Fully exited before shutdown() returned, but not reaped behind its owner's back.
    assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert fake.child.wait(timeout=5) == -9
    assert unregistered == [fake.shutdown]  # SGLang's atexit hook no longer pins the engine
    with pytest.raises(RuntimeError, match="shut down"):
        engine.generate_batch(["x"])
    engine.shutdown()  # idempotent
    assert fake.shutdown_calls == 1


# ---- providers / eviction -----------------------------------------------------


def test_llm_provider_dispatches_on_node_engine(fake_sglang):
    provider = make_llm_provider(gpu_memory_utilization=0.5)
    vllm_engine = provider.resolve(_node("a", "vllm", "m1"))
    sglang_engine = provider.resolve(_node("b", "sglang", "m2"))
    assert isinstance(vllm_engine, FakeVLLMEngine) and vllm_engine.kwargs == {"gpu_memory_utilization": 0.5}
    assert isinstance(sglang_engine, SGLangEngine)
    assert fake_sglang.instances[-1].kwargs == {"model_path": "m2", "mem_fraction_static": 0.5}
    assert provider.resolve(_node("c", "sglang", "m2")) is sglang_engine  # cached per (engine, model)
    with pytest.raises(ValueError, match="Cannot build LLM engine"):
        provider.resolve(_node("d", "db", None))
    with pytest.raises(ValueError, match="missing a model"):
        provider.resolve(_node("e", "sglang", None))
    with pytest.raises(ValueError, match="Cannot build VLLM engine"):
        make_vllm_provider().resolve(_node("f", "sglang", "m"))
    with pytest.raises(ValueError, match="Cannot build SGLang engine"):
        make_sglang_provider().resolve(_node("g", "vllm", "m"))


def test_clear_cache_shuts_down_sglang_engines(fake_sglang):
    provider = make_llm_provider()
    sglang_engine = provider.resolve(_node("b", "sglang", "m2"))
    provider.resolve(_node("a", "vllm", "m1"))  # no shutdown(): left to GC as before
    provider.clear_cache()
    assert fake_sglang.instances[-1].shutdown_calls == 1
    assert provider._cache == {}
    with pytest.raises(RuntimeError, match="shut down"):
        sglang_engine.generate_batch(["x"])


def test_worker_switches_engines_and_evicts(fake_sglang):
    """GPU worker loop: config carries the engine; switch and close shut SGLang down."""
    sglang_node = _node("s", "sglang", "m")
    vllm_node = _node("v", "vllm", "m")
    tasks, results = queue.Queue(), queue.Queue()
    for msg in [
        TaskMessage(node_id="__CONFIG__", node=None, config={"epoch": 0, "model": "m", "engine": "sglang"}),
        TaskMessage(node_id="s", node=sglang_node, context_slices=[{"q": "hi"}], context_indices=[0]),
        TaskMessage(node_id="v", node=vllm_node, context_slices=[{"q": "x"}], context_indices=[1]),
        # No "engine" key (pre-SGLang config message): defaults to vLLM.
        TaskMessage(node_id="__CONFIG__", node=None, config={"epoch": 1, "model": "m"}),
        TaskMessage(node_id="v", node=vllm_node, context_slices=[{"q": "a"}, {"q": "b"}], context_indices=[0, 1]),
        TaskMessage(node_id="__CONFIG__", node=None, config={"epoch": 2, "model": "m2", "engine": "sglang"}),
        TaskMessage(node_id="__STOP__", node=None, is_stop=True),
    ]:
        tasks.put(msg)
    worker_process_loop("gpu-0", "cpu", tasks, results, {}, {})

    sglang_result, mismatch, vllm_result = (results.get_nowait() for _ in range(3))
    assert results.empty()
    assert sglang_result.error is None and sglang_result.outputs[0]["out"].startswith("sglang[m]:")
    assert "configured for engine sglang" in mismatch.error
    assert vllm_result.error is None and [o["out"][:8] for o in vllm_result.outputs] == ["vllm[m]:"] * 2
    # One SGLang engine per (engine, model): the warmup built the engine the task then used.
    assert [e.kwargs["model_path"] for e in fake_sglang.instances] == ["m", "m2"]
    assert [e.shutdown_calls for e in fake_sglang.instances] == [1, 1]  # switch to vLLM; worker close
    assert [e.model for e in FakeVLLMEngine.instances] == ["m"]


def test_opwise_dp_config_carries_engine():
    sent = []

    class _Queue:
        def put(self, msg):
            sent.append(msg.config)

    pool = _DPWorkerPool(worker_ids=["w0"], task_queues={"w0": _Queue()}, result_queue=None, processes={})
    processor = OpwiseGraphProcessor.__new__(OpwiseGraphProcessor)
    processor._configure_dp_model(pool, _node("s", "sglang", "m"))
    processor._configure_dp_model(pool, _node("s2", "sglang", "m"))  # same model + engine: no resend
    processor._configure_dp_model(pool, _node("v", "vllm", "m"))  # engine switch
    assert sent == [
        {"epoch": 0, "model": "m", "engine": "sglang"},
        {"epoch": 0, "model": "m", "engine": "vllm"},
    ]


# ---- end-to-end with fake engines ---------------------------------------------


def test_serial_processor_runs_mixed_template(fake_sglang, tmp_path):
    graph = _parse(tmp_path, MIXED_TEMPLATE)
    plan = GraphOptimizer(num_gpus=1, scheduler_mode="dp", plan_mode="default").build_plan(
        graph, sample_contexts=[{"user_query": "q"}]
    )
    results = SerialGraphProcessor().run_batch(plan, graph, [{"user_query": "q1"}, {"user_query": "q2"}])
    for ctx in results:
        assert ctx["draft"].startswith("vllm[Qwen/Qwen3-0.6B]:")
        assert ctx["final_answer"].startswith("sglang[Qwen/Qwen3-1.7B]:")
    # q2's switch back to vLLM shut the first SGLang engine down; the second is still cached.
    assert [e.shutdown_calls for e in fake_sglang.instances] == [1, 0]


@pytest.mark.skipif(mp.get_start_method() != "fork", reason="fake engines reach GPU workers via fork")
def test_multiprocess_processor_runs_mixed_template(fake_sglang, tmp_path):
    EVENT_LOG[:] = [tmp_path / "events.log"]
    try:
        graph = _parse(tmp_path, MIXED_TEMPLATE)
        # One GPU: both nodes share gpu-0, so the worker must switch vLLM -> SGLang.
        plan = GraphOptimizer(num_gpus=1, scheduler_mode="dp", plan_mode="default").build_plan(
            graph, sample_contexts=[{"user_query": "q"}]
        )
        results = MultiProcessGraphProcessor().run_batch(plan, graph, [{"user_query": f"q{i}"} for i in range(3)])
        events = EVENT_LOG[0].read_text().split()
    finally:
        EVENT_LOG.clear()
    for ctx in results:
        assert ctx["draft"].startswith("vllm[Qwen/Qwen3-0.6B]:")
        assert ctx["final_answer"].startswith("sglang[Qwen/Qwen3-1.7B]:")
    assert events == [
        "vllm:init:Qwen/Qwen3-0.6B",
        "sglang:init:Qwen/Qwen3-1.7B",
        "sglang:shutdown:Qwen/Qwen3-1.7B",  # worker close
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
