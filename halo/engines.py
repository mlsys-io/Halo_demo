from __future__ import annotations

import atexit
import dataclasses
import importlib
import inspect
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence
import time

from .models import Node

LOGGER = logging.getLogger(__name__)


class LLMEngine(Protocol):
    def generate(self, prompt: str, *, label: str | None = None, **kwargs: Any) -> str: ...
    def generate_batch(
        self, prompts: Sequence[str], *, label: str | None = None, **kwargs: Any
    ) -> List[str]: ...


def _shutdown_engine(engine: Any) -> None:
    """Explicitly release an engine that supports it.

    SGLang keeps GPU memory in scheduler subprocesses, so dropping the reference
    is not enough; vLLM engines have no ``shutdown`` and are left to garbage
    collection as before.
    """
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown()
    except Exception:
        LOGGER.warning("Failed to shut down %s", type(engine).__name__, exc_info=True)


@dataclass(slots=True)
class EngineProvider:
    """Caches LLMEngine instances keyed by a customizable function."""

    factory: Callable[[Node], LLMEngine]
    cache_key_fn: Callable[[Node], Any] = field(default=lambda node: (node.engine, node.model))
    _cache: Dict[Any, LLMEngine] = field(init=False, default_factory=dict)

    def resolve(
        self,
        node: Node,
        *,
        on_initialize: Callable[[float], None] | None = None,
    ) -> LLMEngine:
        key = self.cache_key_fn(node)
        if key not in self._cache:
            start = time.perf_counter()
            engine = self.factory(node)
            init_duration = time.perf_counter() - start
            self._cache[key] = engine
            if on_initialize is not None:
                on_initialize(init_duration)
        return self._cache[key]

    def clear_cache(self) -> None:
        """Evict all cached engines (model switch / worker close), shutting down those that support it."""
        engines = list(self._cache.values())
        self._cache.clear()
        for engine in engines:
            _shutdown_engine(engine)


class VLLMEngine:
    """Thin wrapper around vLLM's LLM interface (single device by default)."""

    def __init__(self, model: str, *, allow_tensor_parallel: bool = False, **kwargs: Any):
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "vLLM is required to instantiate VLLMEngine. "
                "Install vllm and ensure GPU access."
            ) from exc

        self._sampling_cls = SamplingParams
        sampling_options = kwargs.pop("sampling_params", {}) or {}
        sampling_defaults = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024}
        sampling_defaults.update(sampling_options)

        # Each worker already constrains visibility to a single device via CUDA_VISIBLE_DEVICES,
        # so we rely on vLLM defaults and do not request tensor parallelism explicitly.
        if not allow_tensor_parallel:
            kwargs.pop("tensor_parallel_size", None)
        self._engine = LLM(model=model, **kwargs)
        self._default_sampling = SamplingParams(**sampling_defaults)

    def generate(self, prompt: str, *, label: str | None = None, **kwargs: Any) -> str:
        return self.generate_batch([prompt], label=label, **kwargs)[0]

    def generate_batch(
        self, prompts: Sequence[str], *, label: str | None = None, **kwargs: Any
    ) -> List[str]:
        if not prompts:
            return []
        sampling_params = kwargs.get("sampling_params") or self._default_sampling
        outputs = self._engine.generate(list(prompts), sampling_params)
        label_str = label or "unknown"
        responses: List[str] = []
        for idx, output in enumerate(outputs):
            result = output.outputs[0]
            text = result.text.strip()
            finish_reason = getattr(result, "finish_reason", "unknown")
            token_ids = getattr(result, "token_ids", None)
            token_count = len(token_ids) if isinstance(token_ids, list) else "?"
            responses.append(text)
        return responses


# vLLM-style engine kwargs (what Halo's ``engine_kwargs`` carry) -> SGLang ``ServerArgs`` names.
_SGLANG_ENGINE_ARG_NAMES = {
    "tensor_parallel_size": "tp_size",
    "pipeline_parallel_size": "pp_size",
    "data_parallel_size": "dp_size",
    "gpu_memory_utilization": "mem_fraction_static",
    "max_model_len": "context_length",
    "max_num_seqs": "max_running_requests",
    "tokenizer": "tokenizer_path",
    "seed": "random_seed",
    "enforce_eager": "disable_cuda_graph",
}
# vLLM-style sampling options -> SGLang sampling-param names.
_SGLANG_SAMPLING_NAMES = {
    "max_tokens": "max_new_tokens",
    "min_tokens": "min_new_tokens",
    "seed": "sampling_seed",
}
# Halo keeps one completion per prompt (VLLMEngine uses ``outputs[0]``), so never ask for more.
_SGLANG_SAMPLING_DROP = frozenset({"n", "best_of"})


def _sglang_param_names(module: str, attr: str) -> frozenset[str] | None:
    """Names accepted by SGLang's ``module.attr`` (dataclass or class); None if unknown."""
    try:
        target = getattr(importlib.import_module(module), attr)
        if dataclasses.is_dataclass(target):
            return frozenset(f.name for f in dataclasses.fields(target) if f.init)
        params = inspect.signature(target).parameters.values()
    except Exception:
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return frozenset(p.name for p in params)


def _to_sglang_kwargs(
    options: Mapping[str, Any],
    renames: Mapping[str, str],
    accepted: frozenset[str] | None,
    *,
    what: str,
    drop: frozenset[str] = frozenset(),
) -> Dict[str, Any]:
    """Rename vLLM-style keys to SGLang's and drop the ones SGLang does not accept."""
    translated: Dict[str, Any] = {}
    dropped: List[str] = []
    for key, value in options.items():
        name = renames.get(key, key)
        if key in drop or (accepted is not None and name not in accepted):
            dropped.append(key)
            continue
        translated[name] = value
    if dropped:
        LOGGER.warning("SGLangEngine: ignoring %s not supported by SGLang: %s", what, sorted(dropped))
    return translated


def _to_sglang_sampling(options: Mapping[str, Any]) -> Dict[str, Any]:
    translated = _to_sglang_kwargs(
        options,
        _SGLANG_SAMPLING_NAMES,
        _sglang_param_names("sglang.srt.sampling.sampling_params", "SamplingParams"),
        what="sampling params",
        drop=_SGLANG_SAMPLING_DROP,
    )
    # Values whose vLLM conventions differ from SGLang's validation.
    if translated.get("top_k") == 0:  # vLLM: 0 disables top-k; SGLang uses -1
        translated["top_k"] = -1
    penalty = translated.get("repetition_penalty")
    if isinstance(penalty, (int, float)) and penalty > 2.0:
        LOGGER.warning("SGLangEngine: repetition_penalty %.2f clipped to SGLang's maximum 2.0", penalty)
        translated["repetition_penalty"] = 2.0
    return translated


def _to_sglang_engine_args(options: Mapping[str, Any]) -> Dict[str, Any]:
    options = dict(options)
    prefix_caching = options.pop("enable_prefix_caching", None)
    if prefix_caching is not None:
        # SGLang's radix (prefix) cache is on by default; only the switch is inverted.
        options["disable_radix_cache"] = not prefix_caching
    return _to_sglang_kwargs(
        options,
        _SGLANG_ENGINE_ARG_NAMES,
        _sglang_param_names("sglang.srt.server_args", "ServerArgs"),
        what="engine args",
    )


class SGLangSubprocessError(RuntimeError):
    """An SGLang scheduler/detokenizer subprocess failed (it signalled SIGQUIT)."""


def _raise_on_sigquit(signum: int, frame: Any) -> None:
    # Replaces SGLang's default handler, which SIGKILLs the whole hosting process
    # tree: raising lets the GPU worker report the failure as a node error.
    raise SGLangSubprocessError("An SGLang subprocess failed; see its log for the cause.")


def _child_processes() -> List[Any]:
    """All descendant processes of this process (psutil ships with SGLang)."""
    try:
        import psutil

        return psutil.Process().children(recursive=True)
    except Exception:
        return []


def _has_exited(proc: Any) -> bool:
    """Whether ``proc`` has fully exited (all threads gone, so its GPU memory is freed).

    A zombie *status* is not enough: the thread-group leader turns zombie while
    other threads may still hold the CUDA context. For our children,
    ``waitid(WNOWAIT)`` reports full exit without reaping them, leaving that to
    their owner (multiprocessing also collects its own resource tracker).
    """
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except (ChildProcessError, AttributeError):  # not our child (or no waitid)
        try:
            return not proc.is_running()
        except Exception:
            return True


def _wait_for_exit(procs: Sequence[Any], timeout: float = 30.0) -> None:
    """Block until ``procs`` have exited, i.e. their GPU memory is back."""
    deadline = time.monotonic() + timeout
    alive = [proc for proc in procs if not _has_exited(proc)]
    while alive and time.monotonic() < deadline:
        time.sleep(0.05)
        alive = [proc for proc in alive if not _has_exited(proc)]
    if alive:
        LOGGER.warning(
            "SGLang subprocesses still alive %.0fs after shutdown: %s",
            timeout,
            [proc.pid for proc in alive],
        )


def _kill_and_wait(procs: Sequence[Any]) -> None:
    """SIGKILL ``procs`` and their descendants, then wait until they have exited."""
    tree = {proc.pid: proc for proc in procs}
    for proc in procs:
        try:
            tree.update((child.pid, child) for child in proc.children(recursive=True))
        except Exception:  # already gone
            pass
    for proc in tree.values():
        try:
            proc.kill()
        except Exception:
            pass
    _wait_for_exit(list(tree.values()))


class SGLangEngine:
    """Thin wrapper around SGLang's offline ``sgl.Engine`` (single device by default).

    Mirrors :class:`VLLMEngine`: it takes the same vLLM-style kwargs, mapped to
    SGLang names (``tensor_parallel_size`` -> ``tp_size``,
    ``gpu_memory_utilization`` -> ``mem_fraction_static``, ``max_tokens`` ->
    ``max_new_tokens``, ...); options SGLang does not accept are dropped.
    """

    def __init__(self, model: str, *, allow_tensor_parallel: bool = False, **kwargs: Any):
        try:
            import sglang as sgl
        except Exception as exc:
            raise RuntimeError(
                "SGLang is required to instantiate SGLangEngine. "
                "Install `sglang[all]` (see README) and ensure GPU access."
            ) from exc

        sampling_options = kwargs.pop("sampling_params", {}) or {}
        sampling_defaults = {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 1024}
        sampling_defaults.update(_to_sglang_sampling(sampling_options))

        # Same device policy as VLLMEngine: one visible GPU per worker, no TP unless allowed.
        if not allow_tensor_parallel:
            kwargs.pop("tensor_parallel_size", None)
            kwargs.pop("tp_size", None)
        engine_args = _to_sglang_engine_args(kwargs)
        # Like vLLM, cap max_new_tokens at the context length instead of failing the request.
        engine_args.setdefault("allow_auto_truncate", True)
        engine_args.setdefault("custom_sigquit_handler", _raise_on_sigquit)
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "SGLangEngine must be created on the main thread (SGLang installs a signal handler)."
            )

        known_pids = {proc.pid for proc in _child_processes()}
        self._engine = None
        self._procs: List[Any] = []
        try:
            self._engine = sgl.Engine(model_path=model, **engine_args)
        except ImportError as exc:  # e.g. a frontend-only `sglang` without its runtime deps
            raise RuntimeError(
                "SGLang failed to import its runtime; install `sglang[all]` (see README)."
            ) from exc
        finally:
            # Scheduler/detokenizer subprocesses of this engine (also on a failed start).
            self._procs = [proc for proc in _child_processes() if proc.pid not in known_pids]
            if self._engine is None:
                _kill_and_wait(self._procs)
        self._default_sampling = sampling_defaults

    def generate(self, prompt: str, *, label: str | None = None, **kwargs: Any) -> str:
        return self.generate_batch([prompt], label=label, **kwargs)[0]

    def generate_batch(
        self, prompts: Sequence[str], *, label: str | None = None, **kwargs: Any
    ) -> List[str]:
        """Generate one completion per prompt.

        A per-call ``sampling_params`` dict (vLLM or SGLang names) is merged over the defaults.
        """
        if not prompts:
            return []
        if self._engine is None:
            raise RuntimeError("SGLangEngine has been shut down.")
        sampling_params = self._default_sampling
        if kwargs.get("sampling_params"):
            sampling_params = {**sampling_params, **_to_sglang_sampling(kwargs["sampling_params"])}
        # One dict per prompt: SGLang may edit a request's params in place (e.g.
        # auto-truncating max_new_tokens), so they must not be shared.
        outputs = self._engine.generate(
            prompt=list(prompts), sampling_params=[dict(sampling_params) for _ in prompts]
        )
        return [(output.get("text") or "").strip() for output in outputs]

    def shutdown(self) -> None:
        """Stop this engine's subprocesses and wait for them to exit, releasing their GPU memory.

        SGLang's own ``Engine.shutdown`` kills every child of the hosting process,
        which would also hit unrelated processes when the engine lives in the
        user's main process (Serial / Opwise processors); kill only our own tree.
        """
        engine, self._engine = self._engine, None
        if engine is None:
            return
        procs, self._procs = self._procs, []
        atexit.unregister(engine.shutdown)  # registered by sgl.Engine; would pin the engine
        _kill_and_wait(procs)


def make_vllm_provider(*, allow_tensor_parallel: bool = False, **engine_kwargs: Any) -> EngineProvider:
    """Convenience helper for creating a VLLM-backed provider (per worker-process)."""

    def factory(node: Node) -> LLMEngine:
        if node.engine != "vllm":
            raise ValueError(
                f"Cannot build VLLM engine for node '{node.id}' with engine {node.engine}"
            )
        if not node.model:
            raise ValueError(f"Node '{node.id}' is missing a model name.")
        return VLLMEngine(
            model=node.model,
            allow_tensor_parallel=allow_tensor_parallel,
            **engine_kwargs,
        )

    return EngineProvider(factory=factory)


def make_sglang_provider(*, allow_tensor_parallel: bool = False, **engine_kwargs: Any) -> EngineProvider:
    """Convenience helper for creating an SGLang-backed provider (per worker-process)."""

    def factory(node: Node) -> LLMEngine:
        if node.engine != "sglang":
            raise ValueError(
                f"Cannot build SGLang engine for node '{node.id}' with engine {node.engine}"
            )
        if not node.model:
            raise ValueError(f"Node '{node.id}' is missing a model name.")
        return SGLangEngine(
            model=node.model,
            allow_tensor_parallel=allow_tensor_parallel,
            **engine_kwargs,
        )

    return EngineProvider(factory=factory)


def make_llm_provider(*, allow_tensor_parallel: bool = False, **engine_kwargs: Any) -> EngineProvider:
    """Provider that builds a VLLMEngine or SGLangEngine per ``node.engine`` (per worker-process).

    ``engine_kwargs`` use vLLM names; SGLangEngine maps them to SGLang's.
    """
    factories = {
        "vllm": make_vllm_provider(allow_tensor_parallel=allow_tensor_parallel, **engine_kwargs).factory,
        "sglang": make_sglang_provider(allow_tensor_parallel=allow_tensor_parallel, **engine_kwargs).factory,
    }

    def factory(node: Node) -> LLMEngine:
        build = factories.get(node.engine)
        if build is None:
            raise ValueError(
                f"Cannot build LLM engine for node '{node.id}' with engine {node.engine}"
            )
        return build(node)

    return EngineProvider(factory=factory)
