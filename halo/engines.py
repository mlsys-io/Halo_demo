from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence
import time

from .models import Node


class LLMEngine(Protocol):
    def generate(self, prompt: str, *, label: str | None = None, **kwargs: Any) -> str: ...
    def generate_batch(
        self, prompts: Sequence[str], *, label: str | None = None, **kwargs: Any
    ) -> List[str]: ...


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
        self._cache.clear()


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
