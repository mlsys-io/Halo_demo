from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping
import threading
import time

from .db import DatabaseExecutor, DefaultDatabaseExecutor
from .engines import make_vllm_provider
from .models import Node
from .executor import DBNodeExecutor, HTTPNodeExecutor, ProcessorNodeExecutor, VLLMNodeExecutor

LOGGER = logging.getLogger(__name__)
_RED = "\033[31m"
_RESET = "\033[0m"


@dataclass(slots=True)
class TaskMessage:
    """从主进程发送给 worker 的消息。"""

    node_id: str
    node: Node | None
    context_slices: List[Dict[str, Any]] = field(default_factory=list)
    context_indices: List[int] = field(default_factory=list)
    is_stop: bool = False
    config: Dict[str, Any] | None = None


@dataclass(slots=True)
class ResultMessage:
    """worker 返回给主进程的结果消息。"""

    node_id: str
    worker_id: str | None = None
    context_indices: List[int] = field(default_factory=list)
    outputs: List[Dict[str, Any]] | None = None
    error: str | None = None
    stats: Dict[str, Any] | None = None


def configure_device_env(device: str) -> None:
    """根据 Worker.device 配置环境（例如 CUDA_VISIBLE_DEVICES）。"""
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":")[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        except Exception:
            pass


def worker_process_loop(
    worker_id: str,
    device: str,
    task_queue: "mp.Queue[TaskMessage]",
    result_queue: "mp.Queue[ResultMessage]",
    engine_kwargs: Dict[str, Any] | None = None,
    executor_kwargs: Dict[str, Any] | None = None,
) -> None:
    """Worker 进程主循环。

    每个 GPU worker 进程拥有自己的 EngineProvider 和 vLLM 引擎 cache。
    """
    engine_kwargs = dict(engine_kwargs or {})
    executor_kwargs = dict(executor_kwargs or {})

    configure_device_env(device)

    engine_provider = make_vllm_provider(**engine_kwargs)
    node_executor = VLLMNodeExecutor(
        engine_provider=engine_provider,
        **executor_kwargs,
    )
    current_model: str | None = None
    current_epoch: int | None = None
    pending_init_stats: Dict[str, Any] | None = None

    LOGGER.info("%sWorker %s started on device %s%s", _RED, worker_id, device, _RESET)

    def _merge_stats(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any]:
        if not a:
            return b or {}
        if not b:
            return dict(a)
        merged = dict(a)
        for key, val in b.items():
            if isinstance(val, (int, float)) and isinstance(merged.get(key), (int, float)):
                merged[key] = merged.get(key, 0) + val
            else:
                merged[key] = val
        return merged

    while True:
        task: TaskMessage = task_queue.get()
        if task.is_stop:
            LOGGER.info("%sWorker %s received stop signal%s", _RED, worker_id, _RESET)
            break

        if task.config is not None:
            epoch = task.config.get("epoch")
            model = task.config.get("model")
            if model != current_model:
                try:
                    node_executor.engine_provider.clear_cache()
                except AttributeError:
                    pass
                current_model = model
                # Warm up vLLM to overlap model init.
                if current_model:
                    try:
                        node_executor.warmup_model(current_model)
                        # Preserve model init stats to return on first real task.
                        pending_init_stats = node_executor.consume_stats()
                    except Exception:
                        LOGGER.warning(
                            "%sWorker %s warmup failed for model %s%s",
                            _RED,
                            worker_id,
                            current_model,
                            _RESET,
                        )
            current_epoch = epoch
            LOGGER.info(
                "%sWorker %s configured for epoch %s (model=%s)%s",
                _RED,
                worker_id,
                current_epoch,
                current_model,
                _RESET,
            )
            continue

        if task.node is None:
            result_queue.put(
                ResultMessage(
                    node_id=task.node_id,
                    worker_id=worker_id,
                    context_indices=list(task.context_indices),
                    outputs=None,
                    error="Missing node",
                )
            )
            continue

        contexts = task.context_slices or []
        if not contexts:
            result_queue.put(
                ResultMessage(
                    node_id=task.node.id,
                    worker_id=worker_id,
                    context_indices=list(task.context_indices),
                    outputs=None,
                    error="Empty context batch",
                )
            )
            continue

        error_message: str | None = None
        outputs_batch: List[Dict[str, Any]] | None = None
        stats_payload: Dict[str, Any] | None = None
        executed = False
        start_time = time.perf_counter()
        try:
            if task.node.engine == "vllm" and current_model is not None:
                node_model = task.node.model or ""
                if node_model != current_model:
                    raise RuntimeError(
                        f"Worker {worker_id} configured for model {current_model} "
                        f"but received node model {node_model}"
                    )
            if len(contexts) == 1:
                executed = True
                outputs_batch = [node_executor.execute(task.node, contexts[0])]
            else:
                executed = True
                outputs_batch = node_executor.execute_batch(task.node, contexts)
        except Exception as exc:
            LOGGER.exception("Worker %s failed to execute node %s", worker_id, task.node.id)
            error_message = str(exc)
        finally:
            if executed:
                total_time = time.perf_counter() - start_time
                stats_payload = node_executor.consume_stats()
                if stats_payload is None:
                    stats_payload = {}
                stats_payload["node_total_time"] = total_time
                if pending_init_stats:
                    stats_payload = _merge_stats(stats_payload, pending_init_stats)
                    pending_init_stats = None
        result_queue.put(
            ResultMessage(
                node_id=task.node.id,
                worker_id=worker_id,
                context_indices=list(task.context_indices),
                outputs=outputs_batch,
                error=error_message,
                stats=stats_payload,
            )
        )


def cpu_worker_loop(
    worker_id: str,
    task_queue: Any,
    result_queue: Any,
    db_executor_factory: Callable[[], DatabaseExecutor] | None = None,
    executor_kwargs: Dict[str, Any] | None = None,
) -> None:
    """CPU worker 线程主循环，仅处理 DB/HTTP/非 LLM 节点。"""
    db_executor_factory = db_executor_factory or DefaultDatabaseExecutor
    executor_kwargs = dict(executor_kwargs or {})
    http_concurrency = executor_kwargs.pop("http_concurrency", None)
    if http_concurrency is None:
        http_concurrency = executor_kwargs.get("db_concurrency", 1) or 1
    http_default_sleep_s = executor_kwargs.pop("http_default_sleep_s", 0.0)
    node_executor = DBNodeExecutor(
        db_executor=db_executor_factory(),
        **executor_kwargs,
    )
    http_executor = HTTPNodeExecutor(
        http_concurrency=http_concurrency,
        default_sleep_s=http_default_sleep_s,
    )
    processor_executor = ProcessorNodeExecutor()
    LOGGER.info("%sCPU worker %s started%s", _RED, worker_id, _RESET)

    while True:
        task: TaskMessage = task_queue.get()
        if task.is_stop:
            LOGGER.info("%sCPU worker %s received stop signal%s", _RED, worker_id, _RESET)
            break
        if task.node is None:
            result_queue.put(
                ResultMessage(
                    node_id=task.node_id,
                    worker_id=worker_id,
                    context_indices=list(task.context_indices),
                    outputs=None,
                    error="Missing node",
                )
            )
            continue

        contexts = task.context_slices or []
        if not contexts:
            result_queue.put(
                ResultMessage(
                    node_id=task.node.id,
                    worker_id=worker_id,
                    context_indices=list(task.context_indices),
                    outputs=None,
                    error="Empty context batch",
                )
            )
            continue

        error_message: str | None = None
        outputs_batch: List[Dict[str, Any]] | None = None
        stats_payload: Dict[str, Any] | None = None
        start_time = time.perf_counter()
        try:
            if task.node.type == "processor":
                outputs_batch = processor_executor.execute_batch(task.node, contexts)
                stats_payload = processor_executor.consume_stats()
            elif task.node.engine == "db":
                outputs_batch = node_executor.execute_batch(task.node, contexts)
                stats_payload = node_executor.consume_stats()
            elif task.node.engine == "http":
                outputs_batch = http_executor.execute_batch(task.node, contexts)
                stats_payload = http_executor.consume_stats()
            elif task.node.engine == "noop":
                outputs_batch = [
                    {name: ctx.get(name) for name in task.node.outputs} for ctx in contexts
                ]
                stats_payload = {}
            else:
                raise RuntimeError(
                    f"Unsupported node for CPU worker: engine={task.node.engine}, type={task.node.type}"
                )
        except Exception as exc:
            LOGGER.exception("CPU worker %s failed to execute node %s", worker_id, task.node.id)
            error_message = str(exc)
        finally:
            total_time = time.perf_counter() - start_time
            if stats_payload is None:
                stats_payload = {}
            stats_payload["node_total_time"] = total_time

        result_queue.put(
            ResultMessage(
                node_id=task.node.id,
                worker_id=worker_id,
                context_indices=list(task.context_indices),
                outputs=outputs_batch,
                error=error_message,
                stats=stats_payload,
            )
        )
