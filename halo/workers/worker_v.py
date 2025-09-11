# worker_vllm.py
from __future__ import annotations

import os
import time
import logging
import queue
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Project-local types (adjust the import path if needed)
from halo.components import Query, Operator, ModelConfig, ExecuteInfo


class vLLMWorker:
    """
    Worker process backed by vLLM.

    Responsibilities:
      - Initialize or switch model/tokenizer per Operator (op).
      - Build input prompts (optionally using HF chat templates).
      - Run batched generation via vLLM.
      - Return results in a uniform dict message to the parent process.

    All messages returned to the parent must be dicts of the form:
        {"command": <str>, "result": <payload>, "elapsed_time": <float>}
    """

    def __init__(
        self,
        id: int,
        physical_gpu_id: int,
        cmd_queue: "queue.Queue",
        result_queue: "queue.Queue",
    ) -> None:
        self.id = id
        self.device = f"cuda:{physical_gpu_id}"

        # Bind this process to a single visible GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        # Runtime state
        self.model_name: Optional[str] = None
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.cmd_queue = cmd_queue
        self.response_queue = result_queue

        # vLLM options (tune to your environment)
        self.enforce_eager: bool = True

        logging.info("vLLMWorker[%s] initialized on device %s", id, self.device)

    # --------------------------------------------------------------------- #
    # Per-Operator initialization
    # --------------------------------------------------------------------- #

    def init_op(self, op: Operator) -> None:
        """
        Initialize per-op runtime state:
          - Sampling parameters
          - Optional system/common messages
          - Switch model/tokenizer if op.model_config.model_name changes
        """
        self.op_name = op.id
        self.is_duplicate = getattr(op, "is_duplicate", False)

        cfg: ModelConfig = op.model_config
        self.use_chat_template: bool = bool(getattr(cfg, "use_chat_template", True))

        # Build sampling params for vLLM
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            min_tokens=getattr(cfg, "min_tokens", 0),
        )

        # (Re)load model & tokenizer if needed
        model_name = cfg.model_name
        if model_name != self.model_name:
            # Release previous instances to free GPU memory
            if self.llm is not None:
                try:
                    del self.llm
                except Exception:
                    pass
                self.llm = None

            if self.tokenizer is not None:
                try:
                    del self.tokenizer
                except Exception:
                    pass
                self.tokenizer = None

            torch.cuda.empty_cache()

            # vLLM accepts dtype as strings: "bfloat16"|"float16"|"float32"|"auto"
            self.llm = LLM(
                model_name,
                dtype=str(getattr(cfg, "dtype", "bfloat16")),
                quantization=getattr(cfg, "quantization", None),
                max_model_len=getattr(cfg, "max_model_len", None),
                enforce_eager=self.enforce_eager,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model_name = model_name

        # System/common messages (optional)
        self.prefix: str = (getattr(cfg, "system_prompt", None) or "").strip()
        self.common_message: str = (getattr(cfg, "common_message", "") or "").strip()

    # --------------------------------------------------------------------- #
    # Commands
    # --------------------------------------------------------------------- #

    @torch.inference_mode()
    def execute(self, exe_info: ExecuteInfo) -> Dict[str, Any]:
        """
        Execute a single round for one Operator:
          1) Initialize per-op state
          2) Build batch inputs (with or without chat template)
          3) Call vLLM.generate
          4) Return structured results

        Returns:
            {
              "item": List[{"id": int, "output": str, "benchmark": (float, float)}],
              "op_name": str,
              "benchmark": {"init_time": float, "prefill_time": float, "generate_time": float}
            }
        """
        init_start = time.perf_counter()
        self.init_op(exe_info.op)

        queries: List[Query] = [
            Query(qid, prompt) for qid, prompt in zip(exe_info.query_ids, exe_info.prompts)
        ]
        init_time = time.perf_counter() - init_start

        # Build inputs
        prefill_start = time.perf_counter()
        messages_batch: List[Any] = []
        if self.use_chat_template:
            # Use HF chat template structure
            sys_text = "\n".join([x for x in [self.prefix, self.common_message] if x]).strip()
            for q in queries:
                messages_batch.append(
                    [
                        {"role": "system", "content": sys_text},
                        {"role": "user", "content": q.prompt},
                    ]
                )
            inputs: List[str] = self.tokenizer.apply_chat_template(
                messages_batch,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Plain string prompts (system/common + user)
            inputs: List[str] = []
            for q in queries:
                header = "\n".join([x for x in [self.prefix, self.common_message] if x]).strip()
                if header:
                    inputs.append(f"{header}\n\n{q.prompt}")
                else:
                    inputs.append(q.prompt)

        # Inference
        outputs = self.llm.generate(inputs, self.sampling_params)  # type: ignore[arg-type]

        # Collect results
        results: List[Dict[str, Any]] = []
        for i, output in enumerate(outputs):
            gen_text = output.outputs[0].text if output.outputs else ""
            # Mirror original behavior: prepend the exact input for reproducibility
            if isinstance(inputs[i], str):
                full_text = inputs[i] + gen_text
            else:
                # Fallback (should not happen if apply_chat_template returned strings)
                full_text = "".join([m.get("content", "") for m in inputs[i]]) + gen_text

            results.append(
                {
                    "id": queries[i].id,
                    "output": full_text,
                    "benchmark": (prefill_start, time.perf_counter()),
                }
            )

        generate_time = time.perf_counter() - prefill_start
        if self.is_duplicate:
            benchmark = {"init_time": 0.0, "prefill_time": 0.0, "generate_time": 0.0}
        else:
            benchmark = {"init_time": init_time, "prefill_time": 0.0, "generate_time": generate_time}

        return {"item": results, "op_name": self.op_name, "benchmark": benchmark}

    def exit(self) -> str:
        """Release model/tokenizer resources and clear CUDA cache."""
        try:
            if self.llm is not None:
                del self.llm
        except Exception:
            pass
        try:
            if self.tokenizer is not None:
                del self.tokenizer
        except Exception:
            pass
        self.llm = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        logging.info("vLLMWorker[%s] exited.", self.id)
        return "Worker exited."

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def run(self, debug: bool = True) -> None:
        """
        Command loop consuming tasks from cmd_queue and writing
        structured results to response_queue.

        Supported commands:
          - "execute": params = (ExecuteInfo,)
          - "exit":    params = ()
        """
        while True:
            msg = self.cmd_queue.get()
            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                self.response_queue.put(
                    {"command": "error", "result": "Unsupported message format", "elapsed_time": 0.0}
                )
                continue

            if command == "exit":
                out = self.exit()
                self.response_queue.put({"command": "exit", "result": out, "elapsed_time": 0.0})
                break

            func = getattr(self, command, None)
            if not callable(func):
                self.response_queue.put(
                    {"command": "error", "result": f"Unknown command: {command}", "elapsed_time": 0.0}
                )
                continue

            start = time.perf_counter()
            try:
                result = func(*params) if params is not None else func()  # type: ignore[misc]
                elapsed = time.perf_counter() - start
                self.response_queue.put({"command": command, "result": result, "elapsed_time": elapsed})
            except Exception as e:
                if debug:
                    # Surface the error to ease debugging; parent receives nothing in this branch.
                    raise
                self.response_queue.put({"command": "error", "result": repr(e), "elapsed_time": 0.0})


if __name__ == '__main__':
    import queue
    from halo.components import Query, Operator, ModelConfig, ExecuteInfo

    query = Query(0, "What is the capital of France?")
    model = "meta-llama/Llama-3.2-3B-Instruct"

    worker = vLLMWorker(
        id='1',
        device='cuda:1',
        cmd_queue=queue.Queue(),
        result_queue=queue.Queue(),
    )
    
    config = ModelConfig(
        model_name=model,
        system_prompt='You are a helpful assistant.',
    )

    op = Operator(
        id='op_0',
        model_config=config,
        keep_cache=False,
    )
    
    exe_info = ExecuteInfo(
        op=op,
        query_ids=[query.id],
        prompts=[query.prompt],
    )
    out = worker.execute(exe_info)
    print(out)