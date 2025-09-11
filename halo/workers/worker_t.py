import time
import queue
import threading
from typing import Dict, List, Tuple, Optional

import torch
import pynvml
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, LlamaForCausalLM

# Project-local types
from halo.components import Query, Operator, ModelConfig, ExecuteInfo
from halo.util import _resolve_dtype


class TransformersWorker:
    """
    Worker that executes an Operator with (optionally) multiple HF models:

      - Initializes models and tokenizers.
      - Builds a reusable KV prefix from an optional system prompt (prefill).
      - Streams generation with batched decode and cache reuse.
      - Offloads/prefetches KV caches CPU<->GPU via background threads.
      - Dynamically adjusts decode batch size by GPU utilization.

    The worker listens to commands on cmd_queue and responds via result_queue.
    All responses are dicts like:
        {"command": "<name|error|exit>", "result": <payload>, "elapsed_time": <float>}
    """

    def __init__(
        self,
        id: int,
        phys_id: int,
        cmd_queue,
        result_queue,
        communication_queues,
        models: List[str],
    ) -> None:
        self.id = id
        self.device = f"cuda:{id}"
        self.cmd_queue = cmd_queue
        self.result_queue = result_queue
        self.communication_queues = communication_queues  # one per device index

        # KV cache store: query_id -> [(K, V), ...]
        self.cache: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Runtime state
        self.model_name: Optional[str] = None
        self.prefill_batch_size: int = 4
        self.min_batch_size: Optional[int] = None  # set from op config on first run
        self.lower_util, self.upper_util = 85, 90  # GPU utilization targets
        self.adjust_interval = 5                   # adjust every N iterations

        # Background thread for offloading KV to CPU pinned memory
        self.offload_queue: "queue.Queue[Optional[Tuple[int, List[Tuple[torch.Tensor, torch.Tensor]]]]]" = queue.Queue()
        self.offload_thread = threading.Thread(target=self._offload_worker, daemon=True)
        self.offload_thread.start()

        # Background thread for preloading KV back to GPU
        self.preload_queue: "queue.Queue[Optional[List[Query]]]" = queue.Queue(maxsize=1)
        self.prefill_queue: "queue.Queue[List[Query]]" = queue.Queue(maxsize=1)
        self.preload_thread = threading.Thread(target=self._preload_worker, daemon=True)
        self.preload_thread.start()

        # Single vs multi-model mode
        if len(models) == 1:
            self.multiple_models = False
            self.init_model(models[0])
        else:
            self.multiple_models = True
            self.init_models(models)

        # NVML: utilization feedback
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.id)

        # Ablations
        self.disable_cache = False
        self.disable_dynamic_batch = False
        if self.disable_dynamic_batch:
            self.min_batch_size = 16

    # ---------------- Initialization ---------------- #

    def init_op(self, op: Operator) -> None:
        """
        Initialize per-op runtime settings and (optionally) warm a system prefix cache.
        """
        self.op_name = op.id
        self.is_duplicate = getattr(op, "is_duplicate", False)
        self.keep_cache = getattr(op, "keep_cache", False)

        cfg: ModelConfig = op.model_config
        self.max_batch_size = cfg.max_batch_size
        if self.min_batch_size is None:
            self.min_batch_size = min(64, max(1, self.max_batch_size // 2))

        self.max_gen_tokens = cfg.max_tokens
        self.use_chat_template = getattr(cfg, "use_chat_template", False)
        self.dtype = _resolve_dtype(cfg.dtype)
        self.lora_config = getattr(cfg, "lora_config", None)
        self.temperature = getattr(cfg, "temperature", 0.7)
        self.quantization = getattr(cfg, "quantization", None)

        # Multi-model: switch active model/tokenizer if needed
        if self.multiple_models and cfg.model_name != self.model_name:
            self.model = self.models[cfg.model_name].to(self.device)
            self.tokenizer = self.tokenizers[cfg.model_name]
            self.model_name = cfg.model_name

        # Optional system prompt prefix caching
        self.prompt = getattr(cfg, "system_prompt", None)
        if self.prompt and not self.disable_cache:
            out = self.prompt_prefill(self.prompt)
            self.sys_kv = out["past_key_values"]
            self.prefix_len = self.sys_kv[0][0].size(2)
        else:
            self.sys_kv = None
            self.prefix_len = 0

    def init_model(self, model_name: str, dtype: str = "bfloat16") -> None:
        """
        Initialize a single model/tokenizer pair and bind to the target device.
        """
        self.model_name = model_name
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,               # 'cuda:0' | 'auto' | dict
            torch_dtype=_resolve_dtype(dtype),
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._pad_token = self.tokenizer.pad_token
        self.eos_id = self.tokenizer.eos_token_id

    def init_models(self, model_list: List[str], dtype: str = "bfloat16", attn: str = "flash_attention_2") -> None:
        """
        Initialize multiple models/tokenizers, keeping models on CPU until selected.
        """
        self.models: Dict[str, LlamaForCausalLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}

        for name in model_list:
            mdl = LlamaForCausalLM.from_pretrained(
                name,
                torch_dtype=_resolve_dtype(dtype),
                attn_implementation=attn,
            ).eval()
            self.models[name] = mdl.to("cpu")

            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            tok.padding_side = "left"
            tok.pad_token = tok.eos_token
            self.tokenizers[name] = tok

        any_tok = next(iter(self.tokenizers.values()))
        self._pad_token = any_tok.pad_token
        self.eos_id = any_tok.eos_token_id

    # ---------------- Cache APIs (used by optimizer) ---------------- #

    def resume_cache(self, cache_dict: dict):
        """Merge a batch of KV slices into this worker's local cache."""
        self.cache_append(cache_dict)
        return "Cache resumed."

    def dump_cache(self):
        """Dump and clear worker's local cache for the current op."""
        if self.disable_cache:
            return {"item": {}, "op_name": self.op_name}
        caches = self.cache.copy()
        self.cache.clear()
        return {"item": caches, "op_name": self.op_name}

    def get_cache(self):
        """Receive a KV dict from the shared communication queue (for this device)."""
        incoming = self.communication_queues[self.id].get()
        self.cache_append(incoming)
        return "Cache received."

    def send_cache(self, target_id: int, query_ids: List[int]):
        """Send only needed queries' KV to another device via communication queue."""
        if self.disable_cache:
            self.communication_queues[target_id].put({})
            return "Cache disabled."
        out = {}
        for qid in query_ids:
            if qid in self.cache:
                out[qid] = self.cache.pop(qid)
        self.communication_queues[target_id].put(out)
        return "Cache sent."

    def complete(self):
        """Clear local cache after finishing an end-op."""
        self.cache.clear()
        return "Cache cleared."

    # ---------------- Cache Utilities ---------------- #

    @staticmethod
    def cache_replicate(cache: List[Tuple[torch.Tensor, torch.Tensor]], bs: int):
        """Replicate a single-sample KV cache to a batch of size `bs`."""
        return [(k.repeat(bs, 1, 1, 1), v.repeat(bs, 1, 1, 1)) for k, v in cache]

    @staticmethod
    def cache_popleft(cache: List[Tuple[torch.Tensor, torch.Tensor]], length: int):
        """Drop the first `length` positions from KV time dimension."""
        return [(k[:, length:, :], v[:, length:, :]) for k, v in cache]

    def cache_append(self, cache_dict: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]) -> None:
        """Merge incoming KV chunks into worker cache by query id."""
        for qid, new_kv in cache_dict.items():
            if qid in self.cache:
                merged = []
                for (ok, ov), (nk, nv) in zip(self.cache[qid], new_kv):
                    merged.append((torch.cat([ok, nk], dim=1), torch.cat([ov, nv], dim=1)))
                self.cache[qid] = merged
            else:
                self.cache[qid] = new_kv

    def cache_save(self, qid: int, cache) -> None:
        """Offload KV cache to CPU pinned memory via background thread."""
        self.offload_queue.put((qid, cache))

    # ---------------- Background Threads ---------------- #

    def _offload_worker(self) -> None:
        """Move KV caches to CPU pinned memory to free GPU RAM."""
        while True:
            item = self.offload_queue.get()
            if item is None:
                break
            qid, past_kvs = item
            offloaded = [(k.detach().cpu().pin_memory(), v.detach().cpu().pin_memory()) for k, v in past_kvs]
            self.cache[qid] = offloaded
            self.offload_queue.task_done()

    def _preload_worker(self) -> None:
        """Bring KV caches back to GPU asynchronously using per-query streams."""
        while True:
            batch = self.preload_queue.get()
            if batch is None:
                break

            streams = {q.id: torch.cuda.Stream(device=self.device) for q in batch}
            for q in batch:
                cached = self.cache.pop(q.id, None)
                if cached is None:
                    q.cache = None
                    continue
                stream = streams[q.id]
                with torch.cuda.stream(stream):
                    q.cache = [(k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True))
                               for k, v in cached]

            for s in streams.values():
                s.synchronize()

            self.prefill_queue.put(batch)
            self.preload_queue.task_done()

    # ---------------- Command Loop ---------------- #

    def run(self, debug: bool = True) -> None:
        """
        Process commands from cmd_queue and push dict messages to result_queue:
          {"command": "<name|error|exit>", "result": <payload>, "elapsed_time": <float>}
        """
        while True:
            msg = self.cmd_queue.get()
            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                self.result_queue.put({"command": "error", "result": "Unsupported message format", "elapsed_time": 0.0})
                continue

            if command == "exit":
                out = self.exit()
                self.result_queue.put({"command": "exit", "result": out, "elapsed_time": 0.0})
                break

            func = getattr(self, command, None)
            if not callable(func):
                self.result_queue.put({"command": "error", "result": f"Unknown command: {command}", "elapsed_time": 0.0})
                continue

            start = time.perf_counter()
            try:
                result = func(*params) if params is not None else func()
                elapsed = time.perf_counter() - start
                self.result_queue.put({"command": command, "result": result, "elapsed_time": elapsed})
            except Exception as e:
                if debug:
                    raise
                self.result_queue.put({"command": "error", "result": repr(e), "elapsed_time": 0.0})

    def exit(self) -> str:
        """
        Gracefully stop background threads and release resources.
        NOTE: do NOT put ('exit', ...) into self.cmd_queue here.
        """
        self.offload_queue.put(None)
        self.preload_queue.put(None)
        self.offload_thread.join()
        self.preload_thread.join()
        return "Worker exited."

    # ---------------- Execute One Round ---------------- #

    @torch.inference_mode()
    def execute(self, exe_info: ExecuteInfo):
        """
        Execute one round:
          - Initialize op and optional system prefix cache.
          - Overlap prefill (batched) and decode (streaming).
          - Dynamically adjust decode batch size by GPU utilization.
        """
        prefill_time, generate_time = 0.0, 0.0

        init_start = time.perf_counter()
        self.init_op(exe_info.op)

        decoding_queries: List[Query] = []
        wait_prefill: List[Query] = [Query(qid, p) for qid, p in zip(exe_info.query_ids, exe_info.prompts)]
        init_time = time.perf_counter() - init_start

        results = []
        iters = 0
        min_bs, max_bs = max(1, self.max_batch_size // 2), self.max_batch_size
        bs = min_bs
        prefill_enabled = True

        # Prime the first preload to overlap transfers and compute
        first = wait_prefill[: self.prefill_batch_size]
        wait_prefill = wait_prefill[self.prefill_batch_size :]
        self.preload_queue.put(first)

        while True:
            # 1) Prefill (batched)
            if prefill_enabled:
                batch = self.prefill_queue.get()
                t0 = time.perf_counter()
                prefilled = self.prefill(batch)
                prefill_time += time.perf_counter() - t0
                decoding_queries.extend(prefilled)

                nxt = wait_prefill[: self.prefill_batch_size]
                if not nxt:
                    prefill_enabled = False
                else:
                    self.preload_queue.put(nxt)
                    wait_prefill = wait_prefill[self.prefill_batch_size :]

            # 2) Decode (streaming one step)
            t0 = time.perf_counter()
            step_batch = decoding_queries[:bs]
            decoding_queries = decoding_queries[bs:]
            active, finished = self.generate(step_batch, max_new_tokens=self.max_gen_tokens)
            results.extend(finished)
            decoding_queries.extend(active)
            generate_time += time.perf_counter() - t0

            # 3) Dynamic batch size schedule
            if not self.disable_dynamic_batch:
                iters += 1
                if iters % self.adjust_interval == 0:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    if util < self.lower_util and bs < max_bs:
                        bs += 1
                    elif util > self.upper_util and bs > min_bs:
                        bs -= 1

            # 4) Exit condition
            if not decoding_queries and not wait_prefill:
                break

        # Benchmarks
        if getattr(self, "is_duplicate", False):
            benchmark = {"init_time": 0.0, "prefill_time": 0.0, "generate_time": 0.0}
        else:
            benchmark = {"init_time": init_time, "prefill_time": prefill_time, "generate_time": generate_time}

        self.min_batch_size = int(bs * 0.75) if bs > 1 else 1
        return {"item": results, "op_name": self.op_name, "benchmark": benchmark}

    # ---------------- Model I/O ---------------- #

    @torch.inference_mode()
    def prompt_prefill(self, prompt: str):
        """Run a forward pass over system prompt to build prefix KV cache."""
        if self.use_chat_template:
            messages = [{"role": "system", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        return self.model(input_ids=input_ids)

    def pre_prefill(self, queries: List[Query]):
        """
        Prepare inputs and padded KVs for prefill.
        Returns: input_ids, attention_mask, padded_past, cache_lens, cache_start_pos, seq_lengths
        """
        bs = len(queries)
        device, dtype = self.device, self.dtype
        prefix_cache = self.sys_kv
        past_key_values = [q.cache for q in queries]

        # 1) Build texts
        texts = []
        for q in queries:
            prompt = q.prompt
            if self.disable_cache and self.prompt:
                prompt = self.prompt + "\n\n" + prompt
            if self.use_chat_template:
                text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
            else:
                text = prompt
            texts.append(text)

        # 2) Tokenize + pad
        toks = self.tokenizer(texts, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = toks.input_ids.to(device)
        attn = toks.attention_mask.to(device)
        seq_lengths = attn.sum(dim=1)

        # 3) Assemble padded past KV (with optional prefix)
        if all(kv is None for kv in past_key_values):
            if prefix_cache is not None:
                padded_past = self.cache_replicate(prefix_cache, bs)
                max_cache = prefix_cache[0][0].shape[2]
                cache_lens = [max_cache] * bs
            else:
                padded_past = None
                cache_lens = [0] * bs
                max_cache = 0
        else:
            pre_len = prefix_cache[0][0].shape[2] if prefix_cache else 0
            cache_lens = []
            for kv in past_key_values:
                cache_lens.append(pre_len if kv is None else kv[0][0].shape[1] + pre_len)
            max_cache = max(cache_lens)

            padded_past = []
            n_layers = len(past_key_values[0] or prefix_cache)
            for l in range(n_layers):
                shape_ref = (
                    past_key_values[0][l][0].shape
                    if past_key_values[0] is not None
                    else prefix_cache[l][0][0].shape
                )
                # [bs, heads, max_cache, dim]
                shape = (bs, shape_ref[0], max_cache, shape_ref[2])
                K = torch.zeros(shape, device=device, dtype=dtype)
                V = torch.zeros_like(K)

                for i, kv in enumerate(past_key_values):
                    if kv is not None:
                        k, v = kv[l]
                        if prefix_cache is not None:
                            k = torch.cat([prefix_cache[l][0][0], k], dim=1)
                            v = torch.cat([prefix_cache[l][1][0], v], dim=1)
                        K[i, :, -k.shape[1] :, :] = k
                        V[i, :, -v.shape[1] :, :] = v
                    elif prefix_cache is not None:
                        pre_k = prefix_cache[l][0][0]
                        pre_v = prefix_cache[l][1][0]
                        K[i, :, -pre_k.shape[1] :, :] = pre_k
                        V[i, :, -pre_v.shape[1] :, :] = pre_v
                padded_past.append((K, V))

        # 4) Attention over [cache + tokens]
        total_len = input_ids.size(1) + max_cache
        idx = torch.arange(total_len, device=device)
        cache_start = torch.tensor([max_cache - c for c in cache_lens], device=device)[:, None]
        mask = ((idx >= cache_start) & (idx < max_cache)) | (idx >= (total_len - seq_lengths).unsqueeze(1))

        return input_ids, mask, padded_past, cache_lens, cache_start.squeeze(1).tolist(), seq_lengths.tolist()

    def prefill(self, batch: List[Query]) -> List[Query]:
        """Run the batched prefill step and attach per-query GPU-local KVs."""
        t0 = time.perf_counter()
        input_ids, attention_mask, past_key_values, cache_lens, cache_pos, seq_lengths = self.pre_prefill(batch)

        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        logits = output["logits"]

        # Greedy next token
        next_ids = logits[:, -1].argmax(dim=-1).unsqueeze(1)
        decoded = self.tokenizer.batch_decode(next_ids, skip_special_tokens=True)

        new_past = output["past_key_values"].to_legacy_cache()
        for i, q in enumerate(batch):
            q.prefix_len = cache_lens[i]
            q.start_time = t0
            q.generated_tokens = seq_lengths[i] + cache_lens[i] + 1
            q.decoded_tokens = decoded[i]
            q.cache = self._retrieve_cache_prefill(new_past, i, seq_lengths[i], cache_lens[i], cache_pos[i])
            q.input_ids = next_ids[i]
        return batch

    @staticmethod
    def _retrieve_cache_prefill(past_key_values, idx: int, generated_tokens: int, cache_len: int, cache_pos: int):
        """Slice the proper [cache + newly generated] window for prefill."""
        start = cache_pos
        end = cache_pos + cache_len

        out = []
        for K_bvd, V_bvd in past_key_values:
            K1 = K_bvd[idx, :, start:end, :]
            K2 = K_bvd[idx, :, -generated_tokens:, :]
            V1 = V_bvd[idx, :, start:end, :]
            V2 = V_bvd[idx, :, -generated_tokens:, :]
            out.append((torch.cat([K1, K2], dim=1), torch.cat([V1, V2], dim=1)))
        return out

    @staticmethod
    def _retrieve_cache(past_key_values, idx: int, generated_tokens: int):
        """Slice the trailing window for streaming decode."""
        start = -generated_tokens
        return [(K_bvd[idx, :, start:, :], V_bvd[idx, :, start:, :]) for K_bvd, V_bvd in past_key_values]

    def _pre_generate(self, queries: List[Query]):
        """Prepare attention mask, padded KVs, and next-token input ids for one decode step."""
        device = self.device
        dtype = self.model.dtype
        bs = len(queries)

        input_ids = torch.stack([q.input_ids for q in queries], dim=0)  # [bs, 1]

        past_kvs = [q.cache for q in queries]
        cache_lens = [kv[0][0].shape[1] for kv in past_kvs]
        max_cache = max(cache_lens)

        # Attention over [cache + next token]
        seq_len = max_cache + 1
        idx = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_start = torch.tensor([max_cache - L for L in cache_lens], device=device).unsqueeze(1)
        attention_mask = idx >= cache_start

        # Pad KVs to [bs, heads, max_cache, dim]
        layers = len(past_kvs[0])
        padded_past = []
        for l in range(layers):
            heads, _, dim = past_kvs[0][l][0].shape
            K = torch.zeros((bs, heads, max_cache, dim), device=device, dtype=dtype)
            V = torch.zeros_like(K)
            for i, kv in enumerate(past_kvs):
                k, v = kv[l]
                seq = k.shape[1]
                K[i, :, max_cache - seq :, :] = k
                V[i, :, max_cache - seq :, :] = v
            padded_past.append((K, V))

        return attention_mask, padded_past, input_ids

    def generate(self, queries: List[Query], max_new_tokens: int = 64):
        """Execute one decode step; return (active_queries, finished_records)."""
        if not queries:
            return [], []

        attention_mask, past_key_values, input_ids = self._pre_generate(queries)
        past = DynamicCache.from_legacy_cache(past_key_values)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past)
        logits = outputs.logits
        new_past = outputs.past_key_values.to_legacy_cache()

        # Sample next tokens
        next_tokens = self.sample(logits[:, -1], temperature=self.temperature)
        decoded_texts = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)

        active, finished = [], []
        for i, q in enumerate(queries):
            token_id = next_tokens[i].item()
            q.decoded_tokens += decoded_texts[i]
            q.generated_tokens += 1
            q.new_tokens += 1

            kv = self._retrieve_cache(new_past, i, q.generated_tokens - 1)

            if token_id == self.eos_id or q.new_tokens >= max_new_tokens:
                end_time = time.perf_counter()
                if self.keep_cache and not self.disable_cache:
                    # Keep KV for downstream ops; return empty output
                    self.cache_save(q.id, kv)
                    output_text = ""
                else:
                    output_text = q.decoded_tokens

                finished.append({"id": q.id, "output": output_text, "benchmark": (q.start_time, end_time)})
                del q
            else:
                q.input_ids = next_tokens[i]
                q.cache = kv
                active.append(q)

        return active, finished

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        """Greedy when T==0; otherwise multinomial softmax sampling."""
        if temperature <= 0.0:
            return logits.argmax(dim=-1).unsqueeze(1)
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

if __name__ == '__main__':
    import queue
    from halo.components import Query, Operator, ModelConfig, ExecuteInfo

    # Example single query
    query = Query(0, "What is the capital of France?")
    model = "meta-llama/Llama-3.2-3B-Instruct"

    # Initialize worker
    worker = TransformersWorker(
        id=0,
        device="cuda:0",
        cmd_queue=queue.Queue(),
        result_queue=queue.Queue(),
        communication_queues={},
        models=[model],
    )
    worker.init_model(model, dtype="bfloat16", attn="eager")

    # Model config and op definition
    config = ModelConfig(
        model_name=model,
        system_prompt="You are a helpful assistant.",
    )
    op = Operator(
        id="op_1",
        model_config=config,
        keep_cache=False,
    )

    # Execution info
    exe_info = ExecuteInfo(
        op=op,
        query_ids=[query.id],
        prompts=[query.prompt],
    )

    # Run and print result
    out = worker.execute(exe_info)
    print(out)
