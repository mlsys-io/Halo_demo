import time
import queue
import torch
import torch.multiprocessing as mp
from logging import getLogger
from typing import Dict, List

from halo.workers import TransformersWorker
from halo.schedulers import schedule_workflows
from halo.parser import load_config, build_ops_from_config
from halo.components import ExecuteInfo, Query  # types only
from halo.util import _visible_physical_gpu_ids

logger = getLogger(__name__)
logger.setLevel("INFO")


class Optimizer:
    """
    Orchestrates a multi-process, multi-GPU workflow over a DAG of ops.

    - Parse YAML config (via parser.py) to build the op graph (Operator instances).
    - Spawn worker processes and manage IPC queues.
    - Delegate device workflows to `schedule_workflows`.
    - Collect results, manage caches, and aggregate per-op benchmarks.
    - NEW: Only dispatch an `execute` task when all its parent-op outputs exist for
      every query in that task (dependency-aware dispatch).
    """

    def __init__(self, config_path: str, **kwargs):
        self.config = load_config(config_path)
        mp.set_start_method("spawn", force=True)

        self.device_cnt = torch.cuda.device_count()
        self.processes: List[mp.Process] = []
        self.cmd_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []
        self.communication_queues: List[mp.Queue] = []
        self.cache: Dict[str, dict] = {}
        self.dp_threshold = self.config.get("dp_threshold", 2)

        # Build ops from config
        self.ops, self.start_ops, self.end_ops, self.models = build_ops_from_config(self.config)
        self._create_workers()

        # Ablation toggles
        self.disable_rerank = True
        logger.info("Optimizer initialized")

    # ---------------- Workers ---------------- #

    def _create_workers(self) -> None:
        """Create per-device queues and spawn worker processes."""
        for _ in range(self.device_cnt):
            self.cmd_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())
            self.communication_queues.append(mp.Queue())

        phys_ids = _visible_physical_gpu_ids()
        if not phys_ids:
            raise RuntimeError("No visible GPUs. Set CUDA_VISIBLE_DEVICES or ensure GPUs are available.")
        phys_ids = phys_ids[: self.device_cnt]

        for i, phys_id in enumerate(phys_ids):
            proc = mp.Process(
                target=worker_process,
                args=(
                    i,
                    phys_id,
                    self.cmd_queues[i],
                    self.result_queues[i],
                    self.communication_queues,
                    list(self.models),
                ),
                daemon=True,
            )
            self.processes.append(proc)
            proc.start()

    # ---------------- Query Handling ---------------- #

    def _optimize_queries(self, queries: List[Query]) -> None:
        """Optional re-ranking: by priority DESC, then prompt length ASC."""
        if self.disable_rerank:
            self.queries = queries
        else:
            self.queries = sorted(queries, key=lambda q: (-q.priority, q.prompt_len))

    # ---------------- Schedule (delegated) ---------------- #

    def schedule(self, queries: List[Query]) -> None:
        """
        Build per-device workflows via the external scheduler.
        Side effects:
          - self.workflows
          - self.req_id_map
        """
        self.workflows = [[] for _ in range(self.device_cnt)]
        self._optimize_queries(queries)
        self.queries_cnt = len(self.queries)
        self.req_id_map = {q.id: q for q in self.queries}

        self.workflows = schedule_workflows(
            device_cnt=self.device_cnt,
            start_ops=self.start_ops,
            end_ops=self.end_ops,
            all_ops=list(self.ops.values()),
            queries=self.queries,
            req_id_map=self.req_id_map,
            dp_threshold=self.dp_threshold,
        )

    # ---------------- Execute (dependency-aware) ---------------- #

    def execute(self, queries: List[Query] = None, return_queries: bool = False, skip_exit: bool = False):
        """
        Push tasks to workers following the workflows, collect results, update caches/benchmarks.

        Dependency guard:
        - Before sending an `execute` task, verify that for ALL queries in that task,
          outputs from ALL parent ops are already available.
        - If not ready, the task is not dispatched yet; after any task completes, the
          optimizer retries dispatch on all workers to avoid stalls.
        """
        if queries is not None:
            _ = time.perf_counter()
            self.schedule(queries)

        finish_flags = [False] * self.device_cnt     # This worker has no more tasks
        inflight = [False] * self.device_cnt         # A task is running on this worker
        worker_pointer = [0] * self.device_cnt       # Next task index per worker

        def _cmd_transfer(task: Dict) -> Dict:
            """
            Translate a logical task into a concrete worker call:
              - resume_cache -> materialize cache payload
              - execute -> build ExecuteInfo(op, query_ids, prompts)
            """
            if task["command"] == "resume_cache":
                cache = self._cache_resume(task["params"][0], task["params"][1])
                task["params"] = (cache,)

            elif task["command"] == "execute":
                op, query_ids = task["params"][0], task["params"][1]
                prompts = []
                for qid in query_ids:
                    prompt = self.req_id_map[qid].prompt
                    if isinstance(prompt, list):
                        step = self.req_id_map[qid].step
                        prompt = prompt[step]
                    history_seqs = [self.req_id_map[qid].op_output.get(inp.id, "") for inp in op.input_ops]
                    history = "".join(history_seqs)
                    prompts.append(prompt + history)
                execute_info = ExecuteInfo(op=op, query_ids=query_ids, prompts=prompts)
                task["params"] = (execute_info,)
            return task

        def _task_ready(task: Dict) -> bool:
            """
            Return True if this task can be dispatched:
            - Non-`execute` commands are always ready.
            - For `execute(op, query_ids)`, all parent outputs must exist
              for every query in `query_ids`.
            """
            if task.get("command") != "execute":
                return True
            op, query_ids = task["params"][0], task["params"][1]
            if not getattr(op, "input_ops", None):
                return True  # start op
            parent_ids = [p.id for p in op.input_ops]
            for qid in query_ids:
                q = self.req_id_map[qid]
                for pid in parent_ids:
                    if pid not in q.op_output:
                        return False
            return True

        def _try_send(i: int) -> None:
            """
            Attempt to send the current task for worker i if:
              - worker not finished,
              - no task currently in flight on that worker,
              - a task exists and dependencies are satisfied.
            """
            if finish_flags[i] or inflight[i]:
                return

            if worker_pointer[i] >= len(self.workflows[i]):
                # No more tasks for this worker
                finish_flags[i] = True
                if not skip_exit:
                    self.cmd_queues[i].put(("exit", ()))
                return

            task = self.workflows[i][worker_pointer[i]]
            if _task_ready(task):
                self.cmd_queues[i].put(_cmd_transfer(task))
                inflight[i] = True  # mark as running

        exe_start = time.perf_counter()

        # Initial dispatch attempt on all workers
        for i in range(self.device_cnt):
            _try_send(i)

        # Main loop: gather results and (re)dispatch tasks
        while not all(finish_flags):
            made_progress = False
            for i in range(self.device_cnt):
                if finish_flags[i] or not inflight[i]:
                    continue  # either finished or idle (no in-flight task yet)

                try:
                    message = self.result_queues[i].get(timeout=0.1)
                except queue.Empty:
                    continue

                # A message arrived -> the in-flight task for worker i is done
                inflight[i] = False
                made_progress = True

                # Defensive: non-dict message (like "error: ...")
                if not isinstance(message, dict):
                    logger.warning("Worker %d returned non-dict message: %r", i, message)
                    # Advance pointer to avoid deadlock on malformed message
                    worker_pointer[i] += 1
                    _try_send(i)
                    # Other workers might now be unblocked
                    for j in range(self.device_cnt):
                        if not finish_flags[j]:
                            _try_send(j)
                    continue

                cmd = message.get("command")

                if cmd == "dump_cache":
                    op_name = message["result"]["op_name"]
                    self.cache[op_name] = message["result"]["item"]

                elif cmd == "execute":
                    result = message["result"]
                    op_name = result["op_name"]
                    for rec in result["item"]:
                        q = self.req_id_map[rec["id"]]
                        q.op_output[op_name] = rec["output"]
                        q.step += 1
                        q.benchmark[op_name] = rec["benchmark"]
                    # Merge per-op benchmark
                    self.ops[op_name].benchmark.update(result["benchmark"])

                elif cmd == "error":
                    logger.error("Worker %d error: %s", i, message.get("result"))

                elif cmd == "exit":
                    # Worker acknowledged exit; nothing else to do here.
                    pass

                # Move pointer and try to send next for this worker
                worker_pointer[i] += 1
                _try_send(i)

                # After any completion, try dispatch on all workers
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

            # If no progress this tick, do a gentle sweep to try unblocked tasks
            if not made_progress:
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

        if return_queries:
            return self.queries
        return time.perf_counter() - exe_start

    # ---------------- Housekeeping ---------------- #

    def exit(self):
        """Close all queues and join processes."""
        for q in self.cmd_queues + self.result_queues + self.communication_queues:
            q.close()
            q.join_thread()
        for p in self.processes:
            p.join()
        logger.info("Optimizer exited")

    # ---------------- Helpers ---------------- #

    def _cache_resume(self, op_id: str, query_ids: List[int]) -> dict:
        """Return cached KV slices for the given op and a subset of query ids."""
        new_cache = {}
        cache = self.cache.get(op_id, None)
        if cache is not None:
            for qid in query_ids:
                if qid in cache:
                    new_cache[qid] = cache[qid]
        return new_cache


def worker_process(id, phys_id, cmd_queue, result_queue, communication_queues, models):
    worker = TransformersWorker(id, phys_id, cmd_queue, result_queue, communication_queues, models)
    worker.run()


if __name__ == "__main__":
    import os, logging

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/halo.log",
        filemode="w",
        format="%(asctime)s %(processName)s[%(process)d] %(levelname)s %(name)s: %(message)s",
    )

    opt = Optimizer("templates/adv_reason.yaml")
    queries = [Query(i, "What is Machine Learning System?") for i in range(8)]
    opt.schedule(queries)
    queries = opt.execute(return_queries=True)
    for q in queries:
        print(f"Query {q.id} result: {q.op_output}")
        break

    for op in opt.ops.values():
        print(f"Op {op.id} benchmark: {op.benchmark}")
    opt.exit()
