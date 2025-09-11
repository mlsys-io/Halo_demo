import time
import queue
import numpy as np
import torch
import torch.multiprocessing as mp
from logging import getLogger
from typing import Dict, List

from halo.workers import vLLMWorker
from halo.parser import load_config, build_ops_from_config
from halo.schedulers import (
    schedule_search,
    schedule_heuristic,
    schedule_rr,
    schedule_by_levels,
    schedule_colocate_parents,
    schedule_dp,
)
from halo.components import Operator, ExecuteInfo, Query  # types only
from halo.util import _visible_physical_gpu_ids

logger = getLogger(__name__)
logger.setLevel("INFO")


class Optimizer:
    """
    Multi-process orchestrator over a DAG of operators.

    - Build Operator graph via parser.py
    - Spawn vLLM workers
    - Schedule via schedulers/* to build device workflows
    - Execute workflows, collect outputs, and compute latency percentiles
    """

    def __init__(self, config_path: str, **kwargs):
        self.config = load_config(config_path)
        mp.set_start_method("spawn", force=True)

        self.device_cnt = torch.cuda.device_count()
        self.processes: List[mp.Process] = []
        self.cmd_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []
        self.dp_threshold = 2

        # Build ops
        self.ops, self.start_ops, self.end_ops, self.models = build_ops_from_config(self.config)
        self._create_workers()
        logger.info("Optimizer initialized")

    # ---------------- Workers ---------------- #

    def _create_workers(self) -> None:
        """
        Spawn one worker per visible physical GPU.
        Each worker process is restricted to exactly one physical GPU by
        narrowing CUDA_VISIBLE_DEVICES inside the child process.
        """
        # Per-worker queues
        for _ in range(self.device_cnt):
            self.cmd_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())

        # Resolve visible physical GPU IDs
        phys_ids = _visible_physical_gpu_ids()
        if not phys_ids:
            raise RuntimeError("No visible GPUs. Set CUDA_VISIBLE_DEVICES or ensure GPUs are available.")
        phys_ids = phys_ids[: self.device_cnt]

        # Spawn processes
        for i, physical_gpu_id in enumerate(phys_ids):
            proc = mp.Process(
                target=worker_process,
                args=(i, physical_gpu_id, self.cmd_queues[i], self.result_queues[i]),
                daemon=False,
            )
            self.processes.append(proc)
            proc.start()

    # ---------------- Query ordering ---------------- #

    def _optimize_queries(self, queries: List[Query]) -> None:
        """Sort queries by priority DESC, then by prompt length ASC."""
        self.queries = sorted(queries, key=lambda x: (-x.priority, x.prompt_len))

    # ---------------- Scheduling (choose strategy) ---------------- #

    def schedule(self, queries: List[Query], strategy: str = "heuristic") -> None:
        """
        Build self.workflows by selected strategy.
        """
        self._optimize_queries(queries)
        self.req_id_map = {q.id: q for q in self.queries}
        if strategy == "search":
            self.workflows = schedule_search(
                device_cnt=self.device_cnt,
                start_ops=self.start_ops,
                end_ops=self.end_ops,
                all_ops=list(self.ops.values()),
                queries=self.queries,
                beam_width=4,
            )
        elif strategy == "heuristic":
            self.workflows = schedule_heuristic(
                device_cnt=self.device_cnt,
                start_ops=self.start_ops,
                end_ops=self.end_ops,
                all_ops=list(self.ops.values()),
                queries=self.queries,
                dp_threshold=self.dp_threshold,
            )
        elif strategy == "rr":
            self.workflows = schedule_rr(self.device_cnt, list(self.ops.values()), self.queries)
        elif strategy == "levels":
            self.workflows = schedule_by_levels(self.device_cnt, list(self.ops.values()), self.queries)
        elif strategy == "colocate_parents":
            self.workflows = schedule_colocate_parents(self.device_cnt, list(self.ops.values()), self.queries)
        elif strategy == "dp":
            self.workflows = schedule_dp(self.device_cnt, list(self.ops.values()), self.queries)
        else:
            raise ValueError(f"Unknown schedule strategy: {strategy}")

    # ---------------- Execution loop ---------------- #

    def execute(self, queries: List[Query] = None, return_queries: bool = False, skip_exit: bool = False):
        """
        Dispatch tasks to workers and collect results.

        Added dependency guard:
        - Before sending a task to a worker, verify that for ALL queries in that task,
          outputs from ALL parent ops are already available.
        - If not ready, do not dispatch yet; retry after other tasks complete.
        """
        if queries is not None:
            self.schedule(queries, strategy="search")

        finish_flags = [False] * self.device_cnt      # Worker finished all tasks
        inflight = [False] * self.device_cnt          # A task is currently running on worker
        worker_pointer = [0] * self.device_cnt        # Index into workflows per worker

        def _cmd_transfer(task: Dict) -> Dict:
            """Translate (op, query_ids) to ExecuteInfo(op=..., query_ids=..., prompts=[...])."""
            if task["command"] == "execute":
                op, query_ids = task["params"][0], task["params"][1]
                prompts = []
                for qid in query_ids:
                    prompt = self.req_id_map[qid].prompt
                    if isinstance(prompt, list):
                        step = self.req_id_map[qid].step
                        prompt = prompt[step]
                    # Concatenate parents' outputs as history
                    history_seqs = [self.req_id_map[qid].op_output.get(inp.id, "") for inp in op.input_ops]
                    history = "".join(history_seqs)
                    prompts.append(prompt + history)
                exe = ExecuteInfo(op=op, query_ids=query_ids, prompts=prompts)
                task["params"] = (exe,)
            return task

        def _task_ready(task: Dict) -> bool:
            """
            True if this task can be dispatched:
            - Non-execute commands are always ready.
            - For execute(op, query_ids): all parents' outputs exist for every query.
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
              - task exists and all dependencies are satisfied.
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

        # Gathering loop
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

                # Defensive: ensure dict
                if not isinstance(message, dict):
                    logger.warning("Worker %d returned non-dict message: %r", i, message)
                    # Advance pointer to avoid deadlock on malformed message
                    worker_pointer[i] += 1
                    _try_send(i)
                    continue

                cmd = message.get("command")
                if cmd == "execute":
                    result = message.get("result", {})
                    op_name = result.get("op_name") or result.get("node_name")  # tolerate legacy key
                    if op_name is None:
                        logger.error("Worker %d result missing op_name/node_name: %r", i, result)
                    else:
                        # Apply per-query outputs
                        for rec in result.get("item", []):
                            q = self.req_id_map[rec["id"]]
                            q.op_output[op_name] = rec["output"]
                            q.step += 1
                            q.benchmark[op_name] = rec["benchmark"]
                        # Merge per-op benchmark
                        if "benchmark" in result and op_name in self.ops:
                            self.ops[op_name].benchmark.update(result["benchmark"])

                elif cmd == "error":
                    logger.error("Worker %d error: %s", i, message.get("result"))

                # Move pointer and try to send next for this worker
                worker_pointer[i] += 1
                _try_send(i)

                # After any task completes, other workers may get unblocked.
                # Try dispatch for *all* idle workers to avoid global stalls.
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

            # Optional: if no progress in this tick, we can attempt a light sweep
            if not made_progress:
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

        if return_queries:
            return self.queries
        return time.perf_counter() - exe_start

    # ---------------- Metrics & Exit ---------------- #

    def print_latency_percentiles(self):
        """Compute and print global P50/P95 latency across queries."""
        all_latencies = []
        for q in self.req_id_map.values():
            start_time = q.create_time
            points = list(q.benchmark.values())
            if not points:
                continue
            end_time = points[-1][-1]
            all_latencies.append(end_time - start_time)
        if not all_latencies:
            return
        p50 = float(np.percentile(all_latencies, 50))
        p95 = float(np.percentile(all_latencies, 95))
        print(f"Latency Percentiles: P50={p50:.3f}s, P95={p95:.3f}s")

    def exit(self):
        for q in self.cmd_queues + self.result_queues:
            q.close()
            q.join_thread()
        for p in self.processes:
            p.join()
        logger.info("Optimizer exited")


def worker_process(id, device, cmd_queue, result_queue):
    worker = vLLMWorker(id, device, cmd_queue, result_queue)
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

    from halo.components import Query

    opt = Optimizer("templates/adv_reason_3.yaml")
    queries = [Query(i, "What is Machine Learning System?") for i in range(8)]
    opt.schedule(queries)
    queries = opt.execute(return_queries=True)
    for q in queries:
        print(f"Query {q.id} result: {q.op_output}")
        break

    for op in opt.ops.values():
        print(f"Op {op.id} benchmark: {op.benchmark}")
    opt.exit()
