#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference runner for Halo.

Parses a declarative graph template, builds an optimized execution plan with the
chosen scheduler (DP by default), then runs a batch of queries through the
multi-process processor and prints throughput / latency.

Examples
--------
    python scripts/batch_inference.py --template templates/example_chain.yaml -n 64
    python scripts/batch_inference.py --scheduler greedy --gpus 4 --input-file queries.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from halo import GraphTemplateParser, GraphOptimizer, MultiProcessGraphProcessor


# A few canned prompts, repeated to reach the requested batch size when no
# --input-file is given.
_SAMPLE_QUERIES = [
    "What is a machine learning system?",
    "Explain prefix caching in LLM serving.",
    "Why does batching improve GPU utilization?",
    "Summarize the trade-offs of speculative decoding.",
]


def load_queries(input_file: str | None, num_queries: int) -> List[str]:
    """Build `num_queries` query strings, from a file (one per line) or samples."""
    if input_file:
        lines = [
            ln.strip()
            for ln in Path(input_file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        pool = lines or _SAMPLE_QUERIES
    else:
        pool = _SAMPLE_QUERIES
    return [pool[i % len(pool)] for i in range(num_queries)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Halo batch inference runner")
    parser.add_argument(
        "--template",
        type=str,
        default="templates/example_chain.yaml",
        help="Path to the YAML graph template",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dp",
        choices=["dp", "rr_topo", "random_topo", "model_first", "greedy", "minswitch", "milp", "auto"],
        help="Optimizer scheduler mode (default: dp)",
    )
    parser.add_argument(
        "--plan-mode",
        type=str,
        default="default",
        choices=["default", "profiled", "baseline"],
        help="'profiled' enables SQL EXPLAIN profiling (needs Postgres); 'default' skips it",
    )
    parser.add_argument("--gpus", type=int, default=None, help="Override detected GPU count")
    parser.add_argument("--cpu-workers", type=int, default=1, help="CPU workers for DB nodes")
    parser.add_argument("-n", "--num-queries", type=int, default=64, help="Batch size to run")
    parser.add_argument("--input-file", type=str, default=None, help="Optional file: one query per line")
    args = parser.parse_args()

    graph = GraphTemplateParser(args.template).parse()
    queries = load_queries(args.input_file, args.num_queries)
    contexts = [{"user_query": q} for q in queries]

    optimizer = GraphOptimizer(
        num_gpus=args.gpus,
        num_cpu_workers=args.cpu_workers,
        scheduler_mode=args.scheduler,
        plan_mode=args.plan_mode,
    )

    t0 = time.perf_counter()
    plan = optimizer.build_plan(graph, sample_contexts=contexts[:1], input_query_count=len(queries))
    t1 = time.perf_counter()
    print(f"[PLAN] scheduler={args.scheduler} build_plan={t1 - t0:.3f}s tasks={len(plan.tasks)}")
    for task in plan.tasks:
        print(f"  node={task.node_id!r} worker={task.worker_id!r} epoch={getattr(task, 'epoch', None)}")

    processor = MultiProcessGraphProcessor(persistent_workers=True)
    try:
        t2 = time.perf_counter()
        results = processor.run_batch(plan, graph, contexts)
        t3 = time.perf_counter()
    finally:
        processor.close()

    exec_s = max(t3 - t2, 1e-6)
    print(f"[RUN] batch={len(queries)} exec={exec_s:.3f}s throughput={len(queries) / exec_s:.2f} q/s")


if __name__ == "__main__":
    main()
