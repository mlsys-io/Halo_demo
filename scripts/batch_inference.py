#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified benchmark runner (clean version, only halo_v and halo_t).

- Single CLI to run different optimizers.
- Loads a YAML template to fetch dataset spec and builds batched requests.
- Runs the selected optimizer and prints summary stats.
"""

from __future__ import annotations

import os
import time
import argparse
from typing import List, Tuple, Dict, Any

import torch
import yaml
from datasets import load_dataset, concatenate_datasets

from halo.components import Query


# -------------------------------------------------------------------
# Optimizer Loader
# -------------------------------------------------------------------

def load_optimizer(name: str):
    """
    Lazy-import and return an optimizer class/factory by name.
    The returned object must be callable like: optimizer_fun(template_path) -> optimizer_instance
    """
    name = name.lower().strip()

    if name == "halo_v":
        from halo.optimizers import Optimizer_v as opt
        return opt
    if name == "halo_t":
        from halo.optimizers import Optimizer_t as opt
        return opt

    raise ValueError(f"Unknown optimizer: {name}")


# -------------------------------------------------------------------
# Dataset & Request Preparation
# -------------------------------------------------------------------

def load_and_prepare_dataset(template_path: str, num_requests: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load the YAML template for dataset spec, fetch the split via datasets.load_dataset,
    and repeat if necessary to reach num_requests.
    Returns (dataset_rows_as_list_of_dicts, template_config).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset(config["dataset"], name=config.get("name"), split="train")

    if num_requests > len(dataset):
        times = num_requests // len(dataset) + 1
        dataset = concatenate_datasets([dataset] * times)

    dataset = dataset.select(range(num_requests))
    return list(dataset), config


def build_requests(samples: List[Dict[str, Any]], text_key: str, max_input_len: int) -> List[Query]:
    """
    Build a list of Query(id, text) objects from dataset samples, extracting
    `text_key` from each row and truncating to `max_input_len` characters.
    """
    requests: List[Query] = []
    for i, row in enumerate(samples):
        raw = row.get(text_key, "")
        text = raw if isinstance(raw, str) else str(raw)
        text = text[:max_input_len] if max_input_len and max_input_len > 0 else text
        requests.append(Query(i, text))
    return requests


# -------------------------------------------------------------------
# Benchmark Summary
# -------------------------------------------------------------------

def summarize_benchmarks(opt) -> None:
    """
    Print per-operator (or per-node) average timings.
    Compatible with both 'opt.ops' and 'opt.nodes' attributes.
    Expects each item to expose a .benchmark dict or object with:
      init_time, prefill_time, generate_time.
    """
    # Collect Operator/Node objects
    items = None
    if hasattr(opt, "ops"):
        items = list(getattr(opt, "ops").values())
    elif hasattr(opt, "nodes"):
        items = list(getattr(opt, "nodes").values())
    else:
        print("[WARN] Optimizer has neither 'ops' nor 'nodes'; skipping per-operator summary.")
        return

    if not items:
        print("[WARN] No operators/nodes found in optimizer; skipping summary.")
        return

    def _get_val(bm, key: str) -> float:
        # Support dict-style benchmark or object-style attributes
        if isinstance(bm, dict):
            return float(bm.get(key, 0.0))
        return float(getattr(bm, key, 0.0))

    init_times, prefill_times, generate_times = [], [], []
    for it in items:
        bm = getattr(it, "benchmark", {})
        init_times.append(_get_val(bm, "init_time"))
        prefill_times.append(_get_val(bm, "prefill_time"))
        generate_times.append(_get_val(bm, "generate_time"))

    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    print("[OPTIMIZER] Operator benchmark (avg ms):")
    print(f"  init_time:     {_avg(init_times):.2f}")
    print(f"  prefill_time:  {_avg(prefill_times):.2f}")
    print(f"  generate_time: {_avg(generate_times):.2f}")


# -------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean benchmark runner (only halo_v and halo_t)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["halo_v", "halo_t"],
        default="halo_v",
        help="Choose optimizer to benchmark (default: halo_v)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="templates/adv_reason_3.yaml",
        help="Path to YAML template",
    )
    parser.add_argument(
        "-n",
        "--num_queries",
        type=int,
        default=250,
        help="Number of queries to process (default: 250)",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Maximum input length per request (default: 256)",
    )

    args = parser.parse_args()

    # Instantiate optimizer
    optimizer_fun = load_optimizer(args.optimizer)
    print(f"[INFO] Using optimizer: {args.optimizer}")

    # Load dataset and build Query objects
    samples, cfg = load_and_prepare_dataset(args.template, args.num_queries)
    column = cfg["column"]
    opt = optimizer_fun(args.template)
    requests = build_requests(samples, column, args.max_input_length)

    # Run benchmark
    t0 = time.perf_counter()
    total_time = opt.execute(requests)
    t1 = time.perf_counter()

    # Print summary
    print(f"[OPTIMIZER] End-to-end time (optimizer): {total_time:.4f}s")
    print(f"[OPTIMIZER] Wall-clock measured:        {(t1 - t0):.4f}s")
    summarize_benchmarks(opt)

    # Cleanup
    opt.exit()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
