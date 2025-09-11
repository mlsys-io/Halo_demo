#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire random prompts at a FastAPI endpoint at a fixed rate.

- Loads a dataset spec from a YAML template (dataset / name / column).
- Picks prompts uniformly at random from the specified column.
- Sends POST requests to /query (configurable) at a target rate (req/s).
- Stops after N requests (if provided) or runs until interrupted.
- Optionally sends a final /exit request to signal graceful shutdown.
- Optionally attaches a per-request "callback_url" for the server to push results back.

Example:
    python send_load.py -f templates/adv_reason.yaml -H localhost -P 8000 -r 2.5 -n 200 --random-priority
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import requests
import yaml
from datasets import load_dataset


def _load_prompts_from_template(tpl_path: str) -> List[str]:
    """Load the dataset per template and return a list of string prompts."""
    with open(tpl_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "dataset" not in cfg:
        raise ValueError("Template missing required key: 'dataset'")
    if "column" not in cfg:
        raise ValueError("Template missing required key: 'column'")

    dataset_name = cfg["dataset"]
    subset_name = cfg.get("name")  # optional
    column = cfg["column"]

    ds = load_dataset(dataset_name, name=subset_name, split="train")
    if column not in ds.column_names:
        raise ValueError(
            f"Column '{column}' not found in dataset. Available: {ds.column_names}"
        )

    # Defensive: coerce to str; also flatten list-like fields if any
    prompts: List[str] = []
    for v in ds[column]:
        if isinstance(v, list):
            v = "".join(str(x) for x in v)
        prompts.append(str(v))
    if not prompts:
        raise ValueError("No prompts found in the selected dataset/column.")
    return prompts


def _build_payload(prompt: str, priority: int, random_priority: bool, callback_url: str | None) -> dict:
    """Build the JSON payload with prompt, priority, and optional callback_url."""
    if random_priority:
        priority = random.randint(1, 5)
    payload = {"prompt": prompt, "priority": priority}
    if callback_url:
        payload["callback_url"] = callback_url
    return payload


def _stable_sleep(next_tick: float, interval: float) -> float:
    """
    Sleep so that requests are sent at fixed intervals, independent of request time.
    Returns the updated 'next_tick' for the next iteration.
    """
    now = time.monotonic()
    sleep_s = next_tick - now
    if sleep_s > 0:
        time.sleep(sleep_s)
    return next_tick + interval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send random prompts to a FastAPI server at a given rate, stopping after N requests."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="templates/adv_reason.yaml",
        help="Path to the YAML template (default: templates/adv_reason.yaml)",
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port", "-P",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--endpoint", "-e",
        type=str,
        default="/query",
        help="Request endpoint path (default: /query)",
    )
    parser.add_argument(
        "--exit-endpoint",
        type=str,
        default="/exit",
        help="Exit endpoint path to call once finished (default: /exit)",
    )
    parser.add_argument(
        "--rate", "-r",
        type=float,
        required=True,
        help="Target request rate in requests per second (e.g., 2.5)",
    )
    parser.add_argument(
        "--priority", "-p",
        type=int,
        default=0,
        help="Default priority included in each request (default: 0)",
    )
    parser.add_argument(
        "--random-priority",
        action="store_true",
        help="If set, priority is randomized uniformly in [1, 5] per request",
    )
    parser.add_argument(
        "--callback-url",
        type=str,
        default=None,
        help="Optional callback URL appended to each request body for server push (e.g., http://client:8001/callback)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        help="Total number of requests to send; if omitted, run until interrupted",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be > 0")

    # Load prompts from template/dataset
    prompts = _load_prompts_from_template(args.file)

    # Build URLs
    base = f"http://{args.host}:{args.port}"
    url = base + args.endpoint
    exit_url = base + args.exit_endpoint
    interval = 1.0 / args.rate

    print(f"→ Sending to {url} at {args.rate:.2f} req/s (interval {interval:.3f}s)")
    print(f"→ Loaded {len(prompts)} prompts from {args.file!r}")
    if args.count:
        print(f"→ Will send {args.count} requests then stop.")
    else:
        print("→ Will run until interrupted (Ctrl+C).")
    if args.random_priority:
        print("→ Random priority enabled: [1..5] per request.")
    if args.callback_url:
        print(f"→ Attaching callback_url={args.callback_url}")
    print()

    sent = 0
    session = requests.Session()  # reuse connections

    try:
        next_tick = time.monotonic()  # first tick reference

        if args.count is not None:
            # Fixed number of requests
            for i in range(1, args.count + 1):
                prompt = random.choice(prompts)
                payload = _build_payload(prompt, args.priority, args.random_priority, args.callback_url)

                try:
                    resp = session.post(url, json=payload, timeout=args.timeout)
                    status = resp.status_code
                except Exception as e:
                    print(f"[{i}/{args.count}] ERR sending request: {e}", flush=True)
                    next_tick = _stable_sleep(next_tick, interval)
                    continue

                if status == 200:
                    print(f"[{i}/{args.count}] OK   prompt={prompt!r}", flush=True)
                else:
                    print(f"[{i}/{args.count}] ERR  status={status} body={resp.text}", flush=True)

                sent += 1
                next_tick = _stable_sleep(next_tick, interval)

            print(f"\n→ Sent {sent} requests. Done.", flush=True)

            # Notify server to exit
            try:
                r = session.post(exit_url, timeout=args.timeout)
                if r.status_code == 200:
                    print("→ Exit signal sent to server.", flush=True)
                else:
                    print(f"→ Exit request failed: {r.status_code} {r.text}", flush=True)
            except Exception as e:
                print(f"→ Error sending exit signal: {e}", flush=True)

        else:
            # Infinite mode
            i = 0
            while True:
                i += 1
                prompt = random.choice(prompts)
                payload = _build_payload(prompt, args.priority, args.random_priority, args.callback_url)
                try:
                    resp = session.post(url, json=payload, timeout=args.timeout)
                    status = resp.status_code
                except Exception as e:
                    print(f"[{i}] ERR sending request: {e}", flush=True)
                    next_tick = _stable_sleep(next_tick, interval)
                    continue

                sent += 1
                if status == 200:
                    print(f"[{i}] OK   prompt={prompt!r}", flush=True)
                else:
                    print(f"[{i}] ERR  status={status} body={resp.text}", flush=True)

                next_tick = _stable_sleep(next_tick, interval)

    except KeyboardInterrupt:
        print(f"\n→ Interrupted by user after sending {sent} requests.", flush=True)


if __name__ == "__main__":
    # Make relative paths (like templates/xxx.yaml) work when double-clicked.
    os.chdir(os.path.dirname(__file__) or ".")
    main()
