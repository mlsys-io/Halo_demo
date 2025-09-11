#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ramp-load client for a FastAPI server.

- Loads prompts from a YAML template (dataset / name / column).
- Over a fixed duration, linearly increases the request rate from --min-rate to --max-rate (req/s).
- Uses a stable tick scheduler (time.monotonic) to honor the target rate.
- Reuses HTTP connections via requests.Session().
- Optionally attaches a per-request "callback_url" for the server to push results back.
- Sends a final /exit request after the run.

Example:
    python ramp_load.py -f templates/adv_reason.yaml -H localhost -P 8000 \
        --min-rate 0.1 --max-rate 10.0 --duration-sec 600
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
    """Load prompts from a HF dataset specified by a YAML template."""
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


def _stable_sleep(next_tick: float, interval: float) -> float:
    """
    Sleep until 'next_tick', then return the next tick (= current next_tick + interval).
    This keeps the sending cadence stable regardless of request latency fluctuations.
    """
    now = time.monotonic()
    remain = next_tick - now
    if remain > 0:
        time.sleep(remain)
    return next_tick + interval


def _build_payload(prompt: str, priority: int, random_priority: bool, callback_url: str | None) -> dict:
    """Build the JSON payload with prompt, priority, and optional callback_url."""
    if random_priority:
        priority = random.randint(1, 5)
    payload = {"prompt": prompt, "priority": priority}
    if callback_url:
        payload["callback_url"] = callback_url
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send random prompts to a FastAPI server while linearly ramping the rate "
            "from --min-rate to --max-rate over a fixed duration."
        )
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="templates/adv_reason.yaml",
        help="Path to the YAML template with dataset spec (default: templates/adv_reason.yaml)",
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
        "--priority", "-p",
        type=int,
        default=0,
        help="Priority field to attach to each request (default: 0)",
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
        "--min-rate", "-r",
        type=float,
        default=0.05,
        help="Minimum request rate in req/s at the start (default: 0.05)",
    )
    parser.add_argument(
        "--max-rate", "-m",
        type=float,
        default=10.0,
        help="Maximum request rate in req/s at the end (default: 10.0)",
    )
    parser.add_argument(
        "--duration-sec", "-d",
        type=float,
        default=600.0,  # 10 minutes
        help="Total ramp duration in seconds (default: 600s = 10 minutes)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout per request in seconds (default: 5.0)",
    )
    args = parser.parse_args()

    # Validate and normalize rates/duration
    if args.min_rate < 0 or args.max_rate <= 0:
        raise ValueError("--min-rate must be >= 0 and --max-rate must be > 0")
    if args.max_rate < max(args.min_rate, 1e-9):
        raise ValueError("--max-rate must be >= --min-rate")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be > 0")

    # Load prompts
    prompts = _load_prompts_from_template(args.file)

    # Build URLs
    base = f"http://{args.host}:{args.port}"
    url = base + args.endpoint
    exit_url = base + args.exit_endpoint

    print(f"→ Target: {url}")
    print(
        f"→ Ramping from {args.min_rate:.4f} to {args.max_rate:.4f} req/s "
        f"over {args.duration_sec:.1f} s (≈ {args.duration_sec/60:.1f} min)"
    )
    print(f"→ Loaded {len(prompts)} prompts from {args.file!r}")
    if args.random_priority:
        print("→ Random priority enabled: [1..5] per request.")
    if args.callback_url:
        print(f"→ Attaching callback_url={args.callback_url}")
    print()

    session = requests.Session()  # reuse connections
    sent = 0

    start = time.monotonic()
    next_tick = start  # first tick immediately
    last_rate_log = -1.0  # last time (seconds since start) we printed the rate

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= args.duration_sec:
                break  # end of ramp

            # Linear ramp: rate(t) = min_rate + (max_rate - min_rate) * (t / duration)
            ratio = max(0.0, min(1.0, elapsed / args.duration_sec))
            current_rate = args.min_rate + (args.max_rate - args.min_rate) * ratio
            # Avoid division by zero (in case min_rate=0 at t=0)
            current_rate = max(current_rate, 1e-9)
            interval = 1.0 / current_rate

            # Log the current target rate at most once every ~5 seconds
            if last_rate_log < 0 or (elapsed - last_rate_log) >= 5.0:
                print(f"→ Current rate: {current_rate:.3f} req/s (interval: {interval:.3f} s)")
                last_rate_log = elapsed

            # Pick a random prompt
            prompt = random.choice(prompts)
            payload = _build_payload(prompt, args.priority, args.random_priority, args.callback_url)

            # Send the request
            try:
                resp = session.post(url, json=payload, timeout=args.timeout)
                status = resp.status_code
            except Exception as e:
                print(f"[{sent+1}] Error sending request: {e}")
                next_tick = _stable_sleep(next_tick, interval)
                continue

            sent += 1
            if status == 200:
                print(f"[{sent}] OK   prompt={prompt!r}")
            else:
                print(f"[{sent}] ERR  status={status} body={resp.text}")

            # Sleep until the next tick
            next_tick = _stable_sleep(next_tick, interval)

        print(f"\n→ Sent {sent} requests in {args.duration_sec:.1f}s. Done.")

        # Notify server to exit (optional; depends on your server implementation)
        try:
            r = session.post(exit_url, timeout=args.timeout)
            if r.status_code == 200:
                print("→ Exit signal sent to server.")
            else:
                print(f"→ Exit request failed: {r.status_code} {r.text}")
        except Exception as e:
            print(f"→ Error sending exit signal: {e}")

    except KeyboardInterrupt:
        print(f"\n→ Interrupted by user after sending {sent} requests.")


if __name__ == "__main__":
    # Make relative paths work when executed from elsewhere
    os.chdir(os.path.dirname(__file__) or ".")
    main()
