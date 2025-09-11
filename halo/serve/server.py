#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI batching server with pluggable optimizer backends and optional HTTP callbacks.

- Choose the optimizer at runtime via --optimizer (same choices as the batch runner).
- Batches queued queries every `interval` seconds and calls optimizer.execute().
- Tracks simple metrics: latency percentiles, weighted score, and output rate.
- Graceful shutdown via /exit: flush remaining queries, return metrics.
- (New) Records client address for every query and optionally POSTs results back via HTTP.

Endpoints
---------
POST /query
    Enqueue a single query. Optional field: callback_url.

POST /batch
    Enqueue multiple queries (array of {prompt, priority[, callback_url]}).

POST /exit
    Trigger graceful stop of background workers (final flush) and return metrics.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from collections import deque
from typing import List, Optional, Dict, Any

import numpy as np
import uvicorn
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

from halo.components import Query


# ----------------------------- Optimizer Loader -----------------------------

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
    if name == "transformers_batch":
        from halo.optimizers import Optimizer_tb as opt
        return opt
    if name == "transformers_single":
        from halo.optimizers import Optimizer_ts as opt
        return opt
    if name == "vllm":
        from halo.optimizers import Optimizer_vllm as opt
        return opt
    if name == "lmcache":
        from halo.optimizers import Optimizer_lmcache as opt
        return opt

    raise ValueError(f"Unknown optimizer: {name}")


# ----------------------------- API Schemas ----------------------------------

class QueryBody(BaseModel):
    """Inbound request payload schema."""
    prompt: str
    priority: int = 0
    callback_url: Optional[str] = None  # optional per-query override


# ----------------------------- Server ---------------------------------------

class Server:
    """
    FastAPI-serving wrapper that owns an optimizer instance and two background workers:
    - batch worker: periodically flushes the buffered requests to optimizer.execute()
    - log worker: periodically prints/logs output rate over a sliding window
    - (New) optional HTTP callback: POST results to the client's callback endpoint

    Parameters
    ----------
    optimizer_name : str
        Which optimizer to load (see --optimizer choices).
    template : str
        Path to the YAML template passed to the optimizer constructor.
    interval : float, default 1.0
        Batch cycle (seconds). Also used by periodic logging worker if not None.
    log_file : str, default "server.log"
        File path for logging.
    output_window : int, default 60
        Sliding window size (seconds) for output rate (req/s).
    push_callback : bool, default False
        Enable HTTP callback after each query is processed.
    callback_port : int, default 8001
        If `callback_url` is not provided in the request, construct a default
        callback endpoint using the client's IP and this port.
    callback_path : str, default "/callback"
        Default callback path (combined with IP and port when callback_url is missing).
    callback_timeout : float, default 5.0
        Timeout (seconds) for the HTTP callback POST.
    """

    def __init__(
        self,
        optimizer_name: str,
        template: str,
        interval: float = 1.0,
        log_file: str = "server.log",
        output_window: int = 60,
        push_callback: bool = False,
        callback_port: int = 8001,
        callback_path: str = "/callback",
        callback_timeout: float = 5.0,
    ):
        # Create the chosen optimizer instance
        optimizer_fun = load_optimizer(optimizer_name)
        self.opt = optimizer_fun(template)

        self.name: str = f"server[{optimizer_name}]"
        self.interval: float = float(interval)
        self.log_interval: float = 10.0  # periodic logging cadence (seconds)

        # Metrics
        self.score: float = 0.0
        self.latencies: List[float] = []
        self.completed_requests: deque[float] = deque()  # timestamps for throughput
        self.output_window: int = int(output_window)

        # Optional callback settings
        self.push_callback: bool = bool(push_callback)
        self.callback_port: int = int(callback_port)
        self.callback_path: str = str(callback_path)
        self.callback_timeout: float = float(callback_timeout)

        # Per-query origin and callback info (sidecar maps)
        # Keyed by query.id to avoid mutating the Query dataclass if it's frozen
        self._origin_info: Dict[str, Dict[str, Any]] = {}  # {id: {"host":..., "port":..., "callback_url":...}}

        # Concurrency control
        self._stop_event = threading.Event()

        # Logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self.logger = logging.getLogger("ServerLogger")
        self.logger.info(
            "Server initialized (optimizer=%s, interval=%.3fs, window=%ds, push_callback=%s).",
            optimizer_name, self.interval, self.output_window, self.push_callback
        )

    # ----------------------------- Metrics -----------------------------------

    def _compute_output_rate(self, window: int) -> float:
        """Compute req/s based on completions within the last `window` seconds."""
        now = time.time()
        cutoff = now - window
        while self.completed_requests and self.completed_requests[0] < cutoff:
            self.completed_requests.popleft()
        count = len(self.completed_requests)
        return count / window if window > 0 else 0.0

    def _update_metrics(self, batch: List[Query]) -> None:
        """
        Update latency statistics, weighted score, and throughput counters for a batch,
        then print & log summary percentiles (P50, P90, P95, P99).
        """
        now = time.time()
        for req in batch:
            try:
                # Assumes req.benchmark[...] uses the same time base as req.create_time
                end_t = list(req.benchmark.values())[-1][-1]
                latency = float(end_t - req.create_time)
                self.score += req.priority * latency
                self.latencies.append(latency)
                self.completed_requests.append(now)
            except Exception as e:
                print(f"[WARN] Failed to compute metrics for req={req.id}: {e}")

        if self.latencies:
            p50 = float(np.percentile(self.latencies, 50))
            p90 = float(np.percentile(self.latencies, 90))
            p95 = float(np.percentile(self.latencies, 95))
            p99 = float(np.percentile(self.latencies, 99))
        else:
            p50 = p90 = p95 = p99 = float("nan")

        output_rate = self._compute_output_rate(self.output_window)
        msg = (
            f"Metrics: P50={p50:.3f}s, P90={p90:.3f}s, P95={p95:.3f}s, P99={p99:.3f}s, "
            f"Score={self.score:.2f}, OutputRate={output_rate:.2f} req/s "
            f"(window={self.output_window}s)"
        )
        print(msg)
        self.logger.info(msg)

    # ---------------------------- HTTP Callback ------------------------------

    def _compute_default_callback(self, qid: str) -> Optional[str]:
        """
        Build a default callback URL using the recorded client host
        if per-query callback_url is not provided.
        """
        info = self._origin_info.get(qid) or {}
        host = info.get("host")
        if not host:
            return None
        # NOTE: We assume the client listens on `callback_port` and `callback_path`.
        # If not, the POST will fail, which is fine for a best-effort callback.
        return f"http://{host}:{self.callback_port}{self.callback_path}"

    def _build_result_payload(self, req: Query) -> Dict[str, Any]:
        """
        Build a JSON-safe result payload for callbacks.
        Includes:
          - id, priority
          - latency (seconds)
          - outputs: dict of op outputs (if any)
        """
        outputs = getattr(req, "op_output", {}) or {}
        try:
            end_t = list(req.benchmark.values())[-1][-1]
            latency = float(end_t - req.create_time)
        except Exception:
            latency = None

        return {
            "id": req.id,
            "priority": req.priority,
            "latency": latency,
            "outputs": outputs,
        }

    def _maybe_send_callback(self, req: Query) -> None:
        """
        If callbacks are enabled, POST the result payload to:
          1) req-level callback_url (if provided), else
          2) computed default callback URL from client IP.
        Best-effort only; errors are logged but do not affect the batch.
        """
        if not self.push_callback:
            return

        info = self._origin_info.get(req.id, {})
        target = info.get("callback_url") or self._compute_default_callback(req.id)
        if not target:
            return  # nothing to do

        payload = self._build_result_payload(req)
        try:
            resp = requests.post(target, json=payload, timeout=self.callback_timeout)
            if resp.status_code >= 300:
                self.logger.warning("Callback failed for %s: %s %s", req.id, resp.status_code, resp.text)
        except Exception as e:
            self.logger.warning("Callback exception for %s to %s: %s", req.id, target, e)

    # -------------------------- Background Workers ---------------------------

    def _batch_worker(self, buffer: List[Query], lock: threading.Lock) -> None:
        """
        Periodically flush a copy of the buffer as a batch to the optimizer.
        Supports graceful shutdown: when `_stop_event` is set, performs a final flush.
        After each batch, optionally POST per-query results to client callbacks.
        """
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            with lock:
                if not buffer:
                    continue
                batch = buffer.copy()
                buffer.clear()

            print(f"[BatchWorker] Processing batch of size {len(batch)}")
            try:
                # Keep optimizer alive while server runs
                self.opt.execute(batch, skip_exit=True)
                # Callbacks (best-effort, per query)
                for q in batch:
                    self._maybe_send_callback(q)
                # Metrics last
                self._update_metrics(batch)

                # Reset any per-op state if needed
                if hasattr(self.opt, "node_reset"):
                    self.opt.node_reset()
                elif hasattr(self.opt, "op_reset"):
                    self.opt.op_reset()
            except Exception as e:
                print(f"[ERROR] Batch execution failed: {e}")
                self.logger.exception("Batch execution failed.")

        # Final flush after stop
        with lock:
            final_batch = buffer.copy()
            buffer.clear()

        if final_batch:
            print(f"[BatchWorker] Processing final batch of size {len(final_batch)}")
            try:
                self.opt.execute(final_batch, skip_exit=False)
                for q in final_batch:
                    self._maybe_send_callback(q)
                self._update_metrics(final_batch)
            except Exception as e:
                print(f"[ERROR] Final batch execution failed: {e}")
                self.logger.exception("Final batch execution failed.")

        print("[BatchWorker] Final batch processed. Shutdown complete.")
        self.logger.info("Final batch processed. Shutdown complete.")

    def _log_worker(self) -> None:
        """
        Periodically log the output rate. Runs until `_stop_event` is set,
        then writes a final line.
        """
        time.sleep(self.log_interval)  # align roughly with batch cadence
        while not self._stop_event.is_set():
            rate = self._compute_output_rate(self.output_window)
            msg = f"[Periodic] OutputRate={rate:.2f} req/s (window={self.output_window}s)"
            print(msg)
            self.logger.info(msg)
            time.sleep(self.log_interval)

        rate = self._compute_output_rate(self.output_window)
        msg = f"[Periodic-Final] OutputRate={rate:.2f} req/s (window={self.output_window}s)"
        print(msg)
        self.logger.info(msg)

    # ----------------------------- FastAPI -----------------------------------

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        interval: Optional[float] = None,
    ) -> None:
        """
        Start the FastAPI app with two background threads:
        - batch worker: collects & executes buffered requests
        - log worker: periodically logs throughput
        """
        if interval is not None:
            self.interval = float(interval)

        app = FastAPI(title=self.name)

        buffer: List[Query] = []
        lock = threading.Lock()

        # Start background workers
        threading.Thread(target=self._batch_worker, args=(buffer, lock), daemon=True).start()
        threading.Thread(target=self._log_worker, daemon=True).start()

        @app.post("/query")
        async def add_query(body: QueryBody, request: Request):
            """
            Enqueue a single query.
            Records client host/port and optional per-query callback_url (if provided).
            """
            query = Query(
                id=str(uuid.uuid4()),
                prompt=body.prompt,
                priority=body.priority,
            )
            client = request.client or None
            host = getattr(client, "host", None)
            port = getattr(client, "port", None)

            # Sidecar origin/callback info
            self._origin_info[query.id] = {
                "host": host,
                "port": port,
                "callback_url": body.callback_url or None,
            }

            with lock:
                buffer.append(query)
            return {"status": "queued", "id": query.id, "client": {"host": host, "port": port}}

        @app.post("/batch")
        async def add_batch(bodies: List[QueryBody], request: Request):
            """
            Enqueue multiple queries in one call.
            Each element may provide its own callback_url; otherwise default is used.
            """
            client = request.client or None
            host = getattr(client, "host", None)
            port = getattr(client, "port", None)

            queries = [
                Query(id=str(uuid.uuid4()), prompt=b.prompt, priority=b.priority)
                for b in bodies
            ]
            for q, b in zip(queries, bodies):
                self._origin_info[q.id] = {
                    "host": host,
                    "port": port,
                    "callback_url": b.callback_url or None,
                }

            with lock:
                buffer.extend(queries)
            return {"status": "queued", "received": len(queries)}

        @app.post("/exit")
        async def exit_server():
            """
            Trigger graceful stop of background threads (final flush included) and
            return current aggregate metrics. The HTTP server stays up; if you need
            process-level termination, handle it externally.
            """
            self._stop_event.set()
            # Give workers a moment to flush
            time.sleep(self.interval * 1.5)

            rate = self._compute_output_rate(self.output_window)
            metrics = {
                "score": self.score,
                "latency_count": len(self.latencies),
                "output_rate_req_per_s": rate,
                "window_s": self.output_window,
            }
            print("Exit requested. Metrics:", metrics)
            logging.getLogger("ServerLogger").info("Exit requested. Metrics: %s", metrics)
            return metrics

        uvicorn.run(app, host=host, port=port, log_level="info")


# ----------------------------- CLI ------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pluggable-optimizer FastAPI server.")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["halo_v", "halo_t", "transformers_batch", "transformers_single", "vllm", "lmcache"],
        default="halo_v",
        help="Which optimizer backend to use.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="templates/adv_reason.yaml",
        help="Path to the template file passed to the optimizer.",
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", "-P",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=None,
        help="Batching interval in seconds. If not set, uses the server default.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="server.log",
        help="Path to the log file (default: server.log).",
    )
    parser.add_argument(
        "--output-window",
        type=int,
        default=15,
        help="Sliding window (seconds) for throughput (req/s) calculation.",
    )
    # --- New callback-related flags ---
    parser.add_argument(
        "--push-callback",
        action="store_true",
        help="If set, POST each query's result to a callback URL (per-request or constructed from client IP).",
    )
    parser.add_argument(
        "--callback-port",
        type=int,
        default=8001,
        help="Default port for constructing callback URL when the client does not provide one.",
    )
    parser.add_argument(
        "--callback-path",
        type=str,
        default="/callback",
        help="Default path for constructing callback URL when the client does not provide one.",
    )
    parser.add_argument(
        "--callback-timeout",
        type=float,
        default=5.0,
        help="Timeout (seconds) for HTTP callback POST requests.",
    )

    args = parser.parse_args()

    server = Server(
        optimizer_name=args.optimizer,
        template=args.template,
        interval=args.interval or 1.0,
        log_file=args.log_file,
        output_window=args.output_window,
        push_callback=args.push_callback,
        callback_port=args.callback_port,
        callback_path=args.callback_path,
        callback_timeout=args.callback_timeout,
    )
    server.serve(host=args.host, port=args.port, interval=args.interval)
