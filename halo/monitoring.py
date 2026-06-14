from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _sanitize_label(label: str) -> str:
    cleaned = []
    for ch in label.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "run"


def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@dataclass
class _CpuTimes:
    idle: int
    total: int


class SystemMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        label: str,
        interval_s: float,
    ) -> None:
        self.output_dir = output_dir
        self.label = _sanitize_label(label)
        self.interval_s = max(0.1, float(interval_s))
        self.output_path = self.output_dir / f"{self.label}_{_now_tag()}.csv"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._lock = threading.Lock()
        self._prev_cpu: _CpuTimes | None = None
        self._gpu_available: Optional[bool] = None

    def start(self) -> "SystemMonitor":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8", buffering=1)
        self._file.write(
            "timestamp,cpu_util_pct,cpu_mem_used_mb,cpu_mem_total_mb,"
            "gpu_index,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb\n"
        )
        self._thread = threading.Thread(target=self._run, name="halo-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._file:
            try:
                self._file.flush()
            finally:
                self._file.close()
                self._file = None

    def _run(self) -> None:
        self._prev_cpu = self._read_cpu_times()
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cpu_util = self._read_cpu_util()
        mem_used, mem_total = self._read_mem_usage()
        gpu_stats = self._read_gpu_stats()

        if not gpu_stats:
            self._write_row(
                timestamp,
                cpu_util,
                mem_used,
                mem_total,
                "",
                None,
                None,
                None,
            )
            return

        for gpu in gpu_stats:
            self._write_row(
                timestamp,
                cpu_util,
                mem_used,
                mem_total,
                str(gpu.get("index", "")),
                gpu.get("util"),
                gpu.get("mem_used"),
                gpu.get("mem_total"),
            )

    def _write_row(
        self,
        timestamp: str,
        cpu_util: float | None,
        mem_used: float | None,
        mem_total: float | None,
        gpu_index: str,
        gpu_util: float | None,
        gpu_mem_used: float | None,
        gpu_mem_total: float | None,
    ) -> None:
        fields = [
            timestamp,
            _fmt(cpu_util),
            _fmt(mem_used),
            _fmt(mem_total),
            gpu_index,
            _fmt(gpu_util),
            _fmt(gpu_mem_used),
            _fmt(gpu_mem_total),
        ]
        line = ",".join(fields) + "\n"
        with self._lock:
            if self._file:
                self._file.write(line)

    def _read_cpu_times(self) -> _CpuTimes | None:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                line = handle.readline()
        except OSError:
            return None
        parts = line.strip().split()
        if not parts or parts[0] != "cpu":
            return None
        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            return None
        if len(values) < 4:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return _CpuTimes(idle=idle, total=total)

    def _read_cpu_util(self) -> float | None:
        current = self._read_cpu_times()
        if self._prev_cpu is None or current is None:
            self._prev_cpu = current
            return None
        total_delta = current.total - self._prev_cpu.total
        idle_delta = current.idle - self._prev_cpu.idle
        self._prev_cpu = current
        if total_delta <= 0:
            return None
        used = max(0.0, float(total_delta - idle_delta))
        return (used / float(total_delta)) * 100.0

    def _read_mem_usage(self) -> tuple[float | None, float | None]:
        mem_total = None
        mem_available = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        mem_total = _parse_kb(line)
                    elif line.startswith("MemAvailable:"):
                        mem_available = _parse_kb(line)
        except OSError:
            return None, None
        if mem_total is None:
            return None, None
        if mem_available is None:
            mem_used = None
        else:
            mem_used = max(0.0, mem_total - mem_available)
        return mem_used, mem_total

    def _read_gpu_stats(self) -> List[dict]:
        if self._gpu_available is False:
            return []
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=2.0,
            )
        except FileNotFoundError:
            self._gpu_available = False
            return []
        except subprocess.SubprocessError:
            return []

        if proc.returncode != 0:
            if "not found" in (proc.stderr or "").lower():
                self._gpu_available = False
            return []

        self._gpu_available = True
        stats: List[dict] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                stats.append(
                    {
                        "index": int(parts[0]),
                        "util": float(parts[1]),
                        "mem_used": float(parts[2]),
                        "mem_total": float(parts[3]),
                    }
                )
            except ValueError:
                continue
        return stats


class ProgressMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        label: str,
        total_units: int,
    ) -> None:
        self.output_dir = output_dir
        self.label = _sanitize_label(label)
        self.total_units = max(0, int(total_units))
        self.output_path = self.output_dir / f"{self.label}_progress_{_now_tag()}.csv"
        self._completed = 0
        self._file = None
        self._lock = threading.Lock()

    def start(self) -> "ProgressMonitor":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8", buffering=1)
        self._file.write("timestamp,completed,total,progress\n")
        self._write_row(0)
        return self

    def stop(self) -> None:
        with self._lock:
            if self._file:
                try:
                    self._file.flush()
                finally:
                    self._file.close()
                    self._file = None

    def record(self, completed_delta: int) -> None:
        if completed_delta <= 0 or self.total_units <= 0:
            return
        with self._lock:
            self._completed = min(self.total_units, self._completed + int(completed_delta))
            self._write_row(self._completed)

    def _write_row(self, completed: int) -> None:
        if not self._file:
            return
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        total = self.total_units
        progress = (completed / total) if total > 0 else 1.0
        self._file.write(f"{timestamp},{completed},{total},{progress:.4f}\n")


def _parse_kb(line: str) -> float | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1]) / 1024.0
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def start_system_monitor(
    label: str,
    *,
    enable: bool | None = None,
    interval_s: float | None = None,
    output_dir: Path | None = None,
) -> SystemMonitor | None:
    if enable is None:
        enable = _env_flag("HALO_MONITOR_ENABLE", True)
    if not enable:
        return None
    if interval_s is None:
        interval_s = _env_float("HALO_MONITOR_INTERVAL_S", 1.0)
    if output_dir is None:
        output_dir = Path(os.getenv("HALO_MONITOR_DIR", "logs/monitor"))
    return SystemMonitor(output_dir=output_dir, label=label, interval_s=interval_s).start()


def start_progress_monitor(
    label: str,
    total_units: int,
    *,
    enable: bool | None = None,
    output_dir: Path | None = None,
) -> ProgressMonitor | None:
    if enable is None:
        enable = _env_flag("HALO_PROGRESS_ENABLE", False)
    if not enable or total_units <= 0:
        return None
    if output_dir is None:
        output_dir = Path(os.getenv("HALO_PROGRESS_DIR", "logs/progress"))
    return ProgressMonitor(output_dir=output_dir, label=label, total_units=total_units).start()
