"""
GPU Profiler Module (Upgraded)
Uses real NVIDIA driver-level metrics via nvidia-smi
"""

import torch
import threading
import time
import subprocess
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class GPUMetrics:
    timestamp: float
    gpu_utilization: float  # %
    gpu_memory_used: float  # MB
    gpu_memory_total: float  # MB
    gpu_power: Optional[float] = None  # Watts
    gpu_temp: Optional[float] = None  # Celsius


class GPUProfiler:
    def __init__(self, polling_interval: float = 0.2):
        self.polling_interval = polling_interval
        self.is_monitoring = False
        self.metrics_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._polling_thread: Optional[threading.Thread] = None

        self.gpu_available = torch.cuda.is_available()
        self.device_name = torch.cuda.get_device_name(0) if self.gpu_available else "CPU"

    # ---------------------------------------------------
    # MONITOR CONTROL
    # ---------------------------------------------------
    def start_monitoring(self):
        if not self.gpu_available:
            print("GPU not available. Monitoring disabled.")
            return

        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop, daemon=True
        )
        self._polling_thread.start()

    def stop_monitoring(self):
        self.is_monitoring = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)

    # ---------------------------------------------------
    # POLLING LOOP
    # ---------------------------------------------------
    def _polling_loop(self):
        while self.is_monitoring:
            metrics = self._collect_metrics()
            if metrics:
                with self._lock:
                    self.metrics_history.append(metrics)
            time.sleep(self.polling_interval)

    # ---------------------------------------------------
    # REAL METRICS COLLECTION (nvidia-smi)
    # ---------------------------------------------------
    def _collect_metrics(self) -> Optional[GPUMetrics]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
            )

            output = result.stdout.strip()
            if not output:
                return None

            values = output.split(",")
            values = [v.strip() for v in values]

            utilization = float(values[0])
            memory_used = float(values[1])
            memory_total = float(values[2])
            power = float(values[3]) if values[3] != "N/A" else None
            temp = float(values[4]) if values[4] != "N/A" else None

            return GPUMetrics(
                timestamp=time.time(),
                gpu_utilization=utilization,
                gpu_memory_used=memory_used,
                gpu_memory_total=memory_total,
                gpu_power=power,
                gpu_temp=temp,
            )

        except Exception as e:
            print(f"GPU metric collection error: {e}")
            return None

    # ---------------------------------------------------
    # ACCESS METHODS
    # ---------------------------------------------------
    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None

    def get_average_metrics(self, window_size: int = 50) -> Optional[Dict]:
        with self._lock:
            if not self.metrics_history:
                return None

            recent = list(self.metrics_history)[-window_size:]
            if not recent:
                return None

            return {
                "avg_utilization": sum(m.gpu_utilization for m in recent) / len(recent),
                "avg_memory_used": sum(m.gpu_memory_used for m in recent) / len(recent),
                "avg_power": (
                    sum(m.gpu_power for m in recent if m.gpu_power)
                    / len([m for m in recent if m.gpu_power])
                    if any(m.gpu_power for m in recent)
                    else None
                ),
                "max_utilization": max(m.gpu_utilization for m in recent),
                "max_memory_used": max(m.gpu_memory_used for m in recent),
            }

    def get_metrics_summary(self) -> Dict:
        return {
            "device_name": self.device_name,
            "gpu_available": self.gpu_available,
            "latest": self.get_latest_metrics(),
            "average": self.get_average_metrics(),
        }
