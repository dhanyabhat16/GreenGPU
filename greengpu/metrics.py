"""
Metrics Module (Upgraded)
Tracks inference performance, GPU efficiency, and power efficiency
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics


# ---------------------------------------------------
# DATA CLASS
# ---------------------------------------------------

@dataclass
class InferenceMetrics:
    inference_id: str
    inference_time: float          # seconds
    throughput: float              # samples/sec
    gpu_memory_used: float         # MB
    gpu_utilization: float         # %
    cpu_utilization: Optional[float] = None  # %
    gpu_power: Optional[float] = None   # Watts
    power_efficiency: Optional[float] = None  # samples per watt


# ---------------------------------------------------
# TIMER (HIGH PRECISION)
# ---------------------------------------------------

class InferenceTimer:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.inference_id: Optional[str] = None

    def start(self, inference_id: str):
        self.inference_id = inference_id
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return elapsed

    def get_elapsed(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        return time.perf_counter() - self.start_time


# ---------------------------------------------------
# METRICS COLLECTOR
# ---------------------------------------------------

class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        self.metrics_history: deque = deque(maxlen=max_history)

    def record_inference(self, inference_metrics: InferenceMetrics):
        self.metrics_history.append(inference_metrics)

    def get_statistics(self) -> Optional[Dict]:
        if not self.metrics_history:
            return None

        times = [m.inference_time for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history]
        memory_used = [m.gpu_memory_used for m in self.metrics_history]
        utilizations = [m.gpu_utilization for m in self.metrics_history]
        cpu_utils = [m.cpu_utilization for m in self.metrics_history if m.cpu_utilization is not None]
        powers = [m.gpu_power for m in self.metrics_history if m.gpu_power is not None]

        stats = {
            "total_inferences": len(self.metrics_history),
            "inference_time": {
                "min": min(times),
                "max": max(times),
                "mean": statistics.mean(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            },
            "throughput": {
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": statistics.mean(throughputs),
            },
            "gpu_memory": {
                "min": min(memory_used),
                "max": max(memory_used),
                "mean": statistics.mean(memory_used),
            },
            "gpu_utilization": {
                "min": min(utilizations),
                "max": max(utilizations),
                "mean": statistics.mean(utilizations),
            },
        }

        if cpu_utils:
            stats["cpu_utilization"] = {
                "min": min(cpu_utils),
                "max": max(cpu_utils),
                "mean": statistics.mean(cpu_utils),
            }

        if powers:
            stats["gpu_power"] = {
                "min": min(powers),
                "max": max(powers),
                "mean": statistics.mean(powers),
            }

        return stats

    def get_recent_metrics(self, window_size: int = 10) -> List[Dict]:
        recent = list(self.metrics_history)[-window_size:]
        return [asdict(m) for m in recent]

    def clear_history(self):
        self.metrics_history.clear()


# ---------------------------------------------------
# EFFICIENCY ANALYZER
# ---------------------------------------------------

class EfficiencyAnalyzer:

    @staticmethod
    def calculate_power_efficiency(throughput: float, power_watts: float) -> float:
        if power_watts is None or power_watts == 0:
            return 0
        return throughput / power_watts

    @staticmethod
    def analyze_utilization(utilization_percent: float) -> Dict:
        if utilization_percent < 20:
            status = "UNDERUTILIZED"
            recommendation = "GPU underutilized → Consider CPU execution or larger batch size."
        elif utilization_percent < 50:
            status = "LOW"
            recommendation = "GPU utilization is low → Optimize batch size."
        elif utilization_percent < 80:
            status = "GOOD"
            recommendation = "GPU utilization is efficient."
        else:
            status = "HIGH"
            recommendation = "GPU highly utilized."

        return {
            "status": status,
            "recommendation": recommendation,
            "efficiency_score": min(utilization_percent / 100, 1.0),
        }

    @staticmethod
    def analyze_memory(used_mb: float, total_mb: float) -> Dict:
        if total_mb <= 0:
            return {
                "usage_percent": 0,
                "status": "UNKNOWN"
            }

        memory_percent = (used_mb / total_mb) * 100

        if memory_percent < 30:
            status = "OPTIMAL"
        elif memory_percent < 70:
            status = "GOOD"
        elif memory_percent < 90:
            status = "HIGH"
        else:
            status = "CRITICAL"

        return {
            "usage_percent": memory_percent,
            "used_mb": used_mb,
            "total_mb": total_mb,
            "status": status,
        }