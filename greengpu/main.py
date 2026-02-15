"""
GreenGPU - Final Clean Orchestrator
Real GPU profiling + accurate timing
"""

import torch
import time
import psutil
import statistics
from typing import Dict
try:
    from .profiler import GPUProfiler
    from .model_loader import ModelLoader
    from .metrics import (
        InferenceTimer,
        MetricsCollector,
        InferenceMetrics,
        EfficiencyAnalyzer,
    )
except ImportError:
    from profiler import GPUProfiler
    from model_loader import ModelLoader
    from metrics import (
        InferenceTimer,
        MetricsCollector,
        InferenceMetrics,
        EfficiencyAnalyzer,
    )


class GreenGPU:
    def __init__(
        self,
        model_name: str = "resnet18",
        polling_interval: float = 0.01,
        auto_switch_to_cpu: bool = True,
        auto_switch_util_threshold: float = 20.0,
        probe_inferences: int = 10,
    ):
        self.gpu_profiler = GPUProfiler(polling_interval=polling_interval)
        self.model_loader = ModelLoader()
        self.inference_timer = InferenceTimer()
        self.metrics_collector = MetricsCollector()
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.model_name = model_name
        self.auto_switch_to_cpu = auto_switch_to_cpu
        self.auto_switch_util_threshold = auto_switch_util_threshold
        self.probe_inferences = probe_inferences
        
        # State tracking
        self.last_run_duration = 0.0
        self.last_device_used = self.model_loader.device
        self.switched_to_cpu = False
        
        # Baselines for calculating savings
        self.baseline_gpu_power = 0.0
        self.baseline_gpu_throughput = 0.0

    # ---------------------------------------------------
    # GPU VERIFICATION
    # ---------------------------------------------------
    def verify_gpu(self):
        print("=" * 60)
        print("GPU VERIFICATION")
        print("=" * 60)

        print(f"PyTorch Version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✓ CUDA is available")
            print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
        else:
            print("✗ CUDA NOT available — running on CPU")

        print("=" * 60 + "\n")

    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    def initialize(self):
        print("Initializing GreenGPU...\n")

        if torch.cuda.is_available():
            self.gpu_profiler.start_monitoring()
            time.sleep(0.3)

        try:
            self.model_loader.load_pretrained_model(self.model_name)
            model_info = self.model_loader.get_model_info()

            print(f"Model Loaded: {model_info['model_name']}")
            print(f"Total Parameters: {model_info['total_parameters']:,}")

        except Exception as e:
            print(f"Model loading failed: {e}")
            return False

        self.model_loader.warmup(num_iterations=5)
        print("Warmup complete\n")

        return True

    def _run_inference_pass(
        self,
        num_inferences: int,
        batch_size: int,
        record_metrics: bool,
        print_progress: bool,
    ) -> Dict[str, float]:
        input_size = (batch_size, 3, 224, 224)
        cpu_samples = []
        gpu_samples = []
        gpu_power_samples = []

        psutil.cpu_percent(interval=None)
        start_time = time.perf_counter()

        for i in range(num_inferences):
            self.inference_timer.start(f"inference_{i}")

            input_tensor = self.model_loader.prepare_input(input_size)
            _ = self.model_loader.inference(input_tensor)

            if self.model_loader.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            inference_time = self.inference_timer.stop()
            throughput = batch_size / inference_time

            cpu_util = psutil.cpu_percent(interval=None)
            cpu_samples.append(cpu_util)

            gpu_metrics = None
            if self.model_loader.device == "cuda" and torch.cuda.is_available():
                gpu_metrics = self.gpu_profiler.get_latest_metrics()

            if gpu_metrics:
                gpu_util = gpu_metrics.gpu_utilization
                gpu_memory = gpu_metrics.gpu_memory_used
                gpu_power = gpu_metrics.gpu_power
            else:
                gpu_util = 0.0
                gpu_memory = 0.0
                gpu_power = 0.0

            gpu_samples.append(gpu_util)
            if gpu_power:
                gpu_power_samples.append(gpu_power)

            power_eff = None
            if gpu_power:
                power_eff = self.efficiency_analyzer.calculate_power_efficiency(
                    throughput, gpu_power
                )

            if record_metrics:
                metrics = InferenceMetrics(
                    inference_id=f"inference_{i}",
                    inference_time=inference_time,
                    throughput=throughput,
                    gpu_memory_used=gpu_memory,
                    gpu_utilization=gpu_util,
                    cpu_utilization=cpu_util,
                    gpu_power=gpu_power,
                    power_efficiency=power_eff,
                )
                self.metrics_collector.record_inference(metrics)

            if print_progress:
                print(
                    f"Inference {i+1:2d}: "
                    f"{inference_time*1000:7.2f} ms | "
                    f"{throughput:8.2f} samples/s | "
                    f"GPU Util: {gpu_util:5.1f}% | "
                    f"GPU Mem: {gpu_memory:6.1f} MB | "
                    f"Power: {gpu_power if gpu_power else 'N/A'} W"
                )

        duration = time.perf_counter() - start_time
        avg_throughput = (num_inferences * batch_size) / duration
        avg_gpu_util = statistics.mean(gpu_samples) if gpu_samples else 0.0
        avg_cpu_util = statistics.mean(cpu_samples) if cpu_samples else 0.0
        avg_gpu_power = statistics.mean(gpu_power_samples) if gpu_power_samples else 0.0

        return {
            "duration": duration,
            "avg_gpu_util": avg_gpu_util,
            "avg_cpu_util": avg_cpu_util,
            "avg_gpu_power": avg_gpu_power,
            "avg_throughput": avg_throughput
        }

    # ---------------------------------------------------
    # INFERENCE LOOP
    # ---------------------------------------------------
    def run_inference(self, num_inferences: int = 2000, batch_size: int = 1):
        print("=" * 60)
        print(f"RUNNING {num_inferences} INFERENCES | Batch Size = {batch_size}")
        print("=" * 60 + "\n")

        self.metrics_collector.clear_history()
        self.switched_to_cpu = False
        
        # Reset baselines
        self.baseline_gpu_power = 0.0
        self.baseline_gpu_throughput = 0.0

        if (
            self.model_loader.device == "cuda"
            and torch.cuda.is_available()
            and self.auto_switch_to_cpu
        ):
            probe_count = max(1, min(self.probe_inferences, num_inferences))
            print(f"Running probe ({probe_count} inferences)...")
            
            probe = self._run_inference_pass(
                num_inferences=probe_count,
                batch_size=batch_size,
                record_metrics=False,
                print_progress=False,
            )
            
            # Capture baseline GPU performance
            self.baseline_gpu_power = probe["avg_gpu_power"]
            self.baseline_gpu_throughput = probe["avg_throughput"]
            
            print(f"Probe Results -> GPU Util: {probe['avg_gpu_util']:.1f}% | Power: {self.baseline_gpu_power:.2f} W | Throughput: {self.baseline_gpu_throughput:.2f} inf/s")

            if probe["avg_gpu_util"] < self.auto_switch_util_threshold:
                print(
                    "\nAuto-switching to CPU due to low GPU utilization "
                    f"({probe['avg_gpu_util']:.1f}% < {self.auto_switch_util_threshold}%).\n"
                )
                self.switched_to_cpu = True
                self.model_loader.set_device("cpu")
                self.gpu_profiler.stop_monitoring() 
                self.last_device_used = "cpu"
            else:
                self.last_device_used = "gpu"
        else:
            self.last_device_used = self.model_loader.device

        # Run the main workload
        run_info = self._run_inference_pass(
            num_inferences=num_inferences,
            batch_size=batch_size,
            record_metrics=True,
            print_progress=True,
        )
        self.last_run_duration = run_info["duration"]

        print()

    # ---------------------------------------------------
    # REPORT
    # ---------------------------------------------------
    def print_report(self):
        print("=" * 60)
        print("FINAL METRICS REPORT")
        print("=" * 60 + "\n")

        stats = self.metrics_collector.get_statistics()
        if not stats:
            print("No metrics collected.")
            return

        print("Inference Time (ms)")
        print(f"  Mean : {stats['inference_time']['mean']*1000:.2f}")

        print("\nThroughput (samples/sec)")
        print(f"  Mean : {stats['throughput']['mean']:.2f}")

        print("\nGPU Utilization (%)")
        print(f"  Mean : {stats['gpu_utilization']['mean']:.2f}")

        print("\nGPU Memory Usage (MB)")
        print(f"  Mean : {stats['gpu_memory']['mean']:.2f}")

        if "gpu_power" in stats:
            print("\nGPU Power (W)")
            print(f"  Mean : {stats['gpu_power']['mean']:.2f}")

        total_memory_mb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if torch.cuda.is_available()
            else 0
        )

        avg_util = stats["gpu_utilization"]["mean"]
        avg_memory = stats["gpu_memory"]["mean"]

        print("\nEFFICIENCY ANALYSIS")

        util_analysis = self.efficiency_analyzer.analyze_utilization(avg_util)
        memory_analysis = self.efficiency_analyzer.analyze_memory(
            avg_memory, total_memory_mb
        )

        print(f"  GPU Status : {util_analysis['status']}")
        print(f"  Recommendation : {util_analysis['recommendation']}")
        print(f"  Memory Usage : {memory_analysis['usage_percent']:.2f}%")

        avg_cpu_util = 0.0
        if "cpu_utilization" in stats:
            avg_cpu_util = stats["cpu_utilization"]["mean"]

        avg_gpu_util = stats["gpu_utilization"]["mean"]

        recommend_cpu = False
        reason = "GPU utilization sufficient; keep GPU."
        if not torch.cuda.is_available():
            recommend_cpu = True
            reason = "No GPU was available; execution was on CPU."
        elif self.last_device_used == "cpu":
            recommend_cpu = True
            reason = "GPU utilization below threshold; switched to CPU."
        elif avg_gpu_util < self.auto_switch_util_threshold:
            recommend_cpu = True
            reason = "GPU utilization is low; consider CPU execution."

        print("\n1. Compute resource analysis")
        print("-" * 40)
        print(f"  Device used during test run: {self.last_device_used}")
        print(f"  Recommend CPU instead of GPU: {recommend_cpu}")
        print(f"  Reason: {reason}")
        print(f"  Avg GPU util: {avg_gpu_util:.1f}%")
        print(f"  Avg CPU util: {avg_cpu_util:.1f}%")
        print(f"  Test run duration: {self.last_run_duration:.2f}s")

        # --- IMPACT CALCULATION ---
        
        # 1. GPU Hours Saved
        if self.last_device_used == "cpu" and torch.cuda.is_available():
            gpu_hours_saved = self.last_run_duration / 3600.0
        else:
            gpu_hours_saved = 0.0
        
        # 2. Energy Saved
        if gpu_hours_saved > 0 and self.baseline_gpu_power > 0:
            energy_saved_wh = (self.baseline_gpu_power * gpu_hours_saved)
        else:
            energy_saved_wh = 0.0

        # 3. Compute Time Reduction
        compute_time_reduction = 0.0
        if self.last_device_used == "cpu" and self.baseline_gpu_throughput > 0:
            # FIX: Use stats['total_inferences'] instead of trying to len() a float
            total_inferences = stats.get("total_inferences", 0)
            
            projected_gpu_duration = total_inferences / self.baseline_gpu_throughput
            
            if projected_gpu_duration > 0:
                compute_time_reduction = ((projected_gpu_duration - self.last_run_duration) / projected_gpu_duration) * 100

        energy_saved_kwh = energy_saved_wh / 1000.0
        carbon_saved_g = energy_saved_kwh * 0.4 * 1000
        cost_saved_cents = energy_saved_kwh * 0.12 * 100

        print("\n3. Energy & impact estimation")
        print("-" * 40)
        print(f"  GPU hours saved: {gpu_hours_saved:.6f}")
        print(f"  Compute time reduction: {compute_time_reduction:.2f}% (vs GPU baseline)")
        print(f"  Energy saved: {energy_saved_wh:.4f} Wh")
        print(f"  Carbon saved: {carbon_saved_g:.4f} g CO2")
        print(f"  Estimated cost saved: {cost_saved_cents:.4f} cents")

        print("\n" + "=" * 60)

    # ---------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------
    def shutdown(self):
        if torch.cuda.is_available():
            self.gpu_profiler.stop_monitoring()
        print("GreenGPU shutdown complete.\n")


# =======================================================
# MAIN ENTRY
# =======================================================

def main():
    greengpu = GreenGPU(model_name="resnet18", polling_interval=0.01)

    try:
        greengpu.verify_gpu()

        if not greengpu.initialize():
            return

        greengpu.run_inference(num_inferences=2000, batch_size=1)
        greengpu.print_report()

    finally:
        greengpu.shutdown()


if __name__ == "__main__":
    main()