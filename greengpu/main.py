"""
GreenGPU - Final Clean Orchestrator
Real GPU profiling + accurate timing
"""

import torch
import time
from profiler import GPUProfiler
from model_loader import ModelLoader
from metrics import (
    InferenceTimer,
    MetricsCollector,
    InferenceMetrics,
    EfficiencyAnalyzer,
)


class GreenGPU:
    def __init__(self, model_name: str = "resnet18", polling_interval: float = 0.01):
        self.gpu_profiler = GPUProfiler(polling_interval=polling_interval)
        self.model_loader = ModelLoader()
        self.inference_timer = InferenceTimer()
        self.metrics_collector = MetricsCollector()
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.model_name = model_name

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

    # ---------------------------------------------------
    # INFERENCE LOOP
    # ---------------------------------------------------
    def run_inference(self, num_inferences: int = 20, batch_size: int = 32):
        print("=" * 60)
        print(f"RUNNING {num_inferences} INFERENCES | Batch Size = {batch_size}")
        print("=" * 60 + "\n")

        input_size = (batch_size, 3, 224, 224)

        for i in range(num_inferences):

            self.inference_timer.start(f"inference_{i}")

            input_tensor = self.model_loader.prepare_input(input_size)
            _ = self.model_loader.inference(input_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            inference_time = self.inference_timer.stop()
            throughput = batch_size / inference_time

            gpu_metrics = (
                self.gpu_profiler.get_latest_metrics()
                if torch.cuda.is_available()
                else None
            )

            if gpu_metrics:
                power_eff = None
                if gpu_metrics.gpu_power:
                    power_eff = self.efficiency_analyzer.calculate_power_efficiency(
                        throughput, gpu_metrics.gpu_power
                    )

                metrics = InferenceMetrics(
                    inference_id=f"inference_{i}",
                    inference_time=inference_time,
                    throughput=throughput,
                    gpu_memory_used=gpu_metrics.gpu_memory_used,
                    gpu_utilization=gpu_metrics.gpu_utilization,
                    gpu_power=gpu_metrics.gpu_power,
                    power_efficiency=power_eff,
                )

                self.metrics_collector.record_inference(metrics)

                print(
                    f"Inference {i+1:2d}: "
                    f"{inference_time*1000:7.2f} ms | "
                    f"{throughput:8.2f} samples/s | "
                    f"GPU Util: {gpu_metrics.gpu_utilization:5.1f}% | "
                    f"GPU Mem: {gpu_metrics.gpu_memory_used:6.1f} MB | "
                    f"Power: {gpu_metrics.gpu_power if gpu_metrics.gpu_power else 'N/A'} W"
                )

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

        greengpu.run_inference(num_inferences=20, batch_size=32)
        greengpu.print_report()

    finally:
        greengpu.shutdown()


if __name__ == "__main__":
    main()
