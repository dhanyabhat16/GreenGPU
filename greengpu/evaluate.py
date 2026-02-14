# import os
# import time
# from typing import List, Tuple
# from transformers import pipeline
# from sklearn.metrics import accuracy_score, f1_score
# from tqdm import tqdm


# # ==============================
# # CONFIGURATION
# # ==============================

# _BASE = os.path.dirname(os.path.abspath(__file__))
# ORIGINAL_PATH = os.path.join(_BASE, "dataset", "test")
# DEDUP_PATH = os.path.join(_BASE, "dataset", "test_deduplicated")

# MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# MAX_LENGTH = 512
# BATCH_SIZE = 32  # Faster than single inference


# # ==============================
# # DATA LOADING
# # ==============================

# def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
#     texts = []
#     labels = []
#     file_counter = 0

#     pos_dir = os.path.join(dataset_path, "pos")
#     neg_dir = os.path.join(dataset_path, "neg")

#     print(f"\nLoading dataset from: {dataset_path}")

#     # Positive
#     if os.path.isdir(pos_dir):
#         for filename in sorted(os.listdir(pos_dir)):
#             if filename.endswith(".txt"):
#                 filepath = os.path.join(pos_dir, filename)
#                 with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
#                     text = f.read().strip()
#                     if text:
#                         texts.append(text)
#                         labels.append(1)

#                 file_counter += 1
#                 if file_counter % 1000 == 0:
#                     print(f"  Loaded {file_counter} files...")

#     # Negative
#     if os.path.isdir(neg_dir):
#         for filename in sorted(os.listdir(neg_dir)):
#             if filename.endswith(".txt"):
#                 filepath = os.path.join(neg_dir, filename)
#                 with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
#                     text = f.read().strip()
#                     if text:
#                         texts.append(text)
#                         labels.append(0)

#                 file_counter += 1
#                 if file_counter % 1000 == 0:
#                     print(f"  Loaded {file_counter} files...")

#     print(f"Finished loading {file_counter} files.")
#     return texts, labels


# # ==============================
# # EVALUATION FUNCTION
# # ==============================

# def evaluate_dataset(texts: List[str], labels: List[int], classifier):

#     predictions = []
#     start_time = time.time()

#     print("\nRunning inference...")

#     for i in tqdm(range(0, len(texts), BATCH_SIZE)):
#         batch = texts[i:i+BATCH_SIZE]
#         batch = [t[:MAX_LENGTH] for t in batch]

#         results = classifier(batch)

#         for result in results:
#             pred = 1 if result["label"] == "POSITIVE" else 0
#             predictions.append(pred)

#         if i % 1000 == 0 and i != 0:
#             print(f"  Processed {i} samples...")

#     end_time = time.time()

#     acc = accuracy_score(labels, predictions)
#     f1 = f1_score(labels, predictions)
#     inference_time = end_time - start_time

#     return acc, f1, inference_time


# # ==============================
# # MAIN
# # ==============================

# def main(original_path=None, dedup_path=None, device=-1):
#     """Run evaluation comparing original vs deduplicated dataset."""
#     orig_path = original_path or ORIGINAL_PATH
#     dedup_path = dedup_path or DEDUP_PATH

#     print("=" * 70)
#     print("GreenGPU Evaluation: Original vs Deduplicated")
#     print("=" * 70)

#     print("\nLoading model...")
#     classifier = pipeline(
#         "sentiment-analysis",
#         model=MODEL_NAME,
#         device=device  # -1 for CPU, 0 for GPU
#     )

#     # Load datasets
#     orig_texts, orig_labels = load_dataset(orig_path)
#     dedup_texts, dedup_labels = load_dataset(dedup_path)

#     print(f"\nOriginal samples: {len(orig_texts)}")
#     print(f"Deduplicated samples: {len(dedup_texts)}")

#     # Evaluate
#     print("\nEvaluating ORIGINAL dataset...")
#     orig_acc, orig_f1, orig_time = evaluate_dataset(
#         orig_texts, orig_labels, classifier
#     )

#     print("\nEvaluating DEDUPLICATED dataset...")
#     dedup_acc, dedup_f1, dedup_time = evaluate_dataset(
#         dedup_texts, dedup_labels, classifier
#     )

#     # Metrics
#     sample_reduction = (
#         (len(orig_texts) - len(dedup_texts)) / len(orig_texts)
#     ) * 100

#     time_reduction = (
#         (orig_time - dedup_time) / orig_time
#     ) * 100 if orig_time > 0 else 0

#     accuracy_drop = orig_acc - dedup_acc

#     print("\n" + "=" * 70)
#     print("RESULTS")
#     print("=" * 70)

#     print("\n--- ORIGINAL ---")
#     print(f"Accuracy: {orig_acc:.4f}")
#     print(f"F1 Score: {orig_f1:.4f}")
#     print(f"Inference Time: {orig_time:.2f} sec")

#     print("\n--- DEDUPLICATED ---")
#     print(f"Accuracy: {dedup_acc:.4f}")
#     print(f"F1 Score: {dedup_f1:.4f}")
#     print(f"Inference Time: {dedup_time:.2f} sec")

#     print("\n--- GREEN GPU IMPACT ---")
#     print(f"Sample Reduction: {sample_reduction:.2f}%")
#     print(f"Inference Time Reduction: {time_reduction:.2f}%")
#     print(f"Accuracy Drop: {accuracy_drop:.4f}")

#     print("=" * 70)


# if __name__ == "__main__":
#     main()


import os
import argparse
import statistics
import numpy as np
import torch
from typing import List, Tuple, Dict
from transformers import pipeline, set_seed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Connect strictly to the rest of the project ecosystem
try:
    from profiler import GPUProfiler
    from metrics import InferenceTimer, MetricsCollector, InferenceMetrics
except ImportError:
    from .profiler import GPUProfiler
    from .metrics import InferenceTimer, MetricsCollector, InferenceMetrics

# ==============================
# DETERMINISM
# ==============================
def set_deterministic_environment(seed: int):
    """Ensures 100% reproducibility across runs for scientific comparison."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed) # HuggingFace seed

# ==============================
# DATA LOADING
# ==============================
def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    file_counter = 0

    pos_dir = os.path.join(dataset_path, "pos")
    neg_dir = os.path.join(dataset_path, "neg")

    print(f"Loading dataset from: {os.path.basename(dataset_path)}")

    if os.path.isdir(pos_dir):
        for filename in sorted(os.listdir(pos_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(pos_dir, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(1)
                file_counter += 1

    if os.path.isdir(neg_dir):
        for filename in sorted(os.listdir(neg_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(neg_dir, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(0)
                file_counter += 1

    print(f"  Loaded {file_counter} files.")
    return texts, labels

# ==============================
# EVALUATION WRAPPER
# ==============================
def evaluate_dataset_single_run(
    texts: List[str], 
    labels: List[int], 
    classifier, 
    profiler: GPUProfiler, 
    timer: InferenceTimer,
    batch_size: int,
    max_length: int,
    run_id: str
) -> Tuple[InferenceMetrics, float]:
    
    profiler.metrics_history.clear()
    predictions = []
    
    # Use the project's standardized timer
    timer.start(run_id)

    for i in tqdm(range(0, len(texts), batch_size), desc="  Inference", leave=False):
        batch = texts[i:i+batch_size]
        batch = [t[:max_length] for t in batch]

        results = classifier(batch)

        for result in results:
            pred = 1 if result["label"] == "POSITIVE" else 0
            predictions.append(pred)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = timer.stop()
    acc = accuracy_score(labels, predictions)
    throughput = len(texts) / inference_time if inference_time > 0 else 0
    
    # Fetch GPU stats
    history_length = len(profiler.metrics_history)
    gpu_stats = profiler.get_average_metrics(window_size=history_length) if history_length > 0 else None
    
    avg_util = gpu_stats["avg_utilization"] if gpu_stats else 0.0
    avg_power = gpu_stats["avg_power"] if gpu_stats and gpu_stats["avg_power"] is not None else 0.0
    avg_mem = gpu_stats["avg_memory_used"] if gpu_stats else 0.0

    # Package into the project's standard InferenceMetrics dataclass
    run_metrics = InferenceMetrics(
        inference_id=run_id,
        inference_time=inference_time,
        throughput=throughput,
        gpu_memory_used=avg_mem,
        gpu_utilization=avg_util,
        cpu_utilization=None,
        gpu_power=avg_power,
        power_efficiency=(throughput / avg_power) if avg_power > 0 else 0
    )

    return run_metrics, acc

def evaluate_n_times(
    dataset_name: str, 
    texts: List[str], 
    labels: List[int], 
    classifier, 
    profiler: GPUProfiler,
    num_runs: int,
    batch_size: int,
    max_length: int
) -> Dict[str, float]:
    print(f"\nEvaluating {dataset_name.upper()} Dataset (Running {num_runs} times)...")
    
    # Utilize the centralized MetricsCollector from metrics.py
    collector = MetricsCollector()
    timer = InferenceTimer()
    acc_list = []
    
    for run_idx in range(num_runs):
        print(f"  Run {run_idx + 1}/{num_runs}:")
        run_id = f"{dataset_name.lower()}_run_{run_idx+1}"
        
        run_metrics, acc = evaluate_dataset_single_run(
            texts, labels, classifier, profiler, timer, batch_size, max_length, run_id
        )
        
        collector.record_inference(run_metrics)
        acc_list.append(acc)
        
    stats = collector.get_statistics()
    
    return {
        "accuracy": statistics.median(acc_list),
        "duration": stats["inference_time"]["mean"],
        "gpu_util": stats["gpu_utilization"]["mean"],
        "gpu_power": stats.get("gpu_power", {}).get("mean", 0.0) if stats else 0.0,
    }

# ==============================
# MAIN ORCHESTRATOR
# ==============================
def main():
    # Setup argparse to eliminate hardcoded values
    parser = argparse.ArgumentParser(description="GreenGPU Energy-Aware Evaluation")
    _BASE = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--original_path", type=str, default=os.path.join(_BASE, "dataset", "test"))
    parser.add_argument("--dedup_path", type=str, default=os.path.join(_BASE, "dataset", "test_deduplicated"))
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--carbon_factor", type=float, default=0.4, help="kg CO2 per kWh")
    parser.add_argument("--cost_factor", type=float, default=0.12, help="USD per kWh")
    
    args = parser.parse_args()

    print("=" * 70)
    print("GreenGPU Energy-Aware Evaluation: Full vs Deduplicated")
    print("=" * 70)

    set_deterministic_environment(args.seed)
    device = 0 if torch.cuda.is_available() else -1
    
    print(f"\n[1] Initializing Model & Profiler (Batch Size: {args.batch_size})...")
    classifier = pipeline(
        "sentiment-analysis",
        model=args.model,
        device=device
    )
    
    profiler = GPUProfiler(polling_interval=0.1) 
    if torch.cuda.is_available():
        profiler.start_monitoring()
        print("  Warming up GPU...")
        classifier(["Warmup sequence"] * args.batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        print("\n[2] Loading Datasets...")
        orig_texts, orig_labels = load_dataset(args.original_path)
        dedup_texts, dedup_labels = load_dataset(args.dedup_path)
        
        orig_sample_count = len(orig_texts)
        dedup_sample_count = len(dedup_texts)

        # 3. Evaluate Datasets
        orig_metrics = evaluate_n_times("Full", orig_texts, orig_labels, classifier, profiler, args.runs, args.batch_size, args.max_length)
        dedup_metrics = evaluate_n_times("Deduplicated", dedup_texts, dedup_labels, classifier, profiler, args.runs, args.batch_size, args.max_length)

    finally:
        if torch.cuda.is_available():
            profiler.stop_monitoring()

    # 4. Compute Impact & Savings (using dynamic CLI factors)
    sample_reduction_pct = ((orig_sample_count - dedup_sample_count) / orig_sample_count) * 100 if orig_sample_count > 0 else 0
    accuracy_diff = orig_metrics["accuracy"] - dedup_metrics["accuracy"]
    
    time_saved_sec = orig_metrics["duration"] - dedup_metrics["duration"]
    time_saved_pct = (time_saved_sec / orig_metrics["duration"]) * 100 if orig_metrics["duration"] > 0 else 0

    orig_energy_wh = orig_metrics["gpu_power"] * (orig_metrics["duration"] / 3600.0)
    dedup_energy_wh = dedup_metrics["gpu_power"] * (dedup_metrics["duration"] / 3600.0)
    
    energy_saved_wh = orig_energy_wh - dedup_energy_wh
    energy_saved_kwh = energy_saved_wh / 1000.0
    
    carbon_saved_g = (energy_saved_kwh * args.carbon_factor) * 1000 
    cost_saved_cents = (energy_saved_kwh * args.cost_factor) * 100

    # 5. Print Report
    print("\n" + "=" * 70)
    print(f"{'EXPERIMENTAL RESULTS':^70}")
    print("=" * 70)
    
    print(f"\n{ 'METRIC':<25} | { 'FULL DATASET':<18} | { 'DEDUPLICATED':<18}")
    print("-" * 70)
    print(f"{ 'Test Samples':<25} | { orig_sample_count:<18} | { dedup_sample_count:<18}")
    print(f"{ 'Accuracy':<25} | { orig_metrics['accuracy'] * 100:.2f}%{'':<12} | { dedup_metrics['accuracy'] * 100:.2f}%")
    print(f"{ 'Inference Duration':<25} | { orig_metrics['duration']:.2f}s{'':<13} | { dedup_metrics['duration']:.2f}s")
    print(f"{ 'Average GPU Util':<25} | { orig_metrics['gpu_util']:.1f}%{'':<13} | { dedup_metrics['gpu_util']:.1f}%")
    print(f"{ 'Average GPU Power':<25} | { orig_metrics['gpu_power']:.2f}W{'':<12} | { dedup_metrics['gpu_power']:.2f}W")
    print(f"{ 'Total Energy Consumed':<25} | { orig_energy_wh:.4f} Wh{'':<9} | { dedup_energy_wh:.4f} Wh")

    print("\n" + "=" * 70)
    print(f"{'GREEN AI IMPACT SUMMARY':^70}")
    print("=" * 70)
    print(f"  Accuracy Preserved    : Difference of {accuracy_diff * 100:+.3f}%")
    print(f"  Data Redundancy Cut   : {sample_reduction_pct:.1f}% fewer samples evaluated")
    print(f"  Compute Time Saved    : {time_saved_sec:.2f} seconds ({time_saved_pct:.1f}% faster)")
    print(f"  Energy Conserved      : {energy_saved_wh:.4f} Wh")
    print(f"  Carbon Prevented      : {carbon_saved_g:.4f} grams CO2")
    print(f"  Cost Avoided          : {cost_saved_cents:.4f} cents")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
