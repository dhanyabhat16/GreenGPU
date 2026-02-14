import os
import time
from typing import List, Tuple
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# ==============================
# CONFIGURATION
# ==============================

_BASE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_PATH = os.path.join(_BASE, "dataset", "test")
DEDUP_PATH = os.path.join(_BASE, "dataset", "test_deduplicated")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512
BATCH_SIZE = 32  # Faster than single inference


# ==============================
# DATA LOADING
# ==============================

def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    file_counter = 0

    pos_dir = os.path.join(dataset_path, "pos")
    neg_dir = os.path.join(dataset_path, "neg")

    print(f"\nLoading dataset from: {dataset_path}")

    # Positive
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
                if file_counter % 1000 == 0:
                    print(f"  Loaded {file_counter} files...")

    # Negative
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
                if file_counter % 1000 == 0:
                    print(f"  Loaded {file_counter} files...")

    print(f"Finished loading {file_counter} files.")
    return texts, labels


# ==============================
# EVALUATION FUNCTION
# ==============================

def evaluate_dataset(texts: List[str], labels: List[int], classifier):

    predictions = []
    start_time = time.time()

    print("\nRunning inference...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        batch = [t[:MAX_LENGTH] for t in batch]

        results = classifier(batch)

        for result in results:
            pred = 1 if result["label"] == "POSITIVE" else 0
            predictions.append(pred)

        if i % 1000 == 0 and i != 0:
            print(f"  Processed {i} samples...")

    end_time = time.time()

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    inference_time = end_time - start_time

    return acc, f1, inference_time


# ==============================
# MAIN
# ==============================

def main(original_path=None, dedup_path=None, device=-1):
    """Run evaluation comparing original vs deduplicated dataset."""
    orig_path = original_path or ORIGINAL_PATH
    dedup_path = dedup_path or DEDUP_PATH

    print("=" * 70)
    print("GreenGPU Evaluation: Original vs Deduplicated")
    print("=" * 70)

    print("\nLoading model...")
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=device  # -1 for CPU, 0 for GPU
    )

    # Load datasets
    orig_texts, orig_labels = load_dataset(orig_path)
    dedup_texts, dedup_labels = load_dataset(dedup_path)

    print(f"\nOriginal samples: {len(orig_texts)}")
    print(f"Deduplicated samples: {len(dedup_texts)}")

    # Evaluate
    print("\nEvaluating ORIGINAL dataset...")
    orig_acc, orig_f1, orig_time = evaluate_dataset(
        orig_texts, orig_labels, classifier
    )

    print("\nEvaluating DEDUPLICATED dataset...")
    dedup_acc, dedup_f1, dedup_time = evaluate_dataset(
        dedup_texts, dedup_labels, classifier
    )

    # Metrics
    sample_reduction = (
        (len(orig_texts) - len(dedup_texts)) / len(orig_texts)
    ) * 100

    time_reduction = (
        (orig_time - dedup_time) / orig_time
    ) * 100 if orig_time > 0 else 0

    accuracy_drop = orig_acc - dedup_acc

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- ORIGINAL ---")
    print(f"Accuracy: {orig_acc:.4f}")
    print(f"F1 Score: {orig_f1:.4f}")
    print(f"Inference Time: {orig_time:.2f} sec")

    print("\n--- DEDUPLICATED ---")
    print(f"Accuracy: {dedup_acc:.4f}")
    print(f"F1 Score: {dedup_f1:.4f}")
    print(f"Inference Time: {dedup_time:.2f} sec")

    print("\n--- GREEN GPU IMPACT ---")
    print(f"Sample Reduction: {sample_reduction:.2f}%")
    print(f"Inference Time Reduction: {time_reduction:.2f}%")
    print(f"Accuracy Drop: {accuracy_drop:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
