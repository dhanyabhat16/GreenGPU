#!/usr/bin/env python3
"""
GreenGPU - Unified Entry Point

GPU utilization profiler + IMDB semantic deduplication + evaluation pipeline.

Commands:
  profile       Run GPU inference profiling and efficiency analysis
  deduplicate   Semantic deduplication of IMDB test dataset
  evaluate      Compare original vs deduplicated dataset metrics
  pipeline      Run deduplicate + evaluate (full sustainability workflow)
  verify        Verify GPU/CUDA setup
"""

import argparse
import os
import sys

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Dataset paths (relative to greengpu package)
GREENGPU_DIR = os.path.join(ROOT, "greengpu")
DATASET_TEST = os.path.join(GREENGPU_DIR, "dataset", "test")
DATASET_DEDUP = os.path.join(GREENGPU_DIR, "dataset", "test_deduplicated")


def cmd_verify(_args):
    """Verify GPU/CUDA setup."""
    import torch

    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("\n✓ CUDA is available")
        print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
    else:
        print("\n✗ CUDA is NOT available - GPU support not detected")
        print("GPU operations will run on CPU instead")

    print("\n" + "=" * 60)


def cmd_profile(args):
    """Run GPU profiling and inference analysis."""
    from greengpu.main import GreenGPU

    greengpu = GreenGPU(
        model_name=args.model,
        polling_interval=args.poll_interval,
        auto_switch_to_cpu=not args.no_auto_switch,
    )

    try:
        greengpu.verify_gpu()
        if not greengpu.initialize():
            return
        greengpu.run_inference(
            num_inferences=args.inferences,
            batch_size=args.batch_size,
        )
        greengpu.print_report()
    finally:
        greengpu.shutdown()


def cmd_deduplicate(args):
    """Run IMDB semantic deduplication."""
    from greengpu.imdb_deduplicator import IMDBTextDeduplicator

    dataset_path = args.dataset or DATASET_TEST
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        print("Run from project root, or specify --dataset path")
        sys.exit(1)

    deduplicator = IMDBTextDeduplicator(dataset_path, threshold=args.threshold)
    result = deduplicator.deduplicate()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\nDescription:")
    print(result["description"])


def cmd_evaluate(args):
    """Evaluate original vs deduplicated dataset."""
    from greengpu.evaluate import main as eval_main

    orig = args.original or DATASET_TEST
    dedup = args.deduplicated or DATASET_DEDUP
    device = 0 if args.gpu else -1

    if not os.path.isdir(orig):
        print(f"Error: Original dataset not found: {orig}")
        sys.exit(1)
    if not os.path.isdir(dedup):
        print(f"Error: Deduplicated dataset not found: {dedup}")
        print("Run 'deduplicate' first to create it.")
        sys.exit(1)

    eval_main(original_path=orig, dedup_path=dedup, device=device)


def cmd_pipeline(args):
    """Run full pipeline: deduplicate then evaluate."""
    print("=" * 70)
    print("GreenGPU Full Pipeline: Deduplicate → Evaluate")
    print("=" * 70)

    dataset_path = args.dataset or DATASET_TEST

    # Step 1: Deduplicate
    from greengpu.imdb_deduplicator import IMDBTextDeduplicator

    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    deduplicator = IMDBTextDeduplicator(dataset_path, threshold=args.threshold)
    deduplicator.deduplicate()

    # Step 2: Evaluate
    from greengpu.evaluate import main as eval_main

    dedup_path = os.path.join(
        os.path.dirname(dataset_path), "test_deduplicated"
    )
    device = 0 if args.gpu else -1
    eval_main(original_path=dataset_path, dedup_path=dedup_path, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="GreenGPU - GPU Profiling & Sustainability Optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # verify
    subparsers.add_parser("verify", help="Verify GPU/CUDA setup")

    # profile
    p_profile = subparsers.add_parser("profile", help="Run GPU inference profiling")
    p_profile.add_argument(
        "--model", default="resnet18", help="Model name (default: resnet18)"
    )
    p_profile.add_argument(
        "--inferences", type=int, default=1000, help="Number of inferences"
    )
    p_profile.add_argument(
        "--batch-size", type=int, default=16, help="Batch size"
    )
    p_profile.add_argument(
        "--poll-interval", type=float, default=0.01, help="GPU polling interval (s)"
    )
    p_profile.add_argument(
        "--no-auto-switch",
        action="store_true",
        help="Disable auto CPU switch on low GPU utilization",
    )

    # deduplicate
    p_dedup = subparsers.add_parser(
        "deduplicate", help="Semantic deduplication of IMDB test dataset"
    )
    p_dedup.add_argument(
        "--dataset",
        default=None,
        help=f"Path to test dataset (default: {DATASET_TEST})",
    )
    p_dedup.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Cosine similarity threshold (default: 0.80)",
    )

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Compare original vs deduplicated dataset",
    )
    p_eval.add_argument(
        "--original",
        default=None,
        help=f"Path to original test dataset (default: {DATASET_TEST})",
    )
    p_eval.add_argument(
        "--deduplicated",
        default=None,
        help=f"Path to deduplicated dataset (default: {DATASET_DEDUP})",
    )
    p_eval.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")

    # pipeline
    p_pipe = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: deduplicate + evaluate",
    )
    p_pipe.add_argument(
        "--dataset",
        default=None,
        help=f"Path to test dataset (default: {DATASET_TEST})",
    )
    p_pipe.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Cosine similarity threshold (default: 0.80)",
    )
    p_pipe.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "verify": cmd_verify,
        "profile": cmd_profile,
        "deduplicate": cmd_deduplicate,
        "evaluate": cmd_evaluate,
        "pipeline": cmd_pipeline,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
