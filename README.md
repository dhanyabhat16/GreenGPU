# GreenGPU – GPU Profiler & Sustainability Toolkit

A single project for GPU monitoring, inference profiling, semantic dataset deduplication, and evaluation—built around sustainability and efficiency.

## Features

| Feature | Description |
|--------|-------------|
| **GPU Profiling** | Real-time GPU metrics, inference timing, throughput, and efficiency analysis |
| **CPU Auto-Switch** | Automatically switch to CPU when GPU utilization is low |
| **IMDB Deduplicator** | Semantic duplicate removal using transformer embeddings and FAISS |
| **Evaluation Pipeline** | Compare original vs deduplicated datasets (accuracy, F1, inference time) |
| **AI Explanations** | Optional Gemini-powered explanations (set `GEMINI_API_KEY`) |

## Project Structure

```
Defaulters/
├── main.py              # Unified CLI entry point
├── config.py            # Config (GEMINI_API_KEY, etc.)
├── requirements.txt
├── verify_gpu.py        # Quick GPU check
└── greengpu/
    ├── profiler.py      # GPU profiling
    ├── model_loader.py  # Model loading & inference
    ├── metrics.py       # Inference metrics & efficiency
    ├── main.py          # GreenGPU orchestrator
    ├── imdb_deduplicator.py  # Semantic deduplication
    ├── gemini_explainer.py   # AI explanations
    ├── evaluate.py      # Original vs dedup evaluation
    └── dataset/
        ├── test/        # IMDB test (pos/, neg/)
        └── test_deduplicated/  # Created by deduplicate
```

## Installation

### 1. GPU Setup (optional)

Verify PyTorch sees your GPU:

```bash
python main.py verify
# or
python verify_gpu.py
```

### 2. Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

All commands run from the project root.

### GPU Profiling

```bash
python main.py profile
```

Options:
- `--model resnet18` (default)
- `--inferences 1000`
- `--batch-size 16`
- `--no-auto-switch` – keep GPU even with low utilization

### IMDB Semantic Deduplication

```bash
python main.py deduplicate
```

Options:
- `--dataset path/to/test` (default: greengpu/dataset/test)
- `--threshold 0.80` (cosine similarity)

### Evaluation (Original vs Deduplicated)

```bash
python main.py evaluate
```

Requires `test_deduplicated` (run `deduplicate` first). Use `--gpu` for GPU inference.

### Full Pipeline

```bash
python main.py pipeline
```

Runs deduplicate → evaluate in one go.

## Usage Examples

### Programmatic

```python
from greengpu.main import GreenGPU

profiler = GreenGPU(model_name="resnet18")
profiler.verify_gpu()
profiler.initialize()
profiler.run_inference(num_inferences=100, batch_size=8)
profiler.print_report()
profiler.shutdown()
```

```python
from greengpu.imdb_deduplicator import IMDBTextDeduplicator

deduplicator = IMDBTextDeduplicator("greengpu/dataset/test", threshold=0.80)
result = deduplicator.deduplicate()
```

## Configuration

- **GEMINI_API_KEY** – Optional. Set for AI explanations (CPU shift, test case removal).
- **GEMINI_MODEL** – Default: `gemini-1.5-flash`.

## Auto-switch Heuristic

The auto-switch decision uses short probe runs to measure GPU utilization. To avoid noisy early measurements we use a robust statistic (configurable as `median` or `p99`) instead of the mean. A hysteresis counter (`auto_switch_hysteresis`, default 3) requires the low/high condition to be sustained across multiple probes before switching devices. The default threshold of 20% was chosen because:

- Below ~20% GPU utilization the device is typically underutilized for inference workloads and energy efficiency favors CPU or larger batch sizes.
- Using a robust statistic (median or P99) avoids flipping due to outlier samples or transient spikes.

You can tune `probe_stat` (`median` or `p99`) and `auto_switch_hysteresis` in `GreenGPU(...)` to match your environment and stability requirements.

```bash
export GEMINI_API_KEY=your_key
```

## Troubleshooting

### GPU Not Detected

1. `nvidia-smi`
2. `python main.py verify`
3. Ensure a CUDA-compatible PyTorch build.

### Out of Memory

- Reduce `--batch-size`
- Use a smaller model (e.g. `resnet18`)

### Dataset Not Found

- Run from the project root.
- Or pass explicit paths: `python main.py deduplicate --dataset path/to/test`.

## License

MIT
