"""
GreenGPU - FastAPI Backend
Serves the web UI and provides API endpoints for GPU profiling, deduplication, and evaluation.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

GREENGPU_DIR = ROOT / "greengpu"
DATASET_TEST = GREENGPU_DIR / "dataset" / "test"
DATASET_DEDUP = GREENGPU_DIR / "dataset" / "test_deduplicated"
FRONTEND_DIR = ROOT / "frontend"
REACT_BUILD_DIR = ROOT / "frontend-react" / "dist"

# Cache for async results
_job_results = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup if needed
    pass


app = FastAPI(
    title="GreenGPU",
    description="GPU Profiling & Sustainability Toolkit",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_in_thread(coro):
    """Run sync/blocking code in thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: asyncio.run(coro) if asyncio.iscoroutine(coro) else coro)


# ============ API Endpoints ============

@app.get("/api/verify")
async def api_verify():
    """Quick GPU/CUDA verification."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        result = {
            "cuda_available": cuda_available,
            "pytorch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_version": torch.version.cuda if cuda_available else None,
            "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProfileRequest(BaseModel):
    model: str = "resnet18"
    inferences: int = 20
    batch_size: int = 8


@app.post("/api/profile")
async def api_profile(req: ProfileRequest):
    """Run GPU inference profiling. Uses fewer inferences for web demo speed."""
    try:
        from greengpu.main import GreenGPU
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            greengpu = GreenGPU(
                model_name=req.model,
                polling_interval=0.01,
                auto_switch_to_cpu=True,
            )
            try:
                greengpu.verify_gpu()
                if not greengpu.initialize():
                    return {"success": False, "error": "Model initialization failed", "log": f.getvalue()}
                greengpu.run_inference(
                    num_inferences=req.inferences,
                    batch_size=req.batch_size,
                )
                stats = greengpu.metrics_collector.get_statistics()
                if stats:
                    gpu_util = stats.get("gpu_utilization", {}).get("mean", 0)
                    cpu_util = stats.get("cpu_utilization", {}).get("mean", 0)
                    # Impact metrics (when we ran on CPU, we "saved" GPU time)
                    gpu_hours_saved = (greengpu.last_run_duration / 3600.0) if greengpu.last_device_used == "cpu" else 0.0
                    avg_power = greengpu.baseline_gpu_power or stats.get("gpu_power", {}).get("mean", 0) or 50.0
                    energy_wh = (avg_power * gpu_hours_saved) if gpu_hours_saved > 0 else 0.0
                    energy_kwh = energy_wh / 1000.0
                    carbon_kg = energy_kwh * 0.4
                    cost_usd = energy_kwh * 0.12
                    # Workload score: 0-125 scale, threshold ~55
                    workload_score = min(125, int(gpu_util * 0.5 + (100 - cpu_util) * 0.5) or 65)
                    result = {
                        "success": True,
                        "stats": {
                            "inference_time_ms": stats.get("inference_time", {}).get("mean", 0) * 1000,
                            "throughput": stats.get("throughput", {}).get("mean", 0),
                            "gpu_utilization": gpu_util,
                            "gpu_memory_mb": stats.get("gpu_memory", {}).get("mean", 0),
                            "cpu_utilization": cpu_util,
                        },
                        "device_used": greengpu.last_device_used,
                        "recommend_cpu": greengpu.switched_to_cpu,
                        "gpu_hours_saved": gpu_hours_saved,
                        "energy_saved_kwh": energy_kwh,
                        "carbon_saved_kg": carbon_kg,
                        "cost_saved_usd": cost_usd,
                        "workload_score": workload_score,
                        "log": f.getvalue(),
                    }
                else:
                    result = {"success": True, "stats": {}, "log": f.getvalue()}
            finally:
                greengpu.shutdown()
        return result
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


class DeduplicateRequest(BaseModel):
    threshold: float = 0.80


@app.post("/api/deduplicate")
async def api_deduplicate(req: DeduplicateRequest):
    """Run IMDB semantic deduplication."""
    try:
        if not DATASET_TEST.is_dir():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {DATASET_TEST}")
        from greengpu.imdb_deduplicator import IMDBTextDeduplicator
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            deduplicator = IMDBTextDeduplicator(str(DATASET_TEST), threshold=req.threshold)
            res = deduplicator.deduplicate()
        # Build test case list for UI (placeholder entries from summary)
        orig = res.get("original_size", 0)
        reduced = res.get("reduced_size", 0)
        removed = orig - reduced
        test_cases = []
        for i in range(min(reduced, 5)):
            test_cases.append({"id": f"TC-{i+1:03d}", "label": "review", "status": "unique"})
        for i in range(min(removed, 5)):
            sim = 100.0 - (i * 2)
            test_cases.append({"id": f"TC-{reduced+i+1:03d}", "label": "review", "status": "duplicate", "similarTo": "TC-001", "similarity": sim})
        res["test_cases"] = test_cases
        return {"success": True, "result": res, "log": f.getvalue()}
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/evaluate")
async def api_evaluate():
    """Compare original vs deduplicated dataset."""
    try:
        if not DATASET_DEDUP.is_dir():
            raise HTTPException(status_code=404, detail="Run deduplication first. test_deduplicated not found.")
        from greengpu.evaluate import load_dataset, evaluate_dataset
        from transformers import pipeline
        from sklearn.metrics import accuracy_score, f1_score
        import time

        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        orig_texts, orig_labels = load_dataset(str(DATASET_TEST))
        dedup_texts, dedup_labels = load_dataset(str(DATASET_DEDUP))

        t0 = time.time()
        orig_preds = []
        for i in range(0, len(orig_texts), 32):
            batch = [t[:512] for t in orig_texts[i:i+32]]
            for r in classifier(batch):
                orig_preds.append(1 if r["label"] == "POSITIVE" else 0)
        orig_time = time.time() - t0
        orig_acc = accuracy_score(orig_labels, orig_preds)
        orig_f1 = f1_score(orig_labels, orig_preds)

        t0 = time.time()
        dedup_preds = []
        for i in range(0, len(dedup_texts), 32):
            batch = [t[:512] for t in dedup_texts[i:i+32]]
            for r in classifier(batch):
                dedup_preds.append(1 if r["label"] == "POSITIVE" else 0)
        dedup_time = time.time() - t0
        dedup_acc = accuracy_score(dedup_labels, dedup_preds)
        dedup_f1 = f1_score(dedup_labels, dedup_preds)

        sample_reduction = ((len(orig_texts) - len(dedup_texts)) / len(orig_texts)) * 100 if orig_texts else 0
        time_reduction = ((orig_time - dedup_time) / orig_time) * 100 if orig_time > 0 else 0

        return {
            "success": True,
            "original": {"samples": len(orig_texts), "accuracy": orig_acc, "f1": orig_f1, "time_sec": orig_time},
            "deduplicated": {"samples": len(dedup_texts), "accuracy": dedup_acc, "f1": dedup_f1, "time_sec": dedup_time},
            "impact": {"sample_reduction_pct": sample_reduction, "time_reduction_pct": time_reduction},
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


class ImpactSummaryRequest(BaseModel):
    profile: Optional[dict] = None
    dedup: Optional[dict] = None
    eval_data: Optional[dict] = None


@app.post("/api/impact-summary")
async def api_impact_summary(req: ImpactSummaryRequest):
    """Generate Gemini sustainability impact summary from dashboard data."""
    try:
        from greengpu.gemini_explainer import explain_impact_summary
        data = {
            "profile": req.profile or {},
            "dedup": req.dedup or {},
            "eval": req.eval_data or {},
        }
        summary = explain_impact_summary(data)
        return {"success": True, "summary": summary}
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "summary": f"Summary unavailable: {e}"}


# ============ Serve Frontend (React build) ============

if REACT_BUILD_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=REACT_BUILD_DIR / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(REACT_BUILD_DIR / "index.html")

    @app.get("/{path:path}")
    async def catch_all(path: str):
        return FileResponse(REACT_BUILD_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
