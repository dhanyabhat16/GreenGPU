"""
Production-ready plotting helpers for GreenGPU CSV metrics.

Produces:
    throughput_vs_batch.png
    power_vs_batch.png
    utilization_vs_batch.png
"""

from typing import Optional
import os
import pandas as pd
import matplotlib.pyplot as plt

def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely convert columns to numeric when possible.
    Compatible across pandas versions.
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df



def plot_metrics(csv_path: str, out_dir: Optional[str] = None, show: bool = False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _convert_numeric(df)

    if df.empty:
        raise ValueError("CSV file is empty.")

    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."

    os.makedirs(out_dir, exist_ok=True)

    # Normalize column naming variations
    column_map = {
        "gpu_utilization": "gpu_util",
        "gpu_util": "gpu_util",
        "power": "power",
        "gpu_power": "power",
    }

    for col in list(df.columns):
        if col in column_map:
            df.rename(columns={col: column_map[col]}, inplace=True)

    # Ensure batch_size exists
    if "batch_size" not in df.columns:
        df["batch_size"] = 0

    # Aggregate safely using numeric_only
    agg = df.groupby("batch_size", as_index=False).mean(numeric_only=True)

    # -----------------------------
    # Throughput vs Batch Size
    # -----------------------------
    if "throughput" in agg.columns:
        plt.figure()
        plt.plot(agg["batch_size"], agg["throughput"], marker="o")
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (samples/sec)")
        plt.title("Throughput vs Batch Size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "throughput_vs_batch.png"))
        if show:
            plt.show()
        plt.close()

    # -----------------------------
    # Power vs Batch Size
    # -----------------------------
    if "power" in agg.columns:
        plt.figure()
        plt.plot(agg["batch_size"], agg["power"], marker="o")
        plt.xlabel("Batch Size")
        plt.ylabel("GPU Power (W)")
        plt.title("Power vs Batch Size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "power_vs_batch.png"))
        if show:
            plt.show()
        plt.close()

    # -----------------------------
    # Utilization vs Batch Size
    # -----------------------------
    if "gpu_util" in agg.columns:
        plt.figure()
        plt.plot(agg["batch_size"], agg["gpu_util"], marker="o")
        plt.xlabel("Batch Size")
        plt.ylabel("GPU Utilization (%)")
        plt.title("GPU Utilization vs Batch Size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "utilization_vs_batch.png"))
        if show:
            plt.show()
        plt.close()

    return True
