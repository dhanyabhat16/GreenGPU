"""Plot the latest metrics CSV in ./metrics/ using greengpu.plots.plot_metrics

Usage:
    python -m greengpu.plot_latest
or
    python -c "from greengpu.plot_latest import plot_latest; plot_latest()"
"""
import os
import glob
from typing import Optional

from .plots import plot_metrics


def find_latest_metrics_csv(metrics_dir: str = None) -> Optional[str]:
    if metrics_dir is None:
        metrics_dir = os.path.join(os.getcwd(), "metrics")
    pattern = os.path.join(metrics_dir, "metrics-*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_latest(metrics_dir: str = None, out_dir: Optional[str] = None, show: bool = False):
    csv_path = find_latest_metrics_csv(metrics_dir)
    if not csv_path:
        raise FileNotFoundError(f"No metrics CSV found in {metrics_dir or './metrics'}")
    return plot_metrics(csv_path, out_dir=out_dir, show=show)


if __name__ == "__main__":
    try:
        plot_latest(show=False)
        print("Plots saved to ./metrics/")
    except Exception as e:
        print(f"Error: {e}")
