"""
plot_results.py — Visualise and compare experiment runs
=========================================================
Generates publication-quality figures from saved metrics.json files.

Usage:
  python -m experiments.plot_results --results_dir results/
"""

import json
import sys
from pathlib import Path
import click

sys.path.insert(0, str(Path(__file__).parent.parent))


@click.command()
@click.option("--results_dir", default="results", type=str)
@click.option("--out_dir", default="results/plots", type=str)
def plot(results_dir, out_dir):
    """Generate comparison plots from all saved runs."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    results_dir = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = {}
    for p in sorted(results_dir.rglob("metrics.json")):
        with open(p) as f:
            data = json.load(f)
        runs[p.parent.name] = data

    if not runs:
        print("No metrics.json files found.")
        return

    all_fns = sorted({fn for data in runs.values() for fn in data})
    run_names = list(runs.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(run_names)))

    for fn in all_fns:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Function: {fn}", fontsize=13, fontweight="bold")

        csrs, p50s, p99s, labels = [], [], [], []
        for run_name, data in runs.items():
            m = data.get(fn)
            if not m:
                continue
            csrs.append(m["cold_start_rate"] * 100)
            p50s.append(m["p50_cold_start_ms"])
            p99s.append(m["p99_cold_start_ms"])
            labels.append(run_name.split("_", 2)[-1])  # strip timestamp

        x = np.arange(len(labels))
        w = 0.5

        # CSR
        bars = axes[0].bar(x, csrs, width=w, color=colors[:len(labels)])
        axes[0].set_title("Cold Start Rate (%)")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[0].set_ylabel("%")
        for bar, v in zip(bars, csrs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

        # P50
        bars = axes[1].bar(x, p50s, width=w, color=colors[:len(labels)])
        axes[1].set_title("Cold Start P50 Latency (ms)")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[1].set_ylabel("ms")
        for bar, v in zip(bars, p50s):
            if v > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                             f"{v:.0f}", ha="center", va="bottom", fontsize=8)

        # P99
        bars = axes[2].bar(x, p99s, width=w, color=colors[:len(labels)])
        axes[2].set_title("Cold Start P99 Latency (ms)")
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[2].set_ylabel("ms")
        for bar, v in zip(bars, p99s):
            if v > 0:
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                             f"{v:.0f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out_path = out / f"{fn}_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    plot()
