"""
Run Multiple Seeds
===================
Run the full ColdBridge experiment 10 times with different random seeds
to produce mean ± std statistics as claimed in the paper.

Usage:
    python scripts/run_multiple_seeds.py
    python scripts/run_multiple_seeds.py --runs 5

Results are saved to experiments/seed_runs/ as JSON files.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ColdBridge experiments with multiple seeds")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed value")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    out_dir = project_root / "experiments" / "seed_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for i in range(args.runs):
        seed = args.start_seed + i
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{args.runs}  (seed={seed})")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, "run_real_data_experiments.py", "--seed", str(seed)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")
            continue

        # Load the combined results file
        results_file = project_root / "results" / "real_data_experiments" / "combined_experiment_results.json"
        if results_file.exists():
            with open(results_file) as f:
                metrics = json.load(f)
            metrics["seed"] = seed
            metrics["run_index"] = i

            run_file = out_dir / f"run_seed_{seed}.json"
            run_file.write_text(json.dumps(metrics, indent=2))
            all_metrics.append(metrics)
            print(f"  Saved → {run_file}")
        else:
            print(f"  WARNING: results file not found at {results_file}")

    # ── Aggregate statistics ──────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"  Aggregate Statistics ({len(all_metrics)} successful runs)")
        print(f"{'='*60}")

        # Collect per-dataset metrics across runs
        datasets = set()
        for m in all_metrics:
            for exp in m.get("experiments", []):
                datasets.add((exp["dataset"], exp["mode"]))

        summary = {}
        for ds_name, mode in sorted(datasets):
            key = f"{ds_name} [{mode}]"
            f1_vals = []
            p99_imp_vals = []
            csr_imp_vals = []

            for m in all_metrics:
                for exp in m.get("experiments", []):
                    if exp["dataset"] == ds_name and exp["mode"] == mode:
                        f1_vals.append(exp["module_a_f1"])
                        p99_imp_vals.append(exp["p99_improvement_pct"])
                        csr_imp_vals.append(exp["csr_improvement_pct"])

            if f1_vals:
                summary[key] = {
                    "f1_mean": round(float(np.mean(f1_vals)), 4),
                    "f1_std": round(float(np.std(f1_vals)), 4),
                    "p99_imp_mean": round(float(np.mean(p99_imp_vals)), 2),
                    "p99_imp_std": round(float(np.std(p99_imp_vals)), 2),
                    "csr_imp_mean": round(float(np.mean(csr_imp_vals)), 2),
                    "csr_imp_std": round(float(np.std(csr_imp_vals)), 2),
                    "n_runs": len(f1_vals),
                }
                print(f"\n  {key}:")
                s = summary[key]
                print(f"    F1:       {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}")
                print(f"    P99 Imp:  {s['p99_imp_mean']:.2f}% +/- {s['p99_imp_std']:.2f}%")
                print(f"    CSR Imp:  {s['csr_imp_mean']:.2f}% +/- {s['csr_imp_std']:.2f}%")

        # Save aggregate
        agg_file = out_dir / "aggregate_statistics.json"
        agg_file.write_text(json.dumps(summary, indent=2))
        print(f"\n  Aggregate saved → {agg_file}")


if __name__ == "__main__":
    main()
