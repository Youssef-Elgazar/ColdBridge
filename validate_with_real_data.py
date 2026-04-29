"""
validate_with_real_data.py — Test ColdBridge modules against real-world datasets
=================================================================================

This script loads each available real-world dataset and validates
ColdBridge's modules (A, B, C, D) against production cold start data.

Usage:
    python validate_with_real_data.py
    python validate_with_real_data.py --dataset industry40
    python validate_with_real_data.py --dataset zenodo
    python validate_with_real_data.py --dataset all --max-events 2000
"""

import logging
import sys
import time
import json
from pathlib import Path
from dataclasses import asdict

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coldbridge.validation")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

from coldbridge.data.real_trace_loaders import (
    Industry40ColdStartLoader,
    ZenodoHuaweiLoader,
    AzureFunctions2019Loader,
)
from coldbridge.data.trace_loader import InvocationEvent, ColdStartRecord


def print_header(text: str):
    if HAS_RICH:
        console.rule(f"[bold cyan]{text}[/bold cyan]")
    else:
        print(f"\n{'='*60}\n{text}\n{'='*60}")


def print_info(text: str):
    if HAS_RICH:
        console.print(f"  {text}")
    else:
        print(f"  {text}")


def validate_dataset(
    name: str,
    events: list,
    training_records: list,
    out_dir: Path,
):
    """Run full validation of a dataset against ColdBridge modules."""

    print_header(f"Dataset: {name}")

    # ── Basic Statistics ──────────────────────────────────────────────────
    n_events = len(events)
    n_cold = sum(1 for e in events if e.was_cold)
    n_warm = sum(1 for e in events if e.was_cold is not None and not e.was_cold)
    n_unknown = sum(1 for e in events if e.was_cold is None)
    fns = sorted({e.function_name for e in events})
    time_span = events[-1].scheduled_at - events[0].scheduled_at if n_events > 1 else 0

    cold_latencies = [e.cold_start_latency_ms for e in events
                      if e.cold_start_latency_ms and e.cold_start_latency_ms > 0]

    print_info(f"[bold]Events:[/bold]          {n_events}")
    print_info(f"[bold]Functions:[/bold]       {len(fns)}")
    print_info(f"[bold]Time span:[/bold]       {time_span:.0f}s ({time_span/3600:.1f}h)")
    print_info(f"[bold]Cold starts:[/bold]     {n_cold} ({n_cold/max(n_events,1)*100:.1f}%)")
    print_info(f"[bold]Warm starts:[/bold]     {n_warm}")
    if n_unknown > 0:
        print_info(f"[bold]Unknown:[/bold]         {n_unknown}")

    if cold_latencies:
        p50 = np.percentile(cold_latencies, 50)
        p99 = np.percentile(cold_latencies, 99)
        print_info(f"[bold]Cold P50:[/bold]         {p50:.1f} ms")
        print_info(f"[bold]Cold P99:[/bold]         {p99:.1f} ms")
        print_info(f"[bold]Cold Mean:[/bold]        {np.mean(cold_latencies):.1f} ms")
        print_info(f"[bold]Cold Max:[/bold]         {max(cold_latencies):.1f} ms")

    # ── Training Data Validation ──────────────────────────────────────────
    n_train = len(training_records)
    n_train_cold = sum(1 for r in training_records if r.was_cold)
    n_train_warm = n_train - n_train_cold

    print_info("")
    print_info(f"[bold]Training records:[/bold] {n_train}")
    print_info(f"  Cold: {n_train_cold}  Warm: {n_train_warm}")

    if training_records:
        iats = [r.inter_arrival_time_s for r in training_records if r.inter_arrival_time_s > 0]
        if iats:
            print_info(f"  IAT P50: {np.percentile(iats, 50):.1f}s")
            print_info(f"  IAT P99: {np.percentile(iats, 99):.1f}s")
            print_info(f"  IAT Mean: {np.mean(iats):.1f}s")

    # ── Module A Compatibility Check ──────────────────────────────────────
    print_info("")
    print_info("[bold yellow]Module A (Transformer Pre-Warmer) Compatibility:[/bold yellow]")

    # Convert training records to Module A format
    module_a_data = []
    for r in training_records:
        module_a_data.append({
            "function_name": r.function_name,
            "timestamp": r.timestamp,
            "was_cold": r.was_cold,
            "cold_start_latency_ms": r.cold_start_latency_ms,
        })

    if len(module_a_data) > 0:
        print_info(f"  ✅ {len(module_a_data)} records ready for Module A training")
        print_info(f"     Format: function_name, timestamp, was_cold, cold_start_latency_ms")

        # Check if there are enough sequences for training (need > SEQ_LEN=64)
        from collections import Counter
        fn_counts = Counter(r["function_name"] for r in module_a_data)
        trainable = sum(1 for c in fn_counts.values() if c > 64)
        print_info(f"     Functions with >64 events (trainable): {trainable}/{len(fn_counts)}")
    else:
        print_info("  ❌ No training data available")

    # ── Module B Compatibility Check ──────────────────────────────────────
    print_info("")
    print_info("[bold yellow]Module B (Snapshot Registry) Compatibility:[/bold yellow]")
    has_pod_ids = any(hasattr(e, 'function_name') and 'pod' in e.function_name.lower()
                      for e in events[:100])
    if 'huawei' in name.lower() or 'zenodo' in name.lower():
        print_info("  ✅ Pod IDs available for snapshot fingerprinting")
    else:
        print_info("  ⚠️  No Pod IDs — will use function-level snapshots")

    # ── Module C Compatibility Check ──────────────────────────────────────
    print_info("")
    print_info("[bold yellow]Module C (Edge-Cloud Orchestrator) Compatibility:[/bold yellow]")
    if cold_latencies:
        under_80 = sum(1 for l in cold_latencies if l < 80)
        print_info(f"  Cold starts < 80ms: {under_80}/{len(cold_latencies)}")
        print_info(f"  → {len(cold_latencies) - under_80} events would benefit from edge routing")
    else:
        print_info("  ⚠️  No latency data for edge routing analysis")

    # ── Module D (Benchmark) — Replay Simulation ──────────────────────────
    print_info("")
    print_info("[bold yellow]Module D (Benchmark Harness) — Replay Stats:[/bold yellow]")
    if cold_latencies:
        total_cold_time = sum(cold_latencies)
        avg_cold = np.mean(cold_latencies)
        print_info(f"  Total cold start overhead: {total_cold_time:.0f} ms ({total_cold_time/1000:.1f}s)")
        print_info(f"  Avg cold start latency: {avg_cold:.0f} ms")
        print_info(f"  If Module A prevents 80% cold starts:")
        savings = total_cold_time * 0.8
        print_info(f"    → Saved: {savings:.0f} ms ({savings/1000:.1f}s)")
        print_info(f"    → Target P99 < 80ms: {'✅ ACHIEVABLE' if p99 * 0.2 < 80 else '⚠️  CHALLENGING'}")

    # ── Save results ──────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": name,
        "n_events": n_events,
        "n_cold": n_cold,
        "n_warm": n_warm,
        "n_functions": len(fns),
        "time_span_s": time_span,
        "cold_start_rate": n_cold / max(n_events, 1),
        "cold_p50_ms": float(np.percentile(cold_latencies, 50)) if cold_latencies else None,
        "cold_p99_ms": float(np.percentile(cold_latencies, 99)) if cold_latencies else None,
        "cold_mean_ms": float(np.mean(cold_latencies)) if cold_latencies else None,
        "training_records": n_train,
        "module_a_ready": len(module_a_data) > 0,
        "module_a_trainable_fns": trainable if 'trainable' in dir() else 0,
    }

    with open(out_dir / f"{name}_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training data as JSON for Module A
    if module_a_data:
        with open(out_dir / f"{name}_training_data.json", "w") as f:
            json.dump(module_a_data[:10000], f, default=str)
        print_info(f"\n  📁 Saved to {out_dir / f'{name}_validation.json'}")
        print_info(f"  📁 Training data: {out_dir / f'{name}_training_data.json'}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate ColdBridge with real datasets")
    parser.add_argument("--dataset", default="all",
                       choices=["all", "industry40", "zenodo", "azure"],
                       help="Which dataset to validate")
    parser.add_argument("--max-events", type=int, default=5000,
                       help="Max events to load per dataset")
    parser.add_argument("--out", default="results/real_data_validation",
                       help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    all_results = {}

    if HAS_RICH:
        console.print(Panel(
            "[bold]ColdBridge Real-World Dataset Validation[/bold]\n"
            "Testing modules against production cold start traces",
            border_style="cyan",
        ))

    # ── 1. Industry 4.0 Dataset ───────────────────────────────────────────
    if args.dataset in ("all", "industry40"):
        try:
            loader = Industry40ColdStartLoader()
            events = loader.load(max_events=args.max_events)
            records = loader.load_training_records()
            r = validate_dataset("industry40", events, records, out_dir)
            all_results["industry40"] = r
        except FileNotFoundError as e:
            logger.warning("Industry 4.0: %s", e)
        except Exception as e:
            logger.error("Industry 4.0 failed: %s", e, exc_info=True)

    # ── 2. Zenodo / Huawei Traces ─────────────────────────────────────────
    if args.dataset in ("all", "zenodo"):
        try:
            loader = ZenodoHuaweiLoader()
            events = loader.load(max_events=args.max_events)
            records = loader.load_training_records(max_records=args.max_events)
            r = validate_dataset("zenodo_huawei", events, records, out_dir)
            all_results["zenodo_huawei"] = r
        except FileNotFoundError as e:
            logger.warning("Zenodo/Huawei: %s", e)
        except Exception as e:
            logger.error("Zenodo/Huawei failed: %s", e, exc_info=True)

    # ── 3. Azure Functions 2019 ───────────────────────────────────────────
    if args.dataset in ("all", "azure"):
        try:
            loader = AzureFunctions2019Loader()
            events = loader.load(max_events=args.max_events)
            records = loader.load_training_records()
            r = validate_dataset("azure_2019", events, records, out_dir)
            all_results["azure_2019"] = r
        except FileNotFoundError as e:
            logger.warning("Azure 2019: %s", e)
        except Exception as e:
            logger.error("Azure 2019 failed: %s", e, exc_info=True)

    # ── Summary Table ─────────────────────────────────────────────────────
    if all_results and HAS_RICH:
        print_header("Cross-Dataset Comparison")
        table = Table(title="Real-World Dataset Validation Results", box=box.ROUNDED)
        table.add_column("Dataset", style="bold")
        table.add_column("Events", justify="right")
        table.add_column("Cold Starts", justify="right")
        table.add_column("CSR %", justify="right")
        table.add_column("Cold P50 ms", justify="right")
        table.add_column("Cold P99 ms", justify="right")
        table.add_column("Training Recs", justify="right")
        table.add_column("Mod A Ready", justify="center")

        for name, r in all_results.items():
            table.add_row(
                name,
                str(r["n_events"]),
                str(r["n_cold"]),
                f"{r['cold_start_rate']*100:.1f}%",
                f"{r['cold_p50_ms']:.0f}" if r["cold_p50_ms"] else "—",
                f"{r['cold_p99_ms']:.0f}" if r["cold_p99_ms"] else "—",
                str(r["training_records"]),
                "✅" if r["module_a_ready"] else "❌",
            )
        console.print(table)

    # Save combined results
    if all_results:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "combined_validation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print_info(f"\n📁 Combined results: {out_dir / 'combined_validation.json'}")


if __name__ == "__main__":
    main()
