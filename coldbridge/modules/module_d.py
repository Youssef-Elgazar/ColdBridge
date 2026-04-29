"""
Module D — Telemetry Collector & Benchmark Driver
===================================================
Drives all experiments, collects metrics, and produces comparison reports.

Responsibilities:
  1. Replay invocation traces (synthetic or Azure) against the worker pool
  2. Wrap any combination of modules (A, B, C) or none (baseline)
  3. Collect per-invocation metrics via MetricsCollector
  4. Label Module A predictions with ground truth after the fact
  5. Produce console summary, JSON metrics, CSV logs, and comparison plots

Modes:
  "baseline"   — no modules; raw Docker cold start
  "module_a"   — Module A prediction + pre-warming only
  "module_b"   — Module B snapshot restore only       (when teammates deliver)
  "module_c"   — Module C edge routing only           (when teammates deliver)
  "full"       — all modules combined

Usage (programmatic):
  harness = ModuleD(pool, mode="module_a", module_a=ModuleA(pool))
  results = harness.run(trace_events)
  harness.save_results(Path("results/run_001"))

Usage (CLI):
  python -m experiments.run_experiment --mode module_a --invocations 60
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

from coldbridge.worker.pool import WorkerPoolAdapter, InvocationResult
from coldbridge.metrics.collector import MetricsCollector, PredictionRecord, RunMetrics
from coldbridge.data.trace_loader import InvocationEvent

logger = logging.getLogger("coldbridge.module_d")
console = Console()


class ModuleD:
    """
    Benchmark harness.

    All experiment modes flow through this class.  Module A/B/C are injected
    as optional dependencies; when absent the run degrades to a lower mode.
    """

    def __init__(
        self,
        worker_pool: WorkerPoolAdapter,
        mode: str = "baseline",
        module_a=None,
        module_b=None,
        module_c=None,
        inter_event_realtime: bool = False,
        ttl_seconds: float = 120.0,
    ):
        """
        Args:
            worker_pool: DockerWorkerPool or any WorkerPoolAdapter
            mode: experiment mode string
            module_a: ModuleA instance or None
            module_b: ModuleB instance or None
            module_c: ModuleC instance or None
            inter_event_realtime: if True, sleep between events to match trace
                                  timing (slow but realistic); False = as fast
                                  as possible
            ttl_seconds: keep-alive TTL for baseline TTL comparison
        """
        self._pool = worker_pool
        self.mode = mode
        self._module_a = module_a
        self._module_b = module_b
        self._module_c = module_c
        self._realtime = inter_event_realtime
        self._ttl = ttl_seconds
        self._collector = MetricsCollector()

    # ── main entry point ──────────────────────────────────────────────────

    def run(self, events: List[InvocationEvent]) -> Dict[str, RunMetrics]:
        """
        Replay the invocation trace and collect metrics.

        Returns dict: function_name → RunMetrics
        """
        console.rule(f"[bold]ColdBridge Experiment — mode: {self.mode}[/bold]")
        console.print(f"  Functions in trace : {len({e.function_name for e in events})}")
        console.print(f"  Total invocations  : {len(events)}")
        console.print(f"  Realtime replay    : {self._realtime}")
        console.print()

        # Start Module A background loop if active
        if self._module_a is not None:
            self._module_a.start()

        trace_start = time.time()
        last_event_t = events[0].scheduled_at if events else 0.0

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running invocations...", total=len(events))

            for event in events:
                # Optionally pace invocations to match trace timing
                if self._realtime:
                    elapsed = time.time() - trace_start
                    target  = event.scheduled_at - events[0].scheduled_at
                    sleep_t = target - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                result = self._invoke(event)
                self._collector.record_invocation(result)

                # Feed result into Module A's state tracker
                if self._module_a is not None:
                    self._module_a.record_invocation(result)

                progress.advance(task)

        # Stop Module A
        if self._module_a is not None:
            self._module_a.stop()

        # Label predictions with ground truth
        if self._module_a is not None:
            self._label_predictions()

        # Compute and return metrics
        metrics = self._collector.compute(self.mode)
        return metrics

    # ── invocation dispatch ───────────────────────────────────────────────

    def _invoke(self, event: InvocationEvent) -> InvocationResult:
        """Dispatch one invocation, routing through active modules."""
        fn = event.function_name

        # Module C: edge vs cloud routing (stub returns "cloud")
        if self._module_c is not None:
            tier = self._module_c.route(fn)
        else:
            tier = "cloud"

        # Module B: attempt snapshot restore before pool invocation
        # (stub returns None, falling through to normal Docker invocation)
        if self._module_b is not None and self._module_b.is_available():
            restored = self._module_b.restore(fn)
            if restored is not None:
                # Use the restored instance — record as warm (cold_start=0)
                return InvocationResult(
                    function_name=fn,
                    was_cold=False,
                    cold_start_latency_ms=0.0,
                    response_latency_ms=0.0,
                    total_latency_ms=0.0,
                    timestamp=time.time(),
                )

        # Standard pool invocation (Docker)
        return self._pool.invoke(fn)

    # ── prediction ground truth labelling ────────────────────────────────

    def _label_predictions(self) -> None:
        """
        For each pre-warm decision made by Module A, determine whether a cold
        start actually occurred within the lookahead window afterwards.
        Appends labelled PredictionRecords to the collector.
        """
        if self._module_a is None:
            return
        prewarm_log = self._module_a.prewarm_log()
        invocations = self._collector._invocations  # raw records

        # For each prewarm event: was there a cold start within lookahead_s?
        lookahead = self._module_a.lookahead_s
        for pw in prewarm_log:
            fn = pw["function_name"]
            t  = pw["timestamp"]
            # Check if cold start occurred in (t, t + lookahead)
            hit = any(
                r["function_name"] == fn
                and r["was_cold"]
                and t <= r["timestamp"] <= t + lookahead
                for r in invocations
            )
            self._collector.record_prediction(PredictionRecord(
                timestamp=t,
                function_name=fn,
                predicted_prob=pw["prob"],
                threshold=self._module_a.theta,
                decision="prewarm",
                ground_truth=hit,
            ))

    # ── saving & reporting ────────────────────────────────────────────────

    def save_results(self, out_dir: Path) -> None:
        """Write all outputs to out_dir/."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._collector.to_csv(out_dir / "cold_starts.csv")
        self._collector.predictions_to_csv(out_dir / "predictions.csv")
        metrics = self._collector.metrics_to_json(out_dir / "metrics.json", self.mode)

        summary_lines = self._build_summary(metrics)
        summary_file = out_dir / "summary.txt"
        if summary_file.exists():
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write("\n\n--- Appended Run ---\n\n" + "\n".join(summary_lines))
        else:
            summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
        console.print(f"\n[green]Results saved -> {out_dir}[/green]")
        return metrics

    def print_summary(self, metrics: Dict[str, RunMetrics]) -> None:
        """Print a Rich table summary to the console."""
        table = Table(title=f"Results — {self.mode}", show_lines=True)
        table.add_column("Function",        style="bold")
        table.add_column("Invocations",     justify="right")
        table.add_column("Cold Starts",     justify="right")
        table.add_column("CSR (%)",         justify="right")
        table.add_column("Cold P50 (ms)",   justify="right")
        table.add_column("Cold P99 (ms)",   justify="right")
        table.add_column("Total P50 (ms)",  justify="right")
        table.add_column("Pred F1",         justify="right")

        for fn, m in sorted(metrics.items()):
            csr_str  = f"{m.cold_start_rate * 100:.1f}%"
            p50_cold = f"{m.p50_cold_start_ms:.0f}" if m.n_cold else "—"
            p99_cold = f"{m.p99_cold_start_ms:.0f}" if m.n_cold else "—"
            f1_str   = f"{m.prediction_f1:.3f}" if m.prediction_f1 is not None else "—"
            table.add_row(
                fn,
                str(m.n_invocations),
                str(m.n_cold),
                csr_str,
                p50_cold,
                p99_cold,
                f"{m.p50_total_ms:.0f}",
                f1_str,
            )
        console.print(table)

    def _build_summary(self, metrics: Dict[str, RunMetrics]) -> List[str]:
        lines = [
            f"ColdBridge Experiment Summary",
            f"Mode: {self.mode}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
        ]
        for fn, m in sorted(metrics.items()):
            lines += [
                f"\nFunction: {fn}",
                f"  Invocations  : {m.n_invocations}",
                f"  Cold starts  : {m.n_cold} ({m.cold_start_rate*100:.1f}%)",
                f"  Cold P50     : {m.p50_cold_start_ms:.0f} ms",
                f"  Cold P99     : {m.p99_cold_start_ms:.0f} ms",
                f"  Total P50    : {m.p50_total_ms:.0f} ms",
                f"  Total P99    : {m.p99_total_ms:.0f} ms",
            ]
            if m.prediction_f1 is not None:
                lines += [
                    f"  Pred Precision: {m.prediction_precision:.3f}",
                    f"  Pred Recall   : {m.prediction_recall:.3f}",
                    f"  Pred F1       : {m.prediction_f1:.3f}",
                    f"  Pre-warms     : {m.prewarm_triggered}",
                ]
        return lines


# ── Comparison utility ────────────────────────────────────────────────────────

def compare_runs(results_dir: Path) -> None:
    """
    Load all metrics.json files under results_dir, print a comparison table,
    and save a comparison.json.
    """
    results_dir = Path(results_dir)
    runs = []
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        with open(metrics_path) as f:
            data = json.load(f)
        run_name = metrics_path.parent.name
        runs.append((run_name, data))

    if not runs:
        console.print("[red]No metrics.json files found in results/[/red]")
        return

    console.rule("[bold]Cross-Run Comparison[/bold]")

    # Find all function names across all runs
    all_fns = set()
    for _, data in runs:
        all_fns.update(data.keys())

    for fn in sorted(all_fns):
        table = Table(title=f"Function: {fn}", show_lines=True)
        table.add_column("Run",           style="bold")
        table.add_column("Mode")
        table.add_column("CSR (%)",       justify="right")
        table.add_column("Cold P50 (ms)", justify="right")
        table.add_column("Cold P99 (ms)", justify="right")
        table.add_column("Total P50 (ms)",justify="right")
        table.add_column("Pred F1",       justify="right")

        baseline_csr = None
        for run_name, data in runs:
            m = data.get(fn)
            if not m:
                continue
            csr = m["cold_start_rate"]
            if baseline_csr is None:
                baseline_csr = csr

            csr_str  = f"{csr*100:.1f}%"
            if baseline_csr and baseline_csr > 0 and csr < baseline_csr:
                reduction = (1 - csr / baseline_csr) * 100
                csr_str += f" (↓{reduction:.0f}%)"

            p50_cold = f"{m['p50_cold_start_ms']:.0f}" if m["n_cold"] else "—"
            p99_cold = f"{m['p99_cold_start_ms']:.0f}" if m["n_cold"] else "—"
            f1_str   = f"{m['prediction_f1']:.3f}" if m.get("prediction_f1") else "—"

            table.add_row(
                run_name, m["mode"],
                csr_str, p50_cold, p99_cold,
                f"{m['p50_total_ms']:.0f}", f1_str,
            )
        console.print(table)

    # Save comparison JSON
    comparison = {run: data for run, data in runs}
    out = results_dir / "comparison.json"
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    console.print(f"\n[green]Comparison saved → {out}[/green]")
