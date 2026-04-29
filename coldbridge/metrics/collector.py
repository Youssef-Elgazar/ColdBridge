"""
Metrics Collector
=================
Accumulates InvocationResults and computes all paper metrics:
  CSR, P50/P99 latency, prediction precision/recall/F1, pool overhead.

Results are exportable to JSON and CSV for Module D's reports.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RunMetrics:
    """Full metric snapshot for one experiment run."""
    mode: str
    function_name: str
    n_invocations: int = 0
    n_cold: int = 0
    n_warm: int = 0
    cold_start_rate: float = 0.0

    # Latency (ms)
    p50_cold_start_ms: float = 0.0
    p99_cold_start_ms: float = 0.0
    mean_cold_start_ms: float = 0.0
    p50_total_ms: float = 0.0
    p99_total_ms: float = 0.0
    mean_total_ms: float = 0.0

    # Module A prediction metrics (filled if module_a active)
    prediction_precision: Optional[float] = None
    prediction_recall: Optional[float] = None
    prediction_f1: Optional[float] = None
    prewarm_triggered: int = 0
    prewarm_hit: int = 0        # prewarm led to actual warm invocation
    prewarm_wasted: int = 0     # prewarm triggered but evicted before use

    # Efficiency
    pool_overhead_pct: float = 0.0   # extra warm instances / min required


@dataclass
class PredictionRecord:
    """Log entry for one Module A prediction decision."""
    timestamp: float
    function_name: str
    predicted_prob: float
    threshold: float
    decision: str       # "prewarm" | "skip"
    ground_truth: bool  # was there actually a cold start soon after?


class MetricsCollector:
    """Thread-safe accumulator for invocation and prediction records."""

    def __init__(self):
        self._invocations: List[dict] = []
        self._predictions: List[PredictionRecord] = []
        self._start_time = time.time()

    # ── recording ──────────────────────────────────────────────────────────

    def record_invocation(self, result) -> None:
        """Accept an InvocationResult from the worker pool."""
        self._invocations.append({
            "timestamp":            result.timestamp,
            "function_name":        result.function_name,
            "was_cold":             result.was_cold,
            "cold_start_latency_ms": result.cold_start_latency_ms,
            "response_latency_ms":  result.response_latency_ms,
            "total_latency_ms":     result.total_latency_ms,
            "success":              result.success,
        })

    def record_prediction(self, pred: PredictionRecord) -> None:
        self._predictions.append(pred)

    # ── computation ───────────────────────────────────────────────────────

    def compute(self, mode: str) -> Dict[str, RunMetrics]:
        """Return a RunMetrics per function name."""
        if not self._invocations:
            return {}

        results: Dict[str, RunMetrics] = {}
        fn_names = {r["function_name"] for r in self._invocations}

        for fn in fn_names:
            rows = [r for r in self._invocations if r["function_name"] == fn]
            cold = [r for r in rows if r["was_cold"]]
            warm = [r for r in rows if not r["was_cold"]]
            preds = [p for p in self._predictions if p.function_name == fn]

            cold_lats = [r["cold_start_latency_ms"] for r in cold]
            total_lats = [r["total_latency_ms"] for r in rows]

            m = RunMetrics(
                mode=mode,
                function_name=fn,
                n_invocations=len(rows),
                n_cold=len(cold),
                n_warm=len(warm),
                cold_start_rate=len(cold) / max(len(rows), 1),
                p50_cold_start_ms=float(np.percentile(cold_lats, 50)) if cold_lats else 0.0,
                p99_cold_start_ms=float(np.percentile(cold_lats, 99)) if cold_lats else 0.0,
                mean_cold_start_ms=float(np.mean(cold_lats)) if cold_lats else 0.0,
                p50_total_ms=float(np.percentile(total_lats, 50)) if total_lats else 0.0,
                p99_total_ms=float(np.percentile(total_lats, 99)) if total_lats else 0.0,
                mean_total_ms=float(np.mean(total_lats)) if total_lats else 0.0,
            )

            # Prediction metrics
            if preds:
                prewarm_preds = [p for p in preds if p.decision == "prewarm"]
                tp = sum(1 for p in prewarm_preds if p.ground_truth)
                fp = sum(1 for p in prewarm_preds if not p.ground_truth)
                fn_preds = [p for p in preds if p.decision == "skip" and p.ground_truth]
                fn_count = len(fn_preds)
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn_count, 1)
                f1 = (2 * precision * recall / max(precision + recall, 1e-9))
                m.prediction_precision = round(precision, 4)
                m.prediction_recall = round(recall, 4)
                m.prediction_f1 = round(f1, 4)
                m.prewarm_triggered = len(prewarm_preds)

            results[fn] = m

        return results

    # ── export ─────────────────────────────────────────────────────────────

    def to_csv(self, path: Path) -> None:
        """Write raw invocation log to CSV."""
        import csv
        if not self._invocations:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._invocations[0].keys())
            w.writeheader()
            w.writerows(self._invocations)

    def predictions_to_csv(self, path: Path) -> None:
        if not self._predictions:
            return
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp","function_name","predicted_prob",
                "threshold","decision","ground_truth"
            ])
            w.writeheader()
            for p in self._predictions:
                w.writerow({
                    "timestamp": p.timestamp,
                    "function_name": p.function_name,
                    "predicted_prob": p.predicted_prob,
                    "threshold": p.threshold,
                    "decision": p.decision,
                    "ground_truth": p.ground_truth,
                })

    def metrics_to_json(self, path: Path, mode: str) -> Dict[str, RunMetrics]:
        m = self.compute(mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {fn: asdict(rm) for fn, rm in m.items()}
        if path.exists():
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
                existing.update(serialisable)
                serialisable = existing
            except Exception:
                pass
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        return m
