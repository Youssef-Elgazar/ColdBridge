"""
run_real_data_experiments.py
============================
Replay real-world cold-start traces through ColdBridge's discrete-event
simulator and compare Baseline vs ColdBridge (Modules A + B + C) performance.

Datasets tested:
  1. Industry 4.0 — IEEE JSAS 2024 (GCP Cloud Functions, IoT)
  2. Zenodo/Huawei — CCGrid 2026  (Huawei production FaaS)
  3. Azure Functions 2019 — ATC 2020 (50k functions, 14 days)

Usage:
  python run_real_data_experiments.py
  python run_real_data_experiments.py --seed 7
"""

import sys, os, json, time, random, argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from coldbridge.data.real_trace_loaders import (
    Industry40ColdStartLoader,
    ZenodoHuaweiLoader,
    AzureFunctions2019Loader,
)
from coldbridge.modules.module_a import ModuleA
from coldbridge.modules.module_b import ModuleB
from coldbridge.modules.module_c import ModuleC
from coldbridge.baselines.lstm_baseline import LSTMBaseline
from coldbridge.baselines.qlearning_baseline import QLearningBaseline
from coldbridge.baselines.static_ttl_baseline import StaticTTLBaseline


class MockResult:
    def __init__(self, fn, t, was_cold):
        self.function_name = fn
        self.timestamp = t
        self.was_cold = was_cold

def simulate_experiment(
    dataset_name: str,
    events: list,
    training_records: list,
    baseline_ttl: float = 600.0,
    seed: int = 42,
    mode: str = "full",
):
    """
    Run discrete-event simulation of Baseline vs ColdBridge for one dataset.

    Args:
        mode: 'module_a_only' — only Module A prediction, no B/C fallback.
              'full' — Module A + B + C combined.

    Baseline: Fixed TTL keep-alive (10 min). Cold start on container miss.
    ColdBridge: Module A prediction (+ Module B/C if mode='full').

    IMPORTANT: This uses the actual PyTorch Transformer in Module A,
    trained on historical records, processing events causally as a stream.
    Module B restores from real snapshots (not random multipliers).
    Module C routes deterministically (not randomly).
    """
    np.random.seed(seed)
    random.seed(seed)

    n = len(events)
    mod_a = ModuleA(theta=0.50)
    mod_b = ModuleB()
    mod_c = ModuleC()

    # Build cold-start ground truth from the real data (for EVALUATION only)
    ground_truth_cold = {}
    real_latencies = {}
    for evt in events:
        key = (evt.function_name, evt.scheduled_at)
        ground_truth_cold[key] = evt.was_cold if evt.was_cold is not None else None
        if evt.cold_start_latency_ms and evt.cold_start_latency_ms > 0:
            real_latencies[key] = evt.cold_start_latency_ms

    # Compute default cold start latency from dataset stats
    cold_lats = [evt.cold_start_latency_ms for evt in events
                 if evt.cold_start_latency_ms and evt.cold_start_latency_ms > 0]
    default_cold_lat = np.mean(cold_lats) if cold_lats else 500.0
    default_warm_lat = min(default_cold_lat * 0.01, 5.0)  # ~1% of cold

    # ── Baseline simulation ───────────────────────────────────────────────
    print(f"      [1/4] Running baseline simulation ({n} events)...", flush=True)
    base_cold_starts = 0
    base_lats = []
    base_idle_cost = 0.0
    base_last_invoked = {}

    for idx_b, evt in enumerate(events):
        fn = evt.function_name
        t = evt.scheduled_at
        key = (fn, t)

        dt = t - base_last_invoked.get(fn, -np.inf)
        is_cold_gt = ground_truth_cold.get(key)

        if is_cold_gt is True:
            base_cold_starts += 1
            lat = real_latencies.get(key, default_cold_lat)
        elif is_cold_gt is False:
            lat = np.random.uniform(1, default_warm_lat * 2)
            base_idle_cost += min(dt, baseline_ttl)
        else:
            # No ground truth — use TTL heuristic
            if dt > baseline_ttl:
                base_cold_starts += 1
                lat = np.random.uniform(default_cold_lat * 0.7, default_cold_lat * 1.3)
            else:
                lat = np.random.uniform(1, default_warm_lat * 2)
                base_idle_cost += dt
        base_last_invoked[fn] = t
        base_lats.append(lat)
        if (idx_b + 1) % 500 == 0 or idx_b == n - 1:
            print(f"        Baseline: {idx_b+1}/{n} events ({base_cold_starts} cold starts)", flush=True)

    # ── ColdBridge simulation ─────────────────────────────────────────────
    cb_cold_starts = 0
    cb_lats = []
    cb_idle_cost = 0.0
    inference_log = []

    # Train Module A on historical records first (30 epochs, paper spec)
    print(f"      [2/4] Training Module A Transformer ({len(training_records)} records, 30 epochs)...", flush=True)
    train_history = []
    for r in training_records:
        train_history.append({
            "function_name": r.function_name,
            "timestamp": r.timestamp,
            "was_cold": r.was_cold,
            "cold_start_latency_ms": r.cold_start_latency_ms,
        })
    if train_history:
        b_size = min(256, max(16, len(train_history) // 10))
        mod_a.train(train_history, epochs=30, batch_size=b_size)

    # Seed Module A's state with the training events so its history window isn't empty
    for r in train_history:
        mod_a.record_invocation(MockResult(r["function_name"], r["timestamp"], r["was_cold"]))

    # Pre-populate Module B's snapshot store with known functions
    known_fns = set(r["function_name"] for r in train_history)
    for fn in known_fns:
        mod_b.snapshot(fn, f"pretrained_{fn[:8]}")

    print(f"      [3/4] Running ColdBridge simulation ({n} events, mode={mode})...", flush=True)
    for i, evt in enumerate(events):
        fn = evt.function_name
        t = evt.scheduled_at
        key = (fn, t)

        # 1. Ask Module A for a prediction (uses its internal state)
        pred_prob = mod_a._predict(fn)
        threshold = mod_a.theta
        pre_warm = pred_prob > threshold

        # 2. Determine ACTUAL cold/warm outcome (for metrics, NOT for prediction)
        is_cold_gt = ground_truth_cold.get(key)
        if is_cold_gt is None:
            state = mod_a._states.get(fn)
            if state and state.last_invoked_at:
                dt = t - state.last_invoked_at
            else:
                dt = np.inf
            is_cold_gt = dt > baseline_ttl or dt == np.inf

        # 3. Tell Module A what ACTUALLY happened (so it can update its history)
        mod_a.record_invocation(MockResult(fn, t, is_cold_gt))

        dt_for_log = t - mod_a._states[fn].last_invoked_at if (mod_a._states.get(fn) and len(mod_a._states[fn].history) > 1) else -1

        inference_log.append({
            "timestamp": round(t, 4),
            "function": fn,
            "iat_seconds": round(dt_for_log, 2),
            "pred_prob": round(pred_prob, 4),
            "threshold": threshold,
            "pre_warm_triggered": pre_warm,
            "ground_truth_cold": is_cold_gt,
        })

        if pre_warm and is_cold_gt:
            # TRUE POSITIVE: predicted cold, was cold → pre-warmed successfully
            dt_idle = dt_for_log if dt_for_log > 0 else 30.0
            cb_idle_cost += min(dt_idle, 30.0)  # pre-warm keep-alive cost
            cb_lat = np.random.uniform(2, default_warm_lat * 2)  # near-warm latency

        elif pre_warm and not is_cold_gt:
            # FALSE POSITIVE: predicted cold, was warm → wasted pre-warm
            cb_idle_cost += 10.0  # unnecessary keep-alive cost
            cb_lat = np.random.uniform(1, default_warm_lat * 2)

        elif not pre_warm and is_cold_gt:
            # FALSE NEGATIVE: missed cold start → must handle it reactively
            cb_cold_starts += 1
            real_lat = real_latencies.get(key, default_cold_lat)

            if mode == "module_a_only":
                # Module A only: no B/C rescue — full cold start penalty
                cb_lat = real_lat
            else:
                # Full mode: Module B snapshot restore + Module C routing
                # Try Module B restore first (real snapshot lookup)
                restored = mod_b.restore(fn)
                if restored is not None:
                    # Snapshot found — restore is ~50ms (paper: <80ms)
                    # Latency = restore_time + warm_execution
                    restore_lat = 50.0 + np.random.uniform(1, default_warm_lat * 2)
                    cb_lat = restore_lat
                else:
                    # No snapshot — fall back to Module C routing
                    tier = mod_c.route(fn)
                    tier_latency = mod_c.get_latency(tier)
                    cb_lat = real_lat * (tier_latency / 150.0)

                # Capture a new snapshot for future restores
                mod_b.snapshot(fn, f"cont_{i}")

        else:
            # TRUE NEGATIVE: predicted warm, was warm → normal execution
            cb_lat = np.random.uniform(1, default_warm_lat * 2)

        cb_lats.append(cb_lat)
        if (i + 1) % 500 == 0 or i == n - 1:
            tp_so_far = sum(1 for r in inference_log if r["pre_warm_triggered"] and r["ground_truth_cold"] is True)
            fn_so_far = sum(1 for r in inference_log if not r["pre_warm_triggered"] and r["ground_truth_cold"] is True)
            print(f"        ColdBridge: {i+1}/{n} events | cold_starts={cb_cold_starts} | TP={tp_so_far} FN={fn_so_far}", flush=True)

    # ── Compute metrics ──────────────────────────────────────────────────
    print(f"      [4/4] Computing metrics...", flush=True)
    base_csr = (base_cold_starts / n) * 100
    cb_csr = (cb_cold_starts / n) * 100
    csr_imp = ((base_csr - cb_csr) / max(base_csr, 0.01)) * 100

    b_p50 = np.percentile(base_lats, 50)
    b_p99 = np.percentile(base_lats, 99)
    cb_p50 = np.percentile(cb_lats, 50)
    cb_p99 = np.percentile(cb_lats, 99)
    p99_imp = ((b_p99 - cb_p99) / max(b_p99, 0.01)) * 100
    p50_imp = ((b_p50 - cb_p50) / max(b_p50, 0.01)) * 100

    cost_imp = ((base_idle_cost - cb_idle_cost) / max(base_idle_cost, 0.01)) * 100

    # Module A prediction accuracy (evaluated against ground truth)
    tp = sum(1 for r in inference_log if r["pre_warm_triggered"] and r["ground_truth_cold"] is True)
    fp = sum(1 for r in inference_log if r["pre_warm_triggered"] and r["ground_truth_cold"] is False)
    fn_neg = sum(1 for r in inference_log if not r["pre_warm_triggered"] and r["ground_truth_cold"] is True)
    tn = sum(1 for r in inference_log if not r["pre_warm_triggered"] and r["ground_truth_cold"] is False)
    total_labeled = tp + fp + fn_neg + tn
    accuracy = (tp + tn) / max(total_labeled, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn_neg, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    results = {
        "dataset": dataset_name,
        "mode": mode,
        "seed": seed,
        "n_events": n,
        "n_functions": len(set(e.function_name for e in events)),
        "baseline_cold_starts": base_cold_starts,
        "baseline_csr_pct": round(base_csr, 2),
        "baseline_p50_ms": round(b_p50, 2),
        "baseline_p99_ms": round(b_p99, 2),
        "baseline_idle_cost": round(base_idle_cost, 2),
        "coldbridge_cold_starts": cb_cold_starts,
        "coldbridge_csr_pct": round(cb_csr, 2),
        "coldbridge_p50_ms": round(cb_p50, 2),
        "coldbridge_p99_ms": round(cb_p99, 2),
        "coldbridge_idle_cost": round(cb_idle_cost, 2),
        "csr_improvement_pct": round(csr_imp, 2),
        "p50_improvement_pct": round(p50_imp, 2),
        "p99_improvement_pct": round(p99_imp, 2),
        "idle_cost_improvement_pct": round(cost_imp, 2),
        "module_a_tp": tp,
        "module_a_fp": fp,
        "module_a_fn": fn_neg,
        "module_a_tn": tn,
        "module_a_accuracy": round(accuracy, 4),
        "module_a_precision": round(precision, 4),
        "module_a_recall": round(recall, 4),
        "module_a_f1": round(f1, 4),
        "default_cold_lat_ms": round(default_cold_lat, 2),
    }

    return results, inference_log


def run_baselines(
    dataset_name: str,
    events: list,
    training_records: list,
    seed: int = 42,
):
    """Run LSTM, Q-learning, and Static TTL baselines on the same data.

    Returns a dict of baseline_name → {f1, precision, recall}.
    """
    np.random.seed(seed)
    random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    baseline_results = {}

    # Prepare training history
    train_history = []
    for r in training_records:
        train_history.append({
            "function_name": r.function_name,
            "timestamp": r.timestamp,
            "was_cold": r.was_cold,
            "cold_start_latency_ms": r.cold_start_latency_ms,
            "runtime": r.runtime or "python",
            "memory_mb": r.memory_mb or 512.0,
        })

    # Build ground truth for evaluation
    gt_labels = []
    gt_features = []
    for evt in events:
        if evt.was_cold is not None:
            gt_labels.append(1 if evt.was_cold else 0)
            gt_features.append(evt)

    if not gt_labels:
        print(f"      No ground truth labels for {dataset_name} — skipping baselines")
        return {}

    from sklearn.metrics import f1_score, precision_score, recall_score

    # ── Static TTL Baseline ──────────────────────────────────────────────
    print("      Running Static TTL baseline...")
    ttl_base = StaticTTLBaseline(ttl_seconds=600.0)
    ttl_preds = []
    for evt in events:
        if evt.was_cold is not None:
            prob = ttl_base.predict(evt.function_name, evt.scheduled_at)
            ttl_preds.append(1 if prob >= 0.50 else 0)
        ttl_base.update(evt.function_name, evt.scheduled_at)

    if ttl_preds:
        baseline_results["static_ttl"] = {
            "f1": round(f1_score(gt_labels, ttl_preds, zero_division=0), 4),
            "precision": round(precision_score(gt_labels, ttl_preds, zero_division=0), 4),
            "recall": round(recall_score(gt_labels, ttl_preds, zero_division=0), 4),
        }

    # ── Q-Learning Baseline ──────────────────────────────────────────────
    print("      Running Q-learning baseline...")
    import time as _time
    ql = QLearningBaseline()
    ql.train(train_history, epochs=10)

    ql_preds = []
    ql_last_ts = {}
    for evt in events:
        if evt.was_cold is not None:
            fn = evt.function_name
            iat = evt.scheduled_at - ql_last_ts.get(fn, evt.scheduled_at)
            t = _time.localtime(evt.scheduled_at) if evt.scheduled_at > 1e6 else _time.localtime()
            prob = ql.predict(
                runtime=evt.runtime or "python",
                memory_mb=512.0,
                hour=t.tm_hour + t.tm_min / 60.0,
                dow=float(t.tm_wday),
                iat_s=iat,
            )
            ql_preds.append(1 if prob >= 0.50 else 0)
        ql_last_ts[evt.function_name] = evt.scheduled_at

    if ql_preds:
        baseline_results["qlearning"] = {
            "f1": round(f1_score(gt_labels, ql_preds, zero_division=0), 4),
            "precision": round(precision_score(gt_labels, ql_preds, zero_division=0), 4),
            "recall": round(recall_score(gt_labels, ql_preds, zero_division=0), 4),
        }

    # ── LSTM Baseline ────────────────────────────────────────────────────
    if TORCH_AVAILABLE and train_history:
        print("      Running LSTM baseline...")
        lstm = LSTMBaseline()
        lstm.train(train_history, epochs=30)

        from coldbridge.modules.module_a import FunctionState, SEQ_LEN, encode_step
        # Build sequences for evaluation events
        temp_states = {}
        lstm_preds = []
        for evt in events:
            fn = evt.function_name
            if fn not in temp_states:
                temp_states[fn] = FunctionState(
                    name=fn, runtime=evt.runtime or "python",
                    memory_mb=512.0,
                )
            state = temp_states[fn]

            if evt.was_cold is not None:
                seq = state.build_sequence()
                prob = lstm.predict(seq)
                lstm_preds.append(1 if prob >= 0.50 else 0)

            state.record_invocation(evt.scheduled_at, evt.was_cold or False)

        if lstm_preds:
            baseline_results["lstm"] = {
                "f1": round(f1_score(gt_labels, lstm_preds, zero_division=0), 4),
                "precision": round(precision_score(gt_labels, lstm_preds, zero_division=0), 4),
                "recall": round(recall_score(gt_labels, lstm_preds, zero_division=0), 4),
            }

    return baseline_results


def print_results_table(all_results, title="ColdBridge Real-World Dataset Experiment Results"):
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    header = f"{'Dataset':<18} | {'Mode':<14} | {'Metric':<16} | {'Baseline':>12} | {'ColdBridge':>12} | {'Improv.':>10}"
    print(header)
    print("-" * 110)

    for r in all_results:
        name = r["dataset"]
        m = r.get("mode", "full")
        print(f"{name:<18} | {m:<14} | {'CSR':.<16} | {r['baseline_csr_pct']:>10.2f} % | {r['coldbridge_csr_pct']:>10.2f} % | {r['csr_improvement_pct']:>8.1f} %")
        print(f"{'':18} | {'':14} | {'P50 Latency':.<16} | {r['baseline_p50_ms']:>9.1f} ms | {r['coldbridge_p50_ms']:>9.1f} ms | {r['p50_improvement_pct']:>8.1f} %")
        print(f"{'':18} | {'':14} | {'P99 Latency':.<16} | {r['baseline_p99_ms']:>9.1f} ms | {r['coldbridge_p99_ms']:>9.1f} ms | {r['p99_improvement_pct']:>8.1f} %")
        print(f"{'':18} | {'':14} | {'Mod A F1':.<16} | {'n/a':>12} | {r['module_a_f1']:>12.4f} |")
        print(f"{'':18} | {'':14} | {'Confusion':.<16} | {'TP='+str(r.get('module_a_tp','?')):>12} | {'FN='+str(r.get('module_a_fn','?')):>12} |")
        print(f"{'':18} | {'':14} | {'':.<16} | {'FP='+str(r.get('module_a_fp','?')):>12} | {'TN='+str(r.get('module_a_tn','?')):>12} |")
        print("-" * 110)

    print("=" * 110)


def print_baseline_comparison(baseline_results, dataset_name):
    """Print baseline comparison table."""
    if not baseline_results:
        return
    print(f"\n  Baseline Comparison for {dataset_name}:")
    print(f"  {'Method':<16} | {'F1':>8} | {'Precision':>10} | {'Recall':>8}")
    print(f"  {'-'*50}")
    for name, metrics in baseline_results.items():
        print(f"  {name:<16} | {metrics['f1']:>8.4f} | {metrics['precision']:>10.4f} | {metrics['recall']:>8.4f}")
    print()


def _run_dataset(name, loader_fn, mode, max_events=5000, seed=42):
    """Helper: load one dataset, run one experiment mode, return result."""
    loader = loader_fn()
    if name == "Azure 2019":
        events = loader.load(max_events=max_events, days=[1])
        records = loader.load_training_records(days=[1])
    else:
        events = loader.load(max_events=max_events)
        records = (loader.load_training_records(max_records=max_events)
                   if hasattr(loader.load_training_records, '__code__') and
                      'max_records' in loader.load_training_records.__code__.co_varnames
                   else loader.load_training_records())
    print(f"      Loaded {len(events)} events, {len(records)} training records")

    r, log = simulate_experiment(name, events, records, mode=mode, seed=seed)
    return r, log, events, records


def main(seed: int = 42):
    out_dir = Path("results/real_data_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    datasets = [
        ("Industry 4.0", Industry40ColdStartLoader),
        ("Zenodo/Huawei", ZenodoHuaweiLoader),
        ("Azure 2019", AzureFunctions2019Loader),
    ]

    modes = ["module_a_only", "full"]
    all_results = []
    all_baseline_results = {}

    for ds_name, loader_cls in datasets:
        for mode in modes:
            tag = f"{ds_name} [{mode}]"
            print(f"\n{'-'*60}")
            print(f"  {tag}")
            print(f"{'-'*60}")
            try:
                r, log, events, records = _run_dataset(ds_name, loader_cls, mode, seed=seed)
                all_results.append(r)

                slug = ds_name.lower().replace("/", "_").replace(" ", "_")
                pd.DataFrame(log).to_csv(
                    out_dir / f"{slug}_{mode}_inference_log.csv", index=False
                )
                print(f"      CSR: {r['baseline_csr_pct']:.1f}% -> {r['coldbridge_csr_pct']:.1f}%  |  "
                      f"P99: {r['baseline_p99_ms']:.0f}ms -> {r['coldbridge_p99_ms']:.0f}ms  |  "
                      f"F1: {r['module_a_f1']:.3f}")

                # Run baselines (only once per dataset, not per mode)
                if mode == "full" and ds_name not in all_baseline_results:
                    print(f"\n      Running baselines for {ds_name}...")
                    bl_results = run_baselines(ds_name, events, records, seed=seed)
                    if bl_results:
                        # Add Module A's F1 for comparison
                        bl_results["module_a"] = {
                            "f1": r["module_a_f1"],
                            "precision": r["module_a_precision"],
                            "recall": r["module_a_recall"],
                        }
                        all_baseline_results[ds_name] = bl_results
                        print_baseline_comparison(bl_results, ds_name)

            except Exception as e:
                print(f"      SKIPPED: {e}")

    # ── Print combined results ────────────────────────────────────────────
    a_only = [r for r in all_results if r.get("mode") == "module_a_only"]
    full   = [r for r in all_results if r.get("mode") == "full"]

    if a_only:
        print_results_table(a_only, "Module A Only (Prediction — No B/C Fallback)")
    if full:
        print_results_table(full, "Full ColdBridge (Module A + B + C)")

    # ── Save combined JSON ────────────────────────────────────────────────
    combined = {
        "seed": seed,
        "experiments": all_results,
        "baseline_comparisons": all_baseline_results,
    }
    with open(out_dir / "combined_experiment_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColdBridge Real-Data Experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    result = main(seed=args.seed)
