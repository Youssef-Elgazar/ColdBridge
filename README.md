# ColdBridge — Cold Start Mitigation Research Platform

**Team:** Nadira Mohamed 22101377 · Basant Awad 22101405 · Youssef Elgazar 20100251

---

## Overview

ColdBridge is a research platform for measuring and mitigating the cold start
problem in serverless/FaaS environments. It runs **real Docker containers**
as worker instances, giving genuine cold start latency numbers rather than
simulations.

### Why Docker (not AWS Lambda)?

Real Docker containers on your local machine give:

- Actual image-pull, runtime-init, and dependency-load latency
- Fully controlled, reproducible experiments (no external scheduling noise)
- Real JVM cold starts (~1.5–3 s), Node cold starts (~400–800 ms), Python (~300–600 ms)
- The same interface that can be swapped to AWS or Firecracker later

---

## Project Structure

```
coldbridge/
├── coldbridge/               ← Core library
│   ├── modules/
│   │   ├── module_a.py       ← Transformer Pre-Warmer        (Youssef) ✓
│   │   ├── module_b.py       ← Snapshot Registry stub        (Basant)
│   │   ├── module_c.py       ← Edge-Cloud Orchestrator stub  (Nadira)
│   │   └── module_d.py       ← Benchmark Harness             (Youssef) ✓
│   ├── worker/
│   │   └── pool.py           ← DockerWorkerPool + adapter interface
│   ├── metrics/
│   │   └── collector.py      ← MetricsCollector (CSR, P50/P99, F1, ...)
│   └── data/
│       └── trace_loader.py   ← Synthetic trace + Azure trace loader
├── functions/
│   ├── python_fn/            ← Python 3.11 worker image
│   ├── node_fn/              ← Node.js 20 worker image
│   └── java_fn/              ← Java 17 (JVM) worker image
├── experiments/
│   ├── run_experiment.py     ← Main CLI entry point
│   └── plot_results.py       ← Generate comparison plots
├── tests/
│   └── test_module_a.py      ← Unit tests (no Docker needed)
├── scripts/
│   ├── build_images.bat      ← Build Docker images (Windows)
│   └── build_images.sh       ← Build Docker images (Linux/macOS)
├── results/                  ← Experiment outputs (auto-created)
├── setup.bat                 ← One-shot Windows setup
└── requirements.txt
```

---

## Quick Start (Windows)

### 1. Prerequisites

- Python 3.11+ — https://python.org
- Docker Desktop — https://docker.com/products/docker-desktop  
  _(WSL2 is optional — not required)_

### 2. Setup (run once)

```bat
setup.bat
```

This creates a virtual environment, installs dependencies, and builds all
three Docker worker images.

### 3. Activate environment

```bat
.venv\Scripts\activate
```

### 4. Run experiments

```bat
REM Baseline — raw cold starts, no modules active
python -m experiments.run_experiment run --mode baseline --invocations 30

REM Module A — transformer prediction + pre-warming
python -m experiments.run_experiment run --mode module_a --invocations 30

REM Full suite (B and C stubbed until teammates integrate)
python -m experiments.run_experiment run --mode full --skip_b --skip_c --invocations 30

REM Compare all runs
python -m experiments.run_experiment compare

REM Plot results
python -m experiments.plot_results --results_dir results/
```

---

## Experiment Modes

| Mode       | Module A | Module B | Module C | Description                           |
| ---------- | -------- | -------- | -------- | ------------------------------------- |
| `baseline` | —        | —        | —        | Raw Docker cold starts (ground truth) |
| `module_a` | ✓        | —        | —        | Prediction + pre-warming only         |
| `module_b` | —        | ✓        | —        | Snapshot restore only                 |
| `module_c` | —        | —        | ✓        | Edge routing only                     |
| `full`     | ✓        | ✓        | ✓        | All modules combined                  |

Add `--skip_b` or `--skip_c` to any mode to disable those stubs while your
teammates are still implementing.

---

## CLI Reference

```
python -m experiments.run_experiment run [OPTIONS]

  --mode          baseline | module_a | module_b | module_c | full
  --functions     all | python_fn | node_fn | java_fn  (comma-separated)
  --invocations   Approximate total invocations [default: 30]
  --duration      Trace window in seconds (overrides --invocations)
  --realtime      Pace events to match trace timing (slower, more realistic)
  --skip_b        Disable Module B
  --skip_c        Disable Module C
  --theta         Module A decision threshold [default: 0.45]
  --ttl           Container keep-alive TTL in seconds [default: 120]
  --seed          Random seed [default: 42]
  --out           Output directory [default: results/<timestamp>_<mode>]

python -m experiments.run_experiment compare [OPTIONS]

  --results_dir   Directory to scan for metrics.json files [default: results/]

python -m experiments.plot_results [OPTIONS]

  --results_dir   [default: results/]
  --out_dir       [default: results/plots/]
```

---

## Output Files

Each run writes to `results/<timestamp>_<mode>/`:

| File              | Contents                                               |
| ----------------- | ------------------------------------------------------ |
| `metrics.json`    | All quantitative metrics per function                  |
| `cold_starts.csv` | Per-invocation log: timestamp, function, cold/warm, ms |
| `predictions.csv` | Module A decisions with ground truth labels            |
| `summary.txt`     | Human-readable text report                             |

`results/comparison.json` is written by the `compare` command.

---

## Running Tests (no Docker needed)

```bat
python -m pytest tests/ -v
```

Tests cover: feature encoding, FunctionState ring buffer, heuristic
predictor, transformer forward pass, save/load, training loop,
MetricsCollector computations, and SyntheticTraceGenerator.

---

## For Teammates: Integrating Modules B and C

When you're ready to integrate your module, replace the stub in
`coldbridge/modules/module_b.py` (or `module_c.py`) with your real
implementation. The interface is minimal:

**Module B** must implement:

```python
def snapshot(self, function_name: str, container_id: str) -> Optional[str]: ...
def restore(self, function_name: str) -> Optional[WorkerInstance]: ...
def list_snapshots(self) -> List[dict]: ...
def is_available(self) -> bool: ...
```

**Module C** must implement:

```python
def route(self, function_name: str) -> str: ...  # returns "edge" or "cloud"
def is_available(self) -> bool: ...
```

Nothing else in the codebase needs to change.

---

## Using the Azure Traces (for paper experiments)

1. Download from https://github.com/Azure/AzurePublicDataset
2. Extract the CSV file
3. Use `AzureTraceLoader` instead of `SyntheticTraceGenerator`:

```python
from coldbridge.data.trace_loader import AzureTraceLoader
loader = AzureTraceLoader("path/to/azure_trace.csv")
events = loader.load(max_functions=3, duration_minutes=60)
```

---

## Reproducibility

- All experiments use `--seed` for deterministic trace generation
- Docker image digests are pinned in the Dockerfiles
- Results are timestamped and self-contained in `results/`
- Model weights saved automatically to `results/<run>/model.pt` when Module A trains
