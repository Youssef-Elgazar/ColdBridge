"""
Trace Loader
============
Provides two invocation sequence sources:

1. SyntheticTraceGenerator  — creates realistic invocation sequences without
   requiring the Azure dataset download. Suitable for initial development and
   unit testing.

2. AzureTraceLoader         — loads the public Azure Functions 2019/2021 traces
   (download separately; see README). Used for paper-quality experiments.

Both produce a list of InvocationEvent objects consumed by Module D's harness.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("coldbridge.trace_loader")


@dataclass
class InvocationEvent:
    """Single scheduled invocation from a trace."""
    function_name: str
    scheduled_at: float          # seconds from trace start
    expected_pattern: str = ""   # "periodic" | "bursty" | "rare" — metadata only

    # ── Real-world trace metadata (optional) ──────────────────────────────
    was_cold: Optional[bool] = None            # Ground-truth cold start label
    cold_start_latency_ms: Optional[float] = None  # Measured cold start latency
    runtime: Optional[str] = None              # "python" | "node" | "java" etc.


@dataclass
class ColdStartRecord:
    """Standardised training record for Module A's cold start predictor."""
    function_name: str
    timestamp: float                           # absolute or relative timestamp
    was_cold: bool = False                     # ground-truth label
    cold_start_latency_ms: float = 0.0         # measured latency in ms
    inter_arrival_time_s: float = 0.0          # time since previous invocation
    memory_mb: Optional[float] = None          # function memory allocation
    runtime: Optional[str] = None              # language runtime


class SyntheticTraceGenerator:
    """
    Generates realistic invocation traces with three workload patterns:

    - periodic   : regular interval ± small jitter  (e.g. health checks, crons)
    - bursty     : long quiet periods interrupted by burst clusters
    - rare        : very infrequent, unpredictable calls

    Inter-arrival times are drawn from a Pareto (heavy-tail) distribution,
    consistent with the findings in Shahrad et al. (2020).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate(
        self,
        duration_seconds: float = 600.0,
        function_configs: Optional[List[dict]] = None,
    ) -> List[InvocationEvent]:
        """
        Args:
            duration_seconds: Total trace window length.
            function_configs: List of dicts with keys:
                name, pattern, mean_iat_s (mean inter-arrival time in seconds)

        Returns:
            Sorted list of InvocationEvent.
        """
        if function_configs is None:
            function_configs = self._default_configs()

        events: List[InvocationEvent] = []
        for cfg in function_configs:
            fn_events = self._generate_for_function(
                name=cfg["name"],
                pattern=cfg["pattern"],
                mean_iat=cfg["mean_iat_s"],
                duration=duration_seconds,
            )
            events.extend(fn_events)

        events.sort(key=lambda e: e.scheduled_at)
        return events

    # ── internal ──────────────────────────────────────────────────────────

    def _generate_for_function(
        self, name: str, pattern: str, mean_iat: float, duration: float
    ) -> List[InvocationEvent]:
        events = []
        t = 0.0

        if pattern == "periodic":
            while t < duration:
                jitter = self.rng.normal(0, mean_iat * 0.05)
                t += max(0.1, mean_iat + jitter)
                if t < duration:
                    events.append(InvocationEvent(name, t, pattern))

        elif pattern == "bursty":
            # Alternating quiet and burst phases
            while t < duration:
                # quiet phase: Pareto-distributed idle
                quiet = float(self.rng.pareto(1.2) * mean_iat * 2)
                t += min(quiet, duration * 0.3)
                if t >= duration:
                    break
                # burst phase: N rapid invocations
                burst_size = int(self.rng.integers(5, 25))
                for _ in range(burst_size):
                    if t >= duration:
                        break
                    gap = self.rng.exponential(mean_iat * 0.1)
                    t += gap
                    events.append(InvocationEvent(name, t, pattern))

        elif pattern == "rare":
            # Very sparse, heavy-tail inter-arrivals
            while t < duration:
                iat = float(self.rng.pareto(0.8) * mean_iat * 3)
                t += iat
                if t < duration:
                    events.append(InvocationEvent(name, t, pattern))

        return events

    @staticmethod
    def _default_configs() -> List[dict]:
        return [
            {"name": "python_fn", "pattern": "periodic", "mean_iat_s": 30.0},
            {"name": "node_fn",   "pattern": "bursty",   "mean_iat_s": 45.0},
            {"name": "java_fn",   "pattern": "rare",     "mean_iat_s": 90.0},
        ]


class AzureTraceLoader:
    """
    Loads the public Azure Functions trace CSV files.

    Download from:
      https://github.com/Azure/AzurePublicDataset
      → AzureFunctionsInvocationTraceForTwoWeeksJan2021.tar.gz

    Expected CSV columns (Azure 2021 format):
      HashOwner, HashApp, HashFunction, Trigger,
      AverageAllocatedMb, AverageAllocatedMb_pct,
      Count, ... (one row per function per minute)

    For development, use SyntheticTraceGenerator instead.
    """

    def __init__(self, csv_path: str, function_name_map: Optional[dict] = None):
        """
        Args:
            csv_path: Path to the Azure trace CSV.
            function_name_map: Optional dict mapping Azure function hashes to
                               our local function names (python_fn, node_fn, java_fn).
                               If None, functions are sampled round-robin.
        """
        self.csv_path = csv_path
        self.function_name_map = function_name_map or {}

    def load(
        self,
        max_functions: int = 3,
        duration_minutes: int = 60,
    ) -> List[InvocationEvent]:
        """Load and convert Azure trace to InvocationEvent list."""
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for AzureTraceLoader")

        df = pd.read_csv(self.csv_path)
        # Select top N functions by invocation count
        top_fns = (
            df.groupby("HashFunction")["Count"]
            .sum()
            .nlargest(max_functions)
            .index.tolist()
        )
        local_names = ["python_fn", "node_fn", "java_fn"]
        events = []
        for i, fn_hash in enumerate(top_fns):
            local_name = self.function_name_map.get(fn_hash, local_names[i % 3])
            fn_df = df[df["HashFunction"] == fn_hash].head(duration_minutes)
            t = 0.0
            for _, row in fn_df.iterrows():
                count = int(row.get("Count", 1))
                minute_start = t
                for _ in range(count):
                    offset = random.uniform(0, 60)
                    events.append(InvocationEvent(local_name, minute_start + offset))
                t += 60.0

        events.sort(key=lambda e: e.scheduled_at)
        return events
