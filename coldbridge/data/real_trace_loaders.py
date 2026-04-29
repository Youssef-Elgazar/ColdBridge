"""
Real-World Trace Loaders
========================
Loaders for production-grade datasets that are directly downloadable
and ready to use with ColdBridge's modules.

Included loaders:
  1. Industry40ColdStartLoader — IEEE JSAS 2024 cold start dataset
     from GCP Cloud Functions in an Industrial IoT scenario.
     Source: github.com/MuhammedGolec/Cold-Start-Dataset-V2

  2. ZenodoHuaweiLoader — Processed subset of Huawei Public Cloud
     traces from LACE-RL (CCGrid'26 artifact).
     Source: zenodo.org/records/18680777

  3. AzureFunctions2019Loader — Azure Functions invocation traces
     from "Serverless in the Wild" (ATC'20).
     Source: github.com/Azure/AzurePublicDataset

All produce InvocationEvent lists compatible with Module D's harness,
and ColdStartRecord lists for Module A training.
"""

from __future__ import annotations

import logging
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coldbridge.data.trace_loader import InvocationEvent, ColdStartRecord

logger = logging.getLogger("coldbridge.real_trace_loaders")


# ── 1. Industry 4.0 Cold Start Dataset ───────────────────────────────────────

class Industry40ColdStartLoader:
    """
    Loads the Industry 4.0 Cold Start Dataset (IEEE JSAS 2024).

    Paper: "ML-based cold start latency prediction framework in serverless
           edge computing environments for industry 4.0"
    Source: github.com/MuhammedGolec/Cold-Start-Dataset-V2

    Schema (1440 rows, 5-min intervals over 6 days):
      DateTime     — timestamp (2024-01-01 00:00 to 2024-01-06 23:55)
      Hour         — hour of day (0-23)
      Day          — day number (1-6)
      Request      — number of concurrent requests in this 5-min window
      Latency      — measured response latency in ms
      CPU_Usage    — CPU utilisation % (0-100)
      Memory_Usage — memory utilisation % (0-100)

    Cold start identification:
      - Latency > 500 ms → cold start (650-850 ms range)
      - Latency 1-50 ms  → warm invocation
      - Request == 0      → idle (no invocation)

    Environment: Python 3.10, 512 MB RAM, GCP Cloud Functions
    """

    COLD_START_THRESHOLD_MS = 500.0

    def __init__(self, xlsx_path: str = "data/industry40_coldstart.xlsx"):
        self.path = Path(xlsx_path)

    def load(
        self,
        max_events: int = 5000,
        include_idle: bool = False,
    ) -> List[InvocationEvent]:
        """
        Load Industry 4.0 traces as InvocationEvent list.

        Each 5-minute window with Request > 0 generates invocation events.
        Cold starts are identified by Latency > 500 ms.

        Args:
            max_events: Max events to return.
            include_idle: If True, include idle windows as zero-request events.

        Returns:
            Sorted list of InvocationEvent with ground-truth cold start labels.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for Industry40ColdStartLoader")

        if not self.path.exists():
            raise FileNotFoundError(
                f"Industry 4.0 dataset not found at {self.path}. "
                "Download from: https://github.com/MuhammedGolec/Cold-Start-Dataset-V2"
            )

        df = pd.read_excel(str(self.path))
        logger.info("Industry 4.0 dataset loaded: %d rows", len(df))

        # Filter to rows with requests (or all if include_idle)
        if not include_idle:
            df = df[df["Request"] > 0]

        events: List[InvocationEvent] = []
        base_ts = df.iloc[0]["DateTime"].timestamp() if len(df) > 0 else 0.0

        for _, row in df.iterrows():
            ts = row["DateTime"].timestamp() - base_ts
            latency = float(row["Latency"])
            n_requests = int(row["Request"])
            is_cold = latency > self.COLD_START_THRESHOLD_MS

            # Generate one event per 5-min window
            # (each window represents a batch of concurrent requests)
            if n_requests > 0:
                events.append(InvocationEvent(
                    function_name="industry40_svc_predict",
                    scheduled_at=ts,
                    expected_pattern="production_iot",
                    was_cold=is_cold,
                    cold_start_latency_ms=latency if is_cold else 0.0,
                    runtime="python",
                ))

            if len(events) >= max_events:
                break

        events.sort(key=lambda e: e.scheduled_at)
        cold_count = sum(1 for e in events if e.was_cold)
        logger.info(
            "Industry 4.0: %d events, %d cold starts (%.1f%%), span=%.0fs",
            len(events), cold_count, cold_count / max(len(events), 1) * 100,
            events[-1].scheduled_at - events[0].scheduled_at if events else 0,
        )
        return events

    def load_training_records(self) -> List[ColdStartRecord]:
        """
        Load as ColdStartRecord for Module A training.
        Returns both cold and warm records with inter-arrival times.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for Industry40ColdStartLoader")

        df = pd.read_excel(str(self.path))
        active = df[df["Request"] > 0].copy()

        records: List[ColdStartRecord] = []
        last_ts: Optional[float] = None

        for _, row in active.iterrows():
            ts = row["DateTime"].timestamp()
            latency = float(row["Latency"])
            is_cold = latency > self.COLD_START_THRESHOLD_MS

            iat = 0.0
            if last_ts is not None:
                iat = ts - last_ts
            last_ts = ts

            records.append(ColdStartRecord(
                function_name="industry40_svc_predict",
                timestamp=ts,
                was_cold=is_cold,
                cold_start_latency_ms=latency,
                inter_arrival_time_s=iat,
                memory_mb=512.0,  # from paper: 512 MB GCP Cloud Functions
                runtime="python",
            ))

        logger.info("Industry 4.0 training records: %d (cold=%d, warm=%d)",
                     len(records),
                     sum(1 for r in records if r.was_cold),
                     sum(1 for r in records if not r.was_cold))
        return records


# ── 2. Zenodo / Huawei Processed Traces (LACE-RL) ────────────────────────────

class ZenodoHuaweiLoader:
    """
    Loads the processed Huawei Public Cloud trace subset from the
    LACE-RL artifact (CCGrid'26).

    Source: zenodo.org/records/18680777
    Paper: "Green or Fast? Learning to Balance Cold Starts and Idle
           Carbon in Serverless Computing"

    The artifact contains:
      - invocations.pkl: 54,375 FunctionInvocation objects with:
          timestamp, pod_id, region_id, exec_time_s, cpu_cores,
          mem_MB, cold_start_latency_s, user_region
      - cs_latency_dict.pkl: cold start latency lookup by pod_id

    This is derived from the Huawei EuroSys 2025 paper data, making it
    ideal for testing Module A (cold start prediction) and Module B
    (snapshot registry via pod_id matching).
    """

    def __init__(
        self,
        artifact_dir: str = "data/zenodo_huawei/lace-rl-artifact",
    ):
        self.artifact_dir = Path(artifact_dir)
        self._invocations = None
        self._cs_dict = None

    def _ensure_loaded(self):
        """Load pickle data on first access."""
        if self._invocations is not None:
            return

        inv_path = self.artifact_dir / "data" / "demo" / "invocations.pkl"
        cs_path = self.artifact_dir / "data" / "demo" / "cs_latency_dict.pkl"

        if not inv_path.exists():
            raise FileNotFoundError(
                f"Zenodo artifact not found at {inv_path}. "
                "Download: zenodo.org/records/18680777"
            )

        # Use the LACE-RL compat unpickler to handle module renaming
        sys.path.insert(0, str(self.artifact_dir))
        try:
            from lace_rl.utils.io import _load_pickle, _to_invocation, _apply_cs_latency
        except ImportError:
            # Fallback: manual unpickle with class remapping
            import pickle as _pkl

            class _CompatUnpickler(_pkl.Unpickler):
                def find_class(self, module, name):
                    if module == "SimulationwithLatency":
                        module = "lace_rl.sim.trace_simulator"
                    return super().find_class(module, name)

            def _load_pickle(path):
                with open(path, "rb") as f:
                    try:
                        return _pkl.load(f)
                    except ModuleNotFoundError:
                        f.seek(0)
                        return _CompatUnpickler(f).load()

            _to_invocation = lambda x: x
            _apply_cs_latency = lambda inv, cs: inv

        raw = _load_pickle(str(inv_path))
        self._invocations = [_to_invocation(x) for x in raw]

        if cs_path.exists():
            self._cs_dict = _load_pickle(str(cs_path))
            if self._cs_dict:
                self._invocations = [
                    _apply_cs_latency(inv, self._cs_dict)
                    for inv in self._invocations
                ]

        logger.info("Zenodo/Huawei loaded: %d invocations", len(self._invocations))

    def load(
        self,
        max_events: int = 5000,
        max_functions: int = 20,
    ) -> List[InvocationEvent]:
        """
        Load Huawei production traces as InvocationEvent list.

        Args:
            max_events: Maximum events to return.
            max_functions: Limit to top N pod_ids by frequency.

        Returns:
            Sorted list of InvocationEvent with cold start latency metadata.
        """
        self._ensure_loaded()

        # Group by pod_id, take top N
        pod_counts: Dict[str, int] = {}
        for inv in self._invocations:
            pod_counts[inv.pod_id] = pod_counts.get(inv.pod_id, 0) + 1

        top_pods = sorted(pod_counts, key=pod_counts.get, reverse=True)[:max_functions]
        top_pod_set = set(top_pods)

        filtered = [inv for inv in self._invocations if inv.pod_id in top_pod_set]
        filtered.sort(key=lambda inv: inv.timestamp)
        filtered = filtered[:max_events]

        base_ts = filtered[0].timestamp if filtered else 0.0

        events: List[InvocationEvent] = []
        for inv in filtered:
            has_cold = inv.cold_start_latency_s > 0
            events.append(InvocationEvent(
                function_name=f"huawei_{inv.pod_id[:20]}",
                scheduled_at=inv.timestamp - base_ts,
                expected_pattern="production",
                was_cold=has_cold,
                cold_start_latency_ms=inv.cold_start_latency_s * 1000.0 if has_cold else 0.0,
                runtime="python",
            ))

        cold_count = sum(1 for e in events if e.was_cold)
        logger.info(
            "Zenodo/Huawei: %d events from %d pods, %d cold starts (%.1f%%)",
            len(events), len(top_pods), cold_count,
            cold_count / max(len(events), 1) * 100,
        )
        return events

    def load_training_records(
        self,
        max_records: int = 50000,
    ) -> List[ColdStartRecord]:
        """
        Load as ColdStartRecord for Module A training.
        Includes pod-level inter-arrival times for all invocations.
        """
        self._ensure_loaded()

        invs = sorted(self._invocations, key=lambda inv: inv.timestamp)[:max_records]
        records: List[ColdStartRecord] = []
        last_ts: Dict[str, float] = {}

        for inv in invs:
            fn = f"huawei_{inv.pod_id[:20]}"
            iat = 0.0
            if fn in last_ts:
                iat = inv.timestamp - last_ts[fn]
            last_ts[fn] = inv.timestamp

            has_cold = inv.cold_start_latency_s > 0
            records.append(ColdStartRecord(
                function_name=fn,
                timestamp=inv.timestamp,
                was_cold=has_cold,
                cold_start_latency_ms=inv.cold_start_latency_s * 1000.0 if has_cold else 0.0,
                inter_arrival_time_s=iat,
                memory_mb=float(inv.mem_MB),
                runtime="python",
            ))

        logger.info("Zenodo/Huawei training: %d records (cold=%d, warm=%d)",
                     len(records),
                     sum(1 for r in records if r.was_cold),
                     sum(1 for r in records if not r.was_cold))
        return records

    def get_pod_metadata(self) -> Dict[str, dict]:
        """Return per-pod summary (useful for Module B snapshot registry)."""
        self._ensure_loaded()
        pods: Dict[str, dict] = {}
        for inv in self._invocations:
            if inv.pod_id not in pods:
                pods[inv.pod_id] = {
                    "pod_id": inv.pod_id,
                    "region": inv.region_id,
                    "cpu_cores": inv.cpu_cores,
                    "mem_MB": inv.mem_MB,
                    "invocation_count": 0,
                    "cold_starts": 0,
                }
            pods[inv.pod_id]["invocation_count"] += 1
            if inv.cold_start_latency_s > 0:
                pods[inv.pod_id]["cold_starts"] += 1
        return pods


# ── 3. Azure Functions 2019 Loader ───────────────────────────────────────────

class AzureFunctions2019Loader:
    """
    Loads Azure Functions invocation traces from the 2019 dataset.

    Paper: "Serverless in the Wild" (USENIX ATC 2020)
    Source: github.com/Azure/AzurePublicDataset
    Download: azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/
              azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz

    Schema (invocations_per_function_md.anon.d[01-14].csv):
      HashOwner    — anonymised owner ID
      HashApp      — anonymised application ID
      HashFunction — anonymised function ID
      Trigger      — trigger type (http, timer, event, queue, etc.)
      1..1440      — per-minute invocation counts for 24h

    This is the standard dataset used in cold-start research. Module A's
    feature engineering (log-normalised IAT + cyclical encoding) was
    designed for this schema.
    """

    def __init__(self, data_dir: str = "data/azure"):
        self.data_dir = Path(data_dir)

    def load(
        self,
        max_functions: int = 10,
        max_events: int = 5000,
        days: Optional[List[int]] = None,
    ) -> List[InvocationEvent]:
        """
        Load Azure traces as InvocationEvent list.

        Args:
            max_functions: Top N functions by total invocations.
            max_events: Max events to return.
            days: Which days to load (1-14). Default: [1].

        Returns:
            Sorted list of InvocationEvent (no cold start labels — Azure
            traces don't include latency, only invocation counts).
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for AzureFunctions2019Loader")

        if days is None:
            days = [1]

        # Find CSV files
        dfs = []
        for day in days:
            pattern = f"invocations_per_function_md.anon.d{day:02d}.csv"
            candidates = list(self.data_dir.rglob(pattern))
            if not candidates:
                logger.warning("Azure trace file not found: %s", pattern)
                continue
            df = pd.read_csv(candidates[0])
            df["_day"] = day
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No Azure trace CSVs found in {self.data_dir}. "
                "Download from: https://azurepublicdatasettraces.blob.core.windows.net/"
                "azurepublicdatasetv2/azurefunctions_dataset2019/"
                "azurefunctions-dataset2019.tar.xz"
            )

        df = pd.concat(dfs, ignore_index=True)
        logger.info("Azure 2019 loaded: %d function-day rows", len(df))

        # Compute total invocations per function across all minutes
        minute_cols = [str(i) for i in range(1, 1441)]
        available_cols = [c for c in minute_cols if c in df.columns]
        df["_total"] = df[available_cols].sum(axis=1)

        # Top functions
        top = df.groupby("HashFunction")["_total"].sum().nlargest(max_functions)
        top_fns = top.index.tolist()
        df = df[df["HashFunction"].isin(top_fns)]

        # Map hash to readable names
        trigger_map = {}
        for _, row in df.drop_duplicates("HashFunction").iterrows():
            trigger_map[row["HashFunction"]] = row.get("Trigger", "unknown")

        # Generate events from per-minute counts
        runtime_cycle = ["python", "node", "java"]
        fn_name_map = {}
        events: List[InvocationEvent] = []

        for fn_idx, fn_hash in enumerate(top_fns):
            fn_name = f"azure_fn_{fn_idx}"
            fn_name_map[fn_hash] = fn_name
            fn_df = df[df["HashFunction"] == fn_hash]

            for _, row in fn_df.iterrows():
                day = int(row["_day"])
                for minute_str in available_cols:
                    minute = int(minute_str)
                    count = int(row.get(minute_str, 0))
                    if count <= 0:
                        continue

                    base_s = (day - 1) * 86400.0 + (minute - 1) * 60.0
                    for _ in range(min(count, 10)):  # cap per-minute expansion
                        offset = random.uniform(0, 60.0)
                        events.append(InvocationEvent(
                            function_name=fn_name,
                            scheduled_at=base_s + offset,
                            expected_pattern=trigger_map.get(fn_hash, "unknown"),
                            runtime=runtime_cycle[fn_idx % 3],
                        ))

                    if len(events) >= max_events:
                        break
                if len(events) >= max_events:
                    break
            if len(events) >= max_events:
                break

        events.sort(key=lambda e: e.scheduled_at)
        logger.info("Azure 2019: %d events from %d functions",
                     len(events), len(top_fns))
        return events

    def load_training_records(
        self,
        max_functions: int = 10,
        days: Optional[List[int]] = None,
    ) -> List[ColdStartRecord]:
        """
        Load as ColdStartRecord for Module A training.

        NOTE: Azure 2019 traces don't include cold start labels or latency.
        We infer cold starts using the standard heuristic: if a function
        hasn't been invoked for > 10 minutes, the next invocation is cold.
        """
        events = self.load(max_functions=max_functions,
                          max_events=50000, days=days)

        IDLE_THRESHOLD_S = 600.0  # 10 min idle → cold start
        records: List[ColdStartRecord] = []
        last_ts: Dict[str, float] = {}

        for evt in events:
            iat = 0.0
            is_cold = True  # first invocation is always cold

            if evt.function_name in last_ts:
                iat = evt.scheduled_at - last_ts[evt.function_name]
                is_cold = iat > IDLE_THRESHOLD_S
            last_ts[evt.function_name] = evt.scheduled_at

            records.append(ColdStartRecord(
                function_name=evt.function_name,
                timestamp=evt.scheduled_at,
                was_cold=is_cold,
                cold_start_latency_ms=750.0 if is_cold else 0.0,  # estimated
                inter_arrival_time_s=iat,
                runtime=evt.runtime or "python",
            ))

        cold_count = sum(1 for r in records if r.was_cold)
        logger.info("Azure 2019 training: %d records (cold=%d inferred, warm=%d)",
                     len(records), cold_count, len(records) - cold_count)
        return records
