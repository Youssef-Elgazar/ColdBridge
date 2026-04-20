"""
tests/test_module_a.py
======================
Unit tests for Module A (transformer model, encoding, heuristic fallback)
and the worker pool data structures.

Run with:  python -m pytest tests/ -v
"""

import math
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from coldbridge.modules.module_a import (
    ModuleA, FunctionState, encode_step,
    SEQ_LEN, D_INPUT, RUNTIME_IDS,
)
from coldbridge.worker.pool import InvocationResult, WorkerInstance


# ── encode_step ───────────────────────────────────────────────────────────────

class TestEncodeStep:
    def test_output_shape(self):
        v = encode_step(30.0, 512.0, "python", 14.5, 3.0)
        assert v.shape == (D_INPUT,), f"Expected ({D_INPUT},), got {v.shape}"

    def test_runtime_onehot_python(self):
        v = encode_step(0.0, 512.0, "python", 0.0, 0.0)
        assert v[2] == 1.0   # python is index 0 in RUNTIME_IDS → slot 2
        assert v[3] == 0.0
        assert v[4] == 0.0

    def test_runtime_onehot_java(self):
        v = encode_step(0.0, 512.0, "java", 0.0, 0.0)
        assert v[4] == 1.0   # java is index 2

    def test_iat_log_normalised(self):
        v0 = encode_step(0.0, 512.0, "python", 0.0, 0.0)
        v1 = encode_step(100.0, 512.0, "python", 0.0, 0.0)
        assert v1[0] > v0[0]
        assert math.isclose(v1[0], math.log1p(100.0), rel_tol=1e-5)

    def test_memory_normalised(self):
        v = encode_step(0.0, 3008.0, "python", 0.0, 0.0)
        assert math.isclose(v[1], 1.0, rel_tol=1e-5)

    def test_temporal_encoding_cyclical(self):
        # hour=0 and hour=24 should give identical features
        v0 = encode_step(1.0, 512.0, "python", 0.0, 0.0)
        v24 = encode_step(1.0, 512.0, "python", 24.0, 0.0)
        assert math.isclose(v0[10], v24[10], abs_tol=1e-5)
        assert math.isclose(v0[11], v24[11], abs_tol=1e-5)

    def test_no_nan_inf(self):
        v = encode_step(1e6, 3008.0, "node", 23.9, 6.9)
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(v))


# ── FunctionState ─────────────────────────────────────────────────────────────

class TestFunctionState:
    def test_initial_empty(self):
        s = FunctionState("python_fn", "python")
        assert len(s.history) == 0
        assert s.last_invoked_at is None

    def test_record_single_invocation(self):
        s = FunctionState("python_fn", "python")
        t = time.time()
        s.record_invocation(t, was_cold=True)
        assert len(s.history) == 1
        assert s.last_invoked_at == t

    def test_iat_computed_correctly(self):
        s = FunctionState("python_fn", "python")
        t1 = 1000.0
        t2 = 1030.0
        s.record_invocation(t1, was_cold=True)
        s.record_invocation(t2, was_cold=False)
        _, iat, _ = list(s.history)[1]
        assert math.isclose(iat, 30.0, rel_tol=1e-5)

    def test_build_sequence_shape(self):
        s = FunctionState("python_fn", "python")
        for i in range(SEQ_LEN + 5):
            s.record_invocation(1000.0 + i * 30.0, was_cold=(i % 5 == 0))
        seq = s.build_sequence()
        assert seq.shape == (SEQ_LEN, D_INPUT)

    def test_build_sequence_no_nan(self):
        s = FunctionState("java_fn", "java")
        for i in range(SEQ_LEN):
            s.record_invocation(1000.0 + i * 90.0, was_cold=(i % 10 == 0))
        seq = s.build_sequence()
        assert not np.any(np.isnan(seq))

    def test_cold_start_labels(self):
        s = FunctionState("node_fn", "node")
        s.record_invocation(1000.0, was_cold=True)
        s.record_invocation(1030.0, was_cold=False)
        s.record_invocation(1090.0, was_cold=True)
        labels = s.cold_start_labels
        assert list(labels) == [1.0, 0.0, 1.0]

    def test_ring_buffer_maxlen(self):
        s = FunctionState("python_fn", "python")
        for i in range(SEQ_LEN * 3):
            s.record_invocation(float(i), was_cold=False)
        # history maxlen is SEQ_LEN * 2
        assert len(s.history) == SEQ_LEN * 2


# ── ModuleA heuristic ─────────────────────────────────────────────────────────

class TestModuleAHeuristic:
    def _make_result(self, fn, was_cold):
        return InvocationResult(
            function_name=fn,
            was_cold=was_cold,
            cold_start_latency_ms=500.0 if was_cold else 0.0,
            response_latency_ms=10.0,
            total_latency_ms=510.0 if was_cold else 10.0,
            timestamp=time.time(),
        )

    def test_unknown_function_high_prob(self):
        m = ModuleA(worker_pool=None)
        p = m._heuristic_predict("unknown_fn")
        assert p > 0.5

    def test_recently_invoked_low_prob(self):
        m = ModuleA(worker_pool=None)
        result = self._make_result("python_fn", False)
        m.record_invocation(result)
        p = m._heuristic_predict("python_fn")
        assert p < 0.5, f"Expected low prob after recent invocation, got {p:.3f}"

    def test_prob_increases_with_idle_time(self):
        m = ModuleA(worker_pool=None)
        # Simulate invocation 200 seconds ago
        result = self._make_result("python_fn", False)
        result = InvocationResult(
            function_name="python_fn",
            was_cold=False,
            cold_start_latency_ms=0,
            response_latency_ms=5,
            total_latency_ms=5,
            timestamp=time.time() - 200.0,
        )
        m.record_invocation(result)
        p_old = m._heuristic_predict("python_fn")

        # Simulate recent invocation
        result2 = InvocationResult(
            function_name="python_fn",
            was_cold=False,
            cold_start_latency_ms=0,
            response_latency_ms=5,
            total_latency_ms=5,
            timestamp=time.time(),
        )
        m.record_invocation(result2)
        p_recent = m._heuristic_predict("python_fn")

        assert p_old > p_recent, "Older idle time should yield higher cold-start probability"

    def test_prediction_bounded(self):
        m = ModuleA(worker_pool=None)
        for fn in ["python_fn", "node_fn", "java_fn"]:
            p = m._heuristic_predict(fn)
            assert 0.0 <= p <= 1.0, f"Probability out of [0,1] for {fn}: {p}"


# ── ModuleA with transformer (requires torch) ─────────────────────────────────

class TestModuleATransformer:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed")

    def test_model_instantiates(self):
        m = ModuleA(worker_pool=None)
        assert m._model is not None

    def test_parameter_count_reasonable(self):
        m = ModuleA(worker_pool=None)
        n = m.parameter_count()
        # d_model=128, L=4, H=8 → roughly 600k–900k parameters
        assert 100_000 < n < 5_000_000, f"Unexpected parameter count: {n:,}"

    def test_predict_returns_probability(self):
        import torch
        m = ModuleA(worker_pool=None)
        # Build enough history for transformer path
        state_fn = "python_fn"
        for i in range(SEQ_LEN + 5):
            r = InvocationResult(
                function_name=state_fn,
                was_cold=(i % 8 == 0),
                cold_start_latency_ms=400.0 if i % 8 == 0 else 0.0,
                response_latency_ms=10.0,
                total_latency_ms=410.0,
                timestamp=1000.0 + i * 30.0,
            )
            m.record_invocation(r)
        p = m._predict(state_fn)
        assert 0.0 <= p <= 1.0, f"Prediction out of range: {p}"

    def test_save_load_roundtrip(self, tmp_path):
        import torch
        m = ModuleA(worker_pool=None)
        path = tmp_path / "model.pt"
        m.save(path)
        assert path.exists()

        m2 = ModuleA(worker_pool=None, model_path=path)
        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            m._model.named_parameters(), m2._model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_training_on_synthetic_data(self):
        import torch
        m = ModuleA(worker_pool=None)

        # Build a small synthetic invocation history
        history = []
        t = 1_000_000.0
        for fn in ["python_fn", "node_fn"]:
            for i in range(SEQ_LEN * 3):
                t += 30.0
                history.append({
                    "function_name": fn,
                    "timestamp": t,
                    "was_cold": (i % 7 == 0),
                    "cold_start_latency_ms": 400.0 if i % 7 == 0 else 0.0,
                    "response_latency_ms": 10.0,
                    "total_latency_ms": 410.0,
                    "success": True,
                })

        result = m.train(history, epochs=3, batch_size=16)
        assert "val_f1" in result
        assert "epochs_run" in result
        assert result["epochs_run"] >= 1


# ── InvocationResult helpers ──────────────────────────────────────────────────

class TestInvocationResult:
    def test_cold_result(self):
        r = InvocationResult(
            function_name="java_fn",
            was_cold=True,
            cold_start_latency_ms=2100.0,
            response_latency_ms=15.0,
            total_latency_ms=2115.0,
        )
        assert r.was_cold
        assert r.cold_start_latency_ms == 2100.0

    def test_warm_result(self):
        r = InvocationResult(
            function_name="python_fn",
            was_cold=False,
            cold_start_latency_ms=0.0,
            response_latency_ms=8.0,
            total_latency_ms=8.0,
        )
        assert not r.was_cold
        assert r.cold_start_latency_ms == 0.0


# ── MetricsCollector ──────────────────────────────────────────────────────────

class TestMetricsCollector:
    def test_empty_collector(self):
        from coldbridge.metrics.collector import MetricsCollector
        c = MetricsCollector()
        assert c.compute("baseline") == {}

    def test_csr_computation(self):
        from coldbridge.metrics.collector import MetricsCollector
        c = MetricsCollector()
        for i in range(10):
            c.record_invocation(InvocationResult(
                function_name="python_fn",
                was_cold=(i < 4),       # 4 cold, 6 warm
                cold_start_latency_ms=400.0 if i < 4 else 0.0,
                response_latency_ms=10.0,
                total_latency_ms=410.0 if i < 4 else 10.0,
                timestamp=float(i),
            ))
        m = c.compute("baseline")
        assert "python_fn" in m
        assert math.isclose(m["python_fn"].cold_start_rate, 0.4, rel_tol=1e-5)
        assert m["python_fn"].n_cold == 4
        assert m["python_fn"].n_warm == 6

    def test_percentile_latency(self):
        from coldbridge.metrics.collector import MetricsCollector
        c = MetricsCollector()
        lats = list(range(100, 200))  # 100 values: 100ms to 199ms
        for lat in lats:
            c.record_invocation(InvocationResult(
                function_name="java_fn",
                was_cold=True,
                cold_start_latency_ms=float(lat),
                response_latency_ms=5.0,
                total_latency_ms=float(lat) + 5.0,
                timestamp=float(lat),
            ))
        m = c.compute("baseline")["java_fn"]
        assert 148 <= m.p50_cold_start_ms <= 151
        assert m.p99_cold_start_ms >= 197

    def test_csv_export(self, tmp_path):
        from coldbridge.metrics.collector import MetricsCollector
        c = MetricsCollector()
        c.record_invocation(InvocationResult(
            function_name="node_fn", was_cold=True,
            cold_start_latency_ms=600.0, response_latency_ms=12.0,
            total_latency_ms=612.0, timestamp=1.0,
        ))
        out = tmp_path / "invocations.csv"
        c.to_csv(out)
        assert out.exists()
        content = out.read_text()
        assert "node_fn" in content
        assert "was_cold" in content


# ── SyntheticTraceGenerator ───────────────────────────────────────────────────

class TestSyntheticTrace:
    def test_generates_events(self):
        from coldbridge.data.trace_loader import SyntheticTraceGenerator
        gen = SyntheticTraceGenerator(seed=0)
        events = gen.generate(duration_seconds=300.0)
        assert len(events) > 0

    def test_events_sorted(self):
        from coldbridge.data.trace_loader import SyntheticTraceGenerator
        gen = SyntheticTraceGenerator(seed=1)
        events = gen.generate(duration_seconds=300.0)
        times = [e.scheduled_at for e in events]
        assert times == sorted(times)

    def test_all_functions_present(self):
        from coldbridge.data.trace_loader import SyntheticTraceGenerator
        gen = SyntheticTraceGenerator(seed=2)
        events = gen.generate(duration_seconds=600.0)
        fns = {e.function_name for e in events}
        assert "python_fn" in fns
        assert "node_fn" in fns
        assert "java_fn" in fns

    def test_within_duration(self):
        from coldbridge.data.trace_loader import SyntheticTraceGenerator
        dur = 120.0
        gen = SyntheticTraceGenerator(seed=3)
        events = gen.generate(duration_seconds=dur)
        assert all(e.scheduled_at <= dur for e in events)

    def test_reproducible_with_same_seed(self):
        from coldbridge.data.trace_loader import SyntheticTraceGenerator
        e1 = SyntheticTraceGenerator(seed=99).generate(300.0)
        e2 = SyntheticTraceGenerator(seed=99).generate(300.0)
        assert len(e1) == len(e2)
        assert all(
            a.function_name == b.function_name and
            math.isclose(a.scheduled_at, b.scheduled_at, rel_tol=1e-9)
            for a, b in zip(e1, e2)
        )
