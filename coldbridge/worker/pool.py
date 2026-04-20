"""
Worker Pool Adapter
===================
Manages a pool of Docker containers representing FaaS worker instances.
Tracks warm/cold state per function type, measures real cold start latency.

Windows note: containers are reachable at 127.0.0.1 via published ports.
"""

from __future__ import annotations

import time
import threading
import logging
import platform
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import docker

logger = logging.getLogger("coldbridge.worker")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FunctionSpec:
    name: str          # e.g. "python_fn"
    image: str         # e.g. "coldbridge/python-fn:latest"
    port: int          # internal container port (always 8080)
    runtime: str       # "python" | "node" | "java"
    ttl_seconds: float = 300.0   # keep-alive window before eviction


@dataclass
class WorkerInstance:
    function_name: str
    container_id: str
    host_port: int
    state: str          # "starting" | "warm" | "evicted"
    created_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None
    last_invoked_at: Optional[float] = None

    @property
    def cold_start_latency_ms(self) -> Optional[float]:
        if self.ready_at and self.created_at:
            return (self.ready_at - self.created_at) * 1000
        return None

    @property
    def idle_seconds(self) -> float:
        ref = self.last_invoked_at or self.ready_at or self.created_at
        return time.time() - ref


@dataclass
class InvocationResult:
    function_name: str
    was_cold: bool
    cold_start_latency_ms: float   # 0 if warm
    response_latency_ms: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None


# ── Abstract interface (swappable to AWS, Firecracker, etc.) ────────────────

class WorkerPoolAdapter(ABC):
    @abstractmethod
    def invoke(self, function_name: str) -> InvocationResult:
        """Invoke a function; cold-start if no warm instance exists."""

    @abstractmethod
    def prewarm(self, function_name: str) -> bool:
        """Proactively start a container for function_name."""

    @abstractmethod
    def warm_count(self, function_name: str) -> int:
        """Number of warm instances currently available."""

    @abstractmethod
    def evict_idle(self) -> int:
        """Remove instances idle beyond their TTL. Returns eviction count."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop all containers and clean up."""


# ── Docker implementation ────────────────────────────────────────────────────

class DockerWorkerPool(WorkerPoolAdapter):
    """
    Manages Docker containers as FaaS worker instances.

    Each invocation either:
      - Finds an existing warm container  → warm invocation
      - Starts a new container            → cold start (measures real Docker
                                            image-start + runtime-init latency)

    TTL eviction runs in a background thread.
    """

    def __init__(
        self,
        function_specs: List[FunctionSpec],
        max_warm_per_fn: int = 3,
        port_start: int = 19000,
        eviction_interval: float = 30.0,
    ):
        self._client = docker.from_env()
        self._specs: Dict[str, FunctionSpec] = {s.name: s for s in function_specs}
        self._max_warm = max_warm_per_fn
        self._port_counter = port_start
        self._port_lock = threading.Lock()

        # function_name → list of WorkerInstance
        self._pool: Dict[str, List[WorkerInstance]] = {s.name: [] for s in function_specs}
        self._pool_lock = threading.Lock()

        # background eviction
        self._eviction_interval = eviction_interval
        self._stop_evt = threading.Event()
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, daemon=True, name="eviction"
        )
        self._eviction_thread.start()

        logger.info("DockerWorkerPool ready with %d function types", len(function_specs))

    # ── public API ──────────────────────────────────────────────────────────

    def invoke(self, function_name: str) -> InvocationResult:
        t0 = time.perf_counter()
        spec = self._specs[function_name]

        with self._pool_lock:
            warm = self._get_warm(function_name)

        if warm:
            # Warm invocation
            cold_ms = 0.0
            instance = warm
        else:
            # Cold start — start container outside lock so we don't block others
            instance, cold_ms = self._cold_start(function_name)
            if instance is None:
                return InvocationResult(
                    function_name=function_name,
                    was_cold=True,
                    cold_start_latency_ms=cold_ms,
                    response_latency_ms=0,
                    total_latency_ms=(time.perf_counter() - t0) * 1000,
                    success=False,
                    error="container failed to start",
                )

        # Actual HTTP call to the container
        resp_ms, ok, err = self._call_container(instance)

        with self._pool_lock:
            instance.last_invoked_at = time.time()

        total_ms = (time.perf_counter() - t0) * 1000
        return InvocationResult(
            function_name=function_name,
            was_cold=(cold_ms > 0),
            cold_start_latency_ms=cold_ms,
            response_latency_ms=resp_ms,
            total_latency_ms=total_ms,
            success=ok,
            error=err,
        )

    def prewarm(self, function_name: str) -> bool:
        """Start a container now so the next invocation is warm."""
        spec = self._specs.get(function_name)
        if not spec:
            return False
        with self._pool_lock:
            current = len([i for i in self._pool[function_name] if i.state == "warm"])
            if current >= self._max_warm:
                return False  # already at capacity
        instance, _ = self._cold_start(function_name, record_as_prewarm=True)
        return instance is not None

    def warm_count(self, function_name: str) -> int:
        with self._pool_lock:
            return len([i for i in self._pool[function_name] if i.state == "warm"])

    def evict_idle(self) -> int:
        evicted = 0
        with self._pool_lock:
            for fn_name, instances in self._pool.items():
                spec = self._specs[fn_name]
                to_evict = [
                    i for i in instances
                    if i.state == "warm" and i.idle_seconds > spec.ttl_seconds
                ]
                for inst in to_evict:
                    self._stop_container(inst)
                    inst.state = "evicted"
                    evicted += 1
                self._pool[fn_name] = [i for i in instances if i.state != "evicted"]
        if evicted:
            logger.debug("Evicted %d idle containers", evicted)
        return evicted

    def shutdown(self) -> None:
        self._stop_evt.set()
        with self._pool_lock:
            for instances in self._pool.values():
                for inst in instances:
                    self._stop_container(inst)
            self._pool = {k: [] for k in self._pool}
        logger.info("DockerWorkerPool shut down")

    # ── internal helpers ────────────────────────────────────────────────────

    def _get_warm(self, function_name: str) -> Optional[WorkerInstance]:
        """Return the first warm instance (LRU by last_invoked_at), or None."""
        candidates = [
            i for i in self._pool[function_name] if i.state == "warm"
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda i: i.last_invoked_at or 0, reverse=True)[0]

    def _alloc_port(self) -> int:
        with self._port_lock:
            p = self._port_counter
            self._port_counter += 1
            return p

    def _cold_start(
        self, function_name: str, record_as_prewarm: bool = False
    ) -> tuple[Optional[WorkerInstance], float]:
        spec = self._specs[function_name]
        host_port = self._alloc_port()
        t_start = time.perf_counter()

        try:
            container = self._client.containers.run(
                spec.image,
                detach=True,
                ports={f"{spec.port}/tcp": ("127.0.0.1", host_port)},
                remove=True,
                labels={"coldbridge": "true", "fn": function_name},
            )
        except Exception as e:
            logger.error("Failed to start container for %s: %s", function_name, e)
            return None, (time.perf_counter() - t_start) * 1000

        inst = WorkerInstance(
            function_name=function_name,
            container_id=container.id[:12],
            host_port=host_port,
            state="starting",
            created_at=t_start,
        )

        # Poll until the HTTP server is up
        ready = self._wait_ready(inst, timeout=60.0)
        cold_ms = (time.perf_counter() - t_start) * 1000
        inst.ready_at = time.perf_counter()

        if not ready:
            self._stop_container(inst)
            logger.error("Container for %s never became ready", function_name)
            return None, cold_ms

        inst.state = "warm"
        with self._pool_lock:
            self._pool[function_name].append(inst)

        logger.debug(
            "Cold start %s → port %d  %.0f ms%s",
            function_name, host_port, cold_ms,
            " [prewarm]" if record_as_prewarm else "",
        )
        return inst, cold_ms

    def _wait_ready(self, inst: WorkerInstance, timeout: float = 60.0) -> bool:
        """Poll the container's health endpoint until it responds."""
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{inst.host_port}/"
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=1.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        return False

    def _call_container(
        self, inst: WorkerInstance
    ) -> tuple[float, bool, Optional[str]]:
        """Send one HTTP request; returns (latency_ms, success, error)."""
        url = f"http://127.0.0.1:{inst.host_port}/"
        t0 = time.perf_counter()
        try:
            r = requests.get(url, timeout=10.0)
            ms = (time.perf_counter() - t0) * 1000
            return ms, r.status_code == 200, None
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return ms, False, str(e)

    def _stop_container(self, inst: WorkerInstance) -> None:
        try:
            c = self._client.containers.get(inst.container_id)
            c.stop(timeout=2)
        except Exception:
            pass

    def _eviction_loop(self) -> None:
        while not self._stop_evt.wait(timeout=self._eviction_interval):
            self.evict_idle()


# ── Registry of function specs ──────────────────────────────────────────────

DEFAULT_FUNCTIONS = [
    FunctionSpec(
        name="python_fn",
        image="coldbridge/python-fn:latest",
        port=8080,
        runtime="python",
        ttl_seconds=120.0,
    ),
    FunctionSpec(
        name="node_fn",
        image="coldbridge/node-fn:latest",
        port=8080,
        runtime="node",
        ttl_seconds=120.0,
    ),
    FunctionSpec(
        name="java_fn",
        image="coldbridge/java-fn:latest",
        port=8080,
        runtime="java",
        ttl_seconds=120.0,
    ),
]
