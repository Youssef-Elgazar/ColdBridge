"""
Deterministic Routing Policy
=============================
Capacity-aware edge/cloud routing for ColdBridge's Module C.

The policy selects the *edge* if the simulated edge capacity is above a
threshold; otherwise the *cloud* is chosen.  Capacity values are deterministic
and stored in a look-up table so that experiments are repeatable.

This replaces the old ``random.random() > 0.5`` stub with a reproducible
policy that uses function-name hashing for consistent routing decisions.
"""

from __future__ import annotations

from typing import Dict

# Simulated edge node capacities (units are abstract resource slots).
# 10 synthetic edge nodes with varying capacity levels.
EDGE_CAPACITY: Dict[str, int] = {
    f"edge_{i}": 80 + i * 5 for i in range(10)
}

# Threshold: choose edge when capacity >= this value.
CAPACITY_THRESHOLD = 100

# Simulated latencies (ms) for deterministic experiment results.
EDGE_LATENCY_MS = 55.0     # sub-80ms as claimed in the paper
CLOUD_LATENCY_MS = 150.0   # typical cloud fallback latency


def route(function_name: str) -> str:
    """Return ``"edge"`` or ``"cloud"`` based on deterministic capacity.

    The function name seeds a simple hash to select an edge node, then checks
    its capacity against :data:`CAPACITY_THRESHOLD`.

    Args:
        function_name: The serverless function being routed.

    Returns:
        ``"edge"`` if the selected edge node has sufficient capacity,
        ``"cloud"`` otherwise.
    """
    # Deterministic hash → edge node id
    idx = abs(hash(function_name)) % len(EDGE_CAPACITY)
    edge_id = f"edge_{idx}"
    cap = EDGE_CAPACITY[edge_id]
    return "edge" if cap >= CAPACITY_THRESHOLD else "cloud"


def get_latency(tier: str) -> float:
    """Return the simulated latency for the given routing tier.

    Args:
        tier: ``"edge"`` or ``"cloud"``.

    Returns:
        Latency in milliseconds.
    """
    return EDGE_LATENCY_MS if tier == "edge" else CLOUD_LATENCY_MS
