"""
Module C — Edge-Cloud Orchestrator
==========================================
Deterministic capacity-aware Edge-Cloud Orchestrator for ColdBridge.

Replaces the old random routing stub with a reproducible policy defined in
:mod:`coldbridge.modules._routing_policy`.

Interface expected by the runner:
  - route(function_name) → "edge" | "cloud"
  - get_latency(tier)    → float (ms)
  - is_available()       → bool
"""

from __future__ import annotations
import logging

from coldbridge.modules._routing_policy import (
    route as deterministic_route,
    get_latency as policy_latency,
)

logger = logging.getLogger("coldbridge.module_c")


class ModuleC:
    """Deterministic capacity-aware Edge-Cloud Orchestrator.

    Routing decisions are based on a consistent hash of the function name
    mapped to simulated edge-node capacities.  This ensures experiments
    produce identical results across runs with the same inputs.
    """

    def __init__(self, *args, **kwargs):
        logger.info("Module C: Deterministic Edge-Cloud Orchestrator active")

    def route(self, function_name: str) -> str:
        """Route a function to edge or cloud based on capacity policy.

        Two-tier routing logic:
          Tier 1 (Edge): Route to geographically proximate Firecracker microVMs
                         if the selected edge node has capacity.
          Tier 2 (Cloud): Fallback to central cloud pools for overflow.
        """
        tier = deterministic_route(function_name)
        latency = policy_latency(tier)
        logger.debug(
            "Routed Function %s to %s — simulated latency: %.0f ms",
            function_name, tier, latency,
        )
        return tier

    def get_latency(self, tier: str) -> float:
        """Return the simulated latency (ms) for a routing tier."""
        return policy_latency(tier)

    def is_available(self) -> bool:
        return True
