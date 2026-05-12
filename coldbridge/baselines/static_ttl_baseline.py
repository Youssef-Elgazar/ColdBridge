"""
Static TTL Baseline
====================
Simple fixed-TTL cold start predictor for ColdBridge.

Predicts a cold start whenever the inter-arrival time exceeds a fixed
threshold (TTL).  This is the standard baseline used in most cold start
research (e.g., Shahrad et al., 2020).

This baseline corresponds to the "Static TTL" row in the paper's Table IV.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger("coldbridge.baselines.static_ttl")


class StaticTTLBaseline:
    """Predicts cold start if idle time > TTL seconds.

    This is the simplest possible baseline: if the time since the last
    invocation of a function exceeds the TTL, predict cold start.
    """

    def __init__(self, ttl_seconds: float = 600.0, theta: float = 0.50):
        self.ttl = ttl_seconds
        self.theta = theta
        self._last_invoked: Dict[str, float] = {}

    def update(self, function_name: str, timestamp: float) -> None:
        """Record an invocation (call after each event)."""
        self._last_invoked[function_name] = timestamp

    def predict(self, function_name: str, current_time: float) -> float:
        """Return cold start probability based on idle time vs TTL.

        Returns 1.0 if idle > TTL, 0.0 otherwise (hard threshold).
        """
        last = self._last_invoked.get(function_name)
        if last is None:
            return 1.0  # first invocation is always cold
        idle = current_time - last
        if idle > self.ttl:
            return 1.0
        # Smooth transition near the TTL boundary
        if idle > self.ttl * 0.8:
            return (idle - self.ttl * 0.8) / (self.ttl * 0.2)
        return 0.0
