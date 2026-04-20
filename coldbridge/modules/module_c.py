"""
Module C — Edge-Cloud Orchestrator (STUB)
==========================================
Placeholder for Basant / Youssef's implementation.

Interface expected by the runner:
  - route(function_name) → "edge" | "cloud"
  - is_available()       → bool

When implementing:
  1. Replace this stub with real Firecracker/edge pool logic
  2. Keep the same method signatures
"""

from __future__ import annotations
import logging

logger = logging.getLogger("coldbridge.module_c")


class ModuleC:
    """Edge-Cloud Orchestrator stub — not yet implemented."""

    def __init__(self, *args, **kwargs):
        logger.info("Module C: STUB active — all requests routed to cloud pool")

    def route(self, function_name: str) -> str:
        return "cloud"

    def is_available(self) -> bool:
        return False
