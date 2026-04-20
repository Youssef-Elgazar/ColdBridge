"""
Module B — Snapshot Registry (STUB)
=====================================
Placeholder for Basant / Youssef's implementation.

Interface that Module A and the orchestrator expect:
  - snapshot(function_name, container_id) → snapshot_id
  - restore(function_name)               → WorkerInstance | None
  - list_snapshots()                     → List[dict]

When teammates implement this module, they should:
  1. Replace this stub with the real Rust service client or Python implementation
  2. Keep the same method signatures so nothing else needs to change
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger("coldbridge.module_b")


class ModuleB:
    """Snapshot Registry stub — not yet implemented."""

    def __init__(self, *args, **kwargs):
        logger.info("Module B: STUB active — no snapshot acceleration")

    def snapshot(self, function_name: str, container_id: str) -> Optional[str]:
        logger.debug("Module B stub: snapshot(%s) → noop", function_name)
        return None

    def restore(self, function_name: str):
        logger.debug("Module B stub: restore(%s) → None", function_name)
        return None

    def list_snapshots(self):
        return []

    def is_available(self) -> bool:
        return False
