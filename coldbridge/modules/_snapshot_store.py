"""
Snapshot Store
==============
Simple in-memory snapshot store for ColdBridge.

The store keeps a mapping ``snap_id → (function_name, container_id, hash, timestamp)``
and can be serialised to a pickle file so that experiments across runs share the
same checkpoints.  This replaces the old mock-hash stub with a deterministic,
persistent implementation.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("coldbridge.snapshot_store")


class SnapshotStore:
    """Lightweight persistent snapshot registry.

    Each snapshot records the *function name*, *container ID*, a SHA-256
    fingerprint (first 16 hex chars), and a wall-clock timestamp.  The store
    can be flushed to disk (pickle) and reloaded on the next run.
    """

    def __init__(self, store_path: Optional[str | Path] = None):
        self.store_path = Path(store_path) if store_path else None
        self._store: Dict[str, Tuple[str, str, str, float]] = {}
        if self.store_path and self.store_path.exists():
            self.load()

    # ── core operations ───────────────────────────────────────────────────

    def _make_hash(self, fn: str, cid: str) -> str:
        """Deterministic SHA-256 fingerprint for (function, container)."""
        return hashlib.sha256(f"{fn}-{cid}".encode()).hexdigest()[:16]

    def snapshot(self, fn: str, container_id: str) -> str:
        """Capture a checkpoint.  Returns a deterministic snapshot ID."""
        snap_id = f"snap_{fn}_{container_id[:8]}"
        h = self._make_hash(fn, container_id)
        self._store[snap_id] = (fn, container_id, h, time.time())
        logger.debug("SnapshotStore: captured %s (hash=%s)", snap_id, h)
        return snap_id

    def restore(self, snap_id: str) -> Optional[Tuple[str, str]]:
        """Restore a previously captured checkpoint.

        Returns ``(function_name, container_id)`` if the snapshot exists,
        otherwise ``None``.
        """
        rec = self._store.get(snap_id)
        if not rec:
            return None
        fn, cid, _h, _ts = rec
        return fn, cid

    def find_best_match(self, function_name: str) -> Optional[str]:
        """Find the snapshot whose function name matches exactly.

        Returns the *most recent* snapshot ID for `function_name`,
        or ``None`` if no match exists.
        """
        candidates = [
            (sid, ts)
            for sid, (fn, _cid, _h, ts) in self._store.items()
            if fn == function_name
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def list_snapshots(self) -> List[dict]:
        return [
            {"id": sid, "function": fn, "container": cid, "hash": h, "ts": ts}
            for sid, (fn, cid, h, ts) in self._store.items()
        ]

    # ── persistence ───────────────────────────────────────────────────────

    def save(self) -> None:
        if self.store_path is None:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "wb") as f:
            pickle.dump(self._store, f)
        logger.debug("SnapshotStore saved → %s (%d entries)", self.store_path, len(self._store))

    def load(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        with open(self.store_path, "rb") as f:
            self._store = pickle.load(f)
        logger.debug("SnapshotStore loaded ← %s (%d entries)", self.store_path, len(self._store))

    def __len__(self) -> int:
        return len(self._store)
