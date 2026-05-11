"""
Module A — Transformer Pre-Warmer (FIXED)
==========================================
Predicts cold start probability for each function over a lookahead window
and triggers proactive pre-warming via the worker pool.

Architecture (matching paper Section III):
  - Transformer Encoder: L=4 layers, d_model=128, H=8 heads
  - Input: sequence of T=288 timesteps × d_in=7 features per function
  - Function embeddings u_i ∈ R^64 (key innovation for cross-function transfer)
  - Output: scalar cold start probability p ∈ (0,1)
  - Decision: prewarm if p > θ (threshold, default 0.50)

Training:
  - Focal loss (α=0.25, γ=2.0) for class-imbalanced cold start detection
  - AdamW optimiser, cosine LR schedule with warm-up
  - Cross-function: single model shared across all function types
    (transfer learning via learned function embedding vectors)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("coldbridge.module_a")

# ── optional torch import (graceful degradation to heuristic) ────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found — Module A will use heuristic fallback.")


# ── Constants (match paper Table II) ──────────────────────────────────────────
SEQ_LEN   = 288       # T: timesteps (24h at Δt=5min)
D_MODEL   = 128       # transformer model dimension
N_HEADS   = 8         # attention heads
N_LAYERS  = 4         # encoder layers
D_INPUT   = 7         # raw feature vector dimension (see _encode_step)
D_EMBED   = 64        # function embedding dimension
DROPOUT   = 0.1
FOCAL_ALPHA = 0.25    # paper: from Lin et al. 2017
FOCAL_GAMMA = 2.0
DEFAULT_THETA = 0.50  # pre-warm decision threshold (paper default)
LOOKAHEAD_S   = 30.0  # prediction horizon in seconds
TICK_S        = 5.0   # how often the prediction loop runs

RUNTIME_IDS = {"python": 0, "node": 1, "java": 2, "go": 3, "dotnet": 4}


# ── Feature encoding (d_in=7 matching paper) ─────────────────────────────────

def encode_step(
    iat_s: float,
    memory_mb: float,
    runtime: str,
    hour: float,
    day_of_week: float,
) -> np.ndarray:
    """
    Build a d=7 feature vector for one timestep (matching paper Section III-B).

    Features:
      [0]    log-normalised inter-arrival time
      [1]    memory (normalised by 3008 MB, Lambda max)
      [2]    runtime one-hot index (normalised)
      [3]    sin(2π·hour/24)
      [4]    cos(2π·hour/24)
      [5]    sin(2π·dow/7)
      [6]    cos(2π·dow/7)
    """
    v = np.zeros(D_INPUT, dtype=np.float32)
    v[0] = math.log1p(max(iat_s, 0.0))
    v[1] = memory_mb / 3008.0
    v[2] = RUNTIME_IDS.get(runtime, 0) / max(len(RUNTIME_IDS) - 1, 1)
    v[3] = math.sin(2 * math.pi * hour / 24.0)
    v[4] = math.cos(2 * math.pi * hour / 24.0)
    v[5] = math.sin(2 * math.pi * day_of_week / 7.0)
    v[6] = math.cos(2 * math.pi * day_of_week / 7.0)
    return v


# ── Transformer model (with function embeddings) ─────────────────────────────

if TORCH_AVAILABLE:
    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class ColdStartTransformer(nn.Module):
        """
        Transformer encoder + function embeddings + MLP head for binary
        cold start prediction.

        Input:  (batch, T, D_INPUT) + function_ids (batch,)
        Output: (batch,)  — scalar probability in (0,1)

        Key innovation: function embeddings u_i allow cross-function
        knowledge sharing and zero-shot transfer for new functions.
        """

        def __init__(
            self,
            d_input: int = D_INPUT,
            d_model: int = D_MODEL,
            d_embed: int = D_EMBED,
            n_heads: int = N_HEADS,
            n_layers: int = N_LAYERS,
            dropout: float = DROPOUT,
            max_functions: int = 1024,
        ):
            super().__init__()
            self.d_embed = d_embed
            self.max_functions = max_functions

            # Function embedding table (paper Section III-C)
            self.func_embedding = nn.Embedding(max_functions, d_embed)
            nn.init.normal_(self.func_embedding.weight, std=0.02)

            # Project raw features + function embedding to d_model
            self.input_proj = nn.Linear(d_input + d_embed, d_model)
            self.pos_enc = _PositionalEncoding(d_model, max_len=SEQ_LEN + 16, dropout=dropout)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,         # pre-LN for training stability
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(
            self,
            x: torch.Tensor,
            func_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                x: (B, T, D_INPUT) — raw feature sequences
                func_ids: (B,) — integer function IDs for embedding lookup
                          If None, uses mean embedding (zero-shot cold embed)
            """
            B, T, _ = x.shape

            if func_ids is not None:
                u = self.func_embedding(func_ids)        # (B, d_embed)
                u = u.unsqueeze(1).expand(B, T, -1)      # (B, T, d_embed)
            else:
                # Zero-shot: use mean of all learned embeddings
                u = self.func_embedding.weight.mean(0)   # (d_embed,)
                u = u.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

            z = torch.cat([x, u], dim=-1)                # (B, T, d_input + d_embed)
            z = self.pos_enc(self.input_proj(z))          # (B, T, d_model)
            z = self.encoder(z)                           # (B, T, d_model)
            logit = self.head(z[:, -1, :])                # (B, 1) — last timestep
            return torch.sigmoid(logit).squeeze(1)        # (B,)

    def focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA,
    ) -> torch.Tensor:
        # Clamp predictions for numerical stability
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.where(target == 1, pred, 1 - pred)
        w   = torch.where(target == 1,
                          torch.full_like(pt, alpha),
                          torch.full_like(pt, 1 - alpha))
        loss = w * (1 - pt) ** gamma * bce
        return loss.mean()


# ── Per-function state ────────────────────────────────────────────────────────

@dataclass
class FunctionState:
    name: str
    runtime: str
    memory_mb: float = 512.0
    last_invoked_at: Optional[float] = None
    history: Deque = None     # ring buffer of (timestamp, iat, cold_start)

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=SEQ_LEN * 2)

    def record_invocation(self, timestamp: float, was_cold: bool) -> None:
        iat = 0.0
        if self.last_invoked_at is not None:
            iat = timestamp - self.last_invoked_at
        self.history.append((timestamp, iat, int(was_cold)))
        self.last_invoked_at = timestamp

    def build_sequence(self) -> np.ndarray:
        """Return (T, D_INPUT) feature matrix from recent history."""
        seq = np.zeros((SEQ_LEN, D_INPUT), dtype=np.float32)
        records = list(self.history)[-SEQ_LEN:]
        for i, (ts, iat, _) in enumerate(records):
            t = time.localtime(ts)
            seq[i] = encode_step(
                iat_s=iat,
                memory_mb=self.memory_mb,
                runtime=self.runtime,
                hour=t.tm_hour + t.tm_min / 60.0,
                day_of_week=float(t.tm_wday),
            )
        return seq

    @property
    def cold_start_labels(self) -> np.ndarray:
        """Binary labels for history (used in supervised training)."""
        records = list(self.history)
        return np.array([r[2] for r in records], dtype=np.float32)


# ── Module A main class ──────────────────────────────────────────────────────

class ModuleA:
    """
    Transformer-based cold start prediction and pre-warming module.

    Lifecycle:
      1. create with a worker pool reference
      2. call start()  — launches background prediction tick
      3. call record_invocation() after each worker pool call
      4. call stop()   — halts background thread
      5. call train()  — train/fine-tune on labelled trace data
      6. call save() / load() — persist model weights
    """

    def __init__(
        self,
        worker_pool=None,
        theta: float = DEFAULT_THETA,
        lookahead_s: float = LOOKAHEAD_S,
        tick_s: float = TICK_S,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._pool = worker_pool
        self.theta = theta
        self.lookahead_s = lookahead_s
        self.tick_s = tick_s

        # Function state tracking
        self._states: Dict[str, FunctionState] = {}
        self._state_lock = threading.Lock()

        # Function name → integer ID mapping for embeddings
        self._func_id_map: Dict[str, int] = {}
        self._next_func_id = 0

        # Pre-warm accounting
        self._prewarm_log: List[dict] = []

        # Model
        if TORCH_AVAILABLE:
            self._device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._model = ColdStartTransformer().to(self._device)
            if model_path and Path(model_path).exists():
                self.load(model_path)
            logger.info("Module A using device: %s", self._device)
        else:
            self._model = None
            self._device = None

        # Background prediction thread
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _get_func_id(self, function_name: str) -> int:
        """Get or assign an integer ID for a function name."""
        if function_name not in self._func_id_map:
            self._func_id_map[function_name] = self._next_func_id
            self._next_func_id += 1
        return self._func_id_map[function_name]

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background prediction loop."""
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._prediction_loop, daemon=True, name="module_a"
        )
        self._thread.start()
        logger.info("Module A started (θ=%.2f, tick=%.0fs)", self.theta, self.tick_s)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Module A stopped")

    # ── invocation recording ─────────────────────────────────────────────

    def record_invocation(self, result) -> None:
        """
        Called after every worker pool invocation.
        Updates per-function history used by the model.
        """
        with self._state_lock:
            fn = result.function_name
            if fn not in self._states:
                runtime = fn.split("_")[0]  # "python_fn" → "python"
                self._states[fn] = FunctionState(name=fn, runtime=runtime)
            self._states[fn].record_invocation(result.timestamp, result.was_cold)
            self._get_func_id(fn)  # ensure ID exists

    # ── prediction (called in background loop) ────────────────────────────

    def _prediction_loop(self) -> None:
        while not self._stop_evt.wait(timeout=self.tick_s):
            with self._state_lock:
                fns = list(self._states.keys())
            for fn in fns:
                prob = self._predict(fn)
                decision = "prewarm" if prob >= self.theta else "skip"
                if decision == "prewarm" and self._pool is not None:
                    triggered = self._pool.prewarm(fn)
                    if triggered:
                        self._prewarm_log.append({
                            "timestamp": time.time(),
                            "function_name": fn,
                            "prob": prob,
                        })
                        logger.debug("Pre-warm triggered for %s (p=%.3f)", fn, prob)

    def _predict(self, function_name: str) -> float:
        """Return cold start probability for function_name."""
        with self._state_lock:
            state = self._states.get(function_name)
        if state is None or len(state.history) < 2:
            return self._heuristic_predict(function_name)

        if TORCH_AVAILABLE and self._model is not None:
            return self._transformer_predict(state, function_name)
        return self._heuristic_predict(function_name)

    def _transformer_predict(self, state: FunctionState, function_name: str) -> float:
        self._model.eval()
        seq = state.build_sequence()  # (T, D_INPUT)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)
        func_id = self._get_func_id(function_name) % self._model.max_functions
        fid = torch.tensor([func_id], dtype=torch.long).to(self._device)
        with torch.no_grad():
            prob = self._model(x, fid).item()
        return float(prob)

    def _heuristic_predict(self, function_name: str) -> float:
        """
        Fallback when model is unavailable or not enough history:
        Predict cold start if the function hasn't been invoked recently.
        """
        with self._state_lock:
            state = self._states.get(function_name)
        if state is None or state.last_invoked_at is None:
            return 0.8   # unknown function → likely cold
        idle = time.time() - state.last_invoked_at
        # Logistic decay: probability rises as idle time approaches TTL (120s)
        ttl = 120.0
        p = 1.0 / (1.0 + math.exp(-6 * (idle / ttl - 0.6)))
        return float(p)

    # ── training ──────────────────────────────────────────────────────────

    def train(
        self,
        invocation_history: List[dict],
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 3e-4,
        patience: int = 5,
        val_fraction: float = 0.15,
        min_seq_len: int = 8,
    ) -> dict:
        """
        Train the transformer on historical invocation data.

        Args:
            invocation_history: List of dicts with keys:
                function_name, timestamp, was_cold, cold_start_latency_ms, ...
            epochs: max training epochs
            batch_size: training batch size
            lr: initial learning rate (cosine annealed)
            patience: early stopping patience on val-F1
            val_fraction: fraction of data held for validation (chronological)
            min_seq_len: minimum sequence length to create a sample

        Returns:
            dict with train_loss, val_loss, best_f1, epochs_run
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — skipping training")
            return {}

        logger.info("Module A training on %d invocations", len(invocation_history))

        # Build per-function state from history
        temp_states: Dict[str, FunctionState] = {}
        for rec in sorted(invocation_history, key=lambda r: r["timestamp"]):
            fn = rec["function_name"]
            if fn not in temp_states:
                runtime = rec.get("runtime", fn.split("_")[0])
                mem = rec.get("memory_mb", 512.0) or 512.0
                temp_states[fn] = FunctionState(name=fn, runtime=runtime, memory_mb=mem)
            temp_states[fn].record_invocation(rec["timestamp"], rec["was_cold"])
            self._get_func_id(fn)  # ensure we have an ID

        # Build (X, y, func_id) tensors
        X_list, y_list, fid_list = [], [], []
        for state in temp_states.values():
            recs = list(state.history)
            func_id = self._get_func_id(state.name) % self._model.max_functions
            for i in range(min_seq_len, len(recs)):
                # Use preceding records as context, predict current record's cold/warm
                seq_recs = recs[max(0, i - SEQ_LEN): i]
                target_rec = recs[i]
                seq = np.zeros((SEQ_LEN, D_INPUT), dtype=np.float32)
                start_idx = SEQ_LEN - len(seq_recs)
                for j, (ts, iat, _) in enumerate(seq_recs):
                    t = time.localtime(ts)
                    seq[start_idx + j] = encode_step(
                        iat, state.memory_mb, state.runtime,
                        t.tm_hour + t.tm_min / 60.0, float(t.tm_wday)
                    )
                X_list.append(seq)
                y_list.append(float(target_rec[2]))  # was_cold label
                fid_list.append(func_id)

        if len(X_list) < 2:
            logger.warning("Not enough training samples (%d)", len(X_list))
            return {"error": "no_data"}
        batch_size = min(batch_size, len(X_list))

        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)
        fids = torch.tensor(fid_list, dtype=torch.long)

        # Train / validation split (chronological — last N samples)
        n_val = max(1, int(len(X) * val_fraction))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        fid_train, fid_val = fids[:-n_val], fids[-n_val:]

        pos_rate = y_train.mean().item()
        logger.info("Training: %d samples (%.1f%% positive), Val: %d samples",
                     len(X_train), pos_rate * 100, len(X_val))

        self._model.train()
        optimiser = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-2)
        total_steps = epochs * max(1, math.ceil(len(X_train) / batch_size))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

        best_val_f1 = -1.0
        best_state_dict = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(epochs):
            # Training
            self._model.train()
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X_train), batch_size):
                idx = perm[start: start + batch_size]
                xb = X_train[idx].to(self._device)
                yb = y_train[idx].to(self._device)
                fb = fid_train[idx].to(self._device)
                pred = self._model(xb, fb)
                loss = focal_loss(pred, yb, FOCAL_ALPHA, FOCAL_GAMMA)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_train = epoch_loss / max(n_batches, 1)

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val.to(self._device), fid_val.to(self._device))
                val_loss = focal_loss(val_pred, y_val.to(self._device)).item()
                val_probs = val_pred.cpu().numpy()

            # Compute F1
            val_preds_bin = (val_probs >= self.theta).astype(int)
            val_labels = y_val.numpy().astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(val_labels, val_preds_bin, zero_division=0)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(f1)

            if f1 > best_val_f1 + 1e-4:
                best_val_f1 = f1
                best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (best F1=%.4f)", epoch + 1, best_val_f1)
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch %3d | train=%.4f | val=%.4f | F1=%.4f",
                    epoch + 1, avg_train, val_loss, f1,
                )

        # Restore best model
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        # Final evaluation
        self._model.eval()
        with torch.no_grad():
            val_probs = self._model(X_val.to(self._device), fid_val.to(self._device)).cpu().numpy()
        val_preds_bin = (val_probs >= self.theta).astype(int)
        val_labels = y_val.numpy().astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1  = f1_score(val_labels, val_preds_bin, zero_division=0)
        prec = precision_score(val_labels, val_preds_bin, zero_division=0)
        rec  = recall_score(val_labels, val_preds_bin, zero_division=0)

        logger.info(
            "Training complete | best_val_F1=%.4f | F1=%.4f P=%.4f R=%.4f",
            best_val_f1, f1, prec, rec,
        )
        return {
            "epochs_run": len(history["train_loss"]),
            "best_val_f1": float(best_val_f1),
            "val_f1": float(f1),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "train_loss_curve": history["train_loss"],
            "val_loss_curve": history["val_loss"],
            "val_f1_curve": history["val_f1"],
        }

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        if not TORCH_AVAILABLE or self._model is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "theta": self.theta,
            "func_id_map": self._func_id_map,
            "config": {
                "d_input": D_INPUT, "d_model": D_MODEL,
                "n_heads": N_HEADS, "n_layers": N_LAYERS,
                "d_embed": D_EMBED,
            }
        }, path)
        logger.info("Module A model saved → %s", path)

    def load(self, path: Path) -> None:
        if not TORCH_AVAILABLE:
            return
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state"])
        self.theta = ckpt.get("theta", DEFAULT_THETA)
        self._func_id_map = ckpt.get("func_id_map", {})
        self._next_func_id = max(self._func_id_map.values(), default=-1) + 1
        logger.info("Module A model loaded ← %s (θ=%.2f)", path, self.theta)

    # ── diagnostics ───────────────────────────────────────────────────────

    def prewarm_log(self) -> List[dict]:
        return list(self._prewarm_log)

    def parameter_count(self) -> int:
        if not TORCH_AVAILABLE or self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
