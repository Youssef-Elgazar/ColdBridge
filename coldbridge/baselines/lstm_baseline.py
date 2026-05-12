"""
LSTM Baseline
=============
Minimal single-layer LSTM for cold start prediction.

Operates on the same (B, T, d_input=7) feature tensors as Module A's
transformer, allowing a fair architecture-only comparison.

The model is deliberately small (1 layer, hidden=64) to keep training
fast while still providing a credible comparison point.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("coldbridge.baselines.lstm")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from coldbridge.modules.module_a import (
    SEQ_LEN, D_INPUT, encode_step, FunctionState, FOCAL_ALPHA, FOCAL_GAMMA,
)

if TORCH_AVAILABLE:
    from coldbridge.modules.module_a import focal_loss


if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """Single-layer LSTM with sigmoid output for binary cold start prediction."""

        def __init__(self, d_input: int = D_INPUT, hidden: int = 64, n_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=d_input, hidden_size=hidden,
                num_layers=n_layers, batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            self.sig = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, T, d_input) → (B,) probabilities."""
            out, _ = self.lstm(x)
            logits = self.head(out[:, -1, :])
            return self.sig(logits).squeeze(-1)


class LSTMBaseline:
    """LSTM-based cold start predictor matching Module A's interface.

    Provides ``train()`` and ``predict()`` methods with the same data format
    as :class:`ModuleA`, enabling direct comparison.
    """

    def __init__(self, theta: float = 0.50, device: Optional[str] = None):
        self.theta = theta
        if TORCH_AVAILABLE:
            self._device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._model = LSTMModel().to(self._device)
        else:
            self._model = None
            self._device = None

    def train(
        self,
        invocation_history: List[dict],
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 5,
        val_fraction: float = 0.15,
        min_seq_len: int = 8,
    ) -> dict:
        """Train the LSTM on historical invocation data.

        Uses the same data preparation as Module A for a fair comparison.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — skipping LSTM training")
            return {}

        logger.info("LSTM baseline training on %d invocations", len(invocation_history))

        # Build per-function state from history (same as Module A)
        temp_states: Dict[str, FunctionState] = {}
        for rec in sorted(invocation_history, key=lambda r: r["timestamp"]):
            fn = rec["function_name"]
            if fn not in temp_states:
                runtime = rec.get("runtime", fn.split("_")[0])
                mem = rec.get("memory_mb", 512.0) or 512.0
                temp_states[fn] = FunctionState(name=fn, runtime=runtime, memory_mb=mem)
            temp_states[fn].record_invocation(rec["timestamp"], rec["was_cold"])

        # Build (X, y) tensors
        X_list, y_list = [], []
        for state in temp_states.values():
            recs = list(state.history)
            for i in range(min_seq_len, len(recs)):
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
                y_list.append(float(target_rec[2]))

        if len(X_list) < 2:
            logger.warning("Not enough training samples (%d)", len(X_list))
            return {"error": "no_data"}

        batch_size = min(batch_size, len(X_list))
        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)

        # Chronological train/val split
        n_val = max(1, int(len(X) * val_fraction))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        self._model.train()
        optimizer = optim.Adam(self._model.parameters(), lr=lr)

        best_val_f1 = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self._model.train()
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X_train), batch_size):
                idx = perm[start: start + batch_size]
                xb = X_train[idx].to(self._device)
                yb = y_train[idx].to(self._device)
                pred = self._model(xb)
                loss = focal_loss(pred, yb, FOCAL_ALPHA, FOCAL_GAMMA)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val.to(self._device))
                val_probs = val_pred.cpu().numpy()

            from sklearn.metrics import f1_score
            val_preds_bin = (val_probs >= self.theta).astype(int)
            val_labels = y_val.numpy().astype(int)
            f1 = f1_score(val_labels, val_preds_bin, zero_division=0)

            if f1 > best_val_f1 + 1e-4:
                best_val_f1 = f1
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("LSTM early stop at epoch %d (best F1=%.4f)", epoch + 1, best_val_f1)
                    break

        if best_state:
            self._model.load_state_dict(best_state)

        logger.info("LSTM training complete — best_val_F1=%.4f", best_val_f1)
        return {"best_val_f1": float(best_val_f1), "epochs_run": epoch + 1}

    def predict(self, sequence: np.ndarray) -> float:
        """Predict cold start probability from a (T, D_INPUT) feature sequence."""
        if not TORCH_AVAILABLE or self._model is None:
            return 0.5
        self._model.eval()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            prob = self._model(x).item()
        return float(prob)

    def parameter_count(self) -> int:
        if not TORCH_AVAILABLE or self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
