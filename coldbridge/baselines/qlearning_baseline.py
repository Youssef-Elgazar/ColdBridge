"""
Q-Learning Baseline
====================
Tabular Q-learning cold start predictor for ColdBridge.

State = (runtime_id, memory_bin, hour_bin, dow_bin)
Action = predict cold (1) or warm (0)

Reward: +1 for correct prediction, -1 for incorrect.

This is a deliberately simple baseline to demonstrate the gap between
classical RL and the transformer-based Module A approach.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("coldbridge.baselines.qlearning")


RUNTIME_IDS = {"python": 0, "node": 1, "java": 2, "go": 3, "dotnet": 4}


class QLearningBaseline:
    """Tabular Q-learning cold start predictor.

    Discretises the feature space into bins and learns a Q-table
    mapping (state, action) → expected reward.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        theta: float = 0.50,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = theta
        self.Q: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(2))
        self._trained = False

    def _state_key(
        self,
        runtime: str,
        memory_mb: float,
        hour: float,
        dow: float,
        iat_s: float,
    ) -> tuple:
        """Discretise the continuous features into a table key."""
        return (
            RUNTIME_IDS.get(runtime, 0),
            int(memory_mb // 256),
            int(hour // 4),         # 6 bins for time of day
            int(dow),               # 7 bins for day of week
            min(int(iat_s // 60), 20),  # 20 bins for inter-arrival time (capped at 20 min)
        )

    def train(
        self,
        invocation_history: List[dict],
        epochs: int = 10,
    ) -> dict:
        """Train Q-table on historical invocation data.

        Each invocation becomes a (state, action, reward) sample.
        We run multiple passes over the data to converge the Q-values.
        """
        logger.info("Q-learning training on %d invocations (%d epochs)",
                     len(invocation_history), epochs)

        sorted_history = sorted(invocation_history, key=lambda r: r["timestamp"])

        # Pre-compute state keys and labels
        samples = []
        last_ts: Dict[str, float] = {}
        for rec in sorted_history:
            fn = rec["function_name"]
            ts = rec["timestamp"]
            iat = ts - last_ts.get(fn, ts)
            last_ts[fn] = ts

            runtime = rec.get("runtime", fn.split("_")[0])
            memory = rec.get("memory_mb", 512.0) or 512.0
            t = time.localtime(ts)
            hour = t.tm_hour + t.tm_min / 60.0
            dow = float(t.tm_wday)

            state = self._state_key(runtime, memory, hour, dow, iat)
            label = 1 if rec["was_cold"] else 0
            samples.append((state, label))

        if not samples:
            return {"error": "no_data"}

        for epoch in range(epochs):
            total_reward = 0.0
            for state, label in samples:
                # epsilon-greedy action
                if np.random.rand() < self.epsilon:
                    action = np.random.choice([0, 1])
                else:
                    action = int(self.Q[state].argmax())

                # Reward: +1 for correct, -1 for incorrect
                reward = 1.0 if action == label else -1.0
                total_reward += reward

                # Q-update (no next state in this simplified formulation)
                best_next = self.Q[state].max()
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * best_next - self.Q[state][action]
                )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_reward = total_reward / len(samples)
                logger.info("Q-learning epoch %d — avg reward: %.3f", epoch + 1, avg_reward)

        self._trained = True
        # Decay epsilon after training
        self.epsilon = max(0.05, self.epsilon * 0.5)

        logger.info("Q-learning training complete — %d states learned", len(self.Q))
        return {"states_learned": len(self.Q), "epochs_run": epochs}

    def predict(
        self,
        runtime: str,
        memory_mb: float,
        hour: float,
        dow: float,
        iat_s: float,
    ) -> float:
        """Predict cold start probability.

        Returns the Q-value for action=1 (cold) normalised to [0, 1].
        """
        state = self._state_key(runtime, memory_mb, hour, dow, iat_s)
        q_vals = self.Q[state]

        # Convert Q-values to probability via softmax
        q_max = max(q_vals)
        if q_max == 0 and min(q_vals) == 0:
            return 0.5  # no information
        exp_vals = np.exp(q_vals - q_max)  # numerically stable softmax
        probs = exp_vals / exp_vals.sum()
        return float(probs[1])  # probability of action=1 (cold)
