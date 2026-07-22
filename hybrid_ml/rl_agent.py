"""
Reinforcement Learning agent (DQN) for refining trade decisions.

The agent takes the gradient booster's prediction plus market/trade state
and outputs an action: skip, quarter-size, half-size, normal position,
or double-size.  The RL reward is the realised PnL of the trade.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.logging import get_logger

logger = get_logger("hybrid_ml.rl_agent")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = object  # type: ignore[assignment]
    optim = object  # type: ignore[assignment]
    _HAS_TORCH = False
    logger.warning("PyTorch not available — DQNAgent will always output 'normal' action")


# ── Actions ──────────────────────────────────────────────────────────────────
# 0 = SKIP, 1 = QUARTER_SIZE, 2 = HALF_SIZE, 3 = NORMAL, 4 = DOUBLE_SIZE
SKIP = 0
QUARTER_SIZE = 1
HALF_SIZE = 2
NORMAL = 3
DOUBLE_SIZE = 4
NUM_ACTIONS = 5

# Action -> position size multiplier
ACTION_MULTIPLIER = {
    SKIP: 0.0,
    QUARTER_SIZE: 0.25,
    HALF_SIZE: 0.5,
    NORMAL: 1.0,
    DOUBLE_SIZE: 1.5,
}

# State dimension
# [booster_prob, booster_conf, balance_ratio, active_trades_ratio,
#  daily_pnl_norm, win_rate, avg_win_norm, avg_loss_norm, vol_ratio]
STATE_DIM = 9


@dataclass
class DQNConfig:
    """DQN hyperparameters."""
    learning_rate: float = 0.001
    gamma: float = 0.95  # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    hidden_dim: int = 128
    tau: float = 0.005  # soft update coefficient


@dataclass
class Experience:
    """Single (state, action, reward, next_state, done) transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer: List[Experience] = []
        self.capacity = capacity
        self._pos = 0

    def push(self, exp: Experience) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self._pos] = exp
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


_DQNBase = nn.Module if _HAS_TORCH else object


class DQN(_DQNBase):
    """Deep Q-Network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        if not _HAS_TORCH:
            return
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        if not _HAS_TORCH:
            return x
        return self.net(x)


class DQNAgent:
    """DQN agent for refining trade decisions."""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        model_dir: str = "models/hybrid_ml",
    ):
        self.config = config or DQNConfig()
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu") if _HAS_TORCH else None
        self._has_torch = _HAS_TORCH

        if self._has_torch:
            self.policy_net = DQN(STATE_DIM, NUM_ACTIONS, self.config.hidden_dim).to(self.device)
            self.target_net = DQN(STATE_DIM, NUM_ACTIONS, self.config.hidden_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        else:
            self.policy_net = None
            self.target_net = None

        self.memory = ReplayBuffer(self.config.memory_size)
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self._is_ready = False

        # Load saved model if available
        self._load_latest()

    def _load_latest(self) -> bool:
        """Load the most recent saved DQN model."""
        model_files = sorted(
            self._model_dir.glob("dqn_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not model_files:
            return False
        if not self._has_torch:
            return False

        try:
            with open(model_files[0], "rb") as f:
                data = pickle.load(f)
            self.policy_net.load_state_dict(data["policy_state"])
            self.target_net.load_state_dict(data["target_state"])
            self.optimizer.load_state_dict(data["optimizer_state"])
            self.epsilon = data.get("epsilon", self.config.epsilon_start)
            self.steps = data.get("steps", 0)
            self._is_ready = True
            logger.info(f"DQNAgent: loaded model from {model_files[0]}")
            return True
        except Exception as e:
            logger.warning(f"DQNAgent: failed to load model: {e}")
            return False

    def _save(self) -> None:
        """Save DQN model to disk."""
        if not self._has_torch:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._model_dir / f"dqn_{timestamp}.pkl"
        data = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        # Cleanup old models
        for old in sorted(self._model_dir.glob("dqn_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)[5:]:
            old.unlink()
        logger.debug(f"DQNAgent: saved to {path}")

    def build_state(
        self,
        booster_prob: float,
        booster_conf: float,
        account_balance_ratio: float,
        active_trades_ratio: float,
        daily_pnl_norm: float,
        win_rate: float,
        avg_win_norm: float,
        avg_loss_norm: float,
        vol_ratio: float,
    ) -> np.ndarray:
        """Build normalised state vector."""
        return np.array([
            float(booster_prob),
            float(booster_conf),
            float(np.clip(account_balance_ratio, 0, 1)),
            float(np.clip(active_trades_ratio, 0, 1)),
            float(np.clip(daily_pnl_norm, -1, 1)),
            float(np.clip(win_rate, 0, 1)),
            float(np.clip(avg_win_norm, 0, 5)),
            float(np.clip(avg_loss_norm, 0, 5)),
            float(np.clip(vol_ratio, 0, 5)),
        ], dtype=np.float32)

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select an action using epsilon-greedy (training) or greedy (deterministic)."""
        if not self._has_torch or not self._is_ready:
            return NORMAL  # default: normal position

        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())

    def get_position_multiplier(self, state: np.ndarray, deterministic: bool = True) -> float:
        """Get position size multiplier from the agent."""
        action = self.select_action(state, deterministic=deterministic)
        return ACTION_MULTIPLIER[action], action

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.memory.push(Experience(state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """Perform one training step. Returns the loss if training occurred."""
        if not self._has_torch or len(self.memory) < self.config.batch_size:
            return None

        batch = self.memory.sample(self.config.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([float(e.done) for e in batch])).to(self.device)

        # Compute Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        # Soft update target network
        self.steps += 1
        if self.steps % self.config.target_update_freq == 0:
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data + (1 - self.config.tau) * target_param.data
                )

        return float(loss.item())

    def train_on_trades(self, trades: List[Dict[str, Any]]) -> int:
        """Train the agent on a batch of completed trade experiences."""
        trained_count = 0
        for trade in trades:
            state = np.array(trade.get("rl_state_before", [0.0] * STATE_DIM))
            action = trade.get("rl_action", NORMAL)
            reward = trade.get("rl_reward", 0.0)
            next_state = np.array(trade.get("rl_state_after", [0.0] * STATE_DIM))
            done = trade.get("status") == "closed"

            if np.any(state) or np.any(next_state):
                self.store_experience(state, action, reward, next_state, done)
                trained_count += 1

        # Run training steps
        steps = 0
        for _ in range(min(trained_count * 2, 200)):
            loss = self.train_step()
            if loss is not None:
                steps += 1

        if steps > 0:
            self._is_ready = True
            self._save()
            logger.info(f"DQNAgent: trained on {trained_count} experiences ({steps} steps), ε={self.epsilon:.3f}")

        return trained_count

    def get_action_label(self, action: int) -> str:
        return {
            SKIP: "SKIP",
            QUARTER_SIZE: "QUARTER_SIZE",
            HALF_SIZE: "HALF_SIZE",
            NORMAL: "NORMAL",
            DOUBLE_SIZE: "DOUBLE_SIZE",
        }.get(action, "UNKNOWN")
