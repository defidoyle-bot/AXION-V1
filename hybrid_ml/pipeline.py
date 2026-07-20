"""
Hybrid ML Pipeline — orchestrates the gradient booster + RL agent.

The pipeline:
  1. Builds trade features from an approved signal.
  2. Runs the gradient booster to predict win probability.
  3. Builds a state vector from the booster output + account context.
  4. Runs the DQN agent to get a position-size multiplier (or skip).
  5. Returns a HybridMLRefinement with the adjusted parameters.
  6. On background ticks, retrains both models from completed trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import get_config
from core.logging import get_logger
from hybrid_ml.gradient_booster import (
    BoosterPrediction,
    GradientBooster,
    TradeFeatureEngineer,
)
from hybrid_ml.rl_agent import DQNAgent, NORMAL, NUM_ACTIONS, STATE_DIM

logger = get_logger("hybrid_ml.pipeline")


@dataclass
class HybridMLRefinement:
    """Output of the hybrid ML pipeline for a single signal."""
    signal_id: str
    symbol: str
    direction: str
    original_position_size: float
    adjusted_position_size: float
    multiplier: float
    rl_action: int
    rl_action_label: str
    booster_probability: float
    booster_confidence: float
    state_before: List[float]
    should_skip: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "original_position_size": self.original_position_size,
            "adjusted_position_size": self.adjusted_position_size,
            "multiplier": self.multiplier,
            "rl_action": self.rl_action,
            "rl_action_label": self.rl_action_label,
            "booster_probability": self.booster_probability,
            "booster_confidence": self.booster_confidence,
            "state_before": self.state_before,
            "should_skip": self.should_skip,
            "timestamp": self.timestamp.isoformat(),
        }


class HybridMLPipeline:
    """Orchestrates the gradient booster + RL agent for trade refinement."""

    def __init__(self):
        self.booster = GradientBooster()
        self.rl_agent = DQNAgent()
        self._pending_experiences: List[Dict[str, Any]] = []
        logger.info("HybridMLPipeline: initialised")

    # ── Public API ────────────────────────────────────────────────────────

    def refine(
        self,
        signal: Dict[str, Any],
        account_info: Optional[Dict[str, Any]] = None,
        deterministic: bool = True,
    ) -> HybridMLRefinement:
        """Run the full hybrid ML pipeline on a single signal."""
        # 1. Build trade features for the gradient booster
        trade_features = TradeFeatureEngineer.create_trade_features(signal)

        # 2. Get booster prediction
        booster_pred = self.booster.predict(trade_features)

        # 3. Build RL state
        account = account_info or {}
        balance = float(account.get("balance", 10000))
        initial_balance = float(account.get("initial_balance", 10000))
        balance_ratio = balance / initial_balance if initial_balance > 0 else 1.0

        active_trades = float(account.get("active_trades", 0))
        max_trades = float(account.get("max_concurrent_trades", 5))
        active_ratio = active_trades / max_trades if max_trades > 0 else 0.0

        daily_pnl = float(account.get("daily_pnl", 0))
        daily_pnl_norm = daily_pnl / initial_balance if initial_balance > 0 else 0.0

        win_rate = float(account.get("win_rate", 0.5))
        avg_win = float(account.get("avg_win", 0))
        avg_win_norm = avg_win / initial_balance * 100 if initial_balance > 0 else 0.0
        avg_loss = float(account.get("avg_loss", 0))
        avg_loss_norm = avg_loss / initial_balance * 100 if initial_balance > 0 else 0.0

        vol = float(account.get("avg_volume", 0))
        vol_ratio = vol / 1_000_000 if vol > 0 else 1.0  # normalise to ~$1M

        state = self.rl_agent.build_state(
            booster_prob=booster_pred.win_probability,
            booster_conf=booster_pred.confidence,
            account_balance_ratio=balance_ratio,
            active_trades_ratio=active_ratio,
            daily_pnl_norm=daily_pnl_norm,
            win_rate=win_rate,
            avg_win_norm=avg_win_norm,
            avg_loss_norm=avg_loss_norm,
            vol_ratio=vol_ratio,
        )

        # 4. Get RL action
        multiplier, action = self.rl_agent.get_position_multiplier(state, deterministic=deterministic)

        # 5. Compute adjusted position size
        original_size = float(signal.get("position_size", 0))
        adjusted_size = original_size * multiplier
        should_skip = action == 0

        # Store experience context for later reward assignment
        if not deterministic:
            self._pending_experiences.append({
                "signal_id": signal.get("signal_id", ""),
                "state_before": state.tolist(),
                "rl_action": action,
            })

        refinement = HybridMLRefinement(
            signal_id=signal.get("signal_id", ""),
            symbol=signal.get("symbol", ""),
            direction=signal.get("direction", ""),
            original_position_size=original_size,
            adjusted_position_size=adjusted_size,
            multiplier=multiplier,
            rl_action=action,
            rl_action_label=self.rl_agent.get_action_label(action),
            booster_probability=booster_pred.win_probability,
            booster_confidence=booster_pred.confidence,
            state_before=state.tolist(),
            should_skip=should_skip,
        )

        logger.info(
            f"HybridML: {refinement.symbol} {refinement.direction} "
            f"booster={booster_pred.win_probability:.2%} "
            f"action={refinement.rl_action_label} "
            f"mult={multiplier:.2f} "
            f"{'⏭ SKIP' if should_skip else '✅ GO'}"
        )
        return refinement

    # ── Retraining ────────────────────────────────────────────────────────

    def retrain_from_trades(self, completed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrain both models from a batch of completed trades."""
        result = {"booster": None, "rl": 0}

        if len(completed_trades) >= 20:
            perf = self.booster.train(completed_trades)
            result["booster"] = perf

        # Attach RL rewards and train
        for trade in completed_trades:
            signal_id = trade.get("signal_id", "")
            # Find matching pending experience
            for exp in self._pending_experiences:
                if exp["signal_id"] == signal_id:
                    trade["rl_state_before"] = exp["state_before"]
                    trade["rl_action"] = exp["rl_action"]
                    trade["rl_reward"] = self._compute_rl_reward(trade)
                    break

        if completed_trades:
            trained = self.rl_agent.train_on_trades(completed_trades)
            result["rl"] = trained

        return result

    def retrain_booster(self, completed_trades: List[Dict[str, Any]]) -> None:
        """Retrain only the gradient booster."""
        self.booster.train(completed_trades)

    def retrain_rl(self, experiences: List[Dict[str, Any]]) -> None:
        """Train the RL agent on a batch of experiences."""
        self.rl_agent.train_on_trades(experiences)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rl_reward(trade: Dict[str, Any]) -> float:
        """Compute normalised reward for a completed trade."""
        pnl = float(trade.get("pnl", 0))
        margin = float(trade.get("margin", 0))
        if margin > 0:
            return np.clip(pnl / margin, -1.0, 2.0)  # -100% to +200% return on margin
        return np.clip(pnl / 10000, -0.1, 0.5)  # fallback normalisation

    def get_pending_experiences_count(self) -> int:
        return len(self._pending_experiences)
