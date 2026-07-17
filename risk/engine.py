"""
AXION QUANT V4 - Risk Management Engine
Capital preservation over profit generation. Veto authority on all trades.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    RiskConfig, PositionSizingMethod, StopLossMethod, TakeProfitMethod,
    RiskProfile, StrategyProfile, get_config,
)
from core.logging import get_logger

logger = get_logger("risk")


# =============================================================================
# RISK DATA MODELS
# =============================================================================

class TradeStatus(Enum):
    """Trade lifecycle status."""
    CREATED = auto()
    VALIDATED = auto()
    APPROVED = auto()
    OPEN = auto()
    PARTIALLY_CLOSED = auto()
    TRAILING_ACTIVE = auto()
    CLOSED = auto()
    REJECTED = auto()
    CANCELLED = auto()


@dataclass(frozen=True, slots=True)
class PositionSize:
    """Calculated position size."""
    size: float
    method: str
    risk_amount: float
    risk_percent: float
    notional_value: float
    margin_required: float


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Complete risk assessment for a potential trade."""
    approved: bool
    rejection_reason: Optional[str]
    position_size: Optional[PositionSize]
    stop_loss: float
    take_profit: List[float]
    risk_reward: float
    liquidation_price: float
    liquidation_distance_percent: float
    margin_required: float
    max_leverage: int
    position_risk_percent: float
    daily_risk_exposure: float
    weekly_risk_exposure: float
    concurrent_trades: int
    symbol_exposure_percent: float
    total_account_risk_percent: float
    emergency_stop_triggered: bool
    leverage: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "position_size": {
                "size": round(self.position_size.size, 6) if self.position_size else None,
                "method": self.position_size.method if self.position_size else None,
                "risk_amount": round(self.position_size.risk_amount, 2) if self.position_size else None,
                "risk_percent": round(self.position_size.risk_percent, 4) if self.position_size else None,
            },
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": round(self.risk_reward, 2),
            "liquidation_price": self.liquidation_price,
            "liquidation_distance_percent": round(self.liquidation_distance_percent, 2),
            "margin_required": round(self.margin_required, 2),
            "max_leverage": self.max_leverage,
            "position_risk_percent": round(self.position_risk_percent, 4),
            "daily_risk_exposure": round(self.daily_risk_exposure, 4),
            "weekly_risk_exposure": round(self.weekly_risk_exposure, 4),
            "concurrent_trades": self.concurrent_trades,
            "symbol_exposure_percent": round(self.symbol_exposure_percent, 4),
            "total_account_risk_percent": round(self.total_account_risk_percent, 4),
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "leverage": self.leverage,
        }


@dataclass
class Trade:
    """Trade record."""
    trade_id: str
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    leverage: int
    margin: float
    status: TradeStatus
    created_at: datetime
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    funding_cost: float = 0.0
    mfe: float = 0.0  # Max favorable excursion
    mae: float = 0.0  # Max adverse excursion

    def calculate_pnl(self, exit_price: float, fee_rate: float = 0.0006) -> float:
        """Calculate P&L for the trade."""
        price_diff = exit_price - self.entry_price
        if self.direction == "SHORT":
            price_diff = -price_diff

        gross_pnl = price_diff * self.position_size
        fees = (self.entry_price + exit_price) * self.position_size * fee_rate

        self.pnl = gross_pnl - fees - self.funding_cost
        self.pnl_percent = (self.pnl / self.margin) * 100 if self.margin > 0 else 0
        self.exit_price = exit_price
        self.fees = fees

        return self.pnl


# =============================================================================
# RISK MANAGEMENT ENGINE
# =============================================================================

class RiskManagementEngine:
    """Institutional-grade risk management with veto authority."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or get_config().risk
        self._active_trades: Dict[str, Trade] = {}
        self._trade_history: List[Trade] = []
        self._daily_pnl: Dict[str, float] = {}  # date -> pnl
        self._weekly_pnl: Dict[str, float] = {}  # week -> pnl
        self._consecutive_losses = 0
        self._emergency_stop = False
        self._emergency_stop_time: Optional[datetime] = None
        self._account_balance = 10000.0  # Will be updated from config/paper trading

    def set_account_balance(self, balance: float) -> None:
        """Set current account balance."""
        self._account_balance = balance

    def validate_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[List[float]] = None,
        atr: Optional[float] = None,
        swing_points: Optional[List[Dict]] = None,
        order_blocks: Optional[List[Dict]] = None,
        contract_info: Optional[Dict] = None,
        strategy_profile: StrategyProfile = StrategyProfile.INTRADAY,
    ) -> RiskAssessment:
        """Validate a potential trade and calculate risk parameters."""

        # 1. Check emergency stop
        if self._emergency_stop:
            if self._should_auto_resume():
                self._emergency_stop = False
                logger.info("Emergency stop auto-resumed")
            else:
                return self._reject("Emergency stop active - trading suspended")

        # 2. Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger_emergency_stop("consecutive_losses")
            return self._reject(f"Max consecutive losses ({self.config.max_consecutive_losses}) reached")

        # 3. Check concurrent trades
        if len(self._active_trades) >= self.config.max_concurrent_trades:
            return self._reject(
                f"Max concurrent trades ({self.config.max_concurrent_trades}) reached"
            )

        # 4. Check daily loss limit
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_loss = abs(min(0, self._daily_pnl.get(today, 0)))
        daily_limit = self._account_balance * (self.config.daily_loss_limit_percent / 100)
        if daily_loss >= daily_limit:
            return self._reject(f"Daily loss limit reached: ${daily_loss:.2f}")

        # 5. Check weekly loss limit
        current_week = datetime.utcnow().strftime("%Y-%W")
        weekly_loss = abs(min(0, self._weekly_pnl.get(current_week, 0)))
        weekly_limit = self._account_balance * (self.config.weekly_loss_limit_percent / 100)
        if weekly_loss >= weekly_limit:
            return self._reject(f"Weekly loss limit reached: ${weekly_loss:.2f}")

        # 6. Calculate leverage
        leverage = self._get_leverage(strategy_profile)

        # 7. Calculate stop loss if not provided
        if stop_loss is None:
            stop_loss = self._calculate_stop_loss(
                entry_price, direction, atr, swing_points, order_blocks
            )

        # 8. Calculate take profit if not provided
        if take_profit is None:
            take_profit = self._calculate_take_profit(
                entry_price, stop_loss, direction, atr
            )

        # 9. Calculate position size
        risk_amount = self._calculate_risk_amount()
        position_size = self._calculate_position_size(
            entry_price, stop_loss, risk_amount, leverage, contract_info
        )

        if position_size is None:
            return self._reject("Position size calculation failed")

        # 10. Calculate liquidation price
        liquidation_price = self._calculate_liquidation_price(
            entry_price, direction, leverage, contract_info
        )

        # 11. Check liquidation safety
        liquidation_distance = self._calculate_liquidation_distance(
            entry_price, liquidation_price, direction
        )

        min_liquidation_distance = 10.0  # Minimum 10% distance
        if liquidation_distance < min_liquidation_distance:
            return self._reject(
                f"Liquidation too close: {liquidation_distance:.2f}% (min: {min_liquidation_distance}%)"
            )

        # 12. Calculate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit[0] - entry_price) if take_profit else risk * 2
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < self.config.min_risk_reward:
            return self._reject(
                f"Risk/Reward too low: {risk_reward:.2f} (min: {self.config.min_risk_reward})"
            )

        # 13. Check symbol exposure
        symbol_exposure = self._calculate_symbol_exposure(symbol)
        max_symbol_exposure = self._account_balance * (self.config.max_exposure_per_symbol_percent / 100)
        if symbol_exposure + position_size.notional_value > max_symbol_exposure:
            return self._reject("Max symbol exposure would be exceeded")

        # 14. Check total account risk
        total_risk = self._calculate_total_account_risk()
        position_risk_percent = (risk_amount / self._account_balance) * 100
        if total_risk + position_risk_percent > self.config.max_total_account_risk_percent:
            return self._reject("Max total account risk would be exceeded")

        # 15. Check drawdown
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown >= self.config.emergency_drawdown_percent:
            self._trigger_emergency_stop("drawdown")
            return self._reject(f"Emergency drawdown limit reached: {current_drawdown:.2f}%")

        # Trade approved
        return RiskAssessment(
            approved=True,
            rejection_reason=None,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            liquidation_price=liquidation_price,
            liquidation_distance_percent=liquidation_distance,
            margin_required=position_size.margin_required,
            max_leverage=contract_info.get("max_leverage", 125) if contract_info else 125,
            position_risk_percent=position_risk_percent,
            daily_risk_exposure=daily_loss / self._account_balance * 100,
            weekly_risk_exposure=weekly_loss / self._account_balance * 100,
            concurrent_trades=len(self._active_trades),
            symbol_exposure_percent=(symbol_exposure / self._account_balance) * 100,
            total_account_risk_percent=total_risk + position_risk_percent,
            emergency_stop_triggered=False,
            leverage=leverage,
        )

    def _reject(self, reason: str) -> RiskAssessment:
        """Create a rejection assessment."""
        logger.warning(f"Trade rejected: {reason}")
        return RiskAssessment(
            approved=False,
            rejection_reason=reason,
            position_size=None,
            stop_loss=0.0,
            take_profit=[],
            risk_reward=0.0,
            liquidation_price=0.0,
            liquidation_distance_percent=0.0,
            margin_required=0.0,
            max_leverage=0,
            position_risk_percent=0.0,
            daily_risk_exposure=0.0,
            weekly_risk_exposure=0.0,
            concurrent_trades=len(self._active_trades),
            symbol_exposure_percent=0.0,
            total_account_risk_percent=0.0,
            emergency_stop_triggered=self._emergency_stop,
            leverage=0,
        )

    def _calculate_risk_amount(self) -> float:
        """Calculate dollar risk amount per trade."""
        method = self.config.position_sizing_method

        if method == PositionSizingMethod.FIXED_RISK_PERCENT:
            return self._account_balance * (self.config.risk_percent_per_trade / 100)

        elif method == PositionSizingMethod.FIXED_DOLLAR_RISK:
            return self.config.fixed_dollar_risk

        elif method == PositionSizingMethod.KELLY_CRITERION and self.config.kelly_enabled:
            # Simplified Kelly - would need actual win rate and avg win/loss
            win_rate = self._calculate_win_rate()
            avg_win = self._calculate_avg_win()
            avg_loss = self._calculate_avg_loss()

            if avg_loss > 0:
                kelly = (win_rate - ((1 - win_rate) / (avg_win / avg_loss)))
                kelly = max(0, kelly * self.config.kelly_fraction)
                return self._account_balance * kelly
            return self._account_balance * (self.config.risk_percent_per_trade / 100)

        else:
            return self._account_balance * (self.config.risk_percent_per_trade / 100)

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_amount: float,
        leverage: int,
        contract_info: Optional[Dict],
    ) -> Optional[PositionSize]:
        """Calculate position size based on risk and stop loss."""
        price_risk = abs(entry_price - stop_loss)

        if price_risk <= 0:
            return None

        # Position size in contracts
        position_size = risk_amount / price_risk

        # Apply leverage
        notional_value = position_size * entry_price
        margin_required = notional_value / leverage

        # Check minimum order size
        if contract_info:
            min_size = contract_info.get("min_order_size", 0.01)
            if position_size < min_size:
                position_size = min_size
                notional_value = position_size * entry_price
                margin_required = notional_value / leverage
                # Recalculate risk
                risk_amount = price_risk * position_size

        # Check if margin exceeds account
        if margin_required > self._account_balance * 0.5:
            position_size = (self._account_balance * 0.5 * leverage) / entry_price
            notional_value = position_size * entry_price
            margin_required = notional_value / leverage
            risk_amount = price_risk * position_size

        risk_percent = (risk_amount / self._account_balance) * 100

        return PositionSize(
            size=position_size,
            method=self.config.position_sizing_method.value,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            notional_value=notional_value,
            margin_required=margin_required,
        )

    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: Optional[float],
        swing_points: Optional[List[Dict]],
        order_blocks: Optional[List[Dict]],
    ) -> float:
        """Calculate stop loss based on configured method."""
        method = self.config.stop_loss_method

        if method == StopLossMethod.ATR and atr:
            atr_multiple = self.config.atr_stop_multiplier
            if direction == "LONG":
                return entry_price - (atr * atr_multiple)
            else:
                return entry_price + (atr * atr_multiple)

        elif method == StopLossMethod.SWING_STRUCTURE and swing_points:
            # Use nearest relevant swing point
            if direction == "LONG":
                relevant_swings = [s for s in swing_points if s.get("is_low", False)]
                if relevant_swings:
                    return min(s["price"] for s in relevant_swings[-3:]) * 0.995
            else:
                relevant_swings = [s for s in swing_points if s.get("is_high", False)]
                if relevant_swings:
                    return max(s["price"] for s in relevant_swings[-3:]) * 1.005

        elif method == StopLossMethod.ORDER_BLOCK and order_blocks:
            if direction == "LONG":
                bullish_obs = [ob for ob in order_blocks if ob.get("ob_type") == "BULLISH"]
                if bullish_obs:
                    return min(ob["bottom"] for ob in bullish_obs) * 0.99
            else:
                bearish_obs = [ob for ob in order_blocks if ob.get("ob_type") == "BEARISH"]
                if bearish_obs:
                    return max(ob["top"] for ob in bearish_obs) * 1.01

        elif method == StopLossMethod.FIXED_PERCENTAGE:
            stop_percent = self.config.fixed_stop_percent / 100
            if direction == "LONG":
                return entry_price * (1 - stop_percent)
            else:
                return entry_price * (1 + stop_percent)

        # Default: ATR-based
        if atr:
            if direction == "LONG":
                return entry_price - (atr * 1.5)
            else:
                return entry_price + (atr * 1.5)

        # Fallback: fixed percentage
        if direction == "LONG":
            return entry_price * 0.98
        else:
            return entry_price * 1.02

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        atr: Optional[float],
    ) -> List[float]:
        """Calculate take profit levels."""
        method = self.config.take_profit_method
        risk = abs(entry_price - stop_loss)

        if method == TakeProfitMethod.FIXED_RR:
            target_rr = self.config.target_risk_reward
            if direction == "LONG":
                return [entry_price + (risk * target_rr)]
            else:
                return [entry_price - (risk * target_rr)]

        elif method == TakeProfitMethod.ATR_TARGET and atr:
            atr_multiple = self.config.atr_tp_multiplier
            if direction == "LONG":
                return [entry_price + (atr * atr_multiple)]
            else:
                return [entry_price - (atr * atr_multiple)]

        # Default: fixed RR
        target_rr = self.config.target_risk_reward
        if direction == "LONG":
            return [entry_price + (risk * target_rr)]
        else:
            return [entry_price - (risk * target_rr)]

    def _calculate_liquidation_price(
        self,
        entry_price: float,
        direction: str,
        leverage: int,
        contract_info: Optional[Dict],
    ) -> float:
        """Calculate estimated liquidation price."""
        # Simplified calculation (maintenance margin ~0.5%)
        maintenance_margin = 0.005

        if direction == "LONG":
            liquidation = entry_price * (1 - (1 / leverage) + maintenance_margin)
        else:
            liquidation = entry_price * (1 + (1 / leverage) - maintenance_margin)

        return liquidation

    def _calculate_liquidation_distance(
        self,
        entry_price: float,
        liquidation_price: float,
        direction: str,
    ) -> float:
        """Calculate distance to liquidation as percentage."""
        if direction == "LONG":
            return ((entry_price - liquidation_price) / entry_price) * 100
        else:
            return ((liquidation_price - entry_price) / entry_price) * 100

    def _get_leverage(self, strategy_profile: StrategyProfile) -> int:
        """Get leverage based on strategy profile."""
        leverage = self.config.leverage_by_strategy.get(
            strategy_profile, self.config.default_leverage
        )
        return min(leverage, self.config.max_leverage)

    def _calculate_symbol_exposure(self, symbol: str) -> float:
        """Calculate current exposure for a symbol."""
        exposure = 0.0
        for trade in self._active_trades.values():
            if trade.symbol == symbol:
                exposure += trade.position_size * trade.entry_price
        return exposure

    def _calculate_total_account_risk(self) -> float:
        """Calculate total current account risk percentage."""
        total_risk = 0.0
        for trade in self._active_trades.values():
            trade_risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
            total_risk += (trade_risk / self._account_balance) * 100
        return total_risk

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak balance."""
        # Simplified - would track peak balance over time
        return 0.0

    def _calculate_win_rate(self) -> float:
        """Calculate recent win rate."""
        if not self._trade_history:
            return 0.5
        recent = self._trade_history[-50:]
        wins = sum(1 for t in recent if t.pnl > 0)
        return wins / len(recent) if recent else 0.5

    def _calculate_avg_win(self) -> float:
        """Calculate average win."""
        wins = [t.pnl for t in self._trade_history if t.pnl > 0]
        return np.mean(wins) if wins else 0.0

    def _calculate_avg_loss(self) -> float:
        """Calculate average loss."""
        losses = [abs(t.pnl) for t in self._trade_history if t.pnl <= 0]
        return np.mean(losses) if losses else 1.0

    def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency trading stop."""
        self._emergency_stop = True
        self._emergency_stop_time = datetime.utcnow()
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

    def _should_auto_resume(self) -> bool:
        """Check if emergency stop should auto-resume."""
        if not self._emergency_stop or not self._emergency_stop_time:
            return False

        elapsed = (datetime.utcnow() - self._emergency_stop_time).total_seconds() / 60
        return elapsed >= self.config.auto_resume_after_minutes

    def add_trade(self, trade: Trade) -> None:
        """Add an active trade."""
        self._active_trades[trade.trade_id] = trade
        logger.info(f"Trade added: {trade.trade_id} {trade.symbol} {trade.direction}")

    def close_trade(self, trade_id: str, exit_price: float, fee_rate: float = 0.0006) -> Trade:
        """Close an active trade."""
        if trade_id not in self._active_trades:
            raise ValueError(f"Trade not found: {trade_id}")

        trade = self._active_trades[trade_id]
        trade.calculate_pnl(exit_price, fee_rate)
        trade.status = TradeStatus.CLOSED
        trade.closed_at = datetime.utcnow()

        # Update tracking
        del self._active_trades[trade_id]
        self._trade_history.append(trade)

        # Update P&L tracking
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        week_key = datetime.utcnow().strftime("%Y-%W")
        self._daily_pnl[date_key] = self._daily_pnl.get(date_key, 0) + trade.pnl
        self._weekly_pnl[week_key] = self._weekly_pnl.get(week_key, 0) + trade.pnl

        # Update consecutive losses
        if trade.pnl <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Update account balance
        self._account_balance += trade.pnl

        logger.info(
            f"Trade closed: {trade_id} P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)"
        )

        return trade

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            "account_balance": round(self._account_balance, 2),
            "active_trades": len(self._active_trades),
            "total_exposure": sum(t.position_size * t.entry_price for t in self._active_trades.values()),
            "daily_pnl": round(self._daily_pnl.get(datetime.utcnow().strftime("%Y-%m-%d"), 0), 2),
            "consecutive_losses": self._consecutive_losses,
            "emergency_stop": self._emergency_stop,
            "win_rate": round(self._calculate_win_rate(), 4),
            "total_trades": len(self._trade_history),
        }
