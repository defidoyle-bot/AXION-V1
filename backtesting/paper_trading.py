"""
AXION QUANT V4 - Paper Trading
Simulated execution with live market data, virtual account tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import PaperTradingConfig, get_config
from core.logging import get_logger
from risk.engine import Trade, TradeStatus

logger = get_logger("paper_trading")


# =============================================================================
# PAPER TRADING ACCOUNT
# =============================================================================

@dataclass
class PaperAccount:
    """Virtual trading account for paper trading."""
    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    margin_available: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    def update_equity(self, timestamp: datetime) -> None:
        """Update equity and track drawdown."""
        self.equity = self.balance + self.unrealized_pnl

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        drawdown = self.peak_equity - self.equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.equity_curve.append({
            "timestamp": timestamp.isoformat(),
            "equity": self.equity,
            "balance": self.balance,
            "unrealized_pnl": self.unrealized_pnl,
            "drawdown": drawdown,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "margin_used": round(self.margin_used, 2),
            "margin_available": round(self.margin_available, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
        }


# =============================================================================
# PAPER TRADING ENGINE
# =============================================================================

class PaperTradingEngine:
    """Paper trading engine using production analysis and risk logic."""

    def __init__(self, config: Optional[PaperTradingConfig] = None):
        self.config = config or get_config().paper_trading
        self.account = PaperAccount(
            balance=self.config.initial_balance,
            equity=self.config.initial_balance,
            unrealized_pnl=0.0,
            margin_used=0.0,
            margin_available=self.config.initial_balance,
            peak_equity=self.config.initial_balance,
        )
        self._active_trades: Dict[str, Trade] = {}
        self._trade_history: List[Trade] = []
        self._enabled = self.config.enabled

    def is_enabled(self) -> bool:
        """Check if paper trading is enabled."""
        return self._enabled

    def get_account(self) -> PaperAccount:
        """Get current paper trading account."""
        return self.account

    def execute_signal(self, signal: Dict[str, Any], current_price: float) -> Optional[Trade]:
        """Execute a signal in paper trading mode."""
        if not self._enabled:
            return None

        symbol = signal["symbol"]
        direction = signal["direction"]
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        take_profit = signal.get("take_profit", [])
        position_size = signal["position_size"]
        leverage = signal.get("leverage", 1)
        margin = signal.get("margin_required", 0)

        # Check if we have enough margin
        if margin > self.account.margin_available:
            logger.warning(f"Insufficient margin for {symbol}: need ${margin:.2f}, have ${self.account.margin_available:.2f}")
            return None

        # Apply simulated slippage
        slippage = self.config.slippage_percent / 100
        if direction == "LONG":
            executed_price = entry_price * (1 + slippage)
        else:
            executed_price = entry_price * (1 - slippage)

        # Create trade
        trade = Trade(
            trade_id=f"paper_{datetime.utcnow().timestamp()}",
            signal_id=signal.get("signal_id", ""),
            symbol=symbol,
            direction=direction,
            entry_price=executed_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            leverage=leverage,
            margin=margin,
            status=TradeStatus.OPEN,
            created_at=datetime.utcnow(),
            opened_at=datetime.utcnow(),
        )

        # Update account
        self.account.margin_used += margin
        self.account.margin_available -= margin
        self.account.total_trades += 1

        self._active_trades[trade.trade_id] = trade

        logger.info(
            f"Paper trade opened: {trade.trade_id} {symbol} {direction} "
            f"@ {executed_price:,.2f} size: {position_size:.4f} margin: ${margin:.2f}"
        )

        return trade

    def update_trades(self, symbol: str, current_price: float, timestamp: datetime) -> List[Trade]:
        """Update active trades with current market price."""
        if not self._enabled:
            return []

        closed_trades = []

        for trade_id, trade in list(self._active_trades.items()):
            if trade.symbol != symbol:
                continue

            # Calculate unrealized P&L
            if trade.direction == "LONG":
                unrealized = (current_price - trade.entry_price) * trade.position_size
                # Check stop loss
                if current_price <= trade.stop_loss:
                    trade = self._close_trade(trade, trade.stop_loss, "stop_loss", timestamp)
                    closed_trades.append(trade)
                    continue
                # Check take profit
                if trade.take_profit and current_price >= trade.take_profit[0]:
                    trade = self._close_trade(trade, trade.take_profit[0], "take_profit", timestamp)
                    closed_trades.append(trade)
                    continue
            else:  # SHORT
                unrealized = (trade.entry_price - current_price) * trade.position_size
                # Check stop loss
                if current_price >= trade.stop_loss:
                    trade = self._close_trade(trade, trade.stop_loss, "stop_loss", timestamp)
                    closed_trades.append(trade)
                    continue
                # Check take profit
                if trade.take_profit and current_price <= trade.take_profit[0]:
                    trade = self._close_trade(trade, trade.take_profit[0], "take_profit", timestamp)
                    closed_trades.append(trade)
                    continue

            # Update MFE/MAE
            if trade.direction == "LONG":
                trade.mfe = max(trade.mfe, (current_price - trade.entry_price) * trade.position_size)
                trade.mae = max(trade.mae, (trade.entry_price - current_price) * trade.position_size)
            else:
                trade.mfe = max(trade.mfe, (trade.entry_price - current_price) * trade.position_size)
                trade.mae = max(trade.mae, (current_price - trade.entry_price) * trade.position_size)

            # Update unrealized P&L
            self.account.unrealized_pnl += unrealized - self.account.unrealized_pnl

        # Update equity curve
        self.account.update_equity(timestamp)

        return closed_trades

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        reason: str,
        timestamp: datetime,
    ) -> Trade:
        """Close a paper trade."""
        # Calculate P&L
        fee_rate = self.config.fee_rate_percent / 100
        trade.calculate_pnl(exit_price, fee_rate)
        trade.status = TradeStatus.CLOSED
        trade.closed_at = timestamp

        # Update account
        self.account.balance += trade.pnl
        self.account.margin_used -= trade.margin
        self.account.margin_available += trade.margin
        self.account.unrealized_pnl = 0
        self.account.total_pnl += trade.pnl

        if trade.pnl > 0:
            self.account.winning_trades += 1
        else:
            self.account.losing_trades += 1

        # Remove from active trades
        del self._active_trades[trade.trade_id]
        self._trade_history.append(trade)

        logger.info(
            f"Paper trade closed: {trade.trade_id} {reason} "
            f"P&L: ${trade.pnl:,.2f} ({trade.pnl_percent:+.2f}%)"
        )

        return trade

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate paper trading performance report."""
        if not self._trade_history:
            return {
                "status": "No trades executed yet",
                "account": self.account.to_dict(),
            }

        wins = [t for t in self._trade_history if t.pnl > 0]
        losses = [t for t in self._trade_history if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "account": self.account.to_dict(),
            "total_trades": len(self._trade_history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": (len(wins) / len(self._trade_history) * 100) if self._trade_history else 0,
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0,
            "total_pnl": round(self.account.total_pnl, 2),
            "max_drawdown": round(self.account.max_drawdown, 2),
            "active_trades": len(self._active_trades),
            "equity_curve_points": len(self.account.equity_curve),
        }

    def reset_account(self) -> None:
        """Reset paper trading account to initial state."""
        self.account = PaperAccount(
            balance=self.config.initial_balance,
            equity=self.config.initial_balance,
            unrealized_pnl=0.0,
            margin_used=0.0,
            margin_available=self.config.initial_balance,
            peak_equity=self.config.initial_balance,
        )
        self._active_trades.clear()
        self._trade_history.clear()
        logger.info("Paper trading account reset")
