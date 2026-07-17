"""
AXION QUANT V4 - Backtesting Engine
Historical replay using production analysis, scoring, and risk logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import BacktestConfig, get_config
from core.logging import get_logger

logger = get_logger("backtest")


# =============================================================================
# BACKTEST DATA MODELS
# =============================================================================

@dataclass
class BacktestTrade:
    """Simulated trade during backtest."""
    entry_index: int
    entry_price: float
    exit_price: Optional[float] = None
    exit_index: Optional[int] = None
    direction: str = "LONG"
    stop_loss: float = 0.0
    take_profit: List[float] = field(default_factory=list)
    position_size: float = 0.0
    leverage: int = 1
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    funding_cost: float = 0.0
    exit_reason: str = ""
    mfe: float = 0.0
    mae: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        return self.pnl <= 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    net_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    recovery_factor: float
    expectancy: float
    avg_r_multiple: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    trades: List[BacktestTrade]
    monthly_returns: Dict[str, float]
    weekly_returns: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(self.final_balance, 2),
            "total_return": round(self.total_return, 4),
            "net_profit": round(self.net_profit, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_percent": round(self.max_drawdown_percent, 4),
            "expectancy": round(self.expectancy, 4),
            "avg_r_multiple": round(self.avg_r_multiple, 4),
        }


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """Backtesting engine using production logic."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or get_config().backtest
        self._results: List[BacktestResult] = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        signal_generator: callable,
        risk_engine: Any,
        initial_balance: Optional[float] = None,
    ) -> BacktestResult:
        """Run a standard backtest on historical data."""
        balance = initial_balance or self.config.initial_balance
        initial = balance

        trades: List[BacktestTrade] = []
        equity_curve = [balance]
        drawdown_curve = [0.0]
        peak_balance = balance

        logger.info(f"Starting backtest with {len(df)} candles, balance: ${balance:,.2f}")

        # Generate signals for each candle (with lookback)
        for i in range(200, len(df) - self.config.prediction_horizon):
            window = df.iloc[:i]

            # Generate signal using production logic
            signal = signal_generator(window)

            if signal and signal.get("approved", False):
                # Simulate trade
                trade = self._simulate_trade(
                    df, i, signal, balance, risk_engine
                )

                if trade:
                    trades.append(trade)
                    balance += trade.pnl

                    # Update peak and drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    drawdown = peak_balance - balance
                    drawdown_percent = (drawdown / peak_balance) * 100

                    equity_curve.append(balance)
                    drawdown_curve.append(drawdown_percent)

        # Calculate metrics
        result = self._calculate_metrics(
            trades, initial, balance, equity_curve, drawdown_curve, df
        )

        self._results.append(result)
        logger.info(f"Backtest complete: {result.total_trades} trades, {result.win_rate:.1%} win rate")

        return result

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        entry_index: int,
        signal: Dict[str, Any],
        balance: float,
        risk_engine: Any,
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade through to completion."""
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        take_profits = signal.get("take_profit", [])
        direction = signal["direction"]
        leverage = signal.get("leverage", 1)

        # Calculate position size
        risk_amount = balance * 0.01  # 1% risk
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk if price_risk > 0 else 0

        # Apply slippage
        slippage = self.config.slippage_percent / 100
        if direction == "LONG":
            entry_price = entry_price * (1 + slippage)
        else:
            entry_price = entry_price * (1 - slippage)

        # Simulate trade progression
        trade = BacktestTrade(
            entry_index=entry_index,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profits,
            position_size=position_size,
            leverage=leverage,
        )

        # Walk forward through candles
        for j in range(entry_index + 1, min(len(df), entry_index + 100)):
            candle = df.iloc[j]

            # Track MFE/MAE
            if direction == "LONG":
                favorable = candle["high"] - entry_price
                adverse = entry_price - candle["low"]
            else:
                favorable = entry_price - candle["low"]
                adverse = candle["high"] - entry_price

            trade.mfe = max(trade.mfe, favorable)
            trade.mae = max(trade.mae, adverse)

            # Check stop loss
            sl_hit = (direction == "LONG" and candle["low"] <= stop_loss) or                      (direction == "SHORT" and candle["high"] >= stop_loss)

            if sl_hit:
                # Apply slippage on exit
                if direction == "LONG":
                    exit_price = stop_loss * (1 - slippage)
                else:
                    exit_price = stop_loss * (1 + slippage)

                trade.exit_price = exit_price
                trade.exit_index = j
                trade.exit_reason = "stop_loss"
                trade.pnl = self._calculate_pnl(trade, exit_price)
                trade.fees = (entry_price + exit_price) * position_size * (self.config.fee_rate_percent / 100)
                trade.pnl -= trade.fees
                return trade

            # Check take profit
            if take_profits:
                tp_hit = (direction == "LONG" and candle["high"] >= take_profits[0]) or                          (direction == "SHORT" and candle["low"] <= take_profits[0])

                if tp_hit:
                    if direction == "LONG":
                        exit_price = take_profits[0] * (1 - slippage)
                    else:
                        exit_price = take_profits[0] * (1 + slippage)

                    trade.exit_price = exit_price
                    trade.exit_index = j
                    trade.exit_reason = "take_profit"
                    trade.pnl = self._calculate_pnl(trade, exit_price)
                    trade.fees = (entry_price + exit_price) * position_size * (self.config.fee_rate_percent / 100)
                    trade.pnl -= trade.fees
                    return trade

        # Trade didn't close within window
        return None

    def _calculate_pnl(self, trade: BacktestTrade, exit_price: float) -> float:
        """Calculate trade P&L."""
        if trade.direction == "LONG":
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price

        return price_diff * trade.position_size

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        initial_balance: float,
        final_balance: float,
        equity_curve: List[float],
        drawdown_curve: List[float],
        df: pd.DataFrame,
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return BacktestResult(
                start_date=df.index[0] if hasattr(df.index, "__getitem__") else datetime.utcnow(),
                end_date=df.index[-1] if hasattr(df.index, "__getitem__") else datetime.utcnow(),
                initial_balance=initial_balance,
                final_balance=final_balance,
                total_return=0.0,
                net_profit=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                recovery_factor=0.0,
                expectancy=0.0,
                avg_r_multiple=0.0,
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                trades=[],
                monthly_returns={},
                weekly_returns={},
            )

        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]

        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        net_profit = final_balance - initial_balance

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        if downside_returns and np.std(downside_returns) > 0:
            sortino = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252)
        else:
            sortino = 0.0

        # Max drawdown
        max_dd = max(drawdown_curve) if drawdown_curve else 0
        max_dd_percent = max_dd

        # Calmar ratio
        calmar = (total_return / 100) / (max_dd_percent / 100) if max_dd_percent > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if trades else 0

        # R-multiple
        r_multiples = []
        for t in trades:
            risk = abs(t.entry_price - t.stop_loss) * t.position_size
            if risk > 0:
                r_multiples.append(t.pnl / risk)
        avg_r = np.mean(r_multiples) if r_multiples else 0

        return BacktestResult(
            start_date=df.index[0] if hasattr(df.index, "__getitem__") else datetime.utcnow(),
            end_date=df.index[-1] if hasattr(df.index, "__getitem__") else datetime.utcnow(),
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            net_profit=net_profit,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_percent,
            recovery_factor=net_profit / max_dd if max_dd > 0 else 0,
            expectancy=expectancy,
            avg_r_multiple=avg_r,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
            monthly_returns={},
            weekly_returns={},
        )

    def monte_carlo_simulation(
        self,
        backtest_result: BacktestResult,
        runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation on backtest results."""
        runs = runs or self.config.monte_carlo_runs
        trades = backtest_result.trades

        if not trades:
            return {"error": "No trades for Monte Carlo simulation"}

        simulations = []

        for _ in range(runs):
            # Randomize trade sequence
            shuffled = random.sample(trades, len(trades))

            balance = backtest_result.initial_balance
            equity = [balance]

            for trade in shuffled:
                # Add random slippage variation
                slippage_var = random.uniform(-0.5, 0.5) * self.config.slippage_percent
                pnl_adjusted = trade.pnl * (1 + slippage_var / 100)
                balance += pnl_adjusted
                equity.append(balance)

            simulations.append({
                "final_balance": balance,
                "return": (balance - backtest_result.initial_balance) / backtest_result.initial_balance,
                "max_drawdown": self._calculate_max_drawdown(equity),
            })

        returns = [s["return"] for s in simulations]
        drawdowns = [s["max_drawdown"] for s in simulations]

        confidence = self.config.monte_carlo_confidence

        return {
            "runs": runs,
            "confidence_level": confidence,
            "avg_return": np.mean(returns),
            "median_return": np.median(returns),
            "worst_case_return": np.percentile(returns, (1 - confidence) * 100),
            "best_case_return": np.percentile(returns, confidence * 100),
            "avg_max_drawdown": np.mean(drawdowns),
            "worst_max_drawdown": np.max(drawdowns),
            "probability_of_profit": sum(1 for r in returns if r > 0) / len(returns),
        }

    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        """Calculate max drawdown from equity curve."""
        peak = equity[0]
        max_dd = 0

        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def walk_forward_test(
        self,
        df: pd.DataFrame,
        signal_generator: callable,
        risk_engine: Any,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> List[BacktestResult]:
        """Run walk-forward analysis."""
        train_size = train_size or self.config.walk_forward_train_size
        test_size = test_size or self.config.walk_forward_test_size

        results = []
        total_len = len(df)
        window_size = int(total_len * (train_size + test_size))
        step_size = int(total_len * test_size)

        start = 0
        while start + window_size < total_len:
            train_end = start + int(window_size * train_size / (train_size + test_size))
            test_end = start + window_size

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]

            # Train on train_df, test on test_df
            # (Simplified - would integrate with actual ML training)
            result = self.run_backtest(test_df, signal_generator, risk_engine)
            results.append(result)

            start += step_size

        return results
