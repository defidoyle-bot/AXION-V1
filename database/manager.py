"""
AXION QUANT V4 - Database Layer
SQLite with SQLAlchemy ORM, abstraction for future PostgreSQL migration.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Boolean, Text, JSON,
    UniqueConstraint, create_engine, event, select, desc, func,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DatabaseConfig, get_config
from core.logging import get_logger

logger = get_logger("database")

Base = declarative_base()


# =============================================================================
# DATABASE MODELS
# =============================================================================

class MarketData(Base):
    """Historical market data storage."""
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float, default=0.0)
    trades = Column(Integer, default=0)
    mark_price = Column(Float, nullable=True)
    funding_rate = Column(Float, nullable=True)
    open_interest = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Unique constraint prevents duplicate candles per symbol/timeframe/timestamp
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uix_market_data"),
    )


class Signal(Base):
    """Trading signal records."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(JSON, default=list)
    position_size = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    score = Column(Integer, nullable=False)
    classification = Column(String(50), nullable=False)
    risk_reward = Column(Float, nullable=False)
    score_breakdown = Column(JSON, default=dict)
    market_regime = Column(String(50), default="unknown")
    ml_probability = Column(Float, default=0.5)
    ml_confidence = Column(Float, default=0.0)
    smc_summary = Column(Text, default="")
    risk_status = Column(String(20), default="pending")
    timeframe = Column(String(10), nullable=False)
    strategy_profile = Column(String(20), default="intraday")
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trade_id = Column(String(50), nullable=True)


class Trade(Base):
    """Trade execution records."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    signal_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(JSON, default=list)
    position_size = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    margin = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    funding_cost = Column(Float, default=0.0)
    mfe = Column(Float, default=0.0)
    mae = Column(Float, default=0.0)
    status = Column(String(20), default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
    opened_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)


class MLPrediction(Base):
    """ML prediction records."""
    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    feature_importance = Column(JSON, default=dict)
    prediction_explanation = Column(Text, default="")
    market_regime = Column(String(50), default="unknown")
    actual_outcome = Column(Boolean, nullable=True)
    outcome_timestamp = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceMetric(Base):
    """Performance tracking metrics."""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_date = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly
    total_return = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    recovery_factor = Column(Float, default=0.0)
    expectancy = Column(Float, default=0.0)
    avg_r_multiple = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """System event logs."""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    level = Column(String(10), nullable=False)
    logger = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    event_data = Column(JSON, nullable=True)
    correlation_id = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConfigurationSnapshot(Base):
    """Configuration snapshots for reproducibility."""
    __tablename__ = "configuration_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String(50), unique=True, nullable=False)
    config_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class HealthMetric(Base):
    """System health metrics."""
    __tablename__ = "health_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    component = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    latency_ms = Column(Float, nullable=True)
    error_count = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_config().database
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database connection."""
        if self._initialized:
            return

        # Convert SQLite URL to async if needed
        db_url = self.config.url
        if db_url.startswith("sqlite:///"):
            db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

        self.engine = create_async_engine(
            db_url,
            echo=self.config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_recycle=self.config.pool_recycle_seconds,
        )

        self.SessionLocal = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        self._initialized = True
        logger.info(f"Database initialized: {db_url}")

    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            self.initialize()

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created")

    async def get_session(self) -> AsyncSession:
        """Get a database session."""
        if not self.SessionLocal:
            self.initialize()
        return self.SessionLocal()

    async def close(self) -> None:
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

    # =================================================================
    # MARKET DATA OPERATIONS
    # =================================================================

    async def store_market_data(self, data: List[Dict[str, Any]]) -> None:
        """Store market data candles, silently skipping duplicates.

        The uix_market_data unique constraint (symbol, timeframe, timestamp) prevents
        duplicate rows.  IntegrityErrors from duplicate inserts are caught per-candle so
        that a single duplicate does not abort the entire batch.
        """
        from sqlalchemy.exc import IntegrityError

        inserted = 0
        skipped = 0
        async with self.get_session() as session:
            for item in data:
                candle = MarketData(
                    symbol=item["symbol"],
                    timeframe=item["timeframe"],
                    timestamp=datetime.fromisoformat(item["timestamp"]) if isinstance(item["timestamp"], str) else item["timestamp"],
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item["volume"],
                    quote_volume=item.get("quote_volume", 0),
                    trades=item.get("trades", 0),
                    mark_price=item.get("mark_price"),
                    funding_rate=item.get("funding_rate"),
                    open_interest=item.get("open_interest"),
                )
                try:
                    session.add(candle)
                    await session.flush()  # detect constraint violation immediately
                    inserted += 1
                except IntegrityError:
                    await session.rollback()
                    skipped += 1
            await session.commit()

        if skipped:
            logger.debug(
                f"store_market_data: inserted={inserted} skipped_duplicates={skipped} "
                f"(symbol={data[0].get('symbol') if data else 'unknown'})"
            )

    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Retrieve market data for a symbol."""
        async with self.get_session() as session:
            result = await session.execute(
                select(MarketData)
                .where(MarketData.symbol == symbol)
                .where(MarketData.timeframe == timeframe)
                .order_by(desc(MarketData.timestamp))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "symbol": r.symbol,
                    "timestamp": r.timestamp.isoformat(),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]

    # =================================================================
    # SIGNAL OPERATIONS
    # =================================================================

    async def store_signal(self, signal: Dict[str, Any]) -> str:
        """Store a trading signal."""
        async with self.get_session() as session:
            db_signal = Signal(
                signal_id=signal.get("signal_id", str(datetime.utcnow().timestamp())),
                symbol=signal["symbol"],
                direction=signal["direction"],
                entry_price=signal["entry_price"],
                stop_loss=signal["stop_loss"],
                take_profit=signal.get("take_profit", []),
                position_size=signal["position_size"],
                leverage=signal.get("leverage", 1),
                score=signal["score"],
                classification=signal["classification"],
                risk_reward=signal["risk_reward"],
                score_breakdown=signal.get("score_breakdown", {}),
                market_regime=signal.get("market_regime", "unknown"),
                ml_probability=signal.get("ml_probability", 0.5),
                ml_confidence=signal.get("ml_confidence", 0.0),
                smc_summary=signal.get("smc_summary", ""),
                risk_status=signal.get("risk_status", "pending"),
                timeframe=signal.get("timeframe", "1h"),
                strategy_profile=signal.get("strategy_profile", "intraday"),
            )
            session.add(db_signal)
            await session.commit()
            return db_signal.signal_id

    async def get_signals(
        self,
        symbol: Optional[str] = None,
        classification: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve signals with optional filters."""
        async with self.get_session() as session:
            query = select(Signal).order_by(desc(Signal.created_at)).limit(limit)

            if symbol:
                query = query.where(Signal.symbol == symbol)
            if classification:
                query = query.where(Signal.classification == classification)

            result = await session.execute(query)
            rows = result.scalars().all()

            return [
                {
                    "signal_id": r.signal_id,
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "score": r.score,
                    "classification": r.classification,
                    "entry_price": r.entry_price,
                    "stop_loss": r.stop_loss,
                    "take_profit": r.take_profit,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # =================================================================
    # TRADE OPERATIONS
    # =================================================================

    async def store_trade(self, trade: Dict[str, Any]) -> str:
        """Store a trade record."""
        async with self.get_session() as session:
            db_trade = Trade(
                trade_id=trade.get("trade_id", str(datetime.utcnow().timestamp())),
                signal_id=trade.get("signal_id", ""),
                symbol=trade["symbol"],
                direction=trade["direction"],
                entry_price=trade["entry_price"],
                exit_price=trade.get("exit_price"),
                stop_loss=trade["stop_loss"],
                take_profit=trade.get("take_profit", []),
                position_size=trade["position_size"],
                leverage=trade.get("leverage", 1),
                margin=trade.get("margin", 0),
                pnl=trade.get("pnl", 0),
                pnl_percent=trade.get("pnl_percent", 0),
                fees=trade.get("fees", 0),
                funding_cost=trade.get("funding_cost", 0),
                status=trade.get("status", "open"),
            )
            session.add(db_trade)
            await session.commit()
            return db_trade.trade_id

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing trade."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.trade_id == trade_id)
            )
            trade = result.scalar_one_or_none()

            if trade:
                for key, value in updates.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                await session.commit()

    async def get_trade_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        async with self.get_session() as session:
            # Total trades
            total_result = await session.execute(select(func.count(Trade.id)))
            total_trades = total_result.scalar()

            # Winning trades
            win_result = await session.execute(
                select(func.count(Trade.id)).where(Trade.pnl > 0)
            )
            winning_trades = win_result.scalar()

            # Total P&L
            pnl_result = await session.execute(select(func.sum(Trade.pnl)))
            total_pnl = pnl_result.scalar() or 0

            # Average P&L
            avg_pnl_result = await session.execute(select(func.avg(Trade.pnl)))
            avg_pnl = avg_pnl_result.scalar() or 0

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades if total_trades else 0,
                "win_rate": (winning_trades / total_trades * 100) if total_trades else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
            }

    # =================================================================
    # ML PREDICTION OPERATIONS
    # =================================================================

    async def store_ml_prediction(self, prediction: Dict[str, Any]) -> None:
        """Store an ML prediction."""
        async with self.get_session() as session:
            db_pred = MLPrediction(
                symbol=prediction["symbol"],
                timeframe=prediction["timeframe"],
                probability=prediction["probability_of_success"],
                confidence=prediction["confidence"],
                model_used=prediction["model_used"],
                model_version=prediction["model_version"],
                feature_importance=prediction.get("feature_importance", {}),
                prediction_explanation=prediction["prediction_explanation"],
                market_regime=prediction.get("market_regime", "unknown"),
            )
            session.add(db_pred)
            await session.commit()

    # =================================================================
    # PERFORMANCE METRICS
    # =================================================================

    async def store_performance_metric(self, metric: Dict[str, Any]) -> None:
        """Store performance metrics."""
        async with self.get_session() as session:
            db_metric = PerformanceMetric(
                metric_date=metric.get("date", datetime.utcnow()),
                period=metric.get("period", "daily"),
                total_return=metric.get("total_return", 0),
                net_profit=metric.get("net_profit", 0),
                win_rate=metric.get("win_rate", 0),
                avg_win=metric.get("avg_win", 0),
                avg_loss=metric.get("avg_loss", 0),
                profit_factor=metric.get("profit_factor", 0),
                sharpe_ratio=metric.get("sharpe_ratio", 0),
                sortino_ratio=metric.get("sortino_ratio", 0),
                max_drawdown=metric.get("max_drawdown", 0),
                total_trades=metric.get("total_trades", 0),
                winning_trades=metric.get("winning_trades", 0),
                losing_trades=metric.get("losing_trades", 0),
            )
            session.add(db_metric)
            await session.commit()

    # =================================================================
    # SYSTEM LOGS
    # =================================================================

    async def store_system_log(self, log_entry: Dict[str, Any]) -> None:
        """Store a system log entry."""
        async with self.get_session() as session:
            db_log = SystemLog(
                timestamp=log_entry.get("timestamp", datetime.utcnow()),
                level=log_entry.get("level", "INFO"),
                logger=log_entry.get("logger", "unknown"),
                message=log_entry.get("message", ""),
                event_data=log_entry.get("event_data"),
                correlation_id=log_entry.get("correlation_id"),
            )
            session.add(db_log)
            await session.commit()

    # =================================================================
    # HEALTH METRICS
    # =================================================================

    async def store_health_metric(self, metric: Dict[str, Any]) -> None:
        """Store a health metric."""
        async with self.get_session() as session:
            db_health = HealthMetric(
                timestamp=metric.get("timestamp", datetime.utcnow()),
                component=metric["component"],
                status=metric["status"],
                latency_ms=metric.get("latency_ms"),
                error_count=metric.get("error_count", 0),
                metadata=metric.get("metadata"),
            )
            session.add(db_health)
            await session.commit()

    # =================================================================
    # CONFIGURATION SNAPSHOTS
    # =================================================================

    async def store_config_snapshot(self, config_data: Dict[str, Any]) -> str:
        """Store a configuration snapshot."""
        snapshot_id = f"config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        async with self.get_session() as session:
            db_snapshot = ConfigurationSnapshot(
                snapshot_id=snapshot_id,
                config_data=config_data,
            )
            session.add(db_snapshot)
            await session.commit()

        return snapshot_id
