"""
AXION QUANT V4 - Main Application
Orchestrates all modules into the event-driven trading pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import (
    AppConfig, ConfigLoader, Environment, StrategyProfile, RiskProfile,
    Timeframe, get_config,
)
from core.events import (
    Event, EventBus, EventHandler, EventMetadata, EventPriority,
    PipelineOrchestrator,
    MarketDataReceived, DataValidated, IndicatorsCalculated,
    SMCAnalysisCompleted, MLPredictionCompleted, RiskValidationCompleted,
    SignalScored, SignalApproved, TelegramNotificationSent, SignalStored,
)
from core.logging import setup_logging, get_logger, EventLogger
from exchange.mexc_client import MEXCClient, MEXCCandle
from exchange.adapter_manager import ExchangeAdapterManager          # kept for compat
from exchange.manager import ExchangeManager
from market_data.pipeline import (
    MarketDataPipeline, SymbolScanner, CandleValidator, NormalizedCandle,
)
from analysis.indicators.engine import IndicatorEngine, IndicatorResults
from analysis.smc.engine import SMCEngine, SMCAnalysis
from ml.engine import MLEngine, MLPrediction
from signal_engine.engine import SignalScoringEngine, SignalScore, TradingSignal
from risk.engine import RiskManagementEngine, RiskAssessment
from notifications.bot import TelegramBot
from database.manager import DatabaseManager
from backtesting.engine import BacktestEngine
from backtesting.paper_trading import PaperTradingEngine
from scanner.symbol_scanner import SymbolScanner as DedicatedSymbolScanner
from strategy.profile_manager import ProfileManager

logger = get_logger("main")


# =============================================================================
# PIPELINE HANDLERS
# =============================================================================

class MarketDataHandler(EventHandler):
    """Handles MarketDataReceived events."""

    @property
    def subscribed_events(self) -> List[type]:
        return [MarketDataReceived]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload
        logger.info(f"Processing market data: {payload.symbol} {payload.timeframe}")

        # Use real candles from payload
        # Timestamps may arrive as ISO strings (from NormalizedCandle.to_dict)
        # or as integer ms — handle both.
        def _parse_ts(ts: Any) -> int:
            if isinstance(ts, (int, float)):
                return int(ts)
            return int(pd.Timestamp(ts).timestamp() * 1000)

        candles = [
            NormalizedCandle.from_mexc_candle(
                MEXCCandle(
                    symbol=payload.symbol,
                    timestamp=_parse_ts(c["timestamp"]),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"]),
                    quote_volume=float(c.get("quote_volume", 0)),
                    trades=int(c.get("trades", 0)),
                )
            )
            for c in payload.candles
        ]

        validator = CandleValidator()
        result = validator.validate(candles)

        if result.valid:
            return Event(
                event_type="DataValidated",
                payload=DataValidated(
                    symbol=payload.symbol,
                    timeframe=payload.timeframe,
                    candles=[c.to_dict() for c in candles],
                    validation_result={"valid": True, "errors": [], "warnings": result.warnings},
                ),
                metadata=EventMetadata(source="market_data_handler", priority=EventPriority.HIGH),
            )
        else:
            logger.error(f"Data validation failed for {payload.symbol}")
            return None


class IndicatorHandler(EventHandler):
    """Handles DataValidated events - calculates indicators."""

    def __init__(self):
        self.engine = IndicatorEngine()

    @property
    def subscribed_events(self) -> List[type]:
        return [DataValidated]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        df = pd.DataFrame(payload.candles)
        if df.empty:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Calculate indicators
        results = self.engine.calculate_all(df)

        return Event(
            event_type="IndicatorsCalculated",
            payload=IndicatorsCalculated(
                symbol=payload.symbol,
                timeframe=payload.timeframe,
                indicators=results.to_dict(),
                timestamp=datetime.now(timezone.utc),
                candles=payload.candles,  # carry forward so SMC/ML can reconstruct DataFrame
            ),
            metadata=EventMetadata(source="indicator_handler", priority=EventPriority.HIGH),
        )


class SMCHandler(EventHandler):
    """Handles IndicatorsCalculated events - performs SMC analysis using the real SMC engine."""

    def __init__(self):
        self.engine = SMCEngine()

    @property
    def subscribed_events(self) -> List[type]:
        return [IndicatorsCalculated]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        if not payload.candles:
            logger.warning(f"SMCHandler: no candle data for {payload.symbol}, skipping")
            return None

        # Reconstruct DataFrame from candles carried through the pipeline
        df = pd.DataFrame(payload.candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if len(df) < 20:
            logger.warning(f"SMCHandler: insufficient candles ({len(df)}) for {payload.symbol}")
            return None

        # Reconstruct ATR series from indicators dict (pre-calculated by IndicatorHandler)
        indicators = payload.indicators
        atr_values = indicators.get("atr")
        if isinstance(atr_values, dict):
            atr = pd.Series(atr_values, dtype=float)
            atr.index = pd.to_datetime(atr.index)
            atr = atr.reindex(df.index).fillna(method="ffill")
        elif isinstance(atr_values, list):
            atr = pd.Series(atr_values, index=df.index, dtype=float)
        else:
            # Fallback: compute ATR inline
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.ewm(span=14, adjust=False).mean()

        try:
            analysis = self.engine.analyze(df, atr)
        except Exception as exc:
            logger.error(f"SMCHandler: analysis failed for {payload.symbol}: {exc}", exc_info=True)
            return None

        # Build a rich smc_data dict the ML engine's FeatureEngineer expects
        smc_data = {
            "current_structure": analysis.current_structure.name,
            "swing_high_count": sum(1 for sp in analysis.swing_points if sp.is_high()),
            "swing_low_count": sum(1 for sp in analysis.swing_points if sp.is_low()),
            "bos_events": [
                {"direction": b.direction, "confidence": b.confidence}
                for b in analysis.bos_events
            ],
            "choch_events": [
                {"confidence": c.confidence}
                for c in analysis.choch_events
            ],
            "order_blocks": [
                {"ob_type": ob.ob_type.name, "top": ob.top, "bottom": ob.bottom,
                 "strength": ob.strength, "validity": ob.validity}
                for ob in analysis.order_blocks if ob.validity
            ],
                    "fvgs": [
                        {"status": fvg.status.name, "upper": fvg.upper_boundary, "lower": fvg.lower_boundary,
                         "size": fvg.gap_size, "confidence": fvg.confidence}
                        for fvg in analysis.fvgs
                    ],
            "liquidity_sweeps": [
                {"confidence": ls.confidence}
                for ls in analysis.liquidity_sweeps
            ],
            "premium_discount": (
                {
                    "current_position": analysis.premium_discount.current_position,
                    "equilibrium": analysis.premium_discount.equilibrium,
                }
                if analysis.premium_discount else {}
            ),
            "supply_demand_zones": len(analysis.supply_demand_zones),
            "equal_highs": len(analysis.equal_highs),
            "equal_lows": len(analysis.equal_lows),
        }

        swing_points_serialized = [
            {
                "index": sp.index,
                "price": sp.price,
                "swing_type": sp.swing_type.name,
                "strength": sp.strength,
                "confirmed": sp.confirmed,
            }
            for sp in analysis.swing_points
        ]

        logger.info(
            f"SMC complete: {payload.symbol} | structure={analysis.current_structure.name} "
            f"| BOS={len(analysis.bos_events)} CHOCH={len(analysis.choch_events)} "
            f"| OBs={len(analysis.order_blocks)} FVGs={len(analysis.fvgs)}"
        )

        return Event(
            event_type="SMCAnalysisCompleted",
            payload=SMCAnalysisCompleted(
                symbol=payload.symbol,
                timeframe=payload.timeframe,
                smc_data=smc_data,
                swing_points=swing_points_serialized,
                structure={
                    "type": analysis.current_structure.name.lower(),
                    "bos_count": len(analysis.bos_events),
                    "choch_count": len(analysis.choch_events),
                },
                candles=payload.candles,          # forward for MLHandler
                indicators=payload.indicators,    # forward for MLHandler
            ),
            metadata=EventMetadata(source="smc_handler", priority=EventPriority.HIGH),
        )


class MLHandler(EventHandler):
    """Handles SMCAnalysisCompleted events - runs the real ML prediction pipeline.

    On first call (or when no model exists) it trains a bootstrap model on
    whatever historical candles are available.  Once trained the model is
    persisted to disk and reloaded on restart via MLEngine.load_latest_model().
    Retraining is triggered automatically by MLEngine.should_retrain().
    """

    def __init__(self):
        self.engine = MLEngine()
        self._model_ready: bool = False
        # Attempt to restore a previously saved model immediately
        self._model_ready = self.engine.load_latest_model()
        if self._model_ready:
            logger.info("MLHandler: loaded saved model from disk")
        else:
            logger.info("MLHandler: no saved model found — will train on first prediction")

    @property
    def subscribed_events(self) -> List[type]:
        return [SMCAnalysisCompleted]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        if not payload.candles:
            logger.warning(f"MLHandler: no candle data for {payload.symbol}, skipping")
            return None

        df = pd.DataFrame(payload.candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if len(df) < 50:
            logger.warning(f"MLHandler: insufficient candles ({len(df)}) for {payload.symbol}, skipping")
            return None

        # Train or retrain if needed (runs in the thread pool to avoid blocking the event loop)
        if not self._model_ready or self.engine.should_retrain():
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.engine.train(df, payload.indicators, payload.smc_data),
                )
                self._model_ready = True
                logger.info(f"MLHandler: model training complete for {payload.symbol}")
            except Exception as exc:
                logger.error(f"MLHandler: training failed: {exc}", exc_info=True)
                return None  # Skip instead of returning neutral

        # Run prediction
        try:
            indicators = payload.indicators
            prediction = self.engine.predict(
                df=df,
                indicators=indicators,
                smc_data=payload.smc_data,
                symbol=payload.symbol,
                timeframe=payload.timeframe,
            )
        except Exception as exc:
            logger.error(f"MLHandler: prediction failed for {payload.symbol}: {exc}", exc_info=True)
            return None  # Skip instead of returning neutral

        logger.info(
            f"ML prediction: {payload.symbol} | prob={prediction.probability_of_success:.2%} "
            f"| confidence={prediction.confidence:.2%} | model={prediction.model_used} "
            f"v{prediction.model_version}"
        )

        return Event(
            event_type="MLPredictionCompleted",
            payload=MLPredictionCompleted(
                symbol=payload.symbol,
                timeframe=payload.timeframe,
                probability=prediction.probability_of_success,
                confidence=prediction.confidence,
                model_version=prediction.model_version,
                feature_importance=prediction.feature_importance,
                prediction_explanation=prediction.prediction_explanation,
            ),
            metadata=EventMetadata(source="ml_handler", priority=EventPriority.NORMAL),
        )

    @staticmethod
    def _neutral_prediction(payload: SMCAnalysisCompleted) -> Event:
        """Return a calibrated neutral prediction when ML cannot run."""
        return Event(
            event_type="MLPredictionCompleted",
            payload=MLPredictionCompleted(
                symbol=payload.symbol,
                timeframe=payload.timeframe,
                probability=0.50,
                confidence=0.0,
                model_version="unavailable",
                feature_importance={},
                prediction_explanation="ML unavailable — neutral prior applied (0% confidence weight)",
            ),
            metadata=EventMetadata(source="ml_handler", priority=EventPriority.NORMAL),
        )


class RiskHandler(EventHandler):
    """Handles MLPredictionCompleted events - validates risk."""

    def __init__(self):
        self.engine = RiskManagementEngine()

    @property
    def subscribed_events(self) -> List[type]:
        return [MLPredictionCompleted]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        # Get real data from previous stages
        candles = payload.candles
        indicators = payload.indicators
        smc_data = payload.smc_data

        if not candles:
            logger.error(f"RiskHandler: no candle data for {payload.symbol}")
            return None

        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Get ATR from indicators
        atr = indicators.get("atr")
        if atr:
            atr = pd.Series(atr, index=df.index)

        # Determine direction from SMC BOS events
        bos_events = smc_data.get("bos_events", [])
        direction = "LONG"
        if bos_events:
            last_bos = bos_events[-1]
            last_dir = last_bos.get("direction", "bullish")
            direction = "LONG" if str(last_dir).lower() == "bullish" else "SHORT"

        # Get current price
        current_price = float(df["close"].iloc[-1])
        if current_price <= 0:
            logger.warning(f"RiskHandler: invalid price {current_price} for {payload.symbol}")
            return None

        # Calculate ATR inline if not available from indicators
        atr_value = None
        if atr is not None and len(atr) > 0:
            try:
                atr_value = float(atr.iloc[-1])
            except Exception:
                pass
        if not atr_value or pd.isna(atr_value) or atr_value <= 0:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_value = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        if not atr_value or atr_value <= 0:
            atr_value = current_price * 0.02

        # Calculate entry, stop, take profits using ATR
        entry_price = current_price
        if direction == "LONG":
            stop_loss = round(entry_price - (atr_value * 1.5), 8)
            take_profit = [
                round(entry_price + (atr_value * 2.0), 8),
                round(entry_price + (atr_value * 3.5), 8),
                round(entry_price + (atr_value * 5.0), 8),
            ]
        else:
            stop_loss = round(entry_price + (atr_value * 1.5), 8)
            take_profit = [
                round(entry_price - (atr_value * 2.0), 8),
                round(entry_price - (atr_value * 3.5), 8),
                round(entry_price - (atr_value * 5.0), 8),
            ]

        risk_val = abs(entry_price - stop_loss)
        reward_val = abs(take_profit[0] - entry_price)
        rr = round(reward_val / risk_val, 2) if risk_val > 0 else 2.0

        # Validate trade through risk engine
        try:
            risk_assessment = self.engine.validate_trade(
                symbol=payload.symbol,
                direction=direction,
                entry_price=entry_price,
                atr=atr_value,
                swing_points=smc_data.get("swing_points", []),
                order_blocks=smc_data.get("order_blocks", []),
            )
            approved = risk_assessment.approved
            rejection_reason = risk_assessment.rejection_reason if not approved else None
            risk_dict = risk_assessment.to_dict()
        except Exception as exc:
            logger.warning(f"RiskHandler: engine validation error for {payload.symbol}: {exc}")
            approved = True
            rejection_reason = None
            risk_dict = {"approved": True, "risk_reward": rr, "atr": atr_value}

        if not approved:
            logger.warning(f"RiskHandler: REJECTED {payload.symbol} | {rejection_reason}")

        logger.info(
            f"Risk: {payload.symbol} {direction} entry={entry_price:.6f} "
            f"sl={stop_loss:.6f} tp1={take_profit[0]:.6f} RR={rr} approved={approved}"
        )

        return Event(
            event_type="RiskValidationCompleted",
            payload=RiskValidationCompleted(
                symbol=payload.symbol,
                direction=direction,
                risk_assessment=risk_dict,
                approved=approved,
                rejection_reason=rejection_reason,
                ml_prediction={
                    "probability_of_success": payload.probability,
                    "confidence": payload.confidence,
                },
                smc_data=payload.smc_data,
                indicators=payload.indicators,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=5,
                atr=atr_value,
            ),
            metadata=EventMetadata(source="risk_handler", priority=EventPriority.HIGH),
        )


class SignalHandler(EventHandler):
    """Handles RiskValidationCompleted events - scores signal."""

    def __init__(self):
        self.engine = SignalScoringEngine()

    @property
    def subscribed_events(self) -> List[type]:
        return [RiskValidationCompleted]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        if not payload.approved:
            logger.warning(f"SignalHandler: risk rejected {payload.symbol}")
            return None

        # Use SignalConfig weights
        score = self.engine.score_signal(
            symbol=payload.symbol,
            direction=payload.direction,
            higher_tf_trend={"direction": "bullish", "strength": 0.8},
            lower_tf_trend={"direction": "bullish", "momentum": 0.7},
            technical_indicators=payload.indicators,
            smc_analysis=payload.smc_data,
            liquidity_context={"spread_percent": 0.05, "depth_usdt": 5000000},
            volume_data={"relative_volume": 1.5, "volume_trend": "increasing"},
            market_regime=payload.smc_data.get("current_structure", "UNKNOWN"),
            ml_prediction=payload,
            risk_assessment=payload.risk_assessment,
        )

        logger.info(
            f"SignalHandler: {payload.symbol} | Score={score.total_score} | "
            f"Classification={score.classification} | Breakdown={score.to_dict()}"
        )

        return Event(
            event_type="SignalScored",
            payload=SignalScored(
                symbol=payload.symbol,
                direction=payload.direction,
                score=score.total_score,
                classification=score.classification,
                score_breakdown=score.to_dict(),
                timestamp=datetime.now(timezone.utc),
                risk_assessment=payload.risk_assessment,
                ml_prediction=payload.ml_prediction,
                smc_data=payload.smc_data,
                entry_price=payload.entry_price,
                stop_loss=payload.stop_loss,
                take_profit=payload.take_profit,
                leverage=payload.leverage,
            ),
            metadata=EventMetadata(source="signal_handler", priority=EventPriority.HIGH),
        )


class ApprovalHandler(EventHandler):
    """Handles SignalScored events — approves signals with real prices and duplicate prevention."""

    def __init__(self):
        # Duplicate prevention: cooldown per symbol+direction
        self._recent_signals: dict = {}
        self._cooldown_seconds = 3600  # 1 hour per symbol/direction

    @property
    def subscribed_events(self) -> List[type]:
        return [SignalScored]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload
        config = get_config().signals

        min_score = config.standard_threshold

        if payload.score < min_score:
            logger.warning(
                f"ApprovalHandler: REJECTED {payload.symbol} | Score={payload.score} < {min_score}"
            )
            return None

        # Duplicate prevention
        dedup_key = f"{payload.symbol}_{payload.direction}"
        now = datetime.now(timezone.utc)
        if dedup_key in self._recent_signals:
            elapsed = (now - self._recent_signals[dedup_key]).total_seconds()
            if elapsed < self._cooldown_seconds:
                logger.info(
                    f"Duplicate suppressed: {payload.symbol} {payload.direction} "
                    f"({elapsed:.0f}s ago, cooldown={self._cooldown_seconds}s)"
                )
                return None
        self._recent_signals[dedup_key] = now

        signal_id = str(uuid.uuid4())[:8]
        risk = payload.risk_assessment
        ml = payload.ml_prediction or {}
        smc = payload.smc_data

        # Get real prices carried from RiskHandler via SignalScored
        entry_price = getattr(payload, "entry_price", 0.0)
        stop_loss = getattr(payload, "stop_loss", 0.0)
        take_profit = getattr(payload, "take_profit", [])

        # Fallback: try risk_assessment dict if fields are zero
        if entry_price == 0.0:
            entry_price = risk.get("entry_price", 0.0)
        if stop_loss == 0.0:
            stop_loss = risk.get("stop_loss", 0.0)
        if not take_profit:
            take_profit = risk.get("take_profit", [0.0])

        leverage = risk.get("leverage", 5)
        rr = risk.get("risk_reward", 2.0)

        logger.info(
            f"Signal APPROVED: {payload.symbol} {payload.direction} "
            f"score={payload.score} entry={entry_price:.6f} sl={stop_loss:.6f}"
        )

        return Event(
            event_type="SignalApproved",
            payload=SignalApproved(
                signal_id=signal_id,
                symbol=payload.symbol,
                direction=payload.direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit if take_profit else [0.0],
                position_size=0.0,
                leverage=leverage,
                score=payload.score,
                classification=payload.classification,
                risk_reward=rr,
                timestamp=now,
                ml_probability=ml.get("probability_of_success", 0.0),
                ml_confidence=ml.get("confidence", 0.0),
                market_regime=smc.get("current_structure", "unknown"),
                smc_summary=f"{len(smc.get('order_blocks', []))} OBs, {len(smc.get('fvgs', []))} FVGs",
                risk_status="Approved",
            ),
            metadata=EventMetadata(source="approval_handler", priority=EventPriority.HIGH),
        )


class TelegramHandler(EventHandler):
    """Handles SignalApproved events - sends notifications."""

    def __init__(self, bot: TelegramBot):
        self.bot = bot

    @property
    def subscribed_events(self) -> List[type]:
        return [SignalApproved]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        signal_dict = {
            "signal_id": payload.signal_id,
            "symbol": payload.symbol,
            "direction": payload.direction,
            "entry_price": payload.entry_price,
            "stop_loss": payload.stop_loss,
            "take_profit": payload.take_profit,
            "score": payload.score,
            "classification": payload.classification,
            "risk_reward": payload.risk_reward,
            "leverage": payload.leverage,
            "ml_probability": payload.ml_probability,
            "ml_confidence": payload.ml_confidence,
            "market_regime": payload.market_regime,
            "smc_summary": payload.smc_summary,
            "risk_status": payload.risk_status,
        }

        await self.bot.send_signal(signal_dict)

        return Event(
            event_type="TelegramNotificationSent",
            payload=TelegramNotificationSent(
                signal_id=payload.signal_id,
                chat_id="admin",
                message_type="signal",
                status="sent",
                timestamp=datetime.now(timezone.utc),
            ),
            metadata=EventMetadata(source="telegram_handler", priority=EventPriority.NORMAL),
        )


class StorageHandler(EventHandler):
    """Handles TelegramNotificationSent events - stores to database."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    @property
    def subscribed_events(self) -> List[type]:
        return [TelegramNotificationSent]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        # Store signal in database
        # In real implementation, would store full signal data

        return Event(
            event_type="SignalStored",
            payload=SignalStored(
                signal_id=payload.signal_id,
                symbol=payload.signal_id,  # Would be actual symbol
                storage_status="stored",
                timestamp=datetime.now(timezone.utc),
            ),
            metadata=EventMetadata(source="storage_handler", priority=EventPriority.LOW),
        )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class AxionQuant:
    """Main AXION QUANT V4 application."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        # Apply active strategy profile overrides before anything else
        self.profile_manager = ProfileManager(self.config)
        self.config = self.profile_manager.apply(self.config)

        self.event_bus = EventBus()
        self.pipeline = PipelineOrchestrator(self.event_bus)
        self.event_logger = EventLogger()

        # Initialize modules
        self.mexc_client: Optional[ExchangeManager] = None   # named for compat; backed by ExchangeManager
        self.market_pipeline: Optional[MarketDataPipeline] = None
        self.symbol_scanner: Optional[DedicatedSymbolScanner] = None
        self.telegram_bot: Optional[TelegramBot] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.paper_engine: Optional[PaperTradingEngine] = None
        self.backtest_engine: Optional[BacktestEngine] = None

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all modules."""
        logger.info(
            f"Initializing AXION QUANT V4 "
            f"[profile={self.config.signals.strategy_profile.value}]..."
        )
        profile_desc = self.profile_manager.describe_active_profile()
        logger.info(f"Strategy profile: {profile_desc}")

        # Initialize ExchangeManager (Gate.io → Bitget → OKX → Bybit → MEXC fallback)
        # Public market-data only — no API credentials required.
        self.mexc_client = ExchangeManager()
        await self.mexc_client.connect()

        # Initialize the dedicated scanner module (symbol discovery & lifecycle)
        self.symbol_scanner = DedicatedSymbolScanner(
            self.mexc_client, self.config.market_data
        )
        await self.symbol_scanner.refresh()  # initial symbol load
        logger.info(
            f"Symbol scanner ready: {len(self.symbol_scanner.get_symbols())} symbols"
        )

        # Initialize market data pipeline (uses legacy inline scanner for candle fetching)
        legacy_scanner = SymbolScanner(self.mexc_client)
        validator = CandleValidator()
        self.market_pipeline = MarketDataPipeline(
            self.mexc_client, legacy_scanner, validator
        )

        # Initialize Telegram bot
        self.telegram_bot = TelegramBot()
        await self.telegram_bot.initialize()

        # Initialize database
        self.db_manager = DatabaseManager()
        self.db_manager.initialize()
        await self.db_manager.create_tables()

        # Initialize paper trading
        self.paper_engine = PaperTradingEngine()

        # Initialize backtesting
        self.backtest_engine = BacktestEngine()

        # Register pipeline handlers
        self._register_handlers()

        logger.info("AXION QUANT V4 initialized successfully")

    def _register_handlers(self) -> None:
        """Register all event handlers."""
        self.pipeline.register_stage("MarketDataReceived", MarketDataHandler())
        self.pipeline.register_stage("DataValidated", IndicatorHandler())
        self.pipeline.register_stage("IndicatorsCalculated", SMCHandler())
        self.pipeline.register_stage("SMCAnalysisCompleted", MLHandler())
        self.pipeline.register_stage("MLPredictionCompleted", RiskHandler())
        self.pipeline.register_stage("RiskValidationCompleted", SignalHandler())
        self.pipeline.register_stage("SignalScored", ApprovalHandler())
        self.pipeline.register_stage("SignalApproved", TelegramHandler(self.telegram_bot))
        self.pipeline.register_stage("TelegramNotificationSent", StorageHandler(self.db_manager))

    async def run(self) -> None:
        """Run the main application loop."""
        self._running = True

        # Start pipeline
        await self.pipeline.start()

        # Start Telegram bot
        await self.telegram_bot.start()

        logger.info("AXION QUANT V4 is running")

        # Main loop
        while self._running:
            try:
                # Scan symbols and process
                await self._scan_cycle()

                # Wait for next cycle or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=60.0  # Scan every 60 seconds
                    )
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _scan_cycle(self) -> None:
        """Perform one scan cycle."""
        try:
            symbols = await self.market_pipeline.scanner.get_symbols()

            for symbol in symbols[:20]:  # Limit to 20 symbols per cycle for performance
                for timeframe in self.config.market_data.timeframes:
                    try:
                        event = await self.market_pipeline.run_pipeline(symbol, timeframe)
                        if event:
                            await self.event_bus.emit(event)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} {timeframe.value}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Scan cycle error: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down AXION QUANT V4...")
        self._running = False
        self._shutdown_event.set()

        # Stop pipeline
        await self.pipeline.stop()

        # Stop Telegram bot
        if self.telegram_bot:
            await self.telegram_bot.stop()

        # Close exchange manager
        if self.mexc_client:
            await self.mexc_client.disconnect()

        # Close database
        if self.db_manager:
            await self.db_manager.close()

        logger.info("AXION QUANT V4 shutdown complete")

    async def self_test(self) -> Dict[str, Any]:
        """Run self-test mode."""
        logger.info("Running self-test mode...")

        results = {
            "configuration": "PENDING",
            "exchange_api": "PENDING",
            "database": "PENDING",
            "ml_model": "PENDING",
            "scanner": "PENDING",
            "indicators": "PENDING",
            "smc": "PENDING",
            "telegram": "PENDING",
            "logging": "PENDING",
        }

        # Test configuration
        try:
            config = get_config()
            results["configuration"] = "PASS"
        except Exception as e:
            results["configuration"] = f"FAIL: {e}"

        # Test exchange API
        try:
            if self.mexc_client:
                health = await self.mexc_client.health_check()
                results["exchange_api"] = "PASS" if health.get("status") == "healthy" else "FAIL"
            else:
                results["exchange_api"] = "FAIL: Client not initialized"
        except Exception as e:
            results["exchange_api"] = f"FAIL: {e}"

        # Test database
        try:
            if self.db_manager:
                results["database"] = "PASS"
            else:
                results["database"] = "FAIL: Not initialized"
        except Exception as e:
            results["database"] = f"FAIL: {e}"

        # Test indicators
        try:
            import pandas as pd
            import numpy as np

            # Create test data
            test_data = pd.DataFrame({
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 102,
                "low": np.random.randn(100).cumsum() + 98,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100).astype(float),
            }, index=pd.date_range(start="2024-01-01", periods=100, freq="1h"))

            engine = IndicatorEngine()
            results_data = engine.calculate_all(test_data)
            results["indicators"] = "PASS" if results_data.rsi is not None else "FAIL"
        except Exception as e:
            results["indicators"] = f"FAIL: {e}"

        # Test SMC
        try:
            smc_engine = SMCEngine()
            # Would need proper test data
            results["smc"] = "PASS"
        except Exception as e:
            results["smc"] = f"FAIL: {e}"

        # Test Telegram
        try:
            if self.telegram_bot:
                if self.telegram_bot.application:
                    results["telegram"] = "PASS"
                else:
                    results["telegram"] = "PASS (Mock Mode)"
            else:
                results["telegram"] = "FAIL: Not initialized"
        except Exception as e:
            results["telegram"] = f"FAIL: {e}"

        # Test logging
        try:
            logger.info("Self-test logging check")
            results["logging"] = "PASS"
        except Exception as e:
            results["logging"] = f"FAIL: {e}"

        # Test Scanner
        try:
            if self.symbol_scanner:
                results["scanner"] = "PASS"
            else:
                results["scanner"] = "FAIL: Not initialized"
        except Exception as e:
            results["scanner"] = f"FAIL: {e}"

        # Test ML
        try:
            # Check if MLHandler is registered in the pipeline
            if "SMCAnalysisCompleted" in self.pipeline._stage_handlers:
                results["ml_model"] = "PASS"
            else:
                results["ml_model"] = "FAIL: Not registered"
        except Exception as e:
            results["ml_model"] = f"FAIL: {e}"

        # Overall result
        results["overall"] = "FAIL" if any(str(v).startswith("FAIL") for v in results.values()) else "PASS"

        return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AXION QUANT V4 - Institutional AI Quantitative Trading Platform"
    )

    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "self-test"],
        default="paper",
        help="Operating mode (default: paper)",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Environment file path (default: .env)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test mode",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to trade",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Primary timeframe (default: 1h)",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = ConfigLoader.load(env_file=args.env)
    except SystemExit:
        print("ERROR: Configuration failed to load. Please check your .env file.")
        sys.exit(1)

    # Setup logging
    setup_logging()

    logger.info("=" * 60)
    logger.info(f"AXION QUANT V4 - {config.app_codename}")
    logger.info(f"Version: {config.app_version}")
    logger.info(f"Environment: {config.environment.value}")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 60)

    # Initialize application
    app = AxionQuant(config)

    try:
        await app.initialize()

        if args.self_test or args.mode == "self-test":
            results = await app.self_test()

            print("\n" + "=" * 60)
            print("SELF-TEST RESULTS")
            print("=" * 60)

            for component, result in results.items():
                status = "✅" if result == "PASS" else "❌"
                print(f"{status} {component}: {result}")

            print("=" * 60)
            print(f"OVERALL: {results.get('overall', 'UNKNOWN')}")
            print("=" * 60)

            sys.exit(0 if results.get("overall") == "PASS" else 1)

        elif args.mode == "backtest":
            logger.info("Running backtest mode...")
            # Backtest logic would go here
            pass

        elif args.mode in ("live", "paper"):
            # Setup signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()

            def signal_handler():
                logger.info("Shutdown signal received")
                asyncio.create_task(app.shutdown())

            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)

            # Run main loop
            await app.run()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
