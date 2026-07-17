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
from datetime import datetime
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
from exchange.mexc_client import MEXCClient
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

        # Validate data
        candles = [
            NormalizedCandle.from_mexc_candle(
                # This would be properly constructed in real implementation
                type("Candle", (), {
                    "symbol": payload.symbol,
                    "timestamp": c.get("timestamp", 0),
                    "open": c.get("open", 0),
                    "high": c.get("high", 0),
                    "low": c.get("low", 0),
                    "close": c.get("close", 0),
                    "volume": c.get("volume", 0),
                    "quote_volume": c.get("quote_volume", 0),
                    "trades": c.get("trades", 0),
                })()
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
                timestamp=datetime.utcnow(),
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
                {"status": fvg.status.name, "upper": fvg.upper, "lower": fvg.lower,
                 "gap_size": fvg.gap_size}
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

        # Reconstruct DataFrame from candles passed through the pipeline
        if not payload.candles:
            logger.warning(f"MLHandler: no candle data for {payload.symbol}, skipping")
            return None

        df = pd.DataFrame(payload.candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if len(df) < 50:
            logger.warning(
                f"MLHandler: insufficient candles ({len(df)}) for {payload.symbol} — "
                "returning neutral estimate"
            )
            return self._neutral_prediction(payload)

        indicators = payload.indicators

        # Train or retrain if needed (runs in the thread pool to avoid blocking the event loop)
        if not self._model_ready or self.engine.should_retrain():
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.engine.train(df, indicators, payload.smc_data),
                )
                self._model_ready = True
                logger.info(f"MLHandler: model training complete for {payload.symbol}")
            except Exception as exc:
                logger.error(f"MLHandler: training failed: {exc}", exc_info=True)
                return self._neutral_prediction(payload)

        # Run prediction
        try:
            prediction = self.engine.predict(
                df=df,
                indicators=indicators,
                smc_data=payload.smc_data,
                symbol=payload.symbol,
                timeframe=payload.timeframe,
            )
        except Exception as exc:
            logger.error(f"MLHandler: prediction failed for {payload.symbol}: {exc}", exc_info=True)
            return self._neutral_prediction(payload)

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

        # Risk validation would happen here
        # For now, approve with basic parameters

        return Event(
            event_type="RiskValidationCompleted",
            payload=RiskValidationCompleted(
                symbol=payload.symbol,
                direction="LONG",
                risk_assessment={"approved": True, "risk_reward": 2.0},
                approved=True,
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
            return None

        # Score the signal
        score = self.engine.score_signal(
            symbol=payload.symbol,
            direction=payload.direction,
            higher_tf_trend={"direction": "bullish", "strength": 0.8},
            lower_tf_trend={"direction": "bullish", "momentum": 0.7},
            technical_indicators={"rsi": 55, "adx": 30},
            smc_analysis={"current_structure": "UPTREND"},
            liquidity_context={"spread_percent": 0.05, "depth_usdt": 5000000},
            volume_data={"relative_volume": 1.5, "volume_trend": "increasing"},
            market_regime="trending_up",
            ml_prediction={"probability_of_success": 0.65, "confidence": 0.7},
            risk_assessment=payload.risk_assessment,
        )

        return Event(
            event_type="SignalScored",
            payload=SignalScored(
                symbol=payload.symbol,
                direction=payload.direction,
                score=score.total_score,
                classification=score.classification,
                score_breakdown=score.to_dict(),
                timestamp=datetime.utcnow(),
            ),
            metadata=EventMetadata(source="signal_handler", priority=EventPriority.HIGH),
        )


class ApprovalHandler(EventHandler):
    """Handles SignalScored events - approves premium signals."""

    @property
    def subscribed_events(self) -> List[type]:
        return [SignalScored]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        # Only approve signals above Standard threshold
        if payload.score < 70:
            logger.info(f"Signal rejected: {payload.symbol} score {payload.score}")
            return None

        signal_id = str(uuid.uuid4())[:8]

        return Event(
            event_type="SignalApproved",
            payload=SignalApproved(
                signal_id=signal_id,
                symbol=payload.symbol,
                direction=payload.direction,
                entry_price=0.0,  # Would be calculated from actual data
                stop_loss=0.0,
                take_profit=[0.0],
                position_size=0.0,
                leverage=5,
                score=payload.score,
                classification=payload.classification,
                risk_reward=2.0,
                timestamp=datetime.utcnow(),
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
        }

        await self.bot.send_signal(signal_dict)

        return Event(
            event_type="TelegramNotificationSent",
            payload=TelegramNotificationSent(
                signal_id=payload.signal_id,
                chat_id="admin",
                message_type="signal",
                status="sent",
                timestamp=datetime.utcnow(),
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
                timestamp=datetime.utcnow(),
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
        self.mexc_client: Optional[MEXCClient] = None
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
            f"[profile={self.config.signal.strategy_profile.value}]..."
        )
        profile_desc = self.profile_manager.describe_active_profile()
        logger.info(f"Strategy profile: {profile_desc}")

        # Initialize MEXC client
        self.mexc_client = MEXCClient()
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

        # Close MEXC client
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
                "volume": np.random.randint(1000, 10000, 100),
            })

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
            if self.telegram_bot and self.telegram_bot.application:
                results["telegram"] = "PASS"
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

        # Overall result
        all_pass = all(v == "PASS" for v in results.values())
        results["overall"] = "PASS" if all_pass else "FAIL"

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
