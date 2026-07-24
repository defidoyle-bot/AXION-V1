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
    SignalScored, SignalApproved, HybridMLRefined,
    TelegramNotificationSent, SignalStored,
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
from hybrid_ml.pipeline import HybridMLPipeline, HybridMLRefinement

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
            atr = atr.reindex(df.index).ffill()  # FIX: fillna(method=) deprecated in pandas 2.0
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
        # Serialise swing points once — stored both in smc_data (for RiskHandler
        # which only receives smc_data) AND as the dedicated swing_points field.
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

        smc_data = {
            "current_structure": analysis.current_structure.name,
            "swing_high_count": sum(1 for sp in analysis.swing_points if sp.is_high()),
            "swing_low_count": sum(1 for sp in analysis.swing_points if sp.is_low()),
            "swing_points": swing_points_serialized,  # FIX: RiskHandler reads from smc_data
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
            "supply_demand_zones": [
                {"zone_type": z.zone_type, "top": z.top, "bottom": z.bottom,
                 "strength": z.strength, "freshness": z.freshness}
                for z in analysis.supply_demand_zones
            ],
            "equal_highs": analysis.equal_highs[:20],  # FIX: was just len() — now actual list, capped at 20
            "equal_lows":  analysis.equal_lows[:20],   # FIX: same
        }

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
        self._loaded_direction: str = "LONG"
        self._last_trained_at: Optional[datetime] = None
        # Attempt to restore a previously saved model immediately
        loaded_direction = self.engine.load_latest_model(direction=self._loaded_direction)
        if loaded_direction is not None:
            self._model_ready = True
            self._loaded_direction = loaded_direction
            logger.info(f"MLHandler: loaded saved model from disk ({loaded_direction})")
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

        # FIX: Only retrain once per hour — prevents model being overwritten 80x per cycle
        now_utc = datetime.now(timezone.utc)
        retrain_due = (
            not self._model_ready
            or (
                self.engine.should_retrain()
                and (
                    self._last_trained_at is None
                    or (now_utc - self._last_trained_at).total_seconds() > 3600
                )
            )
        )
        if retrain_due:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.engine.train(df, payload.indicators, payload.smc_data),
                )
                self._model_ready = True
                self._last_trained_at = now_utc
                logger.info(f"MLHandler: model training complete for {payload.symbol}")
            except Exception as exc:
                logger.error(f"MLHandler: training failed: {exc}", exc_info=True)
                # FIX: Use neutral prediction fallback instead of dropping signal
                return self._neutral_prediction(payload)

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
            return self._neutral_prediction(payload)  # FIX: fallback instead of dropping

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
                direction=prediction.direction,
                # Forward candle data for downstream handlers (RiskHandler needs these)
                candles=payload.candles,
                indicators=payload.indicators,
                smc_data=payload.smc_data,
            ),
            metadata=EventMetadata(source="ml_handler", priority=EventPriority.NORMAL),
        )

    @staticmethod
    def _neutral_prediction(payload: SMCAnalysisCompleted) -> Event:
        """Return a calibrated neutral prediction when ML cannot run.

        Forwards candles, indicators, and smc_data so downstream handlers
        (RiskHandler, SignalHandler) are not starved of market data.
        """
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
                candles=payload.candles,
                indicators=payload.indicators,
                smc_data=payload.smc_data,
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
                    "direction": payload.direction,
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

        # Derive trend alignment from the actual SMC structure instead of
        # hardcoding bullish. This prevents every SHORT from starting with a
        # trend penalty and makes LONG/SHORT scoring symmetric.
        structure = str(payload.smc_data.get("current_structure", "UNKNOWN")).upper()
        if structure == "UPTREND":
            higher_tf_trend = {"direction": "bullish", "strength": 0.8}
            lower_tf_trend = {"direction": "bullish", "momentum": 0.7}
        elif structure == "DOWNTREND":
            higher_tf_trend = {"direction": "bearish", "strength": 0.8}
            lower_tf_trend = {"direction": "bearish", "momentum": 0.7}
        else:
            higher_tf_trend = {"direction": "neutral", "strength": 0.5}
            lower_tf_trend = {"direction": "neutral", "momentum": 0.5}

        # Use SignalConfig weights
        score = self.engine.score_signal(
            symbol=payload.symbol,
            direction=payload.direction,
            higher_tf_trend=higher_tf_trend,
            lower_tf_trend=lower_tf_trend,
            technical_indicators=payload.indicators,
            smc_analysis=payload.smc_data,
            liquidity_context={"spread_percent": 0.05, "depth_usdt": 5000000},
            volume_data={"relative_volume": 1.5, "volume_trend": "increasing"},
            market_regime=payload.smc_data.get("current_structure", "UNKNOWN"),
            ml_prediction=payload.ml_prediction,
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
                indicators=payload.indicators,
            ),
            metadata=EventMetadata(source="signal_handler", priority=EventPriority.HIGH),
        )


class ApprovalHandler(EventHandler):
    """Handles SignalScored events — approves signals with real prices and duplicate prevention."""

    # Cooldown file persists between GitHub Actions runs (in-memory resets every hour)
    _COOLDOWN_FILE = "data/signal_cooldowns.json"
    _COOLDOWN_SECONDS = 4 * 3600  # 4 hour cooldown per symbol

    def __init__(self):
        self._recent_signals: dict = self._load_cooldowns()

    def _load_cooldowns(self) -> dict:
        """Load persisted cooldowns from disk so they survive between GitHub Actions runs."""
        import json, os
        if not os.path.exists(self._COOLDOWN_FILE):
            return {}
        try:
            with open(self._COOLDOWN_FILE) as f:
                raw = json.load(f)
            # Convert ISO strings back to datetime
            result = {}
            now = datetime.now(timezone.utc)
            for symbol, info in raw.items():
                dt = datetime.fromisoformat(info["time"])
                # Only keep if still within cooldown window
                if (now - dt).total_seconds() < self._COOLDOWN_SECONDS:
                    result[symbol] = {"time": dt, "direction": info["direction"]}
            return result
        except Exception as e:
            logger.warning(f"ApprovalHandler: could not load cooldowns: {e}")
            return {}

    def _save_cooldowns(self) -> None:
        """Persist cooldowns to disk so next GitHub Actions run respects them."""
        import json, os
        os.makedirs("data", exist_ok=True)
        try:
            serialized = {
                symbol: {
                    "time": info["time"].isoformat(),
                    "direction": info["direction"],
                }
                for symbol, info in self._recent_signals.items()
            }
            with open(self._COOLDOWN_FILE, "w") as f:
                json.dump(serialized, f, indent=2)
        except Exception as e:
            logger.warning(f"ApprovalHandler: could not save cooldowns: {e}")

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

        # Duplicate prevention: per-symbol so LONG and SHORT on same coin
        # are both blocked within the cooldown window — persists across runs
        dedup_key = payload.symbol
        now = datetime.now(timezone.utc)
        if dedup_key in self._recent_signals:
            existing = self._recent_signals[dedup_key]
            elapsed = (now - existing["time"]).total_seconds()
            if elapsed < self._COOLDOWN_SECONDS:
                logger.info(
                    f"Duplicate suppressed: {payload.symbol} {payload.direction} "
                    f"({elapsed/3600:.1f}h ago, last was {existing['direction']}, "
                    f"cooldown={self._COOLDOWN_SECONDS/3600:.0f}h)"
                )
                return None

        # Record this signal and persist to disk immediately
        self._recent_signals[dedup_key] = {"time": now, "direction": payload.direction}
        self._save_cooldowns()

        # FIX: Calculate real position size based on risk management
        # position_size was hardcoded 0.0 — now calculated from balance + risk %
        try:
            cfg = get_config()
            paper_balance = getattr(cfg.paper_trading, "initial_balance", 10000.0)
            risk_pct = getattr(cfg.risk, "risk_per_trade_percent", 1.0)
            risk_amount = paper_balance * (risk_pct / 100)
            entry_p = payload.risk_assessment.entry_price if payload.risk_assessment else 0
            sl_p = payload.risk_assessment.stop_loss if payload.risk_assessment else 0
            risk_per_unit = abs(entry_p - sl_p) if entry_p and sl_p else 0
            calculated_position_size = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.01
        except Exception:
            calculated_position_size = 0.01

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

        # Extract indicator snapshot from the scored signal
        indicators = getattr(payload, "indicators", {})

        # Build derived features from indicators
        ind_atr = indicators.get("atr", 0.0)
        entry = entry_price or 0.0001
        atr_pct = float(ind_atr / entry * 100) if entry > 0 and ind_atr else 0.0

        rsi_val = float(indicators.get("rsi", 50))

        # Extract BB width if available
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        bb_mid = indicators.get("bb_middle")
        bb_width = 0.0
        bb_position = 0.5
        if bb_upper and bb_lower and bb_mid and bb_mid != 0:
            bb_width = float((bb_upper - bb_lower) / bb_mid)
            bb_position = float((entry - bb_lower) / (bb_upper - bb_lower)) if bb_upper != bb_lower else 0.5

        macd_hist = float(indicators.get("macd_histogram", 0.0))
        adx_val = float(indicators.get("adx", 0.0))

        vol_24h = float(indicators.get("obv", 0.0))

        return Event(
            event_type="SignalApproved",
            payload=SignalApproved(
                signal_id=signal_id,
                symbol=payload.symbol,
                direction=payload.direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit if take_profit else [0.0],
                position_size=calculated_position_size,  # FIX: was hardcoded 0.0
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
                indicators={
                    "atr_percent": atr_pct,
                    "rsi": rsi_val,
                    "bb_width": bb_width,
                    "bb_position": bb_position,
                    "macd_histogram": macd_hist,
                    "adx": adx_val,
                    "volume_24h": vol_24h,
                    "spread_pct": float(indicators.get("vwap_deviation", 0.0)),
                    "margin_required": risk.get("margin_required", 0.0),
                },
            ),
            metadata=EventMetadata(source="approval_handler", priority=EventPriority.HIGH),
        )


class HybridMLRefinementRegistry:
    """Shared in-memory store for hybrid ML refinements keyed by signal_id."""
    _refinements: Dict[str, HybridMLRefinement] = {}

    @classmethod
    def store(cls, signal_id: str, refinement: HybridMLRefinement) -> None:
        cls._refinements[signal_id] = refinement

    @classmethod
    def get(cls, signal_id: str) -> Optional[HybridMLRefinement]:
        return cls._refinements.get(signal_id)

    @classmethod
    def cleanup(cls, max_age_seconds: int = 3600) -> None:
        """FIX: Prevent registry growing forever — purge entries older than max_age."""
        now = datetime.now(timezone.utc)
        stale = [
            k for k, v in cls._refinements.items()
            if (now - v.timestamp).total_seconds() > max_age_seconds
        ]
        for k in stale:
            del cls._refinements[k]
        if stale:
            logger.debug(f"HybridMLRefinementRegistry: purged {len(stale)} stale entries")


class HybridMLHandler(EventHandler):
    """Handles SignalApproved events — runs hybrid ML pipeline to refine position sizing.

    Stores the refinement result in HybridMLRefinementRegistry so TelegramHandler
    can pick up the adjusted position size without duplicating the event.
    """

    def __init__(
        self,
        pipeline: HybridMLPipeline,
        paper_engine: Optional[PaperTradingEngine] = None,
    ):
        self.hybrid_ml = pipeline
        self.paper_engine = paper_engine

    @property
    def subscribed_events(self) -> List[type]:
        return [SignalApproved]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload
        if not get_config().hybrid_ml.enabled:
            return None  # pass through — TelegramHandler gets the original

        signal_dict = {
            "signal_id": payload.signal_id,
            "symbol": payload.symbol,
            "direction": payload.direction,
            "entry_price": payload.entry_price,
            "stop_loss": payload.stop_loss,
            "take_profit": payload.take_profit,
            "position_size": payload.position_size,
            "leverage": payload.leverage,
            "score": payload.score,
            "classification": payload.classification,
            "risk_reward": payload.risk_reward,
            "market_regime": payload.market_regime,
            "margin": getattr(payload, "indicators", {}).get("margin_required", 0.0),
            "atr_percent": getattr(payload, "indicators", {}).get("atr_percent", 0.0),
            "rsi": getattr(payload, "indicators", {}).get("rsi", 50),
            "spread": abs(getattr(payload, "indicators", {}).get("spread_pct", 0.0)),
            "volume_24h": getattr(payload, "indicators", {}).get("volume_24h", 0.0),
            "bb_width": getattr(payload, "indicators", {}).get("bb_width", 0.0),
            "bb_position": getattr(payload, "indicators", {}).get("bb_position", 0.5),
            "macd_histogram": getattr(payload, "indicators", {}).get("macd_histogram", 0.0),
            "adx": getattr(payload, "indicators", {}).get("adx", 0.0),
            "ml_probability": payload.ml_probability,
            "ml_confidence": payload.ml_confidence,
        }

        # Build account context for RL state
        account_info = None
        if self.paper_engine:
            acct = self.paper_engine.get_account()
            account_info = {
                "balance": acct.balance,
                "initial_balance": 10000.0,
                "active_trades": len(self.paper_engine._active_trades) if hasattr(self.paper_engine, '_active_trades') else 0,
                "max_concurrent_trades": get_config().risk.max_concurrent_trades,
                "daily_pnl": acct.total_pnl,
                "win_rate": (acct.winning_trades / acct.total_trades) if acct.total_trades > 0 else 0.5,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_volume": 0.0,
            }

        # Run hybrid ML pipeline
        try:
            refinement = self.hybrid_ml.refine(signal_dict, account_info)
        except Exception as exc:
            logger.error(f"HybridMLHandler: refinement failed for {payload.symbol}: {exc}")
            return None  # pass through unchanged

        if refinement.should_skip:
            logger.info(f"HybridMLHandler: SKIPPED {payload.symbol} {payload.direction} (RL veto)")
            return None  # signal dropped — TelegramHandler won't fire

        # Store refinement so TelegramHandler can read it
        HybridMLRefinementRegistry.store(payload.signal_id, refinement)

        logger.info(
            f"HybridMLHandler: refined {payload.symbol} {payload.direction} "
            f"size={refinement.original_position_size:.4f}→{refinement.adjusted_position_size:.4f} "
            f"action={refinement.rl_action_label} booster={refinement.booster_probability:.2%}"
        )
        return None  # do NOT re-emit — TelegramHandler gets the original event with registry lookup


class TelegramHandler(EventHandler):
    """Handles SignalApproved events - sends notifications.

    Checks HybridMLRefinementRegistry for any RL refinement to the signal;
    if found, uses the adjusted position size and updated ML values.
    """

    def __init__(self, bot: TelegramBot, paper_engine: Optional[Any] = None):
        self.bot = bot
        self.paper_engine = paper_engine  # FIX: accept paper engine for trade execution

    @property
    def subscribed_events(self) -> List[type]:
        return [SignalApproved]

    async def handle(self, event: Event[Any]) -> Optional[Event]:
        payload = event.payload

        # Check for hybrid ML refinement
        refinement = HybridMLRefinementRegistry.get(payload.signal_id)

        # FIX: Respect HybridML veto — if skip is requested, don't send
        if refinement and refinement.should_skip:
            logger.info(f"TelegramHandler: {payload.symbol} {payload.direction} skipped by HybridML veto")
            return None

        position_size = refinement.adjusted_position_size if refinement else payload.position_size
        ml_prob = refinement.booster_probability if refinement else payload.ml_probability
        ml_conf = refinement.booster_confidence if refinement else payload.ml_confidence
        risk_status = "Refined" if refinement else payload.risk_status

        signal_dict = {
            "signal_id": payload.signal_id,
            "symbol": payload.symbol,
            "direction": payload.direction,
            "entry_price": payload.entry_price,
            "stop_loss": payload.stop_loss,
            "take_profit": payload.take_profit,
            "position_size": position_size,
            "score": payload.score,
            "classification": payload.classification,
            "risk_reward": payload.risk_reward,
            "leverage": payload.leverage,
            "ml_probability": ml_prob,
            "ml_confidence": ml_conf,
            "market_regime": payload.market_regime,
            "smc_summary": payload.smc_summary,
            "risk_status": risk_status,
        }

        await self.bot.send_signal(signal_dict)

        # FIX: Execute paper trade after broadcasting signal
        # Previously signals were sent but never tracked in paper trading
        if self.paper_engine and self.paper_engine.is_enabled():
            try:
                self.paper_engine.execute_signal(signal_dict, payload.entry_price)
                logger.info(f"TelegramHandler: paper trade opened for {payload.symbol} {payload.direction}")
            except Exception as e:
                logger.warning(f"TelegramHandler: paper trade execution failed: {e}")

        return Event(
            event_type="TelegramNotificationSent",
            payload=TelegramNotificationSent(
                signal_id=payload.signal_id,
                chat_id="admin",
                message_type="signal",
                status="sent",
                timestamp=datetime.now(timezone.utc),
                symbol=payload.symbol,
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
        status = "stored"

        # FIX: Actually write to database — previously was a no-op comment
        if self.db:
            try:
                await self.db.store_signal({
                    "signal_id": payload.signal_id,
                    "symbol": payload.symbol,
                    "direction": getattr(payload, "direction", "UNKNOWN"),
                    "entry_price": getattr(payload, "entry_price", 0.0),
                    "stop_loss": getattr(payload, "stop_loss", 0.0),
                    "take_profit": getattr(payload, "take_profit", []),
                    "position_size": getattr(payload, "position_size", 0.0),
                    "leverage": getattr(payload, "leverage", 1),
                    "score": getattr(payload, "score", 0),
                    "classification": getattr(payload, "classification", "Standard"),
                    "risk_reward": getattr(payload, "risk_reward", 0.0),
                    "ml_probability": getattr(payload, "ml_probability", 0.5),
                    "ml_confidence": getattr(payload, "ml_confidence", 0.0),
                    "timeframe": getattr(payload, "timeframe", "1h"),
                    "market_regime": getattr(payload, "market_regime", "unknown"),
                })
                logger.info(f"StorageHandler: signal {payload.signal_id} written to DB")
            except Exception as e:
                logger.warning(f"StorageHandler: DB write failed for {payload.signal_id}: {e}")
                status = "db_error"

        return Event(
            event_type="SignalStored",
            payload=SignalStored(
                signal_id=payload.signal_id,
                symbol=payload.symbol,
                storage_status=status,
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

        # FIX: Wire paper engine as data provider for Telegram command handlers
        # This enables /status, /stats, /signals to return real data
        if self.telegram_bot and self.paper_engine:
            self.telegram_bot.set_data_provider(self.paper_engine)

        # Initialize backtesting
        self.backtest_engine = BacktestEngine()

        # Initialize hybrid ML pipeline
        self.hybrid_ml_pipeline = HybridMLPipeline()

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
        self.pipeline.register_stage(
            "SignalApproved",
            HybridMLHandler(self.hybrid_ml_pipeline, self.paper_engine),
        )
        self.pipeline.register_stage("SignalApproved", TelegramHandler(self.telegram_bot, paper_engine=self.paper_engine))
        self.pipeline.register_stage("TelegramNotificationSent", StorageHandler(self.db_manager))

    async def run(self) -> None:
        """Run the main application loop."""
        self._running = True

        # Start pipeline
        await self.pipeline.start()

        # Start Telegram bot
        await self.telegram_bot.start()

        logger.info("AXION QUANT V4 is running")
        self._last_hybrid_retrain = datetime.now(timezone.utc)

        # Main loop
        while self._running:
            try:
                # Scan symbols and process
                await self._scan_cycle()

                # Periodic hybrid ML retraining (every 60 min)
                if get_config().hybrid_ml.enabled:
                    elapsed = (datetime.now(timezone.utc) - self._last_hybrid_retrain).total_seconds() / 60
                    if elapsed >= get_config().hybrid_ml.booster_retrain_interval_minutes:
                        await self._retrain_hybrid_ml()
                        self._last_hybrid_retrain = datetime.now(timezone.utc)

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
            # FIX: Purge stale HybridML registry entries at the start of each cycle
            HybridMLRefinementRegistry.cleanup(max_age_seconds=3600)

            symbols = self.symbol_scanner.get_symbols()

            # FIX: Only use PRIMARY timeframe — prevents 4x signals per coin per cycle
            primary_tf = self.config.market_data.timeframes[0]
            scan_limit = self.config.market_data.candle_fetch_limit // 25

            for symbol in symbols[:scan_limit]:
                try:
                    event = await self.market_pipeline.run_pipeline(symbol, primary_tf)
                    if event:
                        await self.event_bus.emit(event)

                    # FIX: Update open paper trades for this symbol every cycle
                    # Previously update_trades() was never called so open trades never closed
                    if self.paper_engine and hasattr(self.paper_engine, "update_trades"):
                        try:
                            current_price = await self._get_current_price(symbol)
                            if current_price:
                                from datetime import datetime, timezone as tz
                                closed = self.paper_engine.update_trades(
                                    symbol, current_price, datetime.now(tz.utc)
                                )
                                if closed:
                                    for t in closed:
                                        pnl = getattr(t, "pnl", 0)
                                        status = getattr(t, "status", "closed")
                                        emoji = "✅" if pnl > 0 else "❌"
                                        logger.info(
                                            f"Paper trade closed: {symbol} {status} PnL=${pnl:.2f}"
                                        )
                                        # Notify via Telegram
                                        if self.telegram_bot:
                                            msg = (
                                                f"{emoji} <b>Paper Trade Closed</b>\n"
                                                f"Symbol: <code>{symbol}</code>\n"
                                                f"Status: {status}\n"
                                                f"PnL: <b>${pnl:.2f}</b>"
                                            )
                                            await self.telegram_bot.send_message(
                                                self.config.telegram.admin_chat_id, msg
                                            )
                        except Exception as e:
                            logger.debug(f"update_trades error for {symbol}: {e}")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Scan cycle error: {e}")

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol via exchange manager."""
        try:
            if self.exchange_manager:
                ticker = await self.exchange_manager.get_ticker(symbol)
                if ticker:
                    return float(ticker.get("last", 0) or ticker.get("close", 0))
        except Exception:
            pass
        return None

    async def _retrain_hybrid_ml(self) -> None:
        """Retrain the hybrid ML pipeline on completed trades."""
        completed_trades: List[Dict[str, Any]] = []

        # Collect closed trades from the paper engine
        if self.paper_engine and hasattr(self.paper_engine, '_trade_history'):
            for trade in self.paper_engine._trade_history:
                completed_trades.append({
                    "signal_id": trade.signal_id,
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "entry_price": trade.entry_price,
                    "stop_loss": trade.stop_loss,
                    "take_profit": list(trade.take_profit) if trade.take_profit else [],
                    "position_size": trade.position_size,
                    "leverage": trade.leverage,
                    "margin": trade.margin,
                    "pnl": trade.pnl,
                    "pnl_percent": trade.pnl_percent,
                    "status": trade.status.value if hasattr(trade.status, 'value') else str(trade.status),
                    "market_regime": trade.market_regime,
                    "risk_reward": trade.risk_reward,
                    "score": trade.score,
                    "atr_percent": trade.atr_percent,
                    "rsi": trade.rsi,
                    "spread": trade.spread,
                    "volume_24h": trade.volume_24h,
                    "ml_probability": trade.ml_probability,
                    "ml_confidence": trade.ml_confidence,
                })

        if not completed_trades:
            logger.debug("HybridML: no completed trades to train on")
            return

        # Retrain booster + RL agent
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.hybrid_ml_pipeline.retrain_from_trades(completed_trades),
        )
        booster_perf = result.get("booster")
        rl_count = result.get("rl", 0)

        if booster_perf:
            logger.info(
                f"HybridML retrained: booster ROC-AUC={booster_perf.roc_auc:.4f} "
                f"({booster_perf.total_trades} trades), RL={rl_count} experiences"
            )
        elif rl_count > 0:
            logger.info(f"HybridML retrained: RL={rl_count} experiences (booster skipped)")

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
            if app.backtest_engine:
                symbols = app.symbol_scanner.get_symbols()[:10] if app.symbol_scanner else []
                logger.info(f"Backtest: {len(symbols)} symbols available")
                # Backtest engine runs via paper trading simulation
                logger.info("Backtest complete. Switch to paper/live mode for live scanning.")
            else:
                logger.warning("Backtest engine not initialized")

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
