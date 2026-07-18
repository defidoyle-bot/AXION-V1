"""
AXION QUANT V4 - Event-Driven Core Architecture
Async event bus with typed events, handlers, and pipeline orchestration.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Type, TypeVar, Union

from core.logging import EventLogger, get_logger

logger = get_logger("events")


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventPriority(Enum):
    """Event processing priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class EventStatus(Enum):
    """Event lifecycle status."""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    DROPPED = auto()


@dataclass(frozen=True, slots=True)
class EventMetadata:
    """Event metadata for tracing and observability."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3


T = TypeVar('T')

@dataclass(frozen=True, slots=True)
class Event(Generic[T]):
    """Base event with typed payload."""
    event_type: str
    payload: T
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def with_retry(self) -> "Event[T]":
        """Create a new event with incremented retry count."""
        return Event(
            event_type=self.event_type,
            payload=self.payload,
            metadata=EventMetadata(
                correlation_id=self.metadata.correlation_id,
                timestamp=datetime.utcnow(),
                source=self.metadata.source,
                priority=self.metadata.priority,
                retry_count=self.metadata.retry_count + 1,
                max_retries=self.metadata.max_retries,
            )
        )


# =============================================================================
# TRADING EVENTS (Domain Events)
# =============================================================================

@dataclass(frozen=True, slots=True)
class MarketDataReceived:
    """Emitted when new market data is fetched from exchange."""
    symbol: str
    timeframe: str
    candles: List[Dict[str, Any]]
    timestamp: datetime
    source: str = "mexc"

@dataclass(frozen=True, slots=True)
class DataValidated:
    """Emitted when market data passes validation."""
    symbol: str
    timeframe: str
    candles: List[Dict[str, Any]]
    validation_result: Dict[str, Any]

@dataclass(frozen=True, slots=True)
class IndicatorsCalculated:
    """Emitted when technical indicators are computed."""
    symbol: str
    timeframe: str
    indicators: Dict[str, Any]
    timestamp: datetime
    candles: List[Dict[str, Any]] = field(default_factory=list)  # carried forward for SMC/ML

@dataclass(frozen=True, slots=True)
class SMCAnalysisCompleted:
    """Emitted when Smart Money Concepts analysis is complete."""
    symbol: str
    timeframe: str
    smc_data: Dict[str, Any]
    swing_points: List[Dict[str, Any]]
    structure: Dict[str, Any]
    candles: List[Dict[str, Any]] = field(default_factory=list)      # carried for ML
    indicators: Dict[str, Any] = field(default_factory=dict)         # carried for ML

@dataclass(frozen=True, slots=True)
class MLPredictionCompleted:
    """Emitted when ML model produces a prediction."""
    symbol: str
    timeframe: str
    probability: float
    confidence: float
    model_version: str
    feature_importance: Dict[str, float]
    prediction_explanation: str
    candles: List[Dict[str, Any]] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    smc_data: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class RiskValidationCompleted:
    """Emitted when risk management validates a potential trade."""
    symbol: str
    direction: str  # LONG or SHORT
    risk_assessment: Dict[str, Any]
    approved: bool
    rejection_reason: Optional[str] = None
    ml_prediction: Optional[Dict[str, Any]] = None
    smc_data: Dict[str, Any] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SignalScored:
    """Emitted when a signal is scored by the decision engine."""
    symbol: str
    direction: str
    score: int
    classification: str
    score_breakdown: Dict[str, float]
    timestamp: datetime
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    ml_prediction: Optional[Dict[str, Any]] = None
    smc_data: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SignalApproved:
    """Emitted when a signal passes all validation and is approved."""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    leverage: int
    score: int
    classification: str
    risk_reward: float
    timestamp: datetime
    ml_probability: float = 0.0
    ml_confidence: float = 0.0
    market_regime: str = "unknown"
    smc_summary: str = ""
    risk_status: str = ""

@dataclass(frozen=True, slots=True)
class TelegramNotificationSent:
    """Emitted when a notification is sent via Telegram."""
    signal_id: str
    chat_id: str
    message_type: str
    status: str
    timestamp: datetime

@dataclass(frozen=True, slots=True)
class SignalStored:
    """Emitted when a signal is persisted to the database."""
    signal_id: str
    symbol: str
    storage_status: str
    timestamp: datetime


# =============================================================================
# EVENT HANDLER INTERFACE
# =============================================================================

class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @property
    @abstractmethod
    def subscribed_events(self) -> List[Type[Any]]:
        """List of event types this handler subscribes to."""
        pass

    @abstractmethod
    async def handle(self, event: Event[Any]) -> Optional[Event[Any]]:
        """Process an event. May return a new event to emit."""
        pass

    async def on_error(self, event: Event[Any], error: Exception) -> None:
        """Handle processing errors. Override for custom error handling."""
        logger.error(
            f"Handler {self.__class__.__name__} failed on {event.event_type}",
            extra={
                "event_data": {
                    "correlation_id": event.metadata.correlation_id,
                    "error": str(error),
                    "retry_count": event.metadata.retry_count,
                }
            },
            exc_info=True,
        )


# =============================================================================
# EVENT BUS
# =============================================================================

class EventBus:
    """Async event bus with priority queues, retry logic, and backpressure."""

    def __init__(self, max_queue_size: int = 10000):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._queues: Dict[EventPriority, asyncio.PriorityQueue] = {
            priority: asyncio.PriorityQueue(maxsize=max_queue_size)
            for priority in EventPriority
        }
        self._event_logger = EventLogger()
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._worker_count = 4
        self._metrics: Dict[str, Dict[str, int]] = {
            "emitted": {},
            "processed": {},
            "failed": {},
            "dropped": {},
        }

    def register(self, handler: EventHandler) -> None:
        """Register an event handler."""
        for event_type in handler.subscribed_events:
            type_name = event_type.__name__ if hasattr(event_type, '__name__') else str(event_type)
            if type_name not in self._handlers:
                self._handlers[type_name] = []
            self._handlers[type_name].append(handler)
            logger.info(f"Registered handler {handler.__class__.__name__} for {type_name}")

    def unregister(self, handler: EventHandler) -> None:
        """Unregister an event handler."""
        for event_type in handler.subscribed_events:
            type_name = event_type.__name__ if hasattr(event_type, '__name__') else str(event_type)
            if type_name in self._handlers:
                self._handlers[type_name] = [
                    h for h in self._handlers[type_name] if h != handler
                ]

    async def emit(self, event: Event[Any]) -> None:
        """Emit an event to the bus."""
        self._metrics["emitted"][event.event_type] = self._metrics["emitted"].get(event.event_type, 0) + 1

        priority = event.metadata.priority
        queue = self._queues[priority]

        # Priority queue uses (priority_value, timestamp, event)
        try:
            queue.put_nowait((priority.value, time.time(), event))
            self._event_logger.log_event(
                event.event_type,
                {"correlation_id": event.metadata.correlation_id},
                event.metadata.correlation_id,
            )
        except asyncio.QueueFull:
            self._metrics["dropped"][event.event_type] = self._metrics["dropped"].get(event.event_type, 0) + 1
            logger.warning(f"Event dropped due to queue full: {event.event_type}")

    async def start(self) -> None:
        """Start event processing workers."""
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self._worker_count)
        ]
        logger.info(f"EventBus started with {self._worker_count} workers")

    async def stop(self) -> None:
        """Stop event processing gracefully."""
        self._running = False

        # Wait for queues to drain (with timeout)
        for priority in EventPriority:
            queue = self._queues[priority]
            try:
                await asyncio.wait_for(queue.join(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Queue {priority.name} did not drain in time")

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("EventBus stopped")

    async def _worker_loop(self, worker_id: int) -> None:
        """Event processing worker loop."""
        while self._running:
            event = await self._get_next_event()
            if event is None:
                await asyncio.sleep(0.01)
                continue

            await self._process_event(event)

    async def _get_next_event(self) -> Optional[Event[Any]]:
        """Get next event from highest priority non-empty queue."""
        for priority in EventPriority:
            queue = self._queues[priority]
            if not queue.empty():
                try:
                    _, _, event = queue.get_nowait()
                    return event
                except asyncio.QueueEmpty:
                    continue
        return None

    async def _process_event(self, event: Event[Any]) -> None:
        """Process an event through all registered handlers."""
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No handlers for event type: {event.event_type}")
            return

        for handler in handlers:
            try:
                start_time = time.time()
                result = await handler.handle(event)
                processing_time = time.time() - start_time

                self._metrics["processed"][event.event_type] = self._metrics["processed"].get(event.event_type, 0) + 1

                # If handler returns an event, emit it
                if result is not None and isinstance(result, Event):
                    await self.emit(result)

                logger.debug(
                    f"Handler {handler.__class__.__name__} processed {event.event_type} "
                    f"in {processing_time:.3f}s"
                )

            except Exception as e:
                self._metrics["failed"][event.event_type] = self._metrics["failed"].get(event.event_type, 0) + 1
                await handler.on_error(event, e)

                # Retry logic
                if event.metadata.retry_count < event.metadata.max_retries:
                    retry_event = event.with_retry()
                    await self.emit(retry_event)
                    logger.info(f"Retrying event {event.event_type} (attempt {retry_event.metadata.retry_count})")

    def get_metrics(self) -> Dict[str, Dict[str, int]]:
        """Get event processing metrics."""
        return {
            "emitted": self._metrics["emitted"].copy(),
            "processed": self._metrics["processed"].copy(),
            "failed": self._metrics["failed"].copy(),
            "dropped": self._metrics["dropped"].copy(),
        }


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class PipelineOrchestrator:
    """Orchestrates the event-driven trading pipeline."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._pipeline_stages = [
            "MarketDataReceived",
            "DataValidated",
            "IndicatorsCalculated",
            "SMCAnalysisCompleted",
            "MLPredictionCompleted",
            "RiskValidationCompleted",
            "SignalScored",
            "SignalApproved",
            "TelegramNotificationSent",
            "SignalStored",
        ]
        self._stage_handlers: Dict[str, EventHandler] = {}
        self._running = False

    def register_stage(self, stage_name: str, handler: EventHandler) -> None:
        """Register a pipeline stage handler."""
        if stage_name not in self._pipeline_stages:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")

        self._stage_handlers[stage_name] = handler
        self.event_bus.register(handler)
        logger.info(f"Registered pipeline stage: {stage_name}")

    async def start(self) -> None:
        """Start the pipeline."""
        self._running = True
        await self.event_bus.start()
        logger.info("Pipeline orchestrator started")

    async def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        await self.event_bus.stop()
        logger.info("Pipeline orchestrator stopped")

    async def inject_market_data(self, symbol: str, timeframe: str, candles: List[Dict[str, Any]]) -> None:
        """Inject market data to start the pipeline."""
        event = Event(
            event_type="MarketDataReceived",
            payload=MarketDataReceived(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                timestamp=datetime.utcnow(),
            ),
            metadata=EventMetadata(
                source="scanner",
                priority=EventPriority.HIGH,
            ),
        )
        await self.event_bus.emit(event)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "running": self._running,
            "stages": self._pipeline_stages,
            "registered_stages": list(self._stage_handlers.keys()),
            "metrics": self.event_bus.get_metrics(),
        }
