"""
AXION QUANT V4 - Exchange Base Module
Abstract base class and unified data models for all exchange adapters.
All adapters use internal symbol format: BASE_USDT (e.g. BTC_USDT).
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.logging import get_logger

logger = get_logger("exchange.base")


# =============================================================================
# UNIFIED DATA MODELS
# Identical field layout to MEXCCandle/MEXCContractInfo/etc. for backward compat.
# =============================================================================

@dataclass(frozen=True, slots=True)
class UnifiedCandle:
    """Normalized candle format — shared across all exchange adapters."""
    symbol: str
    timestamp: int   # Unix milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int


@dataclass(frozen=True, slots=True)
class UnifiedContractInfo:
    """Unified perpetual futures contract information."""
    symbol: str           # Internal format: BTC_USDT
    base_asset: str
    quote_asset: str
    contract_size: float
    tick_size: float
    min_order_size: float
    max_leverage: int
    status: str
    margin_asset: str


@dataclass(frozen=True, slots=True)
class UnifiedTicker:
    """Unified futures ticker data."""
    symbol: str
    last_price: float
    mark_price: float
    index_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    open_interest: float
    funding_rate: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percent_24h: float


@dataclass(frozen=True, slots=True)
class UnifiedOrderBook:
    """Unified order book snapshot."""
    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: int

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_percent(self) -> float:
        if self.best_bid > 0:
            return (self.spread / self.best_bid) * 100
        return 0.0


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, requests_per_second: float = 10.0):
        self._tokens = requests_per_second
        self._max_tokens = requests_per_second
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self._max_tokens, self._tokens + elapsed * self._max_tokens)
            self._last_update = now
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._max_tokens
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# =============================================================================
# ABSTRACT BASE CLIENT
# =============================================================================

class BaseExchangeClient(ABC):
    """Abstract base for all exchange market-data adapters.

    Contract
    --------
    - All ``symbol`` parameters and return values use the internal format: ``BASE_USDT``
      (e.g. ``BTC_USDT``, ``ETH_USDT``).
    - Only **public** (unauthenticated) endpoints are used.
    - ``get_klines`` returns candles sorted ascending by timestamp.
    """

    exchange_name: str = "base"

    async def __aenter__(self) -> "BaseExchangeClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    @abstractmethod
    async def connect(self) -> None:
        """Initialize underlying HTTP session."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close underlying HTTP session."""
        ...

    @abstractmethod
    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """Return all active USDT-M perpetual futures contracts."""
        ...

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[UnifiedCandle]:
        """Return OHLCV candles (ascending order). Symbol in BASE_USDT format."""
        ...

    @abstractmethod
    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        """Return 24 h ticker(s). Symbol in BASE_USDT format."""
        ...

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        """Return order book snapshot. Symbol in BASE_USDT format."""
        ...

    @abstractmethod
    async def get_funding_rate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return current funding rate info."""
        ...

    @abstractmethod
    async def get_open_interest(self, symbol: str) -> float:
        """Return current open interest (notional USDT)."""
        ...

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Lightweight connectivity check. Returns ``{"status": "healthy"|"unhealthy", ...}``."""
        ...


class ExchangeAdapterError(Exception):
    """Raised when all configured exchange adapters fail."""
    pass
