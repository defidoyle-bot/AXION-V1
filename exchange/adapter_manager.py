"""
AXION QUANT V4 - Exchange Adapter Manager
Priority-based, fault-tolerant exchange adapter with automatic fallback.

Priority order (configurable via EXCHANGE_PRIORITY env var):
  1. Gate.io  (primary  — public endpoints work on Replit)
  2. Bitget   (secondary — public endpoints work on Replit)
  3. OKX      (tertiary  — public endpoints, fixed symbol mapping)
  4. MEXC     (fallback  — may be blocked on Replit)

On each call the manager tries adapters in priority order and returns the
first successful result, logging which exchange was used.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Type

from core.logging import get_logger
from exchange.base import (
    BaseExchangeClient,
    ExchangeAdapterError,
    RateLimiter,
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)

logger = get_logger("exchange.adapter_manager")


def _build_default_adapters() -> List[BaseExchangeClient]:
    """Instantiate adapters in default priority order."""
    import os

    priority_str = os.environ.get("EXCHANGE_PRIORITY", "gateio,bitget,okx,mexc")
    priority = [e.strip().lower() for e in priority_str.split(",") if e.strip()]

    # Import lazily to avoid circular imports at module level
    from exchange.gateio_client import GateioClient
    from exchange.bitget_client import BitgetClient
    from exchange.okx_client import OKXClient
    from exchange.mexc_client import MEXCClient

    registry: Dict[str, Type[BaseExchangeClient]] = {
        "gateio": GateioClient,
        "bitget": BitgetClient,
        "okx": OKXClient,
        "mexc": MEXCClient,
    }

    adapters: List[BaseExchangeClient] = []
    for name in priority:
        cls = registry.get(name)
        if cls is None:
            logger.warning(f"Unknown exchange in EXCHANGE_PRIORITY: {name!r} — skipping")
            continue
        try:
            adapters.append(cls())
            logger.info(f"Registered exchange adapter: {name}")
        except Exception as exc:
            logger.warning(f"Could not instantiate {name} adapter: {exc}")

    if not adapters:
        # Last resort — always try MEXC
        adapters = [MEXCClient()]

    return adapters


class ExchangeAdapterManager(BaseExchangeClient):
    """Transparent, priority-based wrapper around multiple exchange clients.

    Exposes the same interface as a single ``BaseExchangeClient`` but automatically
    tries adapters in priority order and falls back on failure.

    The active exchange selection is sticky per-call (the first successful adapter
    wins) but never permanently committed — if the primary recovers it is preferred
    on the next call.
    """

    exchange_name = "adapter_manager"

    def __init__(self, adapters: Optional[List[BaseExchangeClient]] = None):
        self._adapters = adapters or _build_default_adapters()
        self._last_active: str = self._adapters[0].exchange_name if self._adapters else "none"

    # ------------------------------------------------------------------
    # Session management — delegate to all adapters
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        tasks = [adapter.connect() for adapter in self._adapters]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for adapter, result in zip(self._adapters, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to connect {adapter.exchange_name}: {result}")
            else:
                logger.info(f"Connected to {adapter.exchange_name}")

    async def disconnect(self) -> None:
        tasks = [adapter.disconnect() for adapter in self._adapters]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Core fallback logic
    # ------------------------------------------------------------------

    async def _try_all(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Try each adapter in priority order; return first non-empty result."""
        last_exc: Optional[Exception] = None
        for adapter in self._adapters:
            try:
                result = await getattr(adapter, method)(*args, **kwargs)
                # Consider empty list/dict as a soft failure — try next exchange
                if result is None or result == [] or result == {}:
                    logger.debug(
                        f"{adapter.exchange_name}.{method}() returned empty — trying next exchange"
                    )
                    continue
                if self._last_active != adapter.exchange_name:
                    logger.info(
                        f"Exchange active: {adapter.exchange_name} "
                        f"(was: {self._last_active})"
                    )
                    self._last_active = adapter.exchange_name
                return result
            except Exception as exc:
                logger.warning(
                    f"{adapter.exchange_name}.{method}() failed: {exc} — trying next exchange"
                )
                last_exc = exc
                continue

        raise ExchangeAdapterError(
            f"All exchange adapters failed for {method}. Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public API — delegates via _try_all
    # ------------------------------------------------------------------

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        try:
            return await self._try_all("get_contracts")
        except ExchangeAdapterError as exc:
            logger.error(str(exc))
            return []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[UnifiedCandle]:
        try:
            return await self._try_all("get_klines", symbol, interval, start_time, end_time, limit)
        except ExchangeAdapterError as exc:
            logger.error(str(exc))
            return []

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        try:
            return await self._try_all("get_ticker", symbol)
        except ExchangeAdapterError as exc:
            logger.error(str(exc))
            return []

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        try:
            return await self._try_all("get_order_book", symbol, limit)
        except ExchangeAdapterError:
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000))

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        try:
            return await self._try_all("get_funding_rate", symbol)
        except ExchangeAdapterError:
            return {}

    async def get_open_interest(self, symbol: str) -> float:
        try:
            return await self._try_all("get_open_interest", symbol)
        except ExchangeAdapterError:
            return 0.0

    async def health_check(self) -> Dict[str, Any]:
        """Health check all adapters; report per-exchange status."""
        results: Dict[str, Any] = {}
        any_healthy = False

        tasks = [adapter.health_check() for adapter in self._adapters]
        checks = await asyncio.gather(*tasks, return_exceptions=True)

        for adapter, check in zip(self._adapters, checks):
            if isinstance(check, Exception):
                results[adapter.exchange_name] = {"status": "unhealthy", "error": str(check)}
            else:
                results[adapter.exchange_name] = check
                if check.get("status") == "healthy":
                    any_healthy = True

        overall_status = "healthy" if any_healthy else "unhealthy"
        return {
            "status": overall_status,
            "active_exchange": self._last_active,
            "exchanges": results,
            "timestamp": int(time.time() * 1000),
        }

    @property
    def active_exchange(self) -> str:
        """Name of the most recently successfully used exchange."""
        return self._last_active

    def get_adapter(self, name: str) -> Optional[BaseExchangeClient]:
        """Return a specific adapter by name, or None if not found."""
        for adapter in self._adapters:
            if adapter.exchange_name == name:
                return adapter
        return None
