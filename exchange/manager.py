"""
AXION QUANT V4 — ExchangeManager
Single entry point for all market-data access.

Priority order (overridable via EXCHANGE_PRIORITY env var):
    1. Gate.io  (primary  — public endpoints, works on Replit)
    2. Bitget   (secondary — public endpoints, works on Replit)
    3. OKX      (tertiary  — public endpoints, symbol mapping fixed)
    4. Bybit    (optional  — may be geo-blocked)
    5. MEXC     (last resort — blocked on Replit in most regions)

Failover rules:
    • On each call, start from the currently active adapter.
    • If it raises an exception OR returns an empty result, try the next.
    • Once a working adapter is found, remember it as active.
    • This gives scan consistency: all calls within one scan cycle use the
      same exchange (switching only on hard failure, not on every call).
    • get_open_interest / get_funding_rate never fall back purely on value —
      only on exceptions — because 0.0 is a valid data point.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Union

from core.logging import get_logger
from exchange.base import (
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)

logger = get_logger("exchange.manager")

# ---------------------------------------------------------------------------
# Adapter type alias (duck-typed — any class with the expected methods works)
# ---------------------------------------------------------------------------
Adapter = Any


def _build_adapters(priority: Optional[List[str]] = None) -> List[Adapter]:
    """Instantiate adapters in configured priority order.
    
    Each adapter uses only public market-data endpoints and does not require
    any API credentials for normal operation.
    """
    if priority is None:
        raw = os.environ.get("EXCHANGE_PRIORITY", "gate,bitget,okx,bybit,mexc")
        priority = [e.strip().lower() for e in raw.split(",") if e.strip()]

    # Lazy imports to avoid circular references
    from exchange.gate import GateAdapter
    from exchange.bitget import BitgetAdapter
    from exchange.okx import OKXAdapter
    from exchange.bybit import BybitAdapter
    from exchange.mexc import MexcAdapter

    _registry: Dict[str, Any] = {
        "gate":   GateAdapter,
        "gateio": GateAdapter,   # allow "gateio" alias
        "bitget": BitgetAdapter,
        "okx":    OKXAdapter,
        "bybit":  BybitAdapter,
        "mexc":   MexcAdapter,
    }

    adapters: List[Adapter] = []
    for name in priority:
        cls = _registry.get(name)
        if cls is None:
            logger.warning(f"ExchangeManager: unknown exchange '{name}' in priority list — skipping")
            continue
        try:
            adapters.append(cls())
            logger.info(f"ExchangeManager: registered adapter '{name}'")
        except Exception as exc:
            logger.warning(f"ExchangeManager: could not instantiate '{name}': {exc} — skipping")

    if not adapters:
        from exchange.gate import GateAdapter
        logger.warning("ExchangeManager: no adapters registered from priority list; defaulting to Gate.io")
        adapters = [GateAdapter()]

    return adapters


# ---------------------------------------------------------------------------
# ExchangeManager
# ---------------------------------------------------------------------------


class ExchangeManager:
    """Priority-based, fault-tolerant exchange manager.

    Drop-in replacement for ExchangeAdapterManager.  All scanner, pipeline,
    and ML code passes through this single object.
    """

    def __init__(self, adapters: Optional[List[Adapter]] = None) -> None:
        self._adapters: List[Adapter] = adapters if adapters is not None else _build_adapters()
        self._active_idx: int = 0          # index into _adapters of the current working adapter
        self._last_health_check: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect all adapters (failures are non-fatal for individual adapters)."""
        for adapter in self._adapters:
            try:
                await adapter.connect()
            except Exception as exc:
                logger.warning(f"ExchangeManager: {adapter.name}.connect() failed: {exc}")
        logger.info(
            f"ExchangeManager ready — {len(self._adapters)} adapters, "
            f"active: {self.active_exchange}"
        )

    async def disconnect(self) -> None:
        """Close all adapter sessions."""
        for adapter in self._adapters:
            try:
                await adapter.close()
            except Exception as exc:
                logger.debug(f"ExchangeManager: {adapter.name}.close() failed: {exc}")

    # Keep old name for backward compat
    async def close(self) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def active_exchange(self) -> str:
        if not self._adapters:
            return "none"
        return self._adapters[self._active_idx % len(self._adapters)].name

    # ------------------------------------------------------------------
    # Internal failover engine
    # ------------------------------------------------------------------

    async def _try_adapters(
        self,
        method_name: str,
        *args: Any,
        empty_check: Any = None,   # callable(result) → True if result is "empty"
        fallback: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Try adapters starting from the active one; fall back on failure or empty result."""
        if empty_check is None:
            empty_check = lambda r: not r  # noqa: E731

        n = len(self._adapters)
        if n == 0:
            return fallback

        for i in range(n):
            idx = (self._active_idx + i) % n
            adapter = self._adapters[idx]
            try:
                result = await getattr(adapter, method_name)(*args, **kwargs)
                if not empty_check(result):
                    if i > 0:
                        old = self._adapters[self._active_idx % n].name
                        logger.warning(
                            f"ExchangeManager: {old}.{method_name} returned empty — "
                            f"switched to {adapter.name}"
                        )
                        self._active_idx = idx
                    return result
                # Result is empty — log and try next
                logger.debug(
                    f"ExchangeManager: {adapter.name}.{method_name} returned empty result"
                )
            except Exception as exc:
                logger.warning(
                    f"ExchangeManager: {adapter.name}.{method_name} raised {type(exc).__name__}: {exc}"
                )

        logger.error(
            f"ExchangeManager: all adapters failed for {method_name}; returning fallback"
        )
        return fallback

    async def _try_adapters_nofallback(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Try adapters for methods where any return value (inc. 0 / empty dict) is valid.
        Only falls back on exceptions — not on the value itself.
        """
        n = len(self._adapters)
        if n == 0:
            return None

        for i in range(n):
            idx = (self._active_idx + i) % n
            adapter = self._adapters[idx]
            try:
                return await getattr(adapter, method_name)(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    f"ExchangeManager: {adapter.name}.{method_name} raised {type(exc).__name__}: {exc}"
                )

        logger.error(f"ExchangeManager: all adapters raised for {method_name}")
        return None

    # ------------------------------------------------------------------
    # Public market-data API
    # (all callers — scanner, pipeline, ML — use these methods only)
    # ------------------------------------------------------------------

    async def get_symbols(self) -> List[str]:
        """Return active USDT perpetual symbol strings from the best available exchange."""
        result = await self._try_adapters("get_symbols", fallback=[])
        return result or []

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """Return full contract metadata (backward-compat with existing pipeline/scanner)."""
        result = await self._try_adapters("get_contracts", fallback=[])
        return result or []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedCandle]:
        result = await self._try_adapters(
            "get_klines", symbol, interval,
            limit=limit, start_time=start_time, end_time=end_time,
            fallback=[],
        )
        return result or []

    async def get_ticker(
        self, symbol: Optional[str] = None
    ) -> List[UnifiedTicker]:
        result = await self._try_adapters("get_ticker", symbol, fallback=[])
        return result or []

    async def get_order_book(
        self, symbol: str, limit: int = 5
    ) -> UnifiedOrderBook:
        empty_ob = UnifiedOrderBook(
            symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000)
        )
        result = await self._try_adapters(
            "get_order_book", symbol, limit,
            empty_check=lambda r: not (r.bids or r.asks),
            fallback=empty_ob,
        )
        return result if result is not None else empty_ob

    async def get_open_interest(self, symbol: str) -> float:
        """Returns 0.0 on total failure (value 0.0 from exchange is valid)."""
        result = await self._try_adapters_nofallback("get_open_interest", symbol)
        return float(result) if result is not None else 0.0

    async def get_funding_rate(self, symbol: str) -> Dict:
        """Returns {"fundingRate": 0.0} on total failure."""
        result = await self._try_adapters_nofallback("get_funding_rate", symbol)
        return result if result is not None else {"fundingRate": 0.0, "symbol": symbol}

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Check all adapters and return a summary dict."""
        results: Dict[str, Any] = {}
        any_healthy = False

        checks = {
            adapter.name: adapter.health_check()
            for adapter in self._adapters
        }
        statuses = await asyncio.gather(*checks.values(), return_exceptions=True)

        for name, status in zip(checks.keys(), statuses):
            if isinstance(status, Exception):
                results[name] = {"status": "error", "error": str(status)}
            elif status is True:
                results[name] = {"status": "healthy"}
                any_healthy = True
            else:
                results[name] = {"status": "unhealthy"}

        # Promote to first healthy if current active is down
        if not results.get(self.active_exchange, {}).get("status") == "healthy":
            for i, adapter in enumerate(self._adapters):
                if results.get(adapter.name, {}).get("status") == "healthy":
                    logger.info(
                        f"ExchangeManager health check: promoting {adapter.name} → active"
                    )
                    self._active_idx = i
                    break

        self._last_health_check = time.time()
        return {
            "status": "healthy" if any_healthy else "unhealthy",
            "active_exchange": self.active_exchange,
            "exchanges": results,
        }
