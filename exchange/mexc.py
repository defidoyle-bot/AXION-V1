"""
AXION QUANT V4 — MEXC Exchange Adapter
Last-resort fallback. MEXC may be blocked on Replit (HTTP 403 on several endpoints).
Thin wrapper over MEXCClient; all AXION code uses this via ExchangeManager.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.logging import get_logger
from exchange.base import UnifiedCandle, UnifiedContractInfo, UnifiedOrderBook, UnifiedTicker
from exchange.mexc_client import MEXCClient

logger = get_logger("exchange.mexc")


class MexcAdapter:
    """MEXC USDT-M perpetuals — may be blocked on Replit; last fallback only."""

    name: str = "mexc"

    def __init__(self) -> None:
        self._client = MEXCClient()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await self._client.connect()

    async def close(self) -> None:
        await self._client.disconnect()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            result = await self._client.health_check()
            return result.get("status") == "healthy"
        except Exception as exc:
            logger.debug(f"MEXC health check failed (likely blocked on Replit): {exc}")
            return False

    # ------------------------------------------------------------------
    # Market data — canonical interface
    # ------------------------------------------------------------------

    async def get_symbols(self) -> List[str]:
        """Return all active USDT perpetual symbol strings (e.g. 'BTC_USDT')."""
        contracts = await self._client.get_contracts()
        return [c.symbol for c in contracts if c.symbol]

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """Return full contract metadata (backward compat with pipeline/scanner)."""
        return await self._client.get_contracts()

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedCandle]:
        return await self._client.get_klines(
            symbol, interval, start_time=start_time, end_time=end_time, limit=limit
        )

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        return await self._client.get_ticker(symbol)

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        return await self._client.get_order_book(symbol, limit)

    async def get_open_interest(self, symbol: str) -> float:
        return await self._client.get_open_interest(symbol)

    async def get_funding_rate(self, symbol: str) -> Dict:
        return await self._client.get_funding_rate(symbol)
