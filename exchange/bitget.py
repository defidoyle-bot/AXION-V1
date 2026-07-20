"""
AXION QUANT V4 — Bitget Exchange Adapter
Secondary exchange. Public USDT-M futures endpoints.
Thin wrapper over BitgetClient; all AXION code uses this via ExchangeManager.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from core.logging import get_logger
from exchange.base import UnifiedCandle, UnifiedContractInfo, UnifiedOrderBook, UnifiedTicker
from exchange.bitget_client import BitgetClient

logger = get_logger("exchange.bitget")


class BitgetAdapter:
    """Bitget USDT-M futures — public market data only."""

    name: str = "bitget"

    def __init__(self) -> None:
        self._client = BitgetClient()

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
            logger.debug(f"Bitget health check failed: {exc}")
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
