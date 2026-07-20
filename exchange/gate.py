"""
AXION QUANT V4 — Gate.io Exchange Adapter
Primary exchange. Public USDT-M perpetuals endpoints.
Thin wrapper over GateioClient; all AXION code uses this via ExchangeManager.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from core.logging import get_logger
from exchange.base import UnifiedCandle, UnifiedContractInfo, UnifiedOrderBook, UnifiedTicker
from exchange.gateio_client import GateioClient

logger = get_logger("exchange.gate")


class GateAdapter:
    """Gate.io USDT-M perpetuals — public market data only."""

    name: str = "gate"

    def __init__(self) -> None:
        self._client = GateioClient()

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
            logger.debug(f"Gate.io health check failed: {exc}")
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
