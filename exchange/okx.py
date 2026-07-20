"""
AXION QUANT V4 — OKX Exchange Adapter
Tertiary exchange. Public USDT-SWAP endpoints.
Thin wrapper over OKXClient (symbol mapping fixed: BTC_USDT ↔ BTC-USDT-SWAP).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.logging import get_logger
from exchange.base import UnifiedCandle, UnifiedContractInfo, UnifiedOrderBook, UnifiedTicker
from exchange.okx_client import OKXClient

logger = get_logger("exchange.okx")


class OKXAdapter:
    """OKX USDT-SWAP perpetuals — public market data only.

    Symbol mapping (handled transparently by OKXClient):
        Internal: BTC_USDT  →  OKX instrument: BTC-USDT-SWAP
    """

    name: str = "okx"

    def __init__(self) -> None:
        self._client = OKXClient()

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
            logger.debug(f"OKX health check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Market data — canonical interface
    # ------------------------------------------------------------------

    async def get_symbols(self) -> List[str]:
        """Return all active USDT SWAP symbol strings (e.g. 'BTC_USDT').
        OKX instrument IDs (BTC-USDT-SWAP) are converted back to internal format.
        """
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
