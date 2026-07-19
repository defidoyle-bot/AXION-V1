"""
AXION QUANT V4 - OKX Exchange Adapter
Public market-data client for OKX USDT-M Perpetual Futures (SWAP instruments).

Symbol mapping fix:
  internal BASE_USDT  →  OKX  BASE-USDT-SWAP
  OKX BASE-USDT-SWAP  →  internal BASE_USDT

API docs: https://www.okx.com/docs-v5/en/
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp

from core.logging import get_logger
from exchange.base import (
    BaseExchangeClient,
    RateLimiter,
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)

logger = get_logger("exchange.okx")


# Timeframe mapping: internal → OKX bar parameter
_TF_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


def _to_okx_symbol(internal: str) -> str:
    """BTC_USDT → BTC-USDT-SWAP (the critical fix for OKX symbol mapping)."""
    return internal.replace("_", "-") + "-SWAP"


def _to_internal_symbol(okx_inst_id: str) -> str:
    """BTC-USDT-SWAP → BTC_USDT"""
    # Remove the -SWAP suffix and replace hyphens with underscore
    base = okx_inst_id.removesuffix("-SWAP")
    return base.replace("-", "_")


class OKXClient(BaseExchangeClient):
    """OKX USDT-M perpetual futures (SWAP) — public market-data adapter."""

    exchange_name = "okx"
    BASE_URL = "https://www.okx.com"

    def __init__(self, rate_limit: float = 10.0, timeout: int = 30, max_retries: int = 3):
        self._rate_limiter = RateLimiter(rate_limit)
        self._timeout = timeout
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("OKXClient initialised", extra={"event_data": {"base_url": self.BASE_URL}})

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "AxionQuantV4/4.0.0",
                },
            )
            logger.info("OKX HTTP session established")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("OKX HTTP session closed")

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retry: int = 0,
    ) -> Any:
        await self._rate_limiter.acquire()
        if self._session is None or self._session.closed:
            await self.connect()

        url = f"{self.BASE_URL}{path}"
        try:
            async with self._session.get(url, params=params or {}) as resp:
                if resp.status == 429:
                    if retry < self._max_retries:
                        await asyncio.sleep(2 ** retry)
                        return await self._request(path, params, retry + 1)
                    raise OKXAPIError("Rate limit exceeded")
                if resp.status >= 500:
                    if retry < self._max_retries:
                        await asyncio.sleep(2 ** retry)
                        return await self._request(path, params, retry + 1)
                resp.raise_for_status()
                envelope = await resp.json()
                code = str(envelope.get("code", ""))
                if code != "0":
                    raise OKXAPIError(f"API error {code}: {envelope.get('msg', 'unknown')}")
                return envelope.get("data", [])
        except aiohttp.ClientError as exc:
            if retry < self._max_retries:
                await asyncio.sleep(2 ** retry)
                return await self._request(path, params, retry + 1)
            raise OKXAPIError(f"Request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """GET /api/v5/public/instruments?instType=SWAP — all USDT perpetuals."""
        try:
            data = await self._request(
                "/api/v5/public/instruments", {"instType": "SWAP"}
            )
            contracts: List[UnifiedContractInfo] = []
            for item in data:
                inst_id = item.get("instId", "")
                settle = item.get("settleCcy", "")
                # Only USDT-settled perpetuals
                if not inst_id.endswith("-USDT-SWAP") or settle != "USDT":
                    continue
                internal = _to_internal_symbol(inst_id)
                state = item.get("state", "live")
                status = "TRADING" if state == "live" else state.upper()
                contracts.append(
                    UnifiedContractInfo(
                        symbol=internal,
                        base_asset=item.get("baseCcy", ""),
                        quote_asset="USDT",
                        contract_size=float(item.get("ctVal", 1) or 1),
                        tick_size=float(item.get("tickSz", 0.01) or 0.01),
                        min_order_size=float(item.get("minSz", 1) or 1),
                        max_leverage=int(float(item.get("lever", 125) or 125)),
                        status=status,
                        margin_asset="USDT",
                    )
                )
            logger.info(f"OKX: discovered {len(contracts)} USDT perpetuals")
            return contracts
        except Exception as exc:
            logger.error(f"OKX get_contracts failed: {exc}")
            return []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[UnifiedCandle]:
        """GET /api/v5/market/candles — OKX returns newest first; we reverse to ascending."""
        okx_symbol = _to_okx_symbol(symbol)
        bar = _TF_MAP.get(interval, interval)
        params: Dict[str, Any] = {
            "instId": okx_symbol,
            "bar": bar,
            "limit": str(min(limit, 300)),  # OKX max = 300
        }
        if start_time:
            params["after"] = str(start_time)
        if end_time:
            params["before"] = str(end_time)

        try:
            data = await self._request("/api/v5/market/candles", params)
            candles: List[UnifiedCandle] = []
            for row in data:
                # [ts_ms, open, high, low, close, vol (contracts), volCcy, volCcyQuote, confirm]
                candles.append(
                    UnifiedCandle(
                        symbol=symbol,
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[7]) if len(row) > 7 else 0.0,
                        trades=0,
                    )
                )
            # OKX returns newest-first — reverse to ascending
            candles.reverse()
            return candles
        except Exception as exc:
            logger.error(f"OKX get_klines({symbol}, {interval}) failed: {exc}")
            return []

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        """GET /api/v5/market/ticker or /tickers?instType=SWAP"""
        try:
            if symbol:
                data = await self._request(
                    "/api/v5/market/ticker", {"instId": _to_okx_symbol(symbol)}
                )
            else:
                data = await self._request(
                    "/api/v5/market/tickers", {"instType": "SWAP"}
                )

            tickers: List[UnifiedTicker] = []
            for item in data:
                inst_id = item.get("instId", "")
                if not inst_id.endswith("-USDT-SWAP"):
                    continue
                internal = _to_internal_symbol(inst_id)
                last = float(item.get("last", 0) or 0)
                tickers.append(
                    UnifiedTicker(
                        symbol=internal,
                        last_price=last,
                        mark_price=float(item.get("markPx", last) or last),
                        index_price=float(item.get("idxPx", last) or last),
                        bid_price=float(item.get("bidPx", last) or last),
                        ask_price=float(item.get("askPx", last) or last),
                        volume_24h=float(item.get("vol24h", 0) or 0),
                        open_interest=float(item.get("oi", 0) or 0),
                        funding_rate=float(item.get("fundingRate", 0) or 0),
                        high_24h=float(item.get("high24h", 0) or 0),
                        low_24h=float(item.get("low24h", 0) or 0),
                        price_change_24h=0.0,
                        price_change_percent_24h=0.0,
                    )
                )
            return tickers
        except Exception as exc:
            logger.error(f"OKX get_ticker({symbol}) failed: {exc}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        """GET /api/v5/market/books"""
        try:
            data = await self._request(
                "/api/v5/market/books",
                {"instId": _to_okx_symbol(symbol), "sz": str(limit)},
            )
            item = data[0] if data else {}
            bids = [(float(b[0]), float(b[1])) for b in item.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in item.get("asks", [])]
            return UnifiedOrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=int(item.get("ts", time.time() * 1000)),
            )
        except Exception as exc:
            logger.error(f"OKX get_order_book({symbol}) failed: {exc}")
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000))

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """GET /api/v5/public/funding-rate"""
        try:
            if symbol:
                data = await self._request(
                    "/api/v5/public/funding-rate",
                    {"instId": _to_okx_symbol(symbol)},
                )
                if data:
                    return {"fundingRate": float(data[0].get("fundingRate", 0) or 0)}
            return {}
        except Exception:
            return {}

    async def get_open_interest(self, symbol: str) -> float:
        """GET /api/v5/public/open-interest"""
        try:
            data = await self._request(
                "/api/v5/public/open-interest",
                {"instType": "SWAP", "instId": _to_okx_symbol(symbol)},
            )
            if data:
                return float(data[0].get("oi", 0) or 0)
            return 0.0
        except Exception:
            return 0.0

    async def health_check(self) -> Dict[str, Any]:
        """Ping OKX via the system status endpoint."""
        try:
            start = time.monotonic()
            await self._request("/api/v5/public/time")
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "healthy", "exchange": "okx", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            return {"status": "unhealthy", "exchange": "okx", "error": str(exc)}


class OKXAPIError(Exception):
    """OKX API error."""
    pass
