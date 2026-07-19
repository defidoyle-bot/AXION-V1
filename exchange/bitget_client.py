"""
AXION QUANT V4 - Bitget Exchange Adapter
Public market-data client for Bitget USDT-M Perpetual Futures (V2 API).

Symbol mapping: internal BASE_USDT ↔ Bitget BASEUSDT (underscore removed).
API docs: https://www.bitget.com/api-doc/contract/market/
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

logger = get_logger("exchange.bitget")


# Timeframe mapping: internal → Bitget V2
_TF_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1Dutc",
}

PRODUCT_TYPE = "USDT-FUTURES"


def _to_bitget_symbol(internal: str) -> str:
    """BTC_USDT → BTCUSDT"""
    return internal.replace("_", "")


def _to_internal_symbol(bitget: str) -> str:
    """BTCUSDT → BTC_USDT (assumes quote = USDT)"""
    if bitget.endswith("USDT"):
        return bitget[:-4] + "_USDT"
    return bitget


class BitgetClient(BaseExchangeClient):
    """Bitget USDT-M perpetual futures — public market-data adapter (secondary exchange)."""

    exchange_name = "bitget"
    BASE_URL = "https://api.bitget.com"

    def __init__(self, rate_limit: float = 10.0, timeout: int = 30, max_retries: int = 3):
        self._rate_limiter = RateLimiter(rate_limit)
        self._timeout = timeout
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("BitgetClient initialised", extra={"event_data": {"base_url": self.BASE_URL}})

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
            logger.info("Bitget HTTP session established")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Bitget HTTP session closed")

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
                    raise BitgetAPIError("Rate limit exceeded")
                if resp.status >= 500:
                    if retry < self._max_retries:
                        await asyncio.sleep(2 ** retry)
                        return await self._request(path, params, retry + 1)
                resp.raise_for_status()
                envelope = await resp.json()
                code = str(envelope.get("code", ""))
                if code != "00000":
                    raise BitgetAPIError(f"API error {code}: {envelope.get('msg', 'unknown')}")
                return envelope.get("data", [])
        except aiohttp.ClientError as exc:
            if retry < self._max_retries:
                await asyncio.sleep(2 ** retry)
                return await self._request(path, params, retry + 1)
            raise BitgetAPIError(f"Request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """GET /api/v2/mix/market/contracts?productType=USDT-FUTURES"""
        try:
            data = await self._request(
                "/api/v2/mix/market/contracts", {"productType": PRODUCT_TYPE}
            )
            contracts: List[UnifiedContractInfo] = []
            for item in data:
                raw_sym = item.get("symbol", "")
                internal = _to_internal_symbol(raw_sym)
                if not internal.endswith("_USDT"):
                    continue
                status = item.get("symbolStatus", "")
                contracts.append(
                    UnifiedContractInfo(
                        symbol=internal,
                        base_asset=item.get("baseCoin", ""),
                        quote_asset="USDT",
                        contract_size=float(item.get("sizeMultiplier", 1) or 1),
                        tick_size=float(item.get("priceEndStep", 0.01) or 0.01),
                        min_order_size=float(item.get("minTradeNum", 0.01) or 0.01),
                        max_leverage=int(item.get("maxLever", 125) or 125),
                        status=status.upper() if status else "TRADING",
                        margin_asset="USDT",
                    )
                )
            logger.info(f"Bitget: discovered {len(contracts)} USDT perpetuals")
            return contracts
        except Exception as exc:
            logger.error(f"Bitget get_contracts failed: {exc}")
            return []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[UnifiedCandle]:
        """GET /api/v2/mix/market/candles"""
        bg_symbol = _to_bitget_symbol(symbol)
        bg_interval = _TF_MAP.get(interval, interval)
        params: Dict[str, Any] = {
            "symbol": bg_symbol,
            "productType": PRODUCT_TYPE,
            "granularity": bg_interval,
            "limit": str(min(limit, 1000)),
        }
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)

        try:
            data = await self._request("/api/v2/mix/market/candles", params)
            candles: List[UnifiedCandle] = []
            for row in data:
                # [timestamp_ms, open, high, low, close, base_vol, quote_vol]
                candles.append(
                    UnifiedCandle(
                        symbol=symbol,
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[6]) if len(row) > 6 else 0.0,
                        trades=0,
                    )
                )
            # Bitget returns ascending order — no sort needed
            return candles
        except Exception as exc:
            logger.error(f"Bitget get_klines({symbol}, {interval}) failed: {exc}")
            return []

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        """GET /api/v2/mix/market/tickers or /ticker for a single symbol."""
        try:
            if symbol:
                data = await self._request(
                    "/api/v2/mix/market/ticker",
                    {"symbol": _to_bitget_symbol(symbol), "productType": PRODUCT_TYPE},
                )
                if isinstance(data, dict):
                    data = [data]
            else:
                data = await self._request(
                    "/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE}
                )

            tickers: List[UnifiedTicker] = []
            for item in data:
                raw_sym = item.get("symbol", "")
                internal = _to_internal_symbol(raw_sym)
                # Bitget V2 uses "lastPr" for last price
                last = float(item.get("lastPr", item.get("last", 0)) or 0)
                bid = float(item.get("bidPr", last) or last)
                ask = float(item.get("askPr", last) or last)
                high = float(item.get("high24h", 0) or 0)
                low = float(item.get("low24h", 0) or 0)
                vol = float(item.get("baseVolume", item.get("quoteVolume", 0)) or 0)
                oi = float(item.get("openInterest", 0) or 0)
                funding = float(item.get("fundingRate", 0) or 0)
                change_pct = float(item.get("chgUtc", item.get("change24h", 0)) or 0)
                tickers.append(
                    UnifiedTicker(
                        symbol=internal,
                        last_price=last,
                        mark_price=float(item.get("markPrice", last) or last),
                        index_price=float(item.get("indexPrice", last) or last),
                        bid_price=bid,
                        ask_price=ask,
                        volume_24h=vol,
                        open_interest=oi,
                        funding_rate=funding,
                        high_24h=high,
                        low_24h=low,
                        price_change_24h=0.0,
                        price_change_percent_24h=change_pct * 100,
                    )
                )
            return tickers
        except Exception as exc:
            logger.error(f"Bitget get_ticker({symbol}) failed: {exc}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        """GET /api/v2/mix/market/merge-depth — returns dict directly (not wrapped in list)."""
        try:
            data = await self._request(
                "/api/v2/mix/market/merge-depth",
                {
                    "symbol": _to_bitget_symbol(symbol),
                    "productType": PRODUCT_TYPE,
                    "precision": "scale0",
                    "limit": str(limit),
                },
            )
            # merge-depth returns the book dict directly, not in a list
            book = data if isinstance(data, dict) else (data[0] if data else {})
            bids = [(float(b[0]), float(b[1])) for b in book.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in book.get("asks", [])]
            ts = int(book.get("ts", time.time() * 1000))
            return UnifiedOrderBook(symbol=symbol, bids=bids, asks=asks, timestamp=ts)
        except Exception as exc:
            logger.error(f"Bitget get_order_book({symbol}) failed: {exc}")
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000))

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        try:
            tickers = await self.get_ticker(symbol)
            if tickers:
                return {"fundingRate": tickers[0].funding_rate}
            return {}
        except Exception:
            return {}

    async def get_open_interest(self, symbol: str) -> float:
        try:
            tickers = await self.get_ticker(symbol)
            if tickers:
                return tickers[0].open_interest
            return 0.0
        except Exception:
            return 0.0

    async def health_check(self) -> Dict[str, Any]:
        try:
            start = time.monotonic()
            await self._request("/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE})
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "healthy", "exchange": "bitget", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            return {"status": "unhealthy", "exchange": "bitget", "error": str(exc)}


class BitgetAPIError(Exception):
    """Bitget API error."""
    pass
