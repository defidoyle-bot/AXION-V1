"""
AXION QUANT V4 — Bybit Exchange Adapter
Optional fallback (may be geo-restricted from some environments).
Public USDT perpetual endpoints — Bybit V5 API.

Note: Bybit was found geo-blocked from Replit during initial testing.
      This adapter is included for completeness and will fail-fast gracefully.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp

from core.logging import get_logger
from exchange.base import (
    RateLimiter,
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)

logger = get_logger("exchange.bybit")

# ---------------------------------------------------------------------------
# Bybit V5 constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.bybit.com"
CATEGORY = "linear"          # USDT perpetuals
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)

# Internal interval → Bybit V5 interval string
_INTERVAL_MAP: Dict[str, str] = {
    "1m":  "1",   "3m":  "3",   "5m":  "5",
    "15m": "15",  "30m": "30",
    "1h":  "60",  "2h":  "120", "4h":  "240",
    "6h":  "360", "12h": "720",
    "1d":  "D",   "1w":  "W",   "1M":  "M",
}


def _to_bybit_symbol(internal: str) -> str:
    """BTC_USDT → BTCUSDT"""
    return internal.replace("_", "")


def _to_internal_symbol(bybit: str) -> str:
    """BTCUSDT → BTC_USDT  (only for USDT pairs)"""
    if bybit.endswith("USDT"):
        return bybit[:-4] + "_USDT"
    return bybit


def _to_bybit_interval(interval: str) -> str:
    mapped = _INTERVAL_MAP.get(interval.lower())
    if mapped:
        return mapped
    # Accept already-numeric strings ("60") or Bybit-native letters ("D")
    return interval


# ---------------------------------------------------------------------------
# Bybit Adapter
# ---------------------------------------------------------------------------


class BybitAdapter:
    """Bybit USDT perpetuals — public market data only (Bybit V5 REST)."""

    name: str = "bybit"

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(requests_per_second=10.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=REQUEST_TIMEOUT)
            logger.info("Bybit HTTP session established")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Bybit HTTP session closed")

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        await self._rate_limiter.acquire()
        if self._session is None or self._session.closed:
            await self.connect()

        url = f"{BASE_URL}{endpoint}"
        async with self._session.get(url, params=params or {}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            ret_code = data.get("retCode", -1)
            if ret_code != 0:
                raise RuntimeError(
                    f"Bybit API error {ret_code}: {data.get('retMsg', 'unknown')}"
                )
            return data.get("result", {})

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            t0 = time.monotonic()
            await asyncio.wait_for(
                self._get("/v5/market/time"),
                timeout=10,
            )
            ms = (time.monotonic() - t0) * 1000
            logger.debug(f"Bybit health OK — latency {ms:.0f} ms")
            return True
        except Exception as exc:
            logger.debug(f"Bybit health check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Market data — canonical interface
    # ------------------------------------------------------------------

    async def get_symbols(self) -> List[str]:
        """Return all active USDT perpetual symbol strings (e.g. 'BTC_USDT')."""
        contracts = await self.get_contracts()
        return [c.symbol for c in contracts]

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """Return contract metadata for all active USDT linear perpetuals."""
        result: List[UnifiedContractInfo] = []
        cursor = ""
        seen = 0

        while True:
            params: Dict[str, Any] = {
                "category": CATEGORY,
                "status": "Trading",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._get("/v5/market/instruments-info", params)
            except Exception as exc:
                logger.error(f"Bybit get_contracts failed: {exc}")
                break

            items = data.get("list", [])
            for item in items:
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                internal = _to_internal_symbol(sym)
                base = item.get("baseCoin", sym[:-4])
                lot = item.get("lotSizeFilter", {})
                price_f = item.get("priceFilter", {})
                lev = item.get("leverageFilter", {})
                result.append(
                    UnifiedContractInfo(
                        symbol=internal,
                        base_asset=base,
                        quote_asset="USDT",
                        contract_size=1.0,
                        tick_size=float(price_f.get("tickSize", 0.01) or 0.01),
                        min_order_size=float(lot.get("minOrderQty", 1) or 1),
                        max_leverage=float(lev.get("maxLeverage", 100) or 100),
                        status=item.get("status", "Trading"),
                        margin_asset="USDT",
                    )
                )
            seen += len(items)
            cursor = data.get("nextPageCursor", "")
            if not cursor or len(items) < 1000:
                break

        logger.info(f"Bybit: fetched {len(result)} USDT perpetuals")
        return result

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedCandle]:
        bybit_sym = _to_bybit_symbol(symbol)
        bybit_interval = _to_bybit_interval(interval)
        params: Dict[str, Any] = {
            "category": CATEGORY,
            "symbol": bybit_sym,
            "interval": bybit_interval,
            "limit": min(limit, 200),
        }
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            data = await self._get("/v5/market/kline", params)
        except Exception as exc:
            logger.error(f"Bybit get_klines({symbol}, {interval}) failed: {exc}")
            return []

        # Bybit returns newest-first: [ts, open, high, low, close, volume, turnover]
        candles: List[UnifiedCandle] = []
        for row in reversed(data.get("list", [])):
            try:
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
            except (IndexError, ValueError):
                continue
        return candles

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        params: Dict[str, Any] = {"category": CATEGORY}
        if symbol:
            params["symbol"] = _to_bybit_symbol(symbol)

        try:
            data = await self._get("/v5/market/tickers", params)
        except Exception as exc:
            logger.error(f"Bybit get_ticker({symbol}) failed: {exc}")
            return []

        tickers: List[UnifiedTicker] = []
        for item in data.get("list", []):
            sym = item.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            internal = _to_internal_symbol(sym)
            last = float(item.get("lastPrice", 0) or 0)
            tickers.append(
                UnifiedTicker(
                    symbol=internal,
                    last_price=last,
                    mark_price=float(item.get("markPrice", last) or last),
                    index_price=float(item.get("indexPrice", last) or last),
                    bid_price=float(item.get("bid1Price", last) or last),
                    ask_price=float(item.get("ask1Price", last) or last),
                    volume_24h=float(item.get("volume24h", 0) or 0),
                    open_interest=float(item.get("openInterest", 0) or 0),
                    funding_rate=float(item.get("fundingRate", 0) or 0),
                    high_24h=float(item.get("highPrice24h", 0) or 0),
                    low_24h=float(item.get("lowPrice24h", 0) or 0),
                    price_change_24h=float(item.get("price24hPcnt", 0) or 0) * last,
                    price_change_percent_24h=float(item.get("price24hPcnt", 0) or 0) * 100,
                )
            )
        return tickers

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        try:
            data = await self._get(
                "/v5/market/orderbook",
                {"category": CATEGORY, "symbol": _to_bybit_symbol(symbol), "limit": limit},
            )
            bids = [(float(b[0]), float(b[1])) for b in data.get("b", [])]
            asks = [(float(a[0]), float(a[1])) for a in data.get("a", [])]
            return UnifiedOrderBook(
                symbol=symbol, bids=bids, asks=asks,
                timestamp=int(data.get("ts", time.time() * 1000)),
            )
        except Exception as exc:
            logger.error(f"Bybit get_order_book({symbol}) failed: {exc}")
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000))

    async def get_open_interest(self, symbol: str) -> float:
        try:
            data = await self._get(
                "/v5/market/open-interest",
                {
                    "category": CATEGORY,
                    "symbol": _to_bybit_symbol(symbol),
                    "intervalTime": "5min",
                    "limit": 1,
                },
            )
            lst = data.get("list", [])
            return float(lst[0].get("openInterest", 0)) if lst else 0.0
        except Exception as exc:
            logger.debug(f"Bybit get_open_interest({symbol}) failed: {exc}")
            return 0.0

    async def get_funding_rate(self, symbol: str) -> Dict:
        try:
            data = await self._get(
                "/v5/market/funding/history",
                {"category": CATEGORY, "symbol": _to_bybit_symbol(symbol), "limit": 1},
            )
            lst = data.get("list", [])
            rate = float(lst[0].get("fundingRate", 0)) if lst else 0.0
            return {"fundingRate": rate, "symbol": symbol}
        except Exception as exc:
            logger.debug(f"Bybit get_funding_rate({symbol}) failed: {exc}")
            return {"fundingRate": 0.0, "symbol": symbol}
