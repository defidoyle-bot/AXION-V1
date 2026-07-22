"""
AXION QUANT V4 - Gate.io Exchange Adapter
Public market-data client for Gate.io USDT-M Perpetual Futures.

Symbol format: BASE_USDT (e.g. BTC_USDT) — matches Gate.io natively, no mapping needed.
API docs: https://www.gate.io/docs/developers/apiv4/
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

logger = get_logger("exchange.gateio")


# Timeframe mapping: internal → Gate.io
_TF_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class GateioClient(BaseExchangeClient):
    """Gate.io USDT-M perpetual futures — public market-data adapter (primary exchange)."""

    exchange_name = "gateio"
    BASE_URL = "https://api.gateio.ws"
    FUTURES_PREFIX = "/api/v4/futures/usdt"

    def __init__(self, rate_limit: float = 10.0, timeout: int = 30, max_retries: int = 3):
        self._rate_limiter = RateLimiter(rate_limit)
        self._timeout = timeout
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("GateioClient initialised", extra={"event_data": {"base_url": self.BASE_URL}})

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
            logger.info("Gate.io HTTP session established")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Gate.io HTTP session closed")

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retry: int = 0,
    ) -> Any:
        await self._rate_limiter.acquire()
        if self._session is None or self._session.closed:
            await self.connect()

        url = f"{self.BASE_URL}{self.FUTURES_PREFIX}{endpoint}"
        try:
            async with self._session.get(url, params=params or {}) as resp:
                if resp.status == 429:
                    if retry < self._max_retries:
                        await asyncio.sleep(2 ** retry)
                        return await self._request(endpoint, params, retry + 1)
                    raise GateioAPIError("Rate limit exceeded after max retries")
                if resp.status >= 500:
                    if retry < self._max_retries:
                        await asyncio.sleep(2 ** retry)
                        return await self._request(endpoint, params, retry + 1)
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as exc:
            if retry < self._max_retries:
                await asyncio.sleep(2 ** retry)
                return await self._request(endpoint, params, retry + 1)
            raise GateioAPIError(f"Request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """GET /api/v4/futures/usdt/contracts — list crypto USDT perpetuals only."""
        try:
            data = await self._request("/contracts")
            contracts: List[UnifiedContractInfo] = []
            # Gate.io also lists stocks, forex, metals, commodities, indices as USDT futures.
            # These are not crypto futures and are excluded here.
            non_crypto_categories = {"stocks", "forex", "metals", "commodities", "indices"}
            for item in data:
                name = item.get("name", "")
                # Gate.io USDT linear contracts have type "direct" (not "swap")
                # We already query /futures/usdt/ endpoint so all results are USDT-settled
                if not name or "_" not in name:
                    continue
                category = (item.get("contract_type") or "").lower()
                if category in non_crypto_categories:
                    continue
                # Gate.io uses name like BTC_USDT directly
                status = "TRADING" if not item.get("in_delisting") else "DELISTED"
                contracts.append(
                    UnifiedContractInfo(
                        symbol=name,
                        base_asset=name.split("_")[0] if "_" in name else name,
                        quote_asset="USDT",
                        contract_size=float(item.get("quanto_multiplier", 1) or 1),
                        tick_size=float(item.get("order_price_round", 0.01) or 0.01),
                        min_order_size=float(item.get("order_size_min", 1) or 1),
                        max_leverage=int(item.get("leverage_max", 100) or 100),
                        status=status,
                        margin_asset="USDT",
                        contract_category="crypto" if not category else category,
                    )
                )
            logger.info(f"Gate.io: discovered {len(contracts)} USDT perpetuals")
            return contracts
        except Exception as exc:
            logger.error(f"Gate.io get_contracts failed: {exc}")
            return []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[UnifiedCandle]:
        """GET /api/v4/futures/usdt/candlesticks"""
        gate_interval = _TF_MAP.get(interval, interval)
        params: Dict[str, Any] = {
            "contract": symbol,
            "interval": gate_interval,
            "limit": min(limit, 2000),
        }
        if start_time:
            params["from"] = start_time // 1000  # Gate.io uses seconds
        if end_time:
            params["to"] = end_time // 1000

        try:
            data = await self._request("/candlesticks", params)
            candles: List[UnifiedCandle] = []
            for item in data:
                # Response: {"t": unix_sec, "o": open, "h": high, "l": low, "c": close,
                #            "v": volume_contracts, "sum": quote_volume}
                ts_ms = int(item["t"]) * 1000  # convert seconds → ms
                candles.append(
                    UnifiedCandle(
                        symbol=symbol,
                        timestamp=ts_ms,
                        open=float(item.get("o", 0)),
                        high=float(item.get("h", 0)),
                        low=float(item.get("l", 0)),
                        close=float(item.get("c", 0)),
                        volume=float(item.get("v", 0)),
                        quote_volume=float(item.get("sum", 0)),
                        trades=0,
                    )
                )
            # Gate.io returns ascending — no sort needed
            return candles
        except Exception as exc:
            logger.error(f"Gate.io get_klines({symbol}, {interval}) failed: {exc}")
            return []

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        """GET /api/v4/futures/usdt/tickers"""
        params: Dict[str, Any] = {}
        if symbol:
            params["contract"] = symbol
        try:
            data = await self._request("/tickers", params)
            if not isinstance(data, list):
                data = [data]
            tickers: List[UnifiedTicker] = []
            for item in data:
                last = float(item.get("last", 0) or 0)
                mark = float(item.get("mark_price", last) or last)
                index = float(item.get("index_price", last) or last)
                high = float(item.get("high_24h", 0) or 0)
                low = float(item.get("low_24h", 0) or 0)
                change_pct = float(item.get("change_percentage", 0) or 0)
                vol = float(item.get("volume_24h_base", 0) or 0)
                vol_quote = float(item.get("volume_24h_quote", 0) or 0)
                oi = float(item.get("open_interest", 0) or 0)
                funding = float(item.get("funding_rate", 0) or 0)
                tickers.append(
                    UnifiedTicker(
                        symbol=item.get("contract", ""),
                        last_price=last,
                        mark_price=mark,
                        index_price=index,
                        bid_price=last,   # Gate.io ticker doesn't include bid/ask
                        ask_price=last,
                        volume_24h=vol_quote if vol == 0 else vol,
                        open_interest=oi,
                        funding_rate=funding,
                        high_24h=high,
                        low_24h=low,
                        price_change_24h=0.0,
                        price_change_percent_24h=change_pct,
                    )
                )
            return tickers
        except Exception as exc:
            logger.error(f"Gate.io get_ticker({symbol}) failed: {exc}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        """GET /api/v4/futures/usdt/order_book"""
        params = {"contract": symbol, "limit": limit, "interval": "0"}
        try:
            data = await self._request("/order_book", params)
            bids = [(float(b["p"]), float(b["s"])) for b in data.get("bids", [])]
            asks = [(float(a["p"]), float(a["s"])) for a in data.get("asks", [])]
            return UnifiedOrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=int(time.time() * 1000),
            )
        except Exception as exc:
            logger.error(f"Gate.io get_order_book({symbol}) failed: {exc}")
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000))

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return funding rate from ticker data."""
        try:
            tickers = await self.get_ticker(symbol)
            if tickers:
                return {"fundingRate": tickers[0].funding_rate}
            return {}
        except Exception:
            return {}

    async def get_open_interest(self, symbol: str) -> float:
        """Return open interest from ticker."""
        try:
            tickers = await self.get_ticker(symbol)
            if tickers:
                return tickers[0].open_interest
            return 0.0
        except Exception:
            return 0.0

    async def health_check(self) -> Dict[str, Any]:
        """Ping Gate.io via a minimal contract list call."""
        try:
            start = time.monotonic()
            await self._request("/contracts", {"limit": 1})
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "healthy", "exchange": "gateio", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            return {"status": "unhealthy", "exchange": "gateio", "error": str(exc)}


class GateioAPIError(Exception):
    """Gate.io API error."""
    pass
