"""
AXION QUANT V4 - MEXC Exchange Integration
Async MEXC USDT-M Perpetual Futures API client with rate limiting, retries, and fault tolerance.
Used as the fallback exchange when Gate.io / Bitget / OKX are unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

from config.settings import ExchangeConfig, get_config
from core.logging import get_logger
from exchange.base import (
    BaseExchangeClient,
    RateLimiter as _BaseLimiter,
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)

logger = get_logger("exchange.mexc")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True, slots=True)
class MEXCCandle:
    """Normalized MEXC futures candle."""
    symbol: str
    timestamp: int  # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int

    @classmethod
    def from_api_response(cls, symbol: str, data: List[Any]) -> "MEXCCandle":
        """Create candle from MEXC API response."""
        return cls(
            symbol=symbol,
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            quote_volume=float(data[6]) if len(data) > 6 else 0.0,
            trades=int(data[8]) if len(data) > 8 else 0,
        )


@dataclass(frozen=True, slots=True)
class MEXCContractInfo:
    """MEXC perpetual futures contract information."""
    symbol: str
    base_asset: str
    quote_asset: str
    contract_size: float
    tick_size: float
    min_order_size: float
    max_leverage: int
    status: str
    margin_asset: str

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "MEXCContractInfo":
        return cls(
            symbol=data.get("symbol", ""),
            base_asset=data.get("baseAsset", ""),
            quote_asset=data.get("quoteAsset", ""),
            contract_size=float(data.get("contractSize", 1)),
            tick_size=float(data.get("tickSize", 0.01)),
            min_order_size=float(data.get("minOrderSize", 0.01)),
            max_leverage=int(data.get("maxLeverage", 125)),
            status=str(data.get("state", data.get("status", ""))).upper(),
            margin_asset=data.get("settleCoin", data.get("quoteCoin", "USDT")),
        )


@dataclass(frozen=True, slots=True)
class MEXCTicker:
    """MEXC futures ticker data."""
    symbol: str
    last_price: float
    mark_price: float
    index_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    open_interest: float
    funding_rate: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percent_24h: float

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "MEXCTicker":
        return cls(
            symbol=data.get("symbol", ""),
            last_price=float(data.get("lastPrice", 0)),
            mark_price=float(data.get("fairPrice", data.get("markPrice", 0))),
            index_price=float(data.get("indexPrice", 0)),
            bid_price=float(data.get("bid1", data.get("bidPrice", 0))),
            ask_price=float(data.get("ask1", data.get("askPrice", 0))),
            volume_24h=float(data.get("volume24", data.get("volume24h", 0))),
            open_interest=float(data.get("holdVol", data.get("openInterest", 0))),
            funding_rate=float(data.get("fundingRate", 0)),
            high_24h=float(data.get("high24Price", data.get("high24h", 0))),
            low_24h=float(data.get("lower24Price", data.get("low24h", 0))),
            price_change_24h=float(data.get("riseFallValue", data.get("priceChange", 0))),
            price_change_percent_24h=float(data.get("riseFallRate", data.get("priceChangePercent", 0))),
        )


@dataclass(frozen=True, slots=True)
class MEXCOrderBook:
    """MEXC order book snapshot."""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    timestamp: int

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_percent(self) -> float:
        if self.best_bid > 0:
            return (self.spread / self.best_bid) * 100
        return 0.0


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, requests_per_second: float):
        self._tokens = requests_per_second
        self._max_tokens = requests_per_second
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self._max_tokens, self._tokens + elapsed * self._max_tokens)
            self._last_update = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._max_tokens
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# =============================================================================
# MEXC CLIENT
# =============================================================================

class MEXCClient(BaseExchangeClient):
    """Async MEXC USDT-M Perpetual Futures API client (fallback exchange)."""

    exchange_name = "mexc"

    # API Endpoints
    BASE_URL = "https://contract.mexc.com"
    API_VERSION = "/api/v1/contract"

    def __init__(self, config: Optional[ExchangeConfig] = None):
        try:
            self.config = config or get_config().exchange
        except (Exception, SystemExit):
            # Allow instantiation without valid MEXC credentials (used as fallback).
            # ConfigLoader calls sys.exit(1) on validation failure; we catch that
            # so the adapter manager can still register MEXC without crashing.
            self.config = None  # type: ignore[assignment]
        if self.config and self.config.futures_base_url:
            self.BASE_URL = self.config.futures_base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        rate = self.config.rate_limit_per_second if self.config else 10
        self._rate_limiter = RateLimiter(rate)
        self._lock = asyncio.Lock()

        logger.info(
            "MEXC client initialized",
            extra={"event_data": {
                "base_url": self.BASE_URL,
                "rate_limit": rate,
                "testnet": self.config.testnet if self.config else False,
            }}
        )

    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self._session is None or self._session.closed:
            timeout_secs = self.config.timeout_seconds if self.config else 30
            timeout = aiohttp.ClientTimeout(total=timeout_secs)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            logger.info("MEXC HTTP session established")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("MEXC HTTP session closed")

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC-SHA256 signature for authenticated requests."""
        query_string = urlencode(sorted(params.items()))
        signature = hmac.new(
            self.config.secret_key.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    # Safe defaults used when self.config is None (credential-free fallback mode)
    _DEFAULT_MAX_RETRIES: int = 3
    _DEFAULT_BACKOFF: float = 1.0

    def _max_retries(self) -> int:
        return self.config.max_retries if self.config else self._DEFAULT_MAX_RETRIES

    def _backoff(self) -> float:
        return self.config.retry_backoff_seconds if self.config else self._DEFAULT_BACKOFF

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
        retry_count: int = 0,
        version_override: Optional[str] = None,
    ) -> Any:
        """Make an API request with rate limiting and retry logic."""
        await self._rate_limiter.acquire()

        if self._session is None or self._session.closed:
            await self.connect()

        version = version_override if version_override else self.API_VERSION
        url = f"{self.BASE_URL}{version}{endpoint}"
        request_params = params or {}

        if authenticated:
            if self.config is None:
                raise MEXCAPIError("Authenticated requests require MEXC credentials")
            request_params["timestamp"] = int(time.time() * 1000)
            request_params["recvWindow"] = 5000
            request_params["signature"] = self._generate_signature(request_params)
            request_params["accessKey"] = self.config.access_key

        max_retries = self._max_retries()
        backoff = self._backoff()

        try:
            async with self._session.request(
                method=method,
                url=url,
                params=request_params if method == "GET" else None,
                json=request_params if method != "GET" else None,
            ) as response:

                if response.status == 429:
                    # Rate limited - exponential backoff
                    if retry_count < max_retries:
                        wait = backoff * (2 ** retry_count)
                        logger.warning(f"Rate limited, waiting {wait}s before retry {retry_count + 1}")
                        await asyncio.sleep(wait)
                        return await self._request(method, endpoint, params, authenticated, retry_count + 1, version_override)
                    raise MEXCAPIError("Rate limit exceeded after max retries")

                if response.status >= 500:
                    # Server error - retry
                    if retry_count < max_retries:
                        wait = backoff * (2 ** retry_count)
                        logger.warning(f"Server error {response.status}, retrying in {wait}s")
                        await asyncio.sleep(wait)
                        return await self._request(method, endpoint, params, authenticated, retry_count + 1, version_override)

                response.raise_for_status()
                data = await response.json()

                if data.get("code") != 0 and data.get("success") is not True:
                    raise MEXCAPIError(f"API error: {data.get('msg', 'Unknown error')}")

                return data.get("data", data)

        except aiohttp.ClientError as e:
            if retry_count < max_retries:
                wait = backoff * (2 ** retry_count)
                logger.warning(f"Request failed: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)
                return await self._request(method, endpoint, params, authenticated, retry_count + 1, version_override)
            raise MEXCAPIError(f"Request failed after {max_retries} retries: {e}")

    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        """Get all active crypto USDT-M perpetual futures contracts."""
        data = await self._request("GET", "/detail")
        contracts: List[UnifiedContractInfo] = []
        # MEXC lists stock, commodity and other non-crypto perpetuals under the same endpoint.
        non_crypto_symbols = {
            "XAU_USDT", "SILVER_USDT", "USOIL_USDT", "XAG_USDT", "OIL_USDT", "GOLD_USDT",
            "NSDQ_USDT", "SPX_USDT", "DJI_USDT",
        }
        for item in data:
            raw = MEXCContractInfo.from_api_response(item)
            if len(contracts) < 3:
                logger.debug(f"Contract debug: symbol={raw.symbol}, margin={raw.margin_asset}, status={raw.status}")
            if raw.margin_asset.upper() != "USDT":
                continue
            if "STOCK" in raw.symbol or raw.symbol in non_crypto_symbols:
                continue
            contracts.append(
                UnifiedContractInfo(
                    symbol=raw.symbol,
                    base_asset=raw.base_asset,
                    quote_asset=raw.quote_asset,
                    contract_size=raw.contract_size,
                    tick_size=raw.tick_size,
                    min_order_size=raw.min_order_size,
                    max_leverage=raw.max_leverage,
                    status=raw.status,
                    margin_asset=raw.margin_asset,
                    contract_category="crypto",
                )
            )
        logger.info(f"Discovered {len(contracts)} crypto USDT-M perpetual contracts")
        return contracts

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[UnifiedCandle]:
        """Get OHLCV candlestick data from MEXC Futures API.

        MEXC v1 contract kline uses a path parameter:
            GET /api/v1/contract/kline/{symbol}?interval=Min60&limit=...
        Response shape: { "time": [...], "open": [...], "close": [...], ... }
        """
        params: Dict[str, Any] = {"interval": interval, "limit": limit}
        if start_time: params["startTime"] = start_time
        if end_time: params["endTime"] = end_time

        try:
            data = await self._request("GET", f"/kline/{symbol}", params)
            # _request unwraps the outer {"success", "code", "data"} wrapper, so data is either
            # the series object {"time": [...], ...} or already a list/empty.
            series = data if isinstance(data, dict) and "time" in data else {}
            if not series or not series.get("time"):
                return []

            times = series.get("time", [])
            opens = series.get("open", [])
            highs = series.get("high", [])
            lows = series.get("low", [])
            closes = series.get("close", [])
            vols = series.get("vol", [])
            amounts = series.get("amount", [])
            trades = series.get("count", [])

            candles: List[UnifiedCandle] = []
            for i in range(len(times)):
                candles.append(
                    UnifiedCandle(
                        symbol=symbol,
                        timestamp=int(times[i]) * 1000,  # seconds -> ms
                        open=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close=float(closes[i]),
                        volume=float(vols[i]),
                        quote_volume=float(amounts[i]) if i < len(amounts) else 0.0,
                        trades=int(trades[i]) if i < len(trades) else 0,
                    )
                )
            return candles
        except MEXCAPIError as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return []

    async def get_ticker(self, symbol: Optional[str] = None) -> List[UnifiedTicker]:
        """Get 24h ticker statistics."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        try:
            data = await self._request("GET", "/ticker", params)
            raw_list = [MEXCTicker.from_api_response(data)] if symbol else [MEXCTicker.from_api_response(item) for item in data]
            return [
                UnifiedTicker(
                    symbol=t.symbol,
                    last_price=t.last_price,
                    mark_price=t.mark_price,
                    index_price=t.index_price,
                    bid_price=t.bid_price,
                    ask_price=t.ask_price,
                    volume_24h=t.volume_24h,
                    open_interest=t.open_interest,
                    funding_rate=t.funding_rate,
                    high_24h=t.high_24h,
                    low_24h=t.low_24h,
                    price_change_24h=t.price_change_24h,
                    price_change_percent_24h=t.price_change_percent_24h,
                )
                for t in raw_list
            ]
        except MEXCAPIError:
            return []

    async def get_order_book(self, symbol: str, limit: int = 5) -> UnifiedOrderBook:
        """Get order book depth."""
        params = {"symbol": symbol, "limit": limit}
        data = await self._request("GET", "/depth", params)

        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]

        return UnifiedOrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=data.get("timestamp", int(time.time() * 1000)),
        )

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current funding rate."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        try:
            return await self._request("GET", "/funding_rate", params)
        except MEXCAPIError:
            return {}

    async def get_open_interest(self, symbol: str) -> float:
        """Get current open interest for a symbol."""
        try:
            # Try both possible endpoints for open interest
            try:
                data = await self._request("GET", "/open_interest", {"symbol": symbol})
            except MEXCAPIError:
                data = await self._request("GET", "/open_interest", {"symbol": symbol}, version_override="/api/v1")
            
            if isinstance(data, dict):
                return float(data.get("openInterest", data.get("amount", 0)))
            return 0.0
        except Exception:
            # Return 0 if open interest is not available
            return 0.0

    async def health_check(self) -> Dict[str, Any]:
        """Check exchange connectivity."""
        try:
            start = time.monotonic()
            data = await self._request("GET", "/ping")
            latency = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "timestamp": int(time.time() * 1000),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time() * 1000),
            }


class MEXCAPIError(Exception):
    """MEXC API specific error."""
    pass
