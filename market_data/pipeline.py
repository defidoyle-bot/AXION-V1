"""
AXION QUANT V4 - Market Data Pipeline
High-performance async market data retrieval, validation, normalization, and distribution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.settings import MarketDataConfig, Timeframe, get_config
from core.events import Event, EventMetadata, EventPriority, MarketDataReceived, DataValidated
from core.logging import get_logger
from exchange.mexc_client import MEXCClient, MEXCCandle, MEXCContractInfo, MEXCTicker

logger = get_logger("market_data")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True, slots=True)
class NormalizedCandle:
    """Internal normalized candle format."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    mark_price: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None

    @classmethod
    def from_mexc_candle(
        cls,
        candle: MEXCCandle,
        mark_price: Optional[float] = None,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
    ) -> "NormalizedCandle":
        return cls(
            symbol=candle.symbol,
            timestamp=datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            quote_volume=candle.quote_volume,
            trades=candle.trades,
            mark_price=mark_price,
            funding_rate=funding_rate,
            open_interest=open_interest,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "mark_price": self.mark_price,
            "funding_rate": self.funding_rate,
            "open_interest": self.open_interest,
        }


@dataclass
class ValidationResult:
    """Candle validation result."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


# =============================================================================
# CANDLE VALIDATOR
# =============================================================================

class CandleValidator:
    """Validates candle data before entering analysis pipeline."""

    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or get_config().market_data

    def validate(self, candles: List[NormalizedCandle]) -> ValidationResult:
        """Validate a list of candles."""
        result = ValidationResult(valid=True)

        if not candles:
            result.add_error("Empty candle list")
            return result

        df = pd.DataFrame([c.to_dict() for c in candles])
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Check for missing timestamps
        if not df.empty and df["timestamp"].isna().any():
            result.add_error(f"Found {df['timestamp'].isna().sum()} missing timestamps")

        # Check for duplicate timestamps
        duplicates = df["timestamp"].duplicated().sum()
        if duplicates > 0:
            result.add_error(f"Found {duplicates} duplicate timestamps")

        # Check for invalid OHLC values
        for col in ["open", "high", "low", "close"]:
            if df[col].isna().any():
                result.add_error(f"Found {df[col].isna().sum()} NaN values in {col}")
            if (df[col] <= 0).any():
                result.add_error(f"Found {(df[col] <= 0).sum()} non-positive values in {col}")

        # Validate OHLC relationships
        invalid_high = (df["high"] < df[["open", "high", "low", "close"]].max(axis=1)).sum()
        if invalid_high > 0:
            result.add_error(f"Found {invalid_high} candles where high is not the maximum")

        invalid_low = (df["low"] > df[["open", "high", "low", "close"]].min(axis=1)).sum()
        if invalid_low > 0:
            result.add_error(f"Found {invalid_low} candles where low is not the minimum")

        # Check for negative volume
        if (df["volume"] < 0).any():
            result.add_error(f"Found {(df['volume'] < 0).sum()} negative volume values")

        # Check for incorrect ordering
        if not df["timestamp"].is_monotonic_increasing:
            result.add_error("Timestamps are not in ascending order")

        # Check for time gaps (warnings)
        if len(df) > 1:
            expected_interval = self._detect_interval(df)
            time_diffs = df["timestamp"].diff().dropna()
            gap_count = (time_diffs > expected_interval * 2).sum()
            if gap_count > 0:
                result.add_warning(f"Found {gap_count} time gaps larger than expected interval")

        # Check for corrupted records (all zeros)
        zero_candles = ((df["open"] == 0) & (df["high"] == 0) & (df["low"] == 0) & (df["close"] == 0)).sum()
        if zero_candles > 0:
            result.add_error(f"Found {zero_candles} corrupted candles (all zeros)")

        # Check for NaN values in any column
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            result.add_warning(f"Found {nan_count} total NaN values across all columns")

        return result

    def _detect_interval(self, df: pd.DataFrame) -> timedelta:
        """Detect candle interval from data."""
        if len(df) < 2:
            return timedelta(minutes=1)

        time_diffs = df["timestamp"].diff().dropna()
        median_diff = time_diffs.median()
        return median_diff


# =============================================================================
# SYMBOL SCANNER
# =============================================================================

class SymbolScanner:
    """Discovers and filters active USDT-M perpetual futures symbols."""

    def __init__(
        self,
        client: MEXCClient,
        config: Optional[MarketDataConfig] = None,
    ):
        self.client = client
        self.config = config or get_config().market_data
        self._cached_symbols: List[str] = []
        self._last_refresh: Optional[datetime] = None
        self._symbol_metadata: Dict[str, Dict[str, Any]] = {}

    async def refresh_symbols(self) -> List[str]:
        """Refresh the list of active symbols from exchange."""
        logger.info("Refreshing symbol list from MEXC...")

        try:
            contracts = await self.client.get_contracts()

            # Filter for USDT-M perpetual futures only
            active_statuses = {"TRADING", "ENABLED", "ONLINE", "1", "TRUE", "0"}
            usdt_contracts = [
                c for c in contracts
                if c.margin_asset.upper() == "USDT" and (c.status.upper() in active_statuses or not c.status)
            ]

            symbols = [c.symbol for c in usdt_contracts]
            self._cached_symbols = symbols
            self._last_refresh = datetime.utcnow()

            # Cache metadata
            for contract in usdt_contracts:
                self._symbol_metadata[contract.symbol] = {
                    "contract_size": contract.contract_size,
                    "tick_size": contract.tick_size,
                    "min_order_size": contract.min_order_size,
                    "max_leverage": contract.max_leverage,
                }

            logger.info(f"Symbol refresh complete: {len(symbols)} USDT-M perpetual contracts")
            return symbols

        except Exception as e:
            logger.error(f"Failed to refresh symbols: {e}")
            if self._cached_symbols:
                logger.warning(f"Using cached symbol list ({len(self._cached_symbols)} symbols)")
                return self._cached_symbols
            raise

    async def get_symbols(self, force_refresh: bool = False) -> List[str]:
        """Get active symbols, refreshing if necessary."""
        if force_refresh or self._should_refresh():
            return await self.refresh_symbols()
        return self._cached_symbols

    def _should_refresh(self) -> bool:
        """Check if symbol list needs refresh."""
        if not self._last_refresh:
            return True

        elapsed = datetime.utcnow() - self._last_refresh
        return elapsed > timedelta(minutes=self.config.symbol_refresh_interval_minutes)

    async def filter_symbols(self, symbols: List[str]) -> List[str]:
        """Filter symbols by market criteria."""
        logger.info(f"Filtering {len(symbols)} symbols by market criteria...")

        filtered = []

        for symbol in symbols:
            try:
                ticker = await self.client.get_ticker(symbol)
                if not ticker:
                    continue

                t = ticker[0]

                # Volume filter
                if t.volume_24h < self.config.min_24h_volume_usdt:
                    continue

                # Open interest filter
                if t.open_interest < self.config.min_open_interest_usdt:
                    continue

                # Spread filter
                order_book = await self.client.get_order_book(symbol, limit=5)
                if order_book.spread_percent > self.config.max_spread_percent:
                    continue

                filtered.append(symbol)

            except Exception as e:
                logger.warning(f"Error filtering {symbol}: {e}")
                continue

        logger.info(f"Filtered to {len(filtered)} symbols meeting criteria")
        return filtered

    def get_symbol_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata for a symbol."""
        return self._symbol_metadata.get(symbol)


# =============================================================================
# MARKET DATA PIPELINE
# =============================================================================

class MarketDataPipeline:
    """Orchestrates market data fetching, validation, and distribution."""

    def __init__(
        self,
        client: MEXCClient,
        scanner: SymbolScanner,
        validator: CandleValidator,
        config: Optional[MarketDataConfig] = None,
    ):
        self.client = client
        self.scanner = scanner
        self.validator = validator
        self.config = config or get_config().market_data
        self._cache: Dict[str, Dict[str, List[NormalizedCandle]]] = {}  # symbol -> timeframe -> candles
        self._running = False

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: Optional[int] = None,
    ) -> List[NormalizedCandle]:
        """Fetch and normalize candles for a symbol and timeframe."""
        limit = limit or self.config.candle_fetch_limit

        # Fetch candles
        mexc_candles = await self.client.get_klines(
            symbol=symbol,
            interval=timeframe.value,
            limit=limit,
        )

        # Fetch additional market data
        ticker = await self.client.get_ticker(symbol)
        mark_price = ticker[0].mark_price if ticker else None

        try:
            funding_data = await self.client.get_funding_rate(symbol)
            funding_rate = float(funding_data.get("fundingRate", 0)) if funding_data else None
        except Exception:
            funding_rate = None

        try:
            open_interest = await self.client.get_open_interest(symbol)
        except Exception:
            open_interest = None

        # Normalize
        normalized = [
            NormalizedCandle.from_mexc_candle(
                c, mark_price=mark_price, funding_rate=funding_rate, open_interest=open_interest
            )
            for c in mexc_candles
        ]

        return normalized

    async def validate_and_normalize(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: List[NormalizedCandle],
    ) -> Tuple[List[NormalizedCandle], ValidationResult]:
        """Validate candles and return with result."""
        result = self.validator.validate(candles)

        if not result.valid:
            logger.error(
                f"Validation failed for {symbol} {timeframe.value}",
                extra={"event_data": {"errors": result.errors, "warnings": result.warnings}}
            )
            return [], result

        if result.warnings:
            logger.warning(
                f"Validation warnings for {symbol} {timeframe.value}",
                extra={"event_data": {"warnings": result.warnings}}
            )

        # Cache valid candles
        if symbol not in self._cache:
            self._cache[symbol] = {}
        self._cache[symbol][timeframe.value] = candles

        return candles, result

    async def get_cached_candles(self, symbol: str, timeframe: Timeframe) -> Optional[List[NormalizedCandle]]:
        """Get cached candles if available and fresh."""
        if symbol not in self._cache:
            return None
        if timeframe.value not in self._cache[symbol]:
            return None
        return self._cache[symbol][timeframe.value]

    async def run_pipeline(
        self,
        symbol: str,
        timeframe: Timeframe,
    ) -> Optional[Event]:
        """Run the full market data pipeline for a symbol/timeframe."""
        try:
            # Fetch
            candles = await self.fetch_candles(symbol, timeframe)

            # Validate
            valid_candles, validation = await self.validate_and_normalize(symbol, timeframe, candles)

            if not valid_candles:
                return None

            # Create validated event
            event = Event(
                event_type="DataValidated",
                payload=DataValidated(
                    symbol=symbol,
                    timeframe=timeframe.value,
                    candles=[c.to_dict() for c in valid_candles],
                    validation_result={
                        "valid": validation.valid,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                    },
                ),
                metadata=EventMetadata(
                    source="market_data_pipeline",
                    priority=EventPriority.HIGH,
                ),
            )

            return event

        except Exception as e:
            logger.error(f"Pipeline failed for {symbol} {timeframe.value}: {e}", exc_info=True)
            return None

    async def scan_all_symbols(self, timeframes: Optional[List[Timeframe]] = None) -> List[Event]:
        """Scan all symbols across all timeframes."""
        timeframes = timeframes or self.config.timeframes
        symbols = await self.scanner.get_symbols()

        events = []

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    event = await self.run_pipeline(symbol, timeframe)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.error(f"Failed to process {symbol} {timeframe.value}: {e}")
                    continue

        return events
