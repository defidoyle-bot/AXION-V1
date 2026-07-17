"""
AXION QUANT V4 - Symbol Scanner
Automatic discovery and lifecycle management of MEXC USDT-M Perpetual Futures contracts.

Responsibilities
----------------
- Discover all active USDT perpetual futures contracts via MEXC API.
- Filter out spot, delisted, paused, maintenance, and non-USDT contracts.
- Apply configurable market filters (volume, open interest, spread, etc.).
- Periodically refresh the active symbol list.
- Expose a stable interface consumed by the MarketDataPipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from config.settings import MarketDataConfig, get_config
from core.logging import get_logger

logger = get_logger("scanner")


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class SymbolInfo:
    """Metadata for a single USDT-M perpetual futures contract."""

    symbol: str
    base_asset: str
    quote_asset: str
    contract_type: str          # "PERPETUAL"
    status: str                 # "TRADING"
    tick_size: float
    lot_size: float
    min_qty: float
    max_leverage: int
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)

    def is_active(self) -> bool:
        """Return True only for actively-trading USDT perpetual contracts."""
        return (
            self.status.upper() == "TRADING"
            and self.quote_asset.upper() == "USDT"
            and self.contract_type.upper() == "PERPETUAL"
        )


@dataclass
class ScannerStats:
    """Statistics from the most recent symbol discovery cycle."""

    total_contracts: int = 0
    active_contracts: int = 0
    filtered_contracts: int = 0
    rejected_maintenance: int = 0
    rejected_non_usdt: int = 0
    rejected_low_volume: int = 0
    rejected_low_oi: int = 0
    last_refresh: Optional[datetime] = None

    def log_summary(self) -> None:
        logger.info(
            f"[Scanner] Contracts Found: {self.total_contracts} | "
            f"Active: {self.active_contracts} | "
            f"After Filters: {self.filtered_contracts} | "
            f"Rejected — Maintenance: {self.rejected_maintenance} | "
            f"Non-USDT: {self.rejected_non_usdt} | "
            f"Low Volume: {self.rejected_low_volume} | "
            f"Low OI: {self.rejected_low_oi}"
        )


# =============================================================================
# SYMBOL SCANNER
# =============================================================================


class SymbolScanner:
    """Discovers and maintains the list of tradeable MEXC USDT-M perpetual contracts.

    Usage
    -----
    scanner = SymbolScanner(mexc_client, config)
    await scanner.refresh()          # initial load
    symbols = scanner.get_symbols()  # returns List[str]

    A background refresh loop can be started with:
        asyncio.create_task(scanner.start_refresh_loop())
    """

    def __init__(self, mexc_client: object, config: Optional[MarketDataConfig] = None) -> None:
        self._client = mexc_client
        self._config: MarketDataConfig = config or get_config().market_data
        self._symbols: Dict[str, SymbolInfo] = {}
        self._filtered_symbols: List[str] = []
        self._stats = ScannerStats()
        self._lock = asyncio.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_symbols(self) -> List[str]:
        """Return the current list of filtered, tradeable symbol strings."""
        return list(self._filtered_symbols)

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Return metadata for a specific symbol, or None if unknown."""
        return self._symbols.get(symbol)

    @property
    def stats(self) -> ScannerStats:
        return self._stats

    async def refresh(self) -> List[str]:
        """Perform a single symbol discovery cycle.

        Returns the updated list of filtered symbols.
        """
        async with self._lock:
            logger.info("Scanner: starting symbol discovery cycle…")
            try:
                raw_symbols = await self._fetch_all_contracts()
            except Exception as exc:
                logger.error(f"Scanner: failed to fetch contracts from exchange: {exc}", exc_info=True)
                return self._filtered_symbols  # return stale list rather than crash

            stats = ScannerStats(total_contracts=len(raw_symbols))
            active: Dict[str, SymbolInfo] = {}

            for info in raw_symbols:
                if not info.is_active():
                    if info.status.upper() in ("BREAK", "PRE_DELIVERING", "DELIVERING", "END_OF_DAY"):
                        stats.rejected_maintenance += 1
                    else:
                        stats.rejected_non_usdt += 1
                    continue

                stats.active_contracts += 1
                active[info.symbol] = info

            # Merge with existing registry (preserve discovery timestamps)
            for sym, info in active.items():
                if sym in self._symbols:
                    info.discovered_at = self._symbols[sym].discovered_at
                self._symbols[sym] = info

            # Remove symbols that disappeared from the exchange
            gone = set(self._symbols.keys()) - set(active.keys())
            for sym in gone:
                logger.info(f"Scanner: symbol removed from exchange: {sym}")
                del self._symbols[sym]

            # Apply market filters
            filtered = await self._apply_market_filters(list(active.values()), stats)

            stats.filtered_contracts = len(filtered)
            stats.last_refresh = datetime.utcnow()
            stats.log_summary()

            self._filtered_symbols = [s.symbol for s in filtered]
            self._stats = stats

            return self._filtered_symbols

    async def start_refresh_loop(self) -> None:
        """Run continuous background symbol refresh at the configured interval."""
        self._running = True
        logger.info(
            f"Scanner: starting refresh loop "
            f"(interval={self._config.symbol_refresh_interval_minutes}m)"
        )
        while self._running:
            try:
                await self.refresh()
            except Exception as exc:
                logger.error(f"Scanner: unexpected error in refresh loop: {exc}", exc_info=True)
            await asyncio.sleep(self._config.symbol_refresh_interval_minutes * 60)

    def stop(self) -> None:
        """Signal the refresh loop to stop after the current cycle."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_all_contracts(self) -> List[SymbolInfo]:
        """Retrieve all futures contract definitions from MEXC via the exchange client."""
        try:
            markets = await self._client.get_futures_markets()
        except Exception as exc:
            raise RuntimeError(f"MEXC markets fetch failed: {exc}") from exc

        symbol_infos: List[SymbolInfo] = []
        for m in markets:
            try:
                info = SymbolInfo(
                    symbol=m.get("symbol", ""),
                    base_asset=m.get("baseCoin", m.get("base", "")),
                    quote_asset=m.get("quoteCoin", m.get("quote", "USDT")),
                    contract_type=m.get("contractType", m.get("type", "PERPETUAL")).upper(),
                    status=m.get("status", m.get("state", "TRADING")).upper(),
                    tick_size=float(m.get("priceUnit", m.get("tickSize", 0.01))),
                    lot_size=float(m.get("volUnit", m.get("lotSize", 0.001))),
                    min_qty=float(m.get("minVol", m.get("minQty", 1))),
                    max_leverage=int(m.get("maxLeverage", 100)),
                )
                if info.symbol:
                    symbol_infos.append(info)
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug(f"Scanner: skipped malformed contract entry: {exc}")

        return symbol_infos

    async def _apply_market_filters(
        self, symbols: List[SymbolInfo], stats: ScannerStats
    ) -> List[SymbolInfo]:
        """Apply configurable market filters: volume, open interest, spread.

        Returns the subset of symbols that pass all enabled filters.
        Filters are always active; thresholds are driven by MarketDataConfig.
        Set min_24h_volume_usdt=0 and min_open_interest_usdt=0 in config to disable.
        """
        passed: List[SymbolInfo] = []

        for info in symbols:
            try:
                ticker = await self._client.get_ticker(info.symbol)
            except Exception as exc:
                logger.debug(f"Scanner: could not fetch ticker for {info.symbol}: {exc}")
                # Conservative: exclude symbols whose ticker cannot be fetched
                continue

            vol_24h = float(ticker.get("volume24", ticker.get("quoteVolume", 0)))
            oi = float(ticker.get("openInterest", ticker.get("holdVol", 0)))
            bid = float(ticker.get("bid1", ticker.get("bid", 0)))
            ask = float(ticker.get("ask1", ticker.get("ask", 0)))
            spread = (ask - bid) / bid if bid > 0 else 999

            # Volume filter
            if self._config.min_24h_volume_usdt and vol_24h < self._config.min_24h_volume_usdt:
                stats.rejected_low_volume += 1
                logger.debug(
                    f"Scanner: {info.symbol} rejected — 24h volume {vol_24h:.0f} "
                    f"< min {self._config.min_24h_volume_usdt:.0f}"
                )
                continue

            # Open interest filter
            if self._config.min_open_interest_usdt and oi < self._config.min_open_interest_usdt:
                stats.rejected_low_oi += 1
                logger.debug(
                    f"Scanner: {info.symbol} rejected — OI {oi:.0f} "
                    f"< min {self._config.min_open_interest_usdt:.0f}"
                )
                continue

            # Spread filter
            if self._config.max_spread_percent and spread > self._config.max_spread_percent / 100:
                logger.debug(
                    f"Scanner: {info.symbol} rejected — spread {spread:.4%} "
                    f"> max {self._config.max_spread_percent:.2f}%"
                )
                continue

            passed.append(info)

        return passed
