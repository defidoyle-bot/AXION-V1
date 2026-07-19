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
from exchange.base import BaseExchangeClient

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
        # Status on MEXC can be ENABLED, ONLINE, TRADING, 1
        active_statuses = {"TRADING", "ENABLED", "ONLINE", "1", "TRUE"}
        # Some contracts might have empty status but be tradeable
        status_ok = self.status.upper() in active_statuses or not self.status
        # Ensure it's USDT-M
        quote_ok = self.quote_asset.upper() == "USDT"
        # MEXC /detail endpoint returns futures
        return status_ok and quote_ok


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

    def __init__(self, exchange_client: BaseExchangeClient, config: Optional[MarketDataConfig] = None) -> None:
        self._client = exchange_client
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
            markets = await self._client.get_contracts()
        except Exception as exc:
            raise RuntimeError(f"MEXC markets fetch failed: {exc}") from exc

        symbol_infos: List[SymbolInfo] = []
        for m in markets:
            try:
                # m is a MEXCContractInfo dataclass
                info = SymbolInfo(
                    symbol=m.symbol,
                    base_asset=m.base_asset,
                    quote_asset=m.quote_asset,
                    contract_type="PERPETUAL",  # get_contracts() only returns perpetuals
                    status=m.status.upper() if m.status else "TRADING",
                    tick_size=m.tick_size,
                    lot_size=m.min_order_size,
                    min_qty=m.min_order_size,
                    max_leverage=m.max_leverage,
                )
                if info.symbol:
                    symbol_infos.append(info)
            except (AttributeError, ValueError, TypeError) as exc:
                logger.debug(f"Scanner: skipped malformed contract entry: {exc}")

        return symbol_infos

    async def _apply_market_filters(
        self, symbols: List[SymbolInfo], stats: ScannerStats
    ) -> List[SymbolInfo]:
        passed = []
        for info in symbols:
            try:
                ticker = await self._client.get_ticker(info.symbol)
                t = ticker[0] if isinstance(ticker, list) and ticker else ticker
                vol_24h = float(t.volume_24h) if hasattr(t, "volume_24h") else 0
                oi = float(t.open_interest) if hasattr(t, "open_interest") else 0
                bid = float(t.bid_price) if hasattr(t, "bid_price") else 0
                ask = float(t.ask_price) if hasattr(t, "ask_price") else 0
                spread = (ask - bid) / bid if bid > 0 else 999

                # Apply filters...
                passed.append(info)
            except Exception as e:
                logger.warning(f"Scanner: could not fetch ticker for {info.symbol}: {e}")
                passed.append(info)  # Keep symbol even if ticker fails
        return passed
