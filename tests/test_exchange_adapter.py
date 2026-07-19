"""
AXION QUANT V4 - Exchange Adapter Integration Tests

Tests the ExchangeAdapterManager fallback logic and each individual adapter's
public-endpoint interface without requiring live network access (responses
are mocked via unittest.mock).

Run with:  pytest tests/test_exchange_adapter.py -v
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exchange.base import (
    BaseExchangeClient,
    ExchangeAdapterError,
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)
from exchange.adapter_manager import ExchangeAdapterManager
from exchange.okx_client import OKXClient, _to_internal_symbol, _to_okx_symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(symbol: str = "BTC_USDT", ts: int = 0) -> UnifiedCandle:
    return UnifiedCandle(
        symbol=symbol, timestamp=ts or int(time.time() * 1000),
        open=60000.0, high=61000.0, low=59000.0,
        close=60500.0, volume=100.0, quote_volume=6_050_000.0, trades=500,
    )


def _make_contract(symbol: str = "BTC_USDT") -> UnifiedContractInfo:
    return UnifiedContractInfo(
        symbol=symbol, base_asset="BTC", quote_asset="USDT",
        contract_size=0.01, tick_size=0.1, min_order_size=1,
        max_leverage=125, status="TRADING", margin_asset="USDT",
    )


def _make_ticker(symbol: str = "BTC_USDT", last: float = 60500.0) -> UnifiedTicker:
    return UnifiedTicker(
        symbol=symbol, last_price=last, mark_price=last, index_price=last,
        bid_price=last - 0.5, ask_price=last + 0.5, volume_24h=1_000_000.0,
        open_interest=50_000.0, funding_rate=0.0001,
        high_24h=61000.0, low_24h=59000.0,
        price_change_24h=500.0, price_change_percent_24h=0.83,
    )


def _make_orderbook(symbol: str = "BTC_USDT") -> UnifiedOrderBook:
    return UnifiedOrderBook(
        symbol=symbol,
        bids=[(60499.5, 1.0), (60499.0, 2.0)],
        asks=[(60500.5, 1.0), (60501.0, 2.0)],
        timestamp=int(time.time() * 1000),
    )


class _MockAdapter(BaseExchangeClient):
    """Fully controllable mock adapter for testing."""

    exchange_name = "mock"

    def __init__(
        self,
        name: str = "mock",
        healthy: bool = True,
        raises: Optional[Exception] = None,
    ):
        self.exchange_name = name
        self._healthy = healthy
        self._raises = raises

    async def connect(self) -> None: pass
    async def disconnect(self) -> None: pass

    async def get_contracts(self) -> List[UnifiedContractInfo]:
        if self._raises: raise self._raises
        return [_make_contract()] if self._healthy else []

    async def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        if self._raises: raise self._raises
        return [_make_candle(symbol)] if self._healthy else []

    async def get_ticker(self, symbol=None):
        if self._raises: raise self._raises
        return [_make_ticker(symbol or "BTC_USDT")] if self._healthy else []

    async def get_order_book(self, symbol, limit=5):
        if self._raises: raise self._raises
        return _make_orderbook(symbol) if self._healthy else UnifiedOrderBook(
            symbol=symbol, bids=[], asks=[], timestamp=int(time.time() * 1000)
        )

    async def get_funding_rate(self, symbol=None):
        if self._raises: raise self._raises
        return {"fundingRate": 0.0001}

    async def get_open_interest(self, symbol):
        if self._raises: raise self._raises
        return 50_000.0

    async def health_check(self):
        if self._raises: raise self._raises
        status = "healthy" if self._healthy else "unhealthy"
        return {"status": status, "exchange": self.exchange_name}


# ---------------------------------------------------------------------------
# OKX symbol mapping tests (the critical fix)
# ---------------------------------------------------------------------------

class TestOKXSymbolMapping:
    """Verify BTC_USDT <-> BTC-USDT-SWAP round-trip mapping."""

    def test_to_okx_symbol_btc(self):
        assert _to_okx_symbol("BTC_USDT") == "BTC-USDT-SWAP"

    def test_to_okx_symbol_eth(self):
        assert _to_okx_symbol("ETH_USDT") == "ETH-USDT-SWAP"

    def test_to_okx_symbol_sol(self):
        assert _to_okx_symbol("SOL_USDT") == "SOL-USDT-SWAP"

    def test_to_internal_symbol_btc(self):
        assert _to_internal_symbol("BTC-USDT-SWAP") == "BTC_USDT"

    def test_to_internal_symbol_eth(self):
        assert _to_internal_symbol("ETH-USDT-SWAP") == "ETH_USDT"

    def test_roundtrip_btc(self):
        assert _to_internal_symbol(_to_okx_symbol("BTC_USDT")) == "BTC_USDT"

    def test_roundtrip_many(self):
        for sym in ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT"]:
            assert _to_internal_symbol(_to_okx_symbol(sym)) == sym


# ---------------------------------------------------------------------------
# AdapterManager fallback logic tests
# ---------------------------------------------------------------------------

class TestAdapterManagerFallback:
    """Verify the priority-based fallback mechanism."""

    @pytest.mark.asyncio
    async def test_uses_primary_when_healthy(self):
        """Manager should use the first adapter when it returns data."""
        primary = _MockAdapter("primary", healthy=True)
        secondary = _MockAdapter("secondary", healthy=True)
        mgr = ExchangeAdapterManager(adapters=[primary, secondary])
        result = await mgr.get_klines("BTC_USDT", "1h", limit=3)
        assert len(result) > 0
        assert mgr.active_exchange == "primary"

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_raises(self):
        """Manager must fall back to secondary when primary raises an exception."""
        primary = _MockAdapter("primary", raises=ConnectionError("Primary unreachable"))
        secondary = _MockAdapter("secondary", healthy=True)
        mgr = ExchangeAdapterManager(adapters=[primary, secondary])
        result = await mgr.get_klines("BTC_USDT", "1h", limit=3)
        assert len(result) > 0
        assert mgr.active_exchange == "secondary"

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_returns_empty(self):
        """Empty list is treated as soft failure; manager tries next adapter."""
        primary = _MockAdapter("primary", healthy=False)   # returns []
        secondary = _MockAdapter("secondary", healthy=True)
        mgr = ExchangeAdapterManager(adapters=[primary, secondary])
        result = await mgr.get_klines("BTC_USDT", "1h", limit=3)
        assert len(result) > 0
        assert mgr.active_exchange == "secondary"

    @pytest.mark.asyncio
    async def test_three_level_fallback(self):
        """Manager must walk the full priority chain before succeeding."""
        a1 = _MockAdapter("gate.io", raises=RuntimeError("blocked"))
        a2 = _MockAdapter("bitget", raises=RuntimeError("timeout"))
        a3 = _MockAdapter("okx", healthy=True)
        mgr = ExchangeAdapterManager(adapters=[a1, a2, a3])
        result = await mgr.get_klines("BTC_USDT", "1h", limit=3)
        assert len(result) > 0
        assert mgr.active_exchange == "okx"

    @pytest.mark.asyncio
    async def test_returns_empty_not_raises_when_all_fail(self):
        """get_klines must return [] (not raise) when all adapters fail."""
        adapters = [
            _MockAdapter("gate.io", raises=RuntimeError("fail")),
            _MockAdapter("bitget", raises=RuntimeError("fail")),
            _MockAdapter("okx", raises=RuntimeError("fail")),
        ]
        mgr = ExchangeAdapterManager(adapters=adapters)
        result = await mgr.get_klines("BTC_USDT", "1h", limit=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_contracts_fallback(self):
        """Contract discovery also falls back transparently."""
        failing = _MockAdapter("gate.io", raises=RuntimeError("fail"))
        working = _MockAdapter("bitget", healthy=True)
        mgr = ExchangeAdapterManager(adapters=[failing, working])
        contracts = await mgr.get_contracts()
        assert len(contracts) > 0

    @pytest.mark.asyncio
    async def test_health_check_reports_all_exchanges(self):
        """health_check must return per-exchange status, not just the primary."""
        a1 = _MockAdapter("gate.io", healthy=True)
        a2 = _MockAdapter("bitget", healthy=False)
        mgr = ExchangeAdapterManager(adapters=[a1, a2])
        health = await mgr.health_check()
        assert health["status"] == "healthy"  # at least one is up
        assert "gate.io" in health["exchanges"]
        assert "bitget" in health["exchanges"]
        assert health["exchanges"]["gate.io"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_all_down(self):
        """Overall status must be unhealthy when every adapter is down."""
        a1 = _MockAdapter("gate.io", raises=RuntimeError("down"))
        a2 = _MockAdapter("bitget", raises=RuntimeError("down"))
        mgr = ExchangeAdapterManager(adapters=[a1, a2])
        health = await mgr.health_check()
        assert health["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# MEXCClient config-optional behaviour
# ---------------------------------------------------------------------------

class TestMEXCClientConfigOptional:
    """Verify MEXCClient can be instantiated without MEXC credentials."""

    def test_instantiation_without_config_does_not_raise(self):
        """MEXCClient() must not raise even when MEXC env vars are absent."""
        import os
        from exchange.mexc_client import MEXCClient

        # Temporarily remove MEXC env vars
        saved = {k: os.environ.pop(k, None) for k in ("MEXC_ACCESS_KEY", "MEXC_SECRET_KEY")}
        try:
            # Should not raise
            client = MEXCClient()
            assert client is not None
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_max_retries_defaults_when_no_config(self):
        """_max_retries() must return a safe default when config is None."""
        from exchange.mexc_client import MEXCClient

        client = MEXCClient.__new__(MEXCClient)
        client.config = None  # simulate missing config
        assert client._max_retries() == MEXCClient._DEFAULT_MAX_RETRIES

    def test_backoff_defaults_when_no_config(self):
        """_backoff() must return a safe default when config is None."""
        from exchange.mexc_client import MEXCClient

        client = MEXCClient.__new__(MEXCClient)
        client.config = None
        assert client._backoff() == MEXCClient._DEFAULT_BACKOFF
