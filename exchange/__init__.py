"""
AXION QUANT V4 — exchange package
Canonical entry point: ExchangeManager.

All scanner, pipeline, and ML code must go through ExchangeManager.
Do NOT import individual clients directly from application code.
"""
from exchange.base import (
    BaseExchangeClient,
    ExchangeAdapterError,
    RateLimiter,
    UnifiedCandle,
    UnifiedContractInfo,
    UnifiedOrderBook,
    UnifiedTicker,
)

# --- Canonical manager (use this everywhere) ---
from exchange.manager import ExchangeManager

# --- Named adapters (used internally by ExchangeManager) ---
from exchange.gate import GateAdapter
from exchange.bitget import BitgetAdapter
from exchange.okx import OKXAdapter
from exchange.bybit import BybitAdapter
from exchange.mexc import MexcAdapter

# --- Legacy / backward-compat (kept so existing imports don't break) ---
from exchange.adapter_manager import ExchangeAdapterManager
from exchange.gateio_client import GateioClient
from exchange.bitget_client import BitgetClient
from exchange.okx_client import OKXClient
from exchange.mexc_client import MEXCClient, MEXCCandle, MEXCContractInfo, MEXCTicker, MEXCOrderBook

__all__ = [
    # Unified types
    "BaseExchangeClient",
    "ExchangeAdapterError",
    "RateLimiter",
    "UnifiedCandle",
    "UnifiedContractInfo",
    "UnifiedOrderBook",
    "UnifiedTicker",
    # Canonical manager
    "ExchangeManager",
    # Named adapters
    "GateAdapter",
    "BitgetAdapter",
    "OKXAdapter",
    "BybitAdapter",
    "MexcAdapter",
    # Legacy (backward compat)
    "ExchangeAdapterManager",
    "GateioClient",
    "BitgetClient",
    "OKXClient",
    "MEXCClient",
    "MEXCCandle",
    "MEXCContractInfo",
    "MEXCTicker",
    "MEXCOrderBook",
]
