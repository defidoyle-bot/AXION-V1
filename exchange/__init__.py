"""
AXION QUANT V4 - Exchange package
Exports the unified adapter manager and all individual exchange clients.
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
from exchange.adapter_manager import ExchangeAdapterManager
from exchange.gateio_client import GateioClient
from exchange.bitget_client import BitgetClient
from exchange.okx_client import OKXClient
from exchange.mexc_client import MEXCClient, MEXCCandle, MEXCContractInfo, MEXCTicker, MEXCOrderBook

__all__ = [
    # Unified / abstract
    "BaseExchangeClient",
    "ExchangeAdapterError",
    "RateLimiter",
    "UnifiedCandle",
    "UnifiedContractInfo",
    "UnifiedOrderBook",
    "UnifiedTicker",
    # Manager
    "ExchangeAdapterManager",
    # Individual adapters
    "GateioClient",
    "BitgetClient",
    "OKXClient",
    "MEXCClient",
    # Backward-compat MEXC types
    "MEXCCandle",
    "MEXCContractInfo",
    "MEXCTicker",
    "MEXCOrderBook",
]
