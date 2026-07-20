---
name: ExchangeManager integration
description: ExchangeManager is the single market-data entry point; all code goes through it, not individual clients.
---

## Rule

Every scanner, pipeline, ML, or signal module that needs market data must go through
`ExchangeManager` (from `exchange.manager`), never through individual exchange clients.

## Why

- **Uniform failover** — if Gate.io is down, Bitget is tried transparently; no caller
  needs to know which exchange is active.
- **Scan consistency** — once an exchange is selected for a scan cycle, all calls within
  that cycle use the same exchange (no switching between klines, ticker, and order book).
- **One health check** — the manager probes all registered adapters, promotes a healthy
  one, and reports per-exchange status.

## How to apply

```python
from exchange.manager import ExchangeManager

mgr = ExchangeManager()            # reads EXCHANGE_PRIORITY env var
await mgr.connect()
symbols = await mgr.get_symbols()   # returns List[str]
tickers = await mgr.get_ticker("BTC_USDT")
# ...
await mgr.disconnect()
```

Adapter wrappers (`gate.py`, `bitget.py`, `okx.py`, `bybit.py`, `mexc.py`) follow this
interface:
- `name` → str
- `health_check()` → bool
- `get_symbols()` → `List[str]`
- `get_contracts()` → `List[UnifiedContractInfo]`
- Same signature for klines, ticker, order_book, open_interest, funding_rate
