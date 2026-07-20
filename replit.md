# AXION QUANT V4 — Replit Integration Guide

**Institutional AI Quantitative Trading Platform for USDT-M Perpetual Futures**

## Overview

AXION QUANT V4 is optimized for Replit environments, using a fault-tolerant multi-exchange architecture to bypass regional API restrictions.

## ExchangeManager Architecture (v4.2)

The platform uses a **canonical ExchangeManager** (`exchange/manager.py`) as the single entry point for all market-data access.

### Priority Chain

| Priority | Exchange | Status | Notes |
|----------|----------|--------|-------|
| 1 | **Gate.io** | Primary | Public API, works on Replit |
| 2 | **Bitget** | Secondary | Public API, works on Replit |
| 3 | **OKX** | Tertiary | Fixed symbol mapping |
| 4 | **Bybit** | Optional | Geo-blocked on Replit (graceful failover) |
| 5 | **MEXC** | Fallback | Last resort, public API only |

## Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `EXCHANGE_PRIORITY` | Comma-separated adapter priority list | `gate,bitget,okx,bybit,mexc` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (optional) | |
| `TELEGRAM_ADMIN_CHAT_ID` | Admin chat ID (optional) | |

*Note: `MEXC_ACCESS_KEY` and `MEXC_SECRET_KEY` are no longer required for market data access.*

## Running on Replit

1. Ensure `.env` is configured (optional for paper trading).
2. Run the bot:
   ```bash
   python main.py --mode paper
   ```

## Symbol Mapping

The system uses an internal `BASE_USDT` format (e.g., `BTC_USDT`). The ExchangeManager handles all translation:
- **Gate.io**: `BTC_USDT` (Native)
- **Bitget**: `BTCUSDT`
- **OKX**: `BTC-USDT-SWAP`
- **Bybit**: `BTCUSDT`
- **MEXC**: `BTC_USDT`
