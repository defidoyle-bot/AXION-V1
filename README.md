# AXION QUANT V4

**Institutional AI Quantitative Trading Platform for Multi-Exchange USDT-M Perpetual Futures**

## Overview

AXION QUANT V4 is a production-grade async Python trading signal platform targeting cryptocurrency USDT-M perpetual futures. It uses an event-driven pipeline: market data → technical indicators → SMC analysis → ML scoring → risk management → Telegram signals.

The platform is now exchange-agnostic, using a fault-tolerant **ExchangeManager** that pulls public market data from Gate.io, Bitget, OKX, and Bybit, with MEXC as a last-resort fallback.

## Architecture

```
Event-Driven Pipeline:
MarketDataReceived → DataValidated → IndicatorsCalculated →
SMCAnalysisCompleted → MLPredictionCompleted → RiskValidationCompleted →
SignalScored → SignalApproved → TelegramNotificationSent → SignalStored
```

## ExchangeManager Architecture (v4.2)

The platform uses a **canonical ExchangeManager** (`exchange/manager.py`) as the single entry point for all market-data access.

### Key Features:
- **Adaptive Failover**: Transparently falls back to the next exchange if the primary one fails or returns empty data.
- **Scan Consistency**: Once a working exchange is found, it is used for the entire scan cycle to ensure data consistency across klines, tickers, and order books.
- **Public API Priority**: No exchange API keys are required for market data — all adapters use public REST endpoints.
- **Multi-Exchange Support**:
    1. **Gate.io** (Primary)
    2. **Bitget** (Secondary)
    3. **OKX** (Tertiary)
    4. **Bybit** (Optional)
    5. **MEXC** (Fallback)

## Running the Bot

```bash
python main.py --mode paper       # paper trading (default)
python main.py --mode self-test   # run self-test
python main.py --mode live        # live mode
```

## Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXCHANGE_PRIORITY` | Comma-separated adapter priority list | `gate,bitget,okx,bybit,mexc` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for notifications | (Optional) |
| `TELEGRAM_ADMIN_CHAT_ID` | Admin chat ID for notifications | (Optional) |

*Note: MEXC_ACCESS_KEY and MEXC_SECRET_KEY are no longer required for market data.*

## Key Modules

- `exchange/` — Fault-tolerant ExchangeManager and individual adapters.
- `market_data/` — Market data pipeline and candle validation.
- `scanner/` — Exchange-agnostic symbol discovery and lifecycle management.
- `analysis/indicators/` — 20+ technical indicators.
- `analysis/smc/` — Smart Money Concepts engine.
- `ml/` — XGBoost/LightGBM/CatBoost ML signal scoring.
- `signal_engine/` — Weighted signal scoring and strategy profiles.
- `risk/` — Risk management (position sizing, SL/TP).
- `notifications/` — Telegram bot with institutional formatting.
- `database/` — SQLite via SQLAlchemy async ORM.
