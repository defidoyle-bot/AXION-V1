# AXION QUANT V4

**Institutional AI Quantitative Trading Platform for USDT-M Perpetual Futures**

## Overview

AXION QUANT V4 is a production-grade async Python trading signal platform targeting cryptocurrency USDT-M perpetual futures. It uses an event-driven pipeline: market data → technical indicators → SMC analysis → ML scoring → risk management → Telegram signals.

## Architecture

```
Event-Driven Pipeline:
MarketDataReceived → DataValidated → IndicatorsCalculated →
SMCAnalysisCompleted → MLPredictionCompleted → RiskValidationCompleted →
SignalScored → SignalApproved → TelegramNotificationSent → SignalStored
```

## Exchange Adapter System (v4.1)

The platform uses a **modular Exchange Adapter Manager** (`exchange/adapter_manager.py`) with automatic fallback:

| Priority | Exchange | Status            | Notes                          |
|----------|----------|-------------------|-------------------------------|
| 1        | Gate.io  | Primary           | Public API, works on Replit   |
| 2        | Bitget   | Secondary         | Public API, works on Replit   |
| 3        | OKX      | Tertiary          | Fixed symbol mapping          |
| 4        | MEXC     | Fallback          | May be blocked on Replit      |

Symbol format everywhere is `BASE_USDT` (e.g. `BTC_USDT`). Each adapter handles its own translation:
- Gate.io: `BTC_USDT` → `BTC_USDT` (native match)
- Bitget: `BTC_USDT` → `BTCUSDT`
- **OKX fix**: `BTC_USDT` → `BTC-USDT-SWAP` (was broken before)
- MEXC: `BTC_USDT` → `BTC_USDT`

## Running the Bot

```bash
python main.py --mode paper       # paper trading (default)
python main.py --mode self-test   # run self-test
python main.py --mode live        # live mode
```

## Key Environment Variables

| Variable             | Description                                 |
|----------------------|---------------------------------------------|
| `MEXC_ACCESS_KEY`    | MEXC API access key (for MEXC fallback)      |
| `MEXC_SECRET_KEY`    | MEXC API secret key (for MEXC fallback)      |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for signal notifications |
| `TELEGRAM_ADMIN_CHAT_ID` | Admin chat ID                           |
| `EXCHANGE_PRIORITY`  | Comma-separated exchange priority list      |

## Key Modules

- `exchange/` — Exchange adapter system (Gate.io, Bitget, OKX, MEXC + base class)
- `market_data/` — Market data pipeline and candle validation
- `scanner/` — Symbol discovery and lifecycle management
- `analysis/indicators/` — 20+ technical indicators
- `analysis/smc/` — Smart Money Concepts engine
- `ml/` — XGBoost/LightGBM/CatBoost ML signal scoring
- `signal_engine/` — Weighted signal scoring and strategy profiles
- `risk/` — Risk management (position sizing, SL/TP)
- `notifications/` — Telegram bot with institutional formatting
- `backtesting/` — Backtesting engine and paper trading
- `database/` — SQLite via SQLAlchemy async ORM

## User Preferences

- Keep modular adapter architecture — do not replace working modules
- Gate.io as primary exchange, Bitget secondary, OKX tertiary, MEXC fallback
- All exchange methods use internal `BASE_USDT` symbol format
- Preserve ML engine, SMC analysis, scanner, Telegram integration
