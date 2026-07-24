# AXION QUANT V4 — Replit Setup

**Institutional AI Quantitative Trading Platform for USDT-M Perpetual Futures**

## How to Run

The bot is configured as the **"AXION QUANT — Live"** workflow. Start it from the workflow panel or run:

```bash
python main.py --mode live    # live signal mode
python main.py --mode paper   # paper trading mode
python main.py --mode self-test  # self-test
```

## Required Secrets (Replit Secrets panel)

| Secret | Purpose |
|--------|---------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token — signals are sent here |
| `TELEGRAM_ADMIN_CHAT_ID` | Admin chat ID for notifications |
| `TELEGRAM_CHANNEL_ID` | Channel ID for signal broadcasts |

## Optional Secrets

| Secret | Purpose |
|--------|---------|
| `MEXC_ACCESS_KEY` | MEXC API key (not required for market data) |
| `MEXC_SECRET_KEY` | MEXC API secret |
| `COINGECKO_API_KEY` | Enhanced metadata |
| `COINMARKETCAP_API_KEY` | Enhanced metadata |

## Environment Variables (set in Replit)

| Variable | Value | Notes |
|----------|-------|-------|
| `EXCHANGE_PRIORITY` | `gate,bitget,okx,bybit,mexc` | Priority order |
| `ENVIRONMENT` | `production` | |
| `LOG_LEVEL` | `INFO` | |
| `MEXC_TESTNET` | `false` | |

## ExchangeManager Architecture

| Priority | Exchange | Notes |
|----------|----------|-------|
| 1 | **Gate.io** | Primary — public API, works on Replit |
| 2 | **Bitget** | Secondary |
| 3 | **OKX** | Tertiary |
| 4 | **Bybit** | Geo-blocked on Replit — graceful failover |
| 5 | **MEXC** | Fallback |

## Signal Pipeline

```
MarketDataReceived → DataValidated → IndicatorsCalculated →
SMCAnalysisCompleted → MLPredictionCompleted → RiskValidationCompleted →
SignalScored → SignalApproved → TelegramNotificationSent → SignalStored
```

Signals require a score ≥ 75 to be approved and sent to Telegram.

## Key Modules

- `exchange/` — ExchangeManager + adapters
- `analysis/indicators/engine.py` — 20+ custom indicators (pure pandas/numpy, no ta-lib)
- `analysis/smc/engine.py` — Smart Money Concepts engine
- `ml/engine.py` — XGBoost ML scoring
- `signal_engine/engine.py` — Weighted signal scoring
- `notifications/bot.py` — Telegram bot

## Notes

- `ta-lib` and `pandas-ta` were removed from requirements — all indicators are natively implemented
- The `.env` file is intentionally empty; all credentials are in Replit Secrets
- Models are saved to `models/` and auto-loaded on restart
