# AXION QUANT V4

**Institutional AI Quantitative Trading Platform**

Version: 4.0.0 | Codename: Institutional AI Quantitative Trading Platform

---

## Overview

AXION QUANT V4 is a production-grade, modular, AI-assisted quantitative trading platform designed for institutional-grade cryptocurrency market analysis, signal generation, risk management, and future automated trading. It targets **MEXC USDT-M Perpetual Futures** exclusively.

## Architecture

```
Event-Driven Pipeline:
MarketDataReceived → DataValidated → IndicatorsCalculated → 
SMCAnalysisCompleted → MLPredictionCompleted → RiskValidationCompleted → 
SignalScored → SignalApproved → TelegramNotificationSent → SignalStored
```

## Features

### Part 1: Project Foundation & Architecture
- ✅ Event-driven async architecture
- ✅ Pydantic configuration with validation
- ✅ Production logging with secret masking
- ✅ Modular, extensible design

### Part 2: Market Data Layer
- ✅ MEXC USDT-M Perpetual Futures integration
- ✅ Automatic symbol discovery and filtering
- ✅ Candle validation (10+ checks)
- ✅ Rate limiting and retry logic

### Part 3: Analysis Engine & SMC
- ✅ 20+ technical indicators (EMA, RSI, MACD, Bollinger, ADX, etc.)
- ✅ Smart Money Concepts: Swing Detection, BOS, CHOCH, Order Blocks, FVG, Liquidity Sweeps
- ✅ Supply/Demand Zones, Premium/Discount, Displacement
- ✅ Multi-timeframe analysis

### Part 4: Machine Learning Engine
- ✅ XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Logistic Regression
- ✅ Feature engineering (Technical, Price Action, Market, Time, Trade)
- ✅ Time-series cross-validation, walk-forward testing
- ✅ Probability calibration (Isotonic/Platt)
- ✅ Auto-retraining on performance degradation

### Part 5: Signal Scoring Engine
- ✅ Weighted scoring (SMC: 20%, Trend: 25%, ML: 10%, Risk: 10%, etc.)
- ✅ Adaptive thresholds by market regime
- ✅ Strategy profiles (Scalping, Intraday, Swing)
- ✅ No hard threshold chains - every component contributes

### Part 6: Risk Management Engine
- ✅ Position sizing (Fixed %, Fixed $, Volatility-based, Kelly Criterion)
- ✅ Stop loss methods (ATR, Swing, Structure, OB, Fixed %)
- ✅ Take profit methods (Fixed RR, ATR, Structure, Liquidity)
- ✅ Leverage management by strategy
- ✅ Portfolio controls, loss limits, emergency stops
- ✅ Risk profiles (Conservative, Balanced, Aggressive)

### Part 7: Telegram Notifications
- ✅ Signal notifications with institutional formatting
- ✅ Trade lifecycle updates (TP hit, SL hit, breakeven, trailing)
- ✅ Daily/Weekly/Monthly reports
- ✅ Chart generation
- ✅ Admin commands (/status, /health, /scan, /signals, etc.)

### Part 8: Database & Backtesting
- ✅ SQLite with SQLAlchemy async ORM
- ✅ Models: MarketData, Signals, Trades, ML Predictions, Performance Metrics
- ✅ Backtesting with production logic
- ✅ Monte Carlo simulation, walk-forward testing
- ✅ Paper trading with live market data

### Part 9: DevOps
- ✅ Docker & docker-compose
- ✅ GitHub Actions CI/CD
- ✅ Self-test mode
- ✅ Health monitoring

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd axion-quant-v4
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - MEXC_ACCESS_KEY
# - MEXC_SECRET_KEY
# - COINGECKO_API_KEY
# - TELEGRAM_BOT_TOKEN
# - TELEGRAM_ADMIN_CHAT_ID
# - TELEGRAM_CHANNEL_ID
```

### 3. Run Self-Test

```bash
python main.py --self-test
```

### 4. Run Paper Trading

```bash
python main.py --mode paper
```

### 5. Run with Docker

```bash
docker-compose up -d
```

## Project Structure

```
axion_quant/
├── config/              # Configuration management
├── core/                # Event bus, logging, utilities
├── exchange/            # MEXC exchange client
├── market_data/         # Data pipeline, validation, scanner
├── analysis/            # Indicators and SMC engine
│   ├── indicators/
│   └── smc/
├── ml/                  # Machine learning engine
├── signal_engine/       # Scoring and decision engine
├── risk/                # Risk management engine
├── telegram/            # Telegram bot and notifications
├── database/            # Database models and operations
├── backtesting/         # Backtesting and paper trading
├── tests/               # Unit, integration, e2e tests
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── docker/              # Docker configuration
├── logs/                # Application logs
├── models/              # Saved ML models
└── data/                # Market data and backups
```

## Configuration

All configuration is externalized via `.env` file. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `STRATEGY_PROFILE` | Scalping/Intraday/Swing | `intraday` |
| `RISK_PROFILE` | Conservative/Balanced/Aggressive | `balanced` |
| `PAPER_TRADING` | Enable paper trading | `true` |
| `DEFAULT_RISK_PERCENT` | Risk per trade | `1.0` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Signal Classification

| Score | Classification |
|-------|---------------|
| 95-100 | 🔥 Institutional Grade |
| 90-94 | 💎 Premium Signal |
| 80-89 | ✅ Strong Signal |
| 70-79 | 📊 Standard Signal |
| 60-69 | 👀 Watchlist |
| < 60 | ❌ Reject |

## Risk Management

The Risk Engine has **veto authority**. No trade passes if:
- Daily/weekly loss limits exceeded
- Max concurrent trades reached
- Liquidation too close (< 10% distance)
- Risk/Reward below minimum
- Emergency stop triggered
- Drawdown limit exceeded

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot |
| `/help` | Show commands |
| `/status` | System status |
| `/health` | Health check |
| `/scan` | Force market scan |
| `/signals` | Recent signals |
| `/stats` | Trading statistics |
| `/performance` | Performance metrics |
| `/backtest` | Run backtest |
| `/retrain` | Retrain ML model |
| `/version` | Version info |

## Development

### Code Quality
- **Ruff**: Linting (blocks builds on failure)
- **Black**: Formatting (blocks builds on failure)
- **isort**: Import sorting (blocks builds on failure)
- **MyPy**: Type checking (blocks builds on failure)
- **pytest**: Testing (≥90% coverage required)

### Running Tests
```bash
pytest --cov=axion_quant --cov-report=term-missing --cov-fail-under=90
```

## Security

- API keys stored only in `.env` (never in code)
- Secrets masked in all log output
- No hardcoded credentials
- Input validation on all API responses

## License

Proprietary - All rights reserved.

## Disclaimer

AXION QUANT V4 is for educational and research purposes. Trading cryptocurrencies involves substantial risk. Past performance does not guarantee future results. Always use paper trading before live deployment.

---

**Built for professionals. Not a toy.**
