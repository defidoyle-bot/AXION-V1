"""
AXION QUANT V4 - Core Configuration Module
Production-grade configuration with validation, env loading, and type safety.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class StrategyProfile(str, Enum):
    """Configurable strategy profiles."""
    SCALPING = "scalping"
    INTRADAY = "intraday"
    SWING = "swing"


class RiskProfile(str, Enum):
    """Configurable risk profiles."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# EXCHANGE CONFIGURATION
# =============================================================================

class ExchangeConfig(BaseModel):
    """MEXC exchange configuration."""

    access_key: str = Field(default="", description="MEXC API access key (no longer required — all endpoints are public)")
    secret_key: str = Field(default="", description="MEXC API secret key (no longer required — all endpoints are public)")
    base_url: str = Field(default="https://api.mexc.com", description="MEXC API base URL")
    futures_base_url: str = Field(default="https://contract.mexc.com", description="MEXC futures API base URL")
    testnet: bool = Field(default=True, description="Use testnet environment")
    rate_limit_per_second: int = Field(default=10, ge=1, le=100, description="API rate limit")
    timeout_seconds: int = Field(default=30, ge=5, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    retry_backoff_seconds: float = Field(default=1.0, ge=0.1, le=60.0, description="Retry backoff")

    @field_validator("access_key", "secret_key")
    @classmethod
    def validate_not_placeholder(cls, v: str) -> str:
        """Reject placeholder values but allow empty (credentials are optional)."""
        if v and ("your_" in v.lower() or "test" in v.lower()):
            raise ValueError("Exchange credentials must be provided correctly — no placeholders allowed. Set empty string to skip credential checks.")
        return v.strip() if v else ""


# =============================================================================
# MULTI-EXCHANGE ADAPTER CONFIGURATION
# =============================================================================

class MultiExchangeConfig(BaseModel):
    """Configuration for the modular exchange adapter system.

    Adapters are tried in priority order; the first successful response wins.
    Only public (unauthenticated) market-data endpoints are used.
    """

    # Priority list — lower index = higher priority
    priority: List[str] = Field(
        default=["gateio", "bitget", "okx", "mexc"],
        description="Exchange priority order for market data (comma-separated env var: EXCHANGE_PRIORITY)",
    )
    # Health check interval (seconds) for background monitoring
    health_check_interval_seconds: int = Field(default=60, ge=10, le=3600)
    # Number of consecutive failures before an exchange is considered unhealthy
    failover_after_errors: int = Field(default=3, ge=1, le=20)

    @field_validator("priority", mode="before")
    @classmethod
    def parse_priority_string(cls, v: Any) -> List[str]:
        """Accept comma-separated string from env var."""
        if isinstance(v, str):
            return [e.strip().lower() for e in v.split(",") if e.strip()]
        return v


# =============================================================================
# MARKET DATA CONFIGURATION
# =============================================================================

class Timeframe(str, Enum):
    """Supported timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class MarketDataConfig(BaseModel):
    """Market data pipeline configuration."""

    timeframes: List[Timeframe] = Field(
        default=[Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4],
        description="Active timeframes for analysis"
    )
    primary_timeframe: Timeframe = Field(default=Timeframe.H1, description="Primary analysis timeframe")
    higher_timeframe: Timeframe = Field(default=Timeframe.H4, description="Higher timeframe for trend context")
    lower_timeframe: Timeframe = Field(default=Timeframe.M15, description="Lower timeframe for entry precision")

    min_24h_volume_usdt: float = Field(default=0, ge=0, description="Minimum 24h volume in USDT")
    min_open_interest_usdt: float = Field(default=0, ge=0, description="Minimum open interest in USDT")
    max_spread_percent: float = Field(default=0.5, ge=0, le=10, description="Maximum spread percentage")
    min_liquidity_score: float = Field(default=0.3, ge=0, le=1, description="Minimum liquidity score")
    min_avg_daily_trades: int = Field(default=1000, ge=0, description="Minimum average daily trades")

    symbol_refresh_interval_minutes: int = Field(default=60, ge=1, le=1440, description="Symbol list refresh interval")
    candle_fetch_limit: int = Field(default=500, ge=50, le=2000, description="Candles to fetch per request")
    cache_ttl_seconds: int = Field(default=60, ge=0, le=3600, description="Cache time-to-live")

    coingecko_api_key: Optional[str] = Field(default=None, description="CoinGecko API key for metadata")
    coingecko_base_url: str = Field(default="https://api.coingecko.com/api/v3", description="CoinGecko API URL")

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, v: List[Timeframe]) -> List[Timeframe]:
        if len(v) < 2:
            raise ValueError("At least 2 timeframes required for multi-timeframe analysis")
        return v


# =============================================================================
# TECHNICAL INDICATORS CONFIGURATION
# =============================================================================

class IndicatorConfig(BaseModel):
    """Technical indicator parameters."""

    # Trend
    ema_periods: List[int] = Field(default=[9, 21, 50, 200], description="EMA periods")
    sma_periods: List[int] = Field(default=[50, 200], description="SMA periods")
    vwap_anchor: str = Field(default="D", description="VWAP anchor period")
    supertrend_period: int = Field(default=10, ge=2, le=50)
    supertrend_multiplier: float = Field(default=3.0, ge=0.5, le=10.0)
    ichimoku_tenkan: int = Field(default=9, ge=2, le=50)
    ichimoku_kijun: int = Field(default=26, ge=2, le=100)
    ichimoku_senkou_b: int = Field(default=52, ge=2, le=200)

    # Momentum
    rsi_period: int = Field(default=14, ge=2, le=50)
    stoch_k_period: int = Field(default=14, ge=2, le=50)
    stoch_d_period: int = Field(default=3, ge=1, le=20)
    stoch_smooth: int = Field(default=3, ge=1, le=20)
    macd_fast: int = Field(default=12, ge=2, le=50)
    macd_slow: int = Field(default=26, ge=5, le=100)
    macd_signal: int = Field(default=9, ge=2, le=50)
    cci_period: int = Field(default=20, ge=2, le=50)
    roc_period: int = Field(default=12, ge=2, le=50)

    # Volatility
    atr_period: int = Field(default=14, ge=2, le=50)
    bb_period: int = Field(default=20, ge=2, le=50)
    bb_std: float = Field(default=2.0, ge=0.5, le=5.0)
    keltner_period: int = Field(default=20, ge=2, le=50)
    keltner_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    donchian_period: int = Field(default=20, ge=2, le=100)

    # Volume
    obv_use_close: bool = Field(default=True)
    mfi_period: int = Field(default=14, ge=2, le=50)
    volume_profile_bins: int = Field(default=50, ge=10, le=200)
    relative_volume_period: int = Field(default=20, ge=2, le=100)

    # Trend Strength
    adx_period: int = Field(default=14, ge=2, le=50)
    adx_threshold: float = Field(default=25.0, ge=0, le=100)
    dmi_period: int = Field(default=14, ge=2, le=50)


# =============================================================================
# SMART MONEY CONCEPTS CONFIGURATION
# =============================================================================

class SMCConfig(BaseModel):
    """Smart Money Concepts configuration."""

    # Swing Detection
    pivot_length: int = Field(default=5, ge=2, le=20, description="Bars on each side for pivot")
    major_swing_threshold: float = Field(default=2.0, ge=0.5, le=10.0, description="ATR multiple for major swing")
    minor_swing_threshold: float = Field(default=1.0, ge=0.5, le=5.0, description="ATR multiple for minor swing")

    # Break of Structure
    bos_min_break_distance: float = Field(default=0.3, ge=0.1, le=2.0, description="Min break distance as ATR multiple")
    bos_require_displacement: bool = Field(default=True)
    bos_displacement_atr_multiple: float = Field(default=1.0, ge=0.5, le=5.0)

    # Change of Character
    choch_min_trend_bars: int = Field(default=10, ge=5, le=100, description="Min bars for trend confirmation")
    choch_require_volume: bool = Field(default=True)

    # Order Blocks
    ob_min_candles: int = Field(default=3, ge=2, le=10, description="Min candles for OB formation")
    ob_max_age_candles: int = Field(default=100, ge=10, le=500, description="Max age before OB expires")
    ob_volume_confirmation: bool = Field(default=True)
    ob_strength_threshold: float = Field(default=0.6, ge=0, le=1.0)

    # Fair Value Gaps
    fvg_min_gap_size: float = Field(default=0.2, ge=0.05, le=2.0, description="Min gap as ATR multiple")
    fvg_max_age_candles: int = Field(default=200, ge=10, le=1000)
    fvg_fill_tolerance: float = Field(default=0.1, ge=0, le=0.5, description="Tolerance for considering gap filled")

    # Liquidity Sweeps
    liquidity_equal_tolerance: float = Field(default=0.1, ge=0.01, le=1.0, description="Price tolerance for equal highs/lows")
    liquidity_wick_penetration: float = Field(default=0.3, ge=0.1, le=1.0, description="Min wick penetration")
    liquidity_recovery_bars: int = Field(default=3, ge=1, le=10, description="Bars to confirm recovery")

    # Supply/Demand Zones
    sd_zone_min_width_atr: float = Field(default=0.5, ge=0.1, le=3.0)
    sd_zone_max_age_candles: int = Field(default=150, ge=10, le=500)
    sd_zone_strength_threshold: float = Field(default=0.5, ge=0, le=1.0)

    # Premium/Discount
    premium_discount_lookback: int = Field(default=50, ge=10, le=200)
    premium_zone_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    discount_zone_threshold: float = Field(default=0.3, ge=0, le=0.5)

    # Displacement
    displacement_atr_multiple: float = Field(default=1.5, ge=0.5, le=5.0)
    displacement_body_percent: float = Field(default=0.7, ge=0.3, le=1.0)
    displacement_relative_volume: float = Field(default=1.5, ge=1.0, le=5.0)


# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

class MLModelType(str, Enum):
    """Supported ML model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    LOGISTIC_REGRESSION = "logistic_regression"


class MLConfig(BaseModel):
    """Machine Learning engine configuration."""

    enabled: bool = Field(default=True)
    model_type: MLModelType = Field(default=MLModelType.XGBOOST)
    model_version: str = Field(default="1.0.0")

    # Training
    lookback_periods: int = Field(default=500, ge=100, le=5000, description="Historical bars for training")
    prediction_horizon: int = Field(default=10, ge=1, le=100, description="Bars ahead to predict")
    retrain_interval_hours: int = Field(default=24, ge=1, le=168, description="Auto-retrain interval")
    performance_degrade_threshold: float = Field(default=0.05, ge=0.01, le=0.5, description="Trigger retrain when performance drops by this amount")

    # Validation
    walk_forward_windows: int = Field(default=5, ge=2, le=20)
    walk_forward_train_size: float = Field(default=0.7, ge=0.5, le=0.9)

    # Hyperparameter optimization
    use_optuna: bool = Field(default=True)
    optuna_trials: int = Field(default=100, ge=10, le=1000)
    optuna_timeout_minutes: int = Field(default=60, ge=5, le=300)

    # Feature engineering
    technical_features: bool = Field(default=True)
    price_action_features: bool = Field(default=True)
    market_features: bool = Field(default=True)
    time_features: bool = Field(default=True)
    trade_features: bool = Field(default=True)

    # Output calibration
    calibrate_probabilities: bool = Field(default=True)
    calibration_method: str = Field(default="isotonic", pattern="^(isotonic|platt)$")

    # Model persistence
    model_save_path: str = Field(default="models/")
    max_models_to_keep: int = Field(default=5, ge=1, le=20)

    # Feature importance tracking
    track_feature_importance: bool = Field(default=True)
    feature_importance_threshold: float = Field(default=0.01, ge=0.001, le=0.1)


# =============================================================================
# SIGNAL ENGINE CONFIGURATION
# =============================================================================

class SignalConfig(BaseModel):
    """Signal scoring and decision engine configuration."""

    # Weights (must sum to 100)
    higher_tf_trend_weight: int = Field(default=15, ge=0, le=50)
    lower_tf_trend_weight: int = Field(default=10, ge=0, le=50)
    technical_indicators_weight: int = Field(default=10, ge=0, le=50)
    smc_weight: int = Field(default=20, ge=0, le=50)
    liquidity_context_weight: int = Field(default=10, ge=0, le=50)
    volume_confirmation_weight: int = Field(default=10, ge=0, le=50)
    market_regime_weight: int = Field(default=10, ge=0, le=50)
    ml_weight: int = Field(default=5, ge=0, le=50)
    risk_management_weight: int = Field(default=5, ge=0, le=50)
    trade_quality_bonus_weight: int = Field(default=5, ge=0, le=20)

    # Score thresholds
    institutional_grade_threshold: int = Field(default=90, ge=20, le=100)
    premium_threshold: int = Field(default=85, ge=20, le=95)
    strong_threshold: int = Field(default=80, ge=20, le=90)
    standard_threshold: int = Field(default=75, ge=20, le=80)
    watchlist_threshold: int = Field(default=70, ge=19, le=70)

    # Adaptive thresholds
    adaptive_scoring_enabled: bool = Field(default=True)
    volatility_adjustment_factor: float = Field(default=0.1, ge=0, le=0.5)

    # Strategy profile overrides
    strategy_profile: StrategyProfile = Field(default=StrategyProfile.INTRADAY)

    @model_validator(mode="after")
    def validate_weights(self) -> "SignalConfig":
        total = (
            self.higher_tf_trend_weight + self.lower_tf_trend_weight +
            self.technical_indicators_weight + self.smc_weight +
            self.liquidity_context_weight + self.volume_confirmation_weight +
            self.market_regime_weight + self.ml_weight +
            self.risk_management_weight + self.trade_quality_bonus_weight
        )
        if total != 100:
            # Auto-correct by scaling weights proportionally instead of crashing
            import warnings
            warnings.warn(f"Signal weights sum to {total}, auto-correcting to 100")
            factor = 100 / total
            self.higher_tf_trend_weight = round(self.higher_tf_trend_weight * factor)
            self.lower_tf_trend_weight = round(self.lower_tf_trend_weight * factor)
            self.technical_indicators_weight = round(self.technical_indicators_weight * factor)
            self.smc_weight = round(self.smc_weight * factor)
            self.liquidity_context_weight = round(self.liquidity_context_weight * factor)
            self.volume_confirmation_weight = round(self.volume_confirmation_weight * factor)
            self.market_regime_weight = round(self.market_regime_weight * factor)
            self.ml_weight = round(self.ml_weight * factor)
            self.risk_management_weight = round(self.risk_management_weight * factor)
            self.trade_quality_bonus_weight = 100 - (
                self.higher_tf_trend_weight + self.lower_tf_trend_weight +
                self.technical_indicators_weight + self.smc_weight +
                self.liquidity_context_weight + self.volume_confirmation_weight +
                self.market_regime_weight + self.ml_weight +
                self.risk_management_weight
            )
        return self

    @model_validator(mode="after")
    def validate_thresholds(self) -> "SignalConfig":
        thresholds = [
            (self.watchlist_threshold, self.standard_threshold, "watchlist < standard"),
            (self.standard_threshold, self.strong_threshold, "standard < strong"),
            (self.strong_threshold, self.premium_threshold, "strong < premium"),
            (self.premium_threshold, self.institutional_grade_threshold, "premium < institutional"),
        ]
        for lower, upper, msg in thresholds:
            if lower >= upper:
                raise ValueError(f"Threshold violation: {msg}")
        return self


# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================

class PositionSizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED_RISK_PERCENT = "fixed_risk_percent"
    FIXED_DOLLAR_RISK = "fixed_dollar_risk"
    VOLATILITY_BASED = "volatility_based"
    FIXED_CONTRACT_SIZE = "fixed_contract_size"
    KELLY_CRITERION = "kelly_criterion"


class StopLossMethod(str, Enum):
    """Stop loss methods."""
    SWING_STRUCTURE = "swing_structure"
    ATR = "atr"
    STRUCTURE_BASED = "structure_based"
    ORDER_BLOCK = "order_block"
    FIXED_PERCENTAGE = "fixed_percentage"
    VOLATILITY = "volatility"


class TakeProfitMethod(str, Enum):
    """Take profit methods."""
    FIXED_RR = "fixed_rr"
    ATR_TARGET = "atr_target"
    STRUCTURE_TARGET = "structure_target"
    LIQUIDITY_TARGET = "liquidity_target"
    SUPPLY_DEMAND = "supply_demand"
    FIBONACCI = "fibonacci"


class RiskConfig(BaseModel):
    """Risk management engine configuration."""

    risk_profile: RiskProfile = Field(default=RiskProfile.BALANCED)

    # Position Sizing
    position_sizing_method: PositionSizingMethod = Field(default=PositionSizingMethod.FIXED_RISK_PERCENT)
    risk_percent_per_trade: float = Field(default=1.0, ge=0.1, le=10.0, description="Account % risked per trade")
    fixed_dollar_risk: float = Field(default=100.0, ge=10.0, le=10000.0)
    volatility_atr_multiple: float = Field(default=2.0, ge=0.5, le=5.0)
    fixed_contract_size: float = Field(default=0.01, ge=0.001, le=100.0)
    kelly_fraction: float = Field(default=0.25, ge=0.1, le=1.0, description="Kelly fraction (usually 0.25-0.5)")
    kelly_enabled: bool = Field(default=False)

    # Stop Loss
    stop_loss_method: StopLossMethod = Field(default=StopLossMethod.ATR)
    atr_stop_multiplier: float = Field(default=1.5, ge=0.5, le=5.0)
    fixed_stop_percent: float = Field(default=2.0, ge=0.1, le=10.0)

    # Take Profit
    take_profit_method: TakeProfitMethod = Field(default=TakeProfitMethod.FIXED_RR)
    min_risk_reward: float = Field(default=1.0, ge=1.0, le=10.0)
    target_risk_reward: float = Field(default=2.0, ge=1.0, le=10.0)
    atr_tp_multiplier: float = Field(default=3.0, ge=0.5, le=10.0)

    # Leverage
    default_leverage: int = Field(default=5, ge=1, le=125)
    max_leverage: int = Field(default=20, ge=1, le=125)
    leverage_by_strategy: Dict[StrategyProfile, int] = Field(default_factory=lambda: {
        StrategyProfile.SCALPING: 10,
        StrategyProfile.INTRADAY: 5,
        StrategyProfile.SWING: 3,
    })

    # Portfolio Controls
    max_concurrent_trades: int = Field(default=5, ge=1, le=50)
    max_exposure_per_symbol_percent: float = Field(default=20.0, ge=1.0, le=100.0)
    max_total_account_risk_percent: float = Field(default=5.0, ge=1.0, le=50.0)

    # Loss Limits
    daily_loss_limit_percent: float = Field(default=3.0, ge=0.5, le=20.0)
    weekly_loss_limit_percent: float = Field(default=6.0, ge=1.0, le=30.0)
    max_consecutive_losses: int = Field(default=5, ge=2, le=20)

    # Emergency Controls
    emergency_drawdown_percent: float = Field(default=10.0, ge=3.0, le=50.0)
    emergency_consecutive_losses: int = Field(default=7, ge=3, le=20)
    auto_resume_after_minutes: int = Field(default=60, ge=10, le=1440)

    # Risk profile overrides
    conservative_override: Dict[str, Any] = Field(default_factory=lambda: {
        "risk_percent_per_trade": 0.5,
        "default_leverage": 3,
        "max_concurrent_trades": 3,
        "min_risk_reward": 2.0,
    })
    aggressive_override: Dict[str, Any] = Field(default_factory=lambda: {
        "risk_percent_per_trade": 2.0,
        "default_leverage": 10,
        "max_concurrent_trades": 10,
        "min_risk_reward": 1.2,
    })


# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================

class TelegramConfig(BaseModel):
    """Telegram notification configuration."""

    bot_token: str = Field(..., description="Telegram bot token from BotFather")
    admin_chat_id: str = Field(..., description="Admin user/chat ID for commands")
    channel_id: Optional[str] = Field(default=None, description="Channel ID for public signals")

    # Notification settings
    send_institutional_signals: bool = Field(default=True)
    send_premium_signals: bool = Field(default=True)
    send_strong_signals: bool = Field(default=True)
    send_standard_signals: bool = Field(default=False)
    send_watchlist_signals: bool = Field(default=False)

    # Reports
    daily_report_time: str = Field(default="00:00", pattern=r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$")
    weekly_report_day: int = Field(default=0, ge=0, le=6, description="0=Monday")
    monthly_report_day: int = Field(default=1, ge=1, le=28)

    # Chart generation
    generate_charts: bool = Field(default=True)
    chart_timeframe: Timeframe = Field(default=Timeframe.H1)
    chart_lookback_candles: int = Field(default=100, ge=20, le=500)

    # Rate limiting
    max_messages_per_minute: int = Field(default=20, ge=5, le=60)
    message_queue_max_size: int = Field(default=100, ge=10, le=1000)

    @field_validator("bot_token", "admin_chat_id")
    @classmethod
    def validate_telegram_credentials(cls, v: str) -> str:
        if v and "your_" in v.lower():
            raise ValueError("Telegram credentials cannot be placeholder values")
        return v.strip() if v else ""


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = Field(default="sqlite:///data/axion_quant.db", description="Database connection string")
    echo: bool = Field(default=False, description="Log SQL queries")
    pool_size: int = Field(default=5, ge=1, le=50)
    max_overflow: int = Field(default=10, ge=0, le=50)
    pool_recycle_seconds: int = Field(default=3600, ge=300, le=7200)

    # Retention
    market_data_retention_days: int = Field(default=90, ge=7, le=365)
    signal_retention_days: int = Field(default=365, ge=30, le=1825)
    log_retention_days: int = Field(default=30, ge=7, le=90)

    # Backup
    auto_backup: bool = Field(default=True)
    backup_interval_hours: int = Field(default=24, ge=1, le=168)
    backup_path: str = Field(default="data/backups/")


# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================

class BacktestConfig(BaseModel):
    """Backtesting engine configuration."""

    initial_balance: float = Field(default=10000.0, ge=1000.0, le=1000000.0)
    fee_rate_percent: float = Field(default=0.06, ge=0.0, le=1.0, description="MEXC taker fee %")
    funding_rate_enabled: bool = Field(default=True)
    slippage_model: str = Field(default="fixed", pattern="^(fixed|variable|none)$")
    slippage_percent: float = Field(default=0.01, ge=0, le=1.0)

    # Simulation
    monte_carlo_runs: int = Field(default=1000, ge=100, le=10000)
    monte_carlo_confidence: float = Field(default=0.95, ge=0.8, le=0.99)

    # Walk-forward
    walk_forward_enabled: bool = Field(default=True)
    walk_forward_train_size: float = Field(default=0.7, ge=0.5, le=0.9)
    walk_forward_test_size: float = Field(default=0.3, ge=0.1, le=0.5)

    # Stress testing
    stress_test_scenarios: List[str] = Field(default_factory=lambda: [
        "high_volatility",
        "low_liquidity",
        "flash_crash",
        "sideways_market",
    ])


# =============================================================================
# PAPER TRADING CONFIGURATION
# =============================================================================

class PaperTradingConfig(BaseModel):
    """Paper trading configuration."""

    enabled: bool = Field(default=True)
    initial_balance: float = Field(default=10000.0, ge=1000.0, le=1000000.0)
    fee_rate_percent: float = Field(default=0.06, ge=0.0, le=1.0)
    funding_rate_enabled: bool = Field(default=True)
    slippage_percent: float = Field(default=0.01, ge=0, le=1.0)

    # Sync with live
    use_live_market_data: bool = Field(default=True)
    signal_delay_ms: int = Field(default=0, ge=0, le=5000, description="Simulated execution delay")


# =============================================================================
# MAIN APPLICATION CONFIGURATION
# =============================================================================

class AppConfig(BaseSettings):
    """Main application configuration combining all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="AXION QUANT V4", frozen=True)
    app_version: str = Field(default="4.0.0", frozen=True)
    app_codename: str = Field(default="Institutional AI Quantitative Trading Platform", frozen=True)
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # Sub-configurations
    exchange: ExchangeConfig = Field(default=None)
    multi_exchange: MultiExchangeConfig = Field(default_factory=MultiExchangeConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    smc: SMCConfig = Field(default_factory=SMCConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    telegram: TelegramConfig = Field(default=None)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)

    @model_validator(mode="before")
    @classmethod
    def load_credentials_from_env(cls, values: dict) -> dict:
        """Build sub-configs that require env var credentials."""
        if not isinstance(values, dict):
            values = {}
        # Exchange credentials
        if "exchange" not in values or values["exchange"] is None:
            values["exchange"] = {
                "access_key": os.environ.get("MEXC_ACCESS_KEY", ""),
                "secret_key": os.environ.get("MEXC_SECRET_KEY", ""),
            }
        # Multi-exchange adapter config
        if "multi_exchange" not in values or values["multi_exchange"] is None:
            priority_str = os.environ.get("EXCHANGE_PRIORITY", "gateio,bitget,okx,mexc")
            values["multi_exchange"] = {
                "priority": [e.strip().lower() for e in priority_str.split(",") if e.strip()],
            }
        # Telegram credentials
        telegram = values.get("telegram") or {}
        telegram["bot_token"] = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if "admin_chat_id" not in telegram:
            telegram["admin_chat_id"] = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "")
        if "channel_id" not in telegram:
            telegram["channel_id"] = os.environ.get("TELEGRAM_CHANNEL_ID", None)
        values["telegram"] = telegram
        # paper_trading must be a dict/object not a bool
        if "paper_trading" in values and isinstance(values["paper_trading"], bool):
            values["paper_trading"] = PaperTradingConfig(enabled=values["paper_trading"])
        return values

    # Strategy
    strategy_profile: StrategyProfile = Field(default=StrategyProfile.INTRADAY)
    risk_profile: RiskProfile = Field(default=RiskProfile.BALANCED)

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))

    # Performance
    max_workers: int = Field(default=4, ge=1, le=16)
    analysis_timeout_ms: int = Field(default=500, ge=100, le=5000)

    @model_validator(mode="after")
    def validate_paths(self) -> "AppConfig":
        """Ensure required directories exist."""
        for path_attr in ["data_dir", "models_dir", "logs_dir"]:
            path = getattr(self, path_attr)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, path_attr, path)
        return self


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class ConfigLoader:
    """Loads and validates application configuration."""

    _instance: Optional[AppConfig] = None

    @classmethod
    def load(cls, env_file: Optional[str] = None, force_reload: bool = False) -> AppConfig:
        """Load configuration from environment and files."""
        if cls._instance is not None and not force_reload:
            return cls._instance

        env_path = Path(env_file) if env_file else Path(".env")
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)

        # .env is optional when env vars are injected directly (e.g. GitHub Actions secrets)
        try:
            config = AppConfig(_env_file=env_path if env_path.exists() else None)
            cls._instance = config
            return config
        except Exception as e:
            print(f"ERROR: Failed to load configuration: {e}")
            sys.exit(1)

    @classmethod
    def reload(cls) -> AppConfig:
        """Force reload configuration."""
        cls._instance = None
        return cls.load(force_reload=True)

    @classmethod
    def get(cls) -> AppConfig:
        """Get current configuration instance."""
        if cls._instance is None:
            return cls.load()
        return cls._instance


# Global accessor (lazy-loaded, no global state mutation)
def get_config() -> AppConfig:
    """Get the current application configuration."""
    return ConfigLoader.get()
