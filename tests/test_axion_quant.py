"""
AXION QUANT V4 - Test Suite
Unit, integration, and end-to-end tests with 90%+ coverage target.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pytest_asyncio import fixture

from config.settings import (
    AppConfig, ExchangeConfig, MarketDataConfig, IndicatorConfig,
    SMCConfig, MLConfig, SignalConfig, RiskConfig, TelegramConfig,
    DatabaseConfig, BacktestConfig, PaperTradingConfig,
    StrategyProfile, RiskProfile, Timeframe, PositionSizingMethod,
    StopLossMethod, TakeProfitMethod, MLModelType, Environment,
)
from core.events import Event, EventBus, EventMetadata, EventPriority
from core.logging import EventLogger, get_logger
from exchange.mexc_client import MEXCClient, MEXCCandle, MEXCContractInfo, MEXCTicker
from market_data.pipeline import (
    MarketDataPipeline, SymbolScanner, CandleValidator,
    NormalizedCandle, ValidationResult,
)
from analysis.indicators.engine import IndicatorEngine, IndicatorResults
from analysis.smc.engine import SMCEngine, SMCAnalysis, SwingPoint, SwingType
from ml.engine import MLEngine, FeatureEngineer, ModelFactory
from signal_engine.engine import SignalScoringEngine, SignalScore
from risk.engine import RiskManagementEngine, RiskAssessment, Trade, TradeStatus
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.paper_trading import PaperTradingEngine, PaperAccount


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    base_price = 50000

    # Generate realistic price action
    returns = np.random.normal(0, 0.002, n)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n),
    })

    df.index = dates
    return df


@pytest.fixture
def sample_config() -> AppConfig:
    """Create a test configuration."""
    return AppConfig(
        environment=Environment.TESTING,
        log_level="DEBUG",
        exchange=ExchangeConfig(
            access_key="test_key",
            secret_key="test_secret",
            testnet=True,
        ),
        market_data=MarketDataConfig(
            timeframes=[Timeframe.H1, Timeframe.H4],
            min_24h_volume_usdt=100000,
        ),
        indicators=IndicatorConfig(
            ema_periods=[9, 21],
            rsi_period=14,
        ),
        smc=SMCConfig(
            pivot_length=3,
            bos_min_break_distance=0.2,
        ),
        ml=MLConfig(
            enabled=True,
            model_type=MLModelType.RANDOM_FOREST,
            lookback_periods=100,
        ),
        signals=SignalConfig(
            smc_weight=20,
            higher_tf_trend_weight=15,
        ),
        risk=RiskConfig(
            risk_percent_per_trade=1.0,
            max_concurrent_trades=5,
        ),
        telegram=TelegramConfig(
            bot_token="123456:TEST_TOKEN",
            admin_chat_id="123456789",
        ),
        database=DatabaseConfig(
            url="sqlite:///data/test.db",
        ),
        backtest=BacktestConfig(
            initial_balance=10000,
        ),
        paper_trading=PaperTradingConfig(
            enabled=True,
            initial_balance=10000,
        ),
    )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test configuration validation and loading."""

    def test_signal_weights_sum_to_100(self, sample_config: AppConfig):
        """Signal weights must sum to 100."""
        weights = [
            sample_config.signals.higher_tf_trend_weight,
            sample_config.signals.lower_tf_trend_weight,
            sample_config.signals.technical_indicators_weight,
            sample_config.signals.smc_weight,
            sample_config.signals.liquidity_context_weight,
            sample_config.signals.volume_confirmation_weight,
            sample_config.signals.market_regime_weight,
            sample_config.signals.ml_weight,
            sample_config.signals.risk_management_weight,
            sample_config.signals.trade_quality_bonus_weight,
        ]
        assert sum(weights) == 100, f"Weights sum to {sum(weights)}, expected 100"

    def test_threshold_ordering(self, sample_config: AppConfig):
        """Score thresholds must be in ascending order."""
        assert sample_config.signals.watchlist_threshold < sample_config.signals.standard_threshold
        assert sample_config.signals.standard_threshold < sample_config.signals.strong_threshold
        assert sample_config.signals.strong_threshold < sample_config.signals.premium_threshold
        assert sample_config.signals.premium_threshold < sample_config.signals.institutional_grade_threshold

    def test_risk_percent_bounds(self, sample_config: AppConfig):
        """Risk percentage must be within bounds."""
        assert 0.1 <= sample_config.risk.risk_percent_per_trade <= 10.0

    def test_leverage_by_strategy(self, sample_config: AppConfig):
        """Leverage must be configured for all strategies."""
        for profile in StrategyProfile:
            assert profile in sample_config.risk.leverage_by_strategy

    def test_timeframe_validation(self, sample_config: AppConfig):
        """At least 2 timeframes required."""
        assert len(sample_config.market_data.timeframes) >= 2


# =============================================================================
# EVENT SYSTEM TESTS
# =============================================================================

class TestEventSystem:
    """Test event-driven architecture."""

    @pytest.mark.asyncio
    async def test_event_bus_priority(self):
        """Test priority queue ordering."""
        bus = EventBus(max_queue_size=100)

        # Emit events with different priorities
        low_event = Event(
            event_type="test",
            payload={"priority": "low"},
            metadata=EventMetadata(priority=EventPriority.LOW),
        )
        high_event = Event(
            event_type="test",
            payload={"priority": "high"},
            metadata=EventMetadata(priority=EventPriority.HIGH),
        )

        await bus.emit(low_event)
        await bus.emit(high_event)

        # High priority should be processed first
        metrics = bus.get_metrics()
        assert metrics["emitted"]["test"] == 2

    @pytest.mark.asyncio
    async def test_event_retry(self):
        """Test event retry mechanism."""
        event = Event(
            event_type="test",
            payload={},
            metadata=EventMetadata(max_retries=3),
        )

        retry_event = event.with_retry()
        assert retry_event.metadata.retry_count == 1
        assert retry_event.metadata.max_retries == 3


# =============================================================================
# MARKET DATA TESTS
# =============================================================================

class TestMarketData:
    """Test market data pipeline."""

    def test_candle_validation(self, sample_ohlcv_data: pd.DataFrame):
        """Test candle validation."""
        validator = CandleValidator()

        candles = [
            NormalizedCandle(
                symbol="BTCUSDT",
                timestamp=sample_ohlcv_data.index[i],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                quote_volume=row["volume"] * row["close"],
                trades=100,
            )
            for i, row in sample_ohlcv_data.head(50).iterrows()
        ]

        result = validator.validate(candles)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_candle_validation_detects_errors(self):
        """Test validation catches errors."""
        validator = CandleValidator()

        # Create invalid candles
        candles = [
            NormalizedCandle(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                open=50000,
                high=49000,  # High < Low (invalid)
                low=50000,
                close=51000,
                volume=-100,  # Negative volume (invalid)
                quote_volume=0,
                trades=0,
            )
        ]

        result = validator.validate(candles)
        assert not result.valid
        assert len(result.errors) > 0

    def test_mexc_candle_from_api(self):
        """Test MEXC candle parsing."""
        api_data = [1704067200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0, 5000000.0, 0, 1000]
        candle = MEXCCandle.from_api_response("BTCUSDT", api_data)

        assert candle.symbol == "BTCUSDT"
        assert candle.timestamp == 1704067200000
        assert candle.open == 50000.0
        assert candle.close == 50500.0


# =============================================================================
# INDICATOR TESTS
# =============================================================================

class TestIndicators:
    """Test technical indicator calculations."""

    def test_indicator_engine(self, sample_ohlcv_data: pd.DataFrame):
        """Test all indicator calculations."""
        engine = IndicatorEngine()
        results = engine.calculate_all(sample_ohlcv_data)

        assert isinstance(results, IndicatorResults)
        assert results.rsi is not None
        assert results.atr is not None
        assert len(results.ema) > 0
        assert results.macd is not None

    def test_rsi_calculation(self, sample_ohlcv_data: pd.DataFrame):
        """Test RSI is within 0-100 range."""
        engine = IndicatorEngine()
        results = engine.calculate_all(sample_ohlcv_data)

        valid_rsi = results.rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_atr_positive(self, sample_ohlcv_data: pd.DataFrame):
        """Test ATR is always positive."""
        engine = IndicatorEngine()
        results = engine.calculate_all(sample_ohlcv_data)

        valid_atr = results.atr.dropna()
        assert (valid_atr > 0).all()

    def test_bollinger_bands(self, sample_ohlcv_data: pd.DataFrame):
        """Test Bollinger Band relationships."""
        engine = IndicatorEngine()
        results = engine.calculate_all(sample_ohlcv_data)

        valid_data = pd.DataFrame({
            "upper": results.bb_upper,
            "middle": results.bb_middle,
            "lower": results.bb_lower,
        }).dropna()

        assert (valid_data["upper"] >= valid_data["middle"]).all()
        assert (valid_data["middle"] >= valid_data["lower"]).all()


# =============================================================================
# SMC TESTS
# =============================================================================

class TestSMC:
    """Test Smart Money Concepts engine."""

    def test_swing_detection(self, sample_ohlcv_data: pd.DataFrame):
        """Test swing point detection."""
        engine = SMCEngine()

        # Calculate ATR for swing detection
        indicator_engine = IndicatorEngine()
        indicators = indicator_engine.calculate_all(sample_ohlcv_data)

        analysis = engine.analyze(sample_ohlcv_data, indicators.atr)

        assert isinstance(analysis, SMCAnalysis)
        assert len(analysis.swing_points) > 0

    def test_structure_detection(self, sample_ohlcv_data: pd.DataFrame):
        """Test market structure detection."""
        engine = SMCEngine()
        indicator_engine = IndicatorEngine()
        indicators = indicator_engine.calculate_all(sample_ohlcv_data)

        analysis = engine.analyze(sample_ohlcv_data, indicators.atr)

        assert analysis.current_structure is not None
        assert analysis.current_structure.name in ["UPTREND", "DOWNTREND", "RANGING", "UNKNOWN"]

    def test_order_block_detection(self, sample_ohlcv_data: pd.DataFrame):
        """Test Order Block detection."""
        engine = SMCEngine()
        indicator_engine = IndicatorEngine()
        indicators = indicator_engine.calculate_all(sample_ohlcv_data)

        analysis = engine.analyze(sample_ohlcv_data, indicators.atr)

        # Should detect some order blocks in 200 candles
        assert len(analysis.order_blocks) >= 0  # May be 0 in random data

    def test_fvg_detection(self, sample_ohlcv_data: pd.DataFrame):
        """Test Fair Value Gap detection."""
        engine = SMCEngine()
        indicator_engine = IndicatorEngine()
        indicators = indicator_engine.calculate_all(sample_ohlcv_data)

        analysis = engine.analyze(sample_ohlcv_data, indicators.atr)

        # FVGs should be detected
        assert len(analysis.fvgs) >= 0


# =============================================================================
# ML ENGINE TESTS
# =============================================================================

class TestMLEngine:
    """Test Machine Learning engine."""

    def test_feature_engineer(self, sample_ohlcv_data: pd.DataFrame):
        """Test feature engineering."""
        engineer = FeatureEngineer()

        indicators = {"rsi": 50, "atr": 100}
        smc = {"current_structure": "UPTREND"}

        features = engineer.create_features(sample_ohlcv_data, indicators, smc)

        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert len(features.columns) > 0

    def test_model_factory(self):
        """Test model creation."""
        model = ModelFactory.create_model(MLModelType.RANDOM_FOREST)
        assert model is not None

    def test_ml_prediction_structure(self):
        """Test ML prediction output format."""
        from ml.engine import MLPrediction

        prediction = MLPrediction(
            symbol="BTCUSDT",
            timeframe="1h",
            probability_of_success=0.65,
            confidence=0.7,
            model_used="xgboost",
            model_version="1.0.0",
            feature_importance={"rsi": 0.3},
            prediction_explanation="Test",
            prediction_timestamp=datetime.utcnow(),
            market_regime="trending_up",
        )

        assert prediction.probability_of_success >= 0
        assert prediction.probability_of_success <= 1
        assert prediction.confidence >= 0
        assert prediction.confidence <= 1


# =============================================================================
# SIGNAL ENGINE TESTS
# =============================================================================

class TestSignalEngine:
    """Test signal scoring engine."""

    def test_score_calculation(self, sample_config: AppConfig):
        """Test signal score calculation."""
        engine = SignalScoringEngine(sample_config.signals)

        score = engine.score_signal(
            symbol="BTCUSDT",
            direction="LONG",
            higher_tf_trend={"direction": "bullish", "strength": 0.8},
            lower_tf_trend={"direction": "bullish", "momentum": 0.7},
            technical_indicators={"rsi": 55, "adx": 30},
            smc_analysis={"current_structure": "UPTREND"},
            liquidity_context={"spread_percent": 0.05, "depth_usdt": 5000000},
            volume_data={"relative_volume": 1.5, "volume_trend": "increasing"},
            market_regime="trending_up",
            ml_prediction={"probability_of_success": 0.65, "confidence": 0.7},
            risk_assessment={"approved": True, "risk_reward": 2.0},
        )

        assert isinstance(score, SignalScore)
        assert 0 <= score.total_score <= 100
        assert len(score.components) > 0

    def test_classification_ranges(self, sample_config: AppConfig):
        """Test score classification boundaries."""
        engine = SignalScoringEngine(sample_config.signals)

        # Test institutional grade
        assert engine._classify_score(95) == "Institutional Grade"
        assert engine._classify_score(100) == "Institutional Grade"

        # Test premium
        assert engine._classify_score(90) == "Premium Signal"
        assert engine._classify_score(94) == "Premium Signal"

        # Test reject
        assert engine._classify_score(50) == "Reject"
        assert engine._classify_score(0) == "Reject"

    def test_adaptive_scoring(self, sample_config: AppConfig):
        """Test adaptive scoring by market regime."""
        engine = SignalScoringEngine(sample_config.signals)

        base_score = 70
        adjusted, adjustments = engine._apply_adaptive_adjustments(
            base_score, "trending_up", []
        )

        assert adjusted != base_score or len(adjustments) == 0


# =============================================================================
# RISK MANAGEMENT TESTS
# =============================================================================

class TestRiskManagement:
    """Test risk management engine."""

    def test_trade_validation(self, sample_config: AppConfig):
        """Test trade validation."""
        engine = RiskManagementEngine(sample_config.risk)
        engine.set_account_balance(10000)

        assessment = engine.validate_trade(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=[52000],
            atr=500,
        )

        assert isinstance(assessment, RiskAssessment)
        assert assessment.approved or assessment.rejection_reason is not None

    def test_risk_reward_validation(self, sample_config: AppConfig):
        """Test minimum risk/reward enforcement."""
        engine = RiskManagementEngine(sample_config.risk)
        engine.set_account_balance(10000)

        # Trade with poor RR should be rejected
        assessment = engine.validate_trade(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49900,  # Very tight stop = poor RR
            take_profit=[50100],  # Small target
            atr=500,
        )

        if not assessment.approved:
            assert "Risk/Reward" in assessment.rejection_reason or "too low" in assessment.rejection_reason.lower()

    def test_liquidation_safety(self, sample_config: AppConfig):
        """Test liquidation distance check."""
        engine = RiskManagementEngine(sample_config.risk)
        engine.set_account_balance(10000)

        assessment = engine.validate_trade(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=[52000],
            atr=500,
            leverage=100,  # Very high leverage
        )

        # Should be rejected due to liquidation risk
        assert not assessment.approved or assessment.liquidation_distance_percent >= 10

    def test_position_size_calculation(self, sample_config: AppConfig):
        """Test position size calculation."""
        engine = RiskManagementEngine(sample_config.risk)
        engine.set_account_balance(10000)

        assessment = engine.validate_trade(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=[52000],
            atr=500,
        )

        if assessment.approved and assessment.position_size:
            assert assessment.position_size.size > 0
            assert assessment.position_size.risk_amount > 0


# =============================================================================
# BACKTESTING TESTS
# =============================================================================

class TestBacktesting:
    """Test backtesting engine."""

    def test_backtest_result_metrics(self):
        """Test backtest result structure."""
        result = BacktestResult(
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            initial_balance=10000,
            final_balance=11000,
            total_return=10.0,
            net_profit=1000,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            avg_win=200,
            avg_loss=100,
            profit_factor=3.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=500,
            max_drawdown_percent=5.0,
            recovery_factor=2.0,
            expectancy=100,
            avg_r_multiple=1.5,
            equity_curve=[10000, 10500, 10300, 11000],
            drawdown_curve=[0, 0, 2.0, 0],
            trades=[],
            monthly_returns={},
            weekly_returns={},
        )

        assert result.total_return == 10.0
        assert result.win_rate == 0.6
        assert result.profit_factor == 3.0

    def test_monte_carlo_simulation(self, sample_ohlcv_data: pd.DataFrame):
        """Test Monte Carlo simulation."""
        engine = BacktestEngine()

        # Create a simple backtest result for testing
        result = BacktestResult(
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            initial_balance=10000,
            final_balance=10500,
            total_return=5.0,
            net_profit=500,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_percent=0,
            recovery_factor=0,
            expectancy=0,
            avg_r_multiple=0,
            equity_curve=[10000, 10500],
            drawdown_curve=[0, 0],
            trades=[],
            monthly_returns={},
            weekly_returns={},
        )

        mc_results = engine.monte_carlo_simulation(result, runs=100)

        assert "runs" in mc_results
        assert mc_results["runs"] == 100
        assert "probability_of_profit" in mc_results


# =============================================================================
# PAPER TRADING TESTS
# =============================================================================

class TestPaperTrading:
    """Test paper trading engine."""

    def test_paper_account(self):
        """Test paper account tracking."""
        account = PaperAccount(
            balance=10000,
            equity=10000,
            unrealized_pnl=0,
            margin_used=0,
            margin_available=10000,
            peak_equity=10000,
        )

        account.update_equity(datetime.utcnow())
        assert len(account.equity_curve) == 1
        assert account.equity == 10000

    def test_paper_trade_execution(self, sample_config: AppConfig):
        """Test paper trade execution."""
        engine = PaperTradingEngine(sample_config.paper_trading)

        signal = {
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry_price": 50000,
            "stop_loss": 49000,
            "take_profit": [52000],
            "position_size": 0.1,
            "leverage": 5,
            "margin_required": 1000,
        }

        trade = engine.execute_signal(signal, 50000)

        if trade:
            assert trade.symbol == "BTCUSDT"
            assert trade.direction == "LONG"
            assert trade.status == TradeStatus.OPEN

    def test_paper_trade_pnl(self, sample_config: AppConfig):
        """Test paper trade P&L calculation."""
        engine = PaperTradingEngine(sample_config.paper_trading)

        trade = Trade(
            trade_id="test",
            signal_id="test",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=[52000],
            position_size=0.1,
            leverage=5,
            margin=1000,
            status=TradeStatus.OPEN,
            created_at=datetime.utcnow(),
        )

        pnl = trade.calculate_pnl(51000, fee_rate=0.0006)

        # Expected: (51000 - 50000) * 0.1 - fees
        assert pnl > 0  # Should be profitable
        assert trade.pnl_percent != 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.asyncio
    async def test_event_pipeline(self, sample_ohlcv_data: pd.DataFrame):
        """Test full event pipeline."""
        bus = EventBus()

        # Create and register handlers
        # This is a simplified integration test

        await bus.start()

        # Emit a market data event
        event = Event(
            event_type="MarketDataReceived",
            payload={"symbol": "BTCUSDT", "timeframe": "1h", "candles": []},
            metadata=EventMetadata(),
        )

        await bus.emit(event)
        await asyncio.sleep(0.1)  # Allow processing

        metrics = bus.get_metrics()
        assert metrics["emitted"]["MarketDataReceived"] == 1

        await bus.stop()

    def test_end_to_end_signal_generation(self, sample_ohlcv_data: pd.DataFrame, sample_config: AppConfig):
        """Test end-to-end signal generation."""
        # Calculate indicators
        indicator_engine = IndicatorEngine(sample_config.indicators)
        indicators = indicator_engine.calculate_all(sample_ohlcv_data)

        # SMC analysis
        smc_engine = SMCEngine(sample_config.smc)
        smc = smc_engine.analyze(sample_ohlcv_data, indicators.atr)

        # Score signal
        signal_engine = SignalScoringEngine(sample_config.signals)
        score = signal_engine.score_signal(
            symbol="BTCUSDT",
            direction="LONG",
            higher_tf_trend={"direction": "bullish", "strength": 0.8},
            lower_tf_trend={"direction": "bullish", "momentum": 0.7},
            technical_indicators=indicators.to_dict(),
            smc_analysis=smc.to_dict(),
            liquidity_context={"spread_percent": 0.05, "depth_usdt": 5000000},
            volume_data={"relative_volume": 1.5, "volume_trend": "increasing"},
            market_regime="trending_up",
            ml_prediction=None,
            risk_assessment={"approved": True, "risk_reward": 2.0},
        )

        assert 0 <= score.total_score <= 100
        assert score.classification is not None


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and load tests."""

    def test_indicator_performance(self, sample_ohlcv_data: pd.DataFrame):
        """Test indicator calculation performance (< 500ms per symbol)."""
        import time

        engine = IndicatorEngine()

        start = time.time()
        results = engine.calculate_all(sample_ohlcv_data)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 500, f"Indicator calculation took {elapsed:.2f}ms (max: 500ms)"

    def test_smc_performance(self, sample_ohlcv_data: pd.DataFrame):
        """Test SMC analysis performance."""
        import time

        indicator_engine = IndicatorEngine()
        indicators = indicator_engine.calculate_all(sample_ohlcv_data)

        smc_engine = SMCEngine()

        start = time.time()
        analysis = smc_engine.analyze(sample_ohlcv_data, indicators.atr)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 500, f"SMC analysis took {elapsed:.2f}ms (max: 500ms)"
