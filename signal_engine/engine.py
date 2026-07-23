"""
AXION QUANT V4 - Signal Scoring & Decision Engine
Adaptive, explainable scoring that eliminates rigid all-or-nothing threshold chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config.settings import (
    SignalConfig, StrategyProfile, Timeframe, get_config,
)
from core.logging import get_logger

logger = get_logger("signal_engine")


# =============================================================================
# SIGNAL DATA MODELS
# =============================================================================

class SignalClassification(Enum):
    """Signal quality classification."""
    INSTITUTIONAL_GRADE = "Institutional Grade"
    PREMIUM = "Premium Signal"
    STRONG = "Strong Signal"
    STANDARD = "Standard Signal"
    WATCHLIST = "Watchlist"
    REJECT = "Reject"


class MarketRegime(Enum):
    """Detected market regime."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    """Individual scoring component."""
    name: str
    raw_score: float  # 0-100
    weight: int
    weighted_score: float
    details: str

    @property
    def contribution(self) -> float:
        return self.weighted_score


@dataclass(frozen=True, slots=True)
class SignalScore:
    """Complete signal score breakdown."""
    symbol: str
    direction: str  # LONG or SHORT
    total_score: int
    classification: str
    components: List[ScoreComponent]
    market_regime: str
    adaptive_adjustments: Dict[str, float]
    trade_quality_bonus: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "total_score": self.total_score,
            "classification": self.classification,
            "components": [
                {
                    "name": c.name,
                    "raw_score": round(c.raw_score, 2),
                    "weight": c.weight,
                    "weighted_score": round(c.weighted_score, 2),
                    "details": c.details,
                }
                for c in self.components
            ],
            "market_regime": self.market_regime,
            "adaptive_adjustments": self.adaptive_adjustments,
            "trade_quality_bonus": round(self.trade_quality_bonus, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TradingSignal:
    """Final approved trading signal."""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    leverage: int
    score: int
    classification: str
    risk_reward: float
    score_breakdown: Dict[str, Any]
    market_regime: str
    ml_probability: float
    ml_confidence: float
    smc_summary: str
    risk_status: str
    timestamp: datetime
    timeframe: str
    strategy_profile: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": round(self.position_size, 6),
            "leverage": self.leverage,
            "score": self.score,
            "classification": self.classification,
            "risk_reward": round(self.risk_reward, 2),
            "score_breakdown": self.score_breakdown,
            "market_regime": self.market_regime,
            "ml_probability": round(self.ml_probability, 4),
            "ml_confidence": round(self.ml_confidence, 4),
            "smc_summary": self.smc_summary,
            "risk_status": self.risk_status,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "strategy_profile": self.strategy_profile,
        }


# =============================================================================
# SIGNAL SCORING ENGINE
# =============================================================================

class SignalScoringEngine:
    """Adaptive scoring engine combining all analysis components."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or get_config().signals

    def score_signal(
        self,
        symbol: str,
        direction: str,
        higher_tf_trend: Dict[str, Any],
        lower_tf_trend: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        smc_analysis: Dict[str, Any],
        liquidity_context: Dict[str, Any],
        volume_data: Dict[str, Any],
        market_regime: str,
        ml_prediction: Optional[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
    ) -> SignalScore:
        """Calculate weighted signal score from all components."""
        components = []
        adaptive_adjustments = {}

        # 1. Higher Timeframe Trend (15 points)
        ht_score = self._score_higher_tf_trend(higher_tf_trend, direction)
        components.append(ScoreComponent(
            name="Higher Timeframe Trend",
            raw_score=ht_score,
            weight=self.config.higher_tf_trend_weight,
            weighted_score=ht_score * (self.config.higher_tf_trend_weight / 100),
            details=self._ht_trend_details(higher_tf_trend),
        ))

        # 2. Lower Timeframe Trend (10 points)
        lt_score = self._score_lower_tf_trend(lower_tf_trend, direction)
        components.append(ScoreComponent(
            name="Lower Timeframe Trend",
            raw_score=lt_score,
            weight=self.config.lower_tf_trend_weight,
            weighted_score=lt_score * (self.config.lower_tf_trend_weight / 100),
            details=self._lt_trend_details(lower_tf_trend),
        ))

        # 3. Technical Indicators (10 points)
        ti_score = self._score_technical_indicators(technical_indicators, direction)
        components.append(ScoreComponent(
            name="Technical Indicators",
            raw_score=ti_score,
            weight=self.config.technical_indicators_weight,
            weighted_score=ti_score * (self.config.technical_indicators_weight / 100),
            details=self._ti_details(technical_indicators),
        ))

        # 4. Smart Money Concepts (20 points) - HIGHEST WEIGHT
        smc_score = self._score_smc(smc_analysis, direction)
        components.append(ScoreComponent(
            name="Smart Money Concepts",
            raw_score=smc_score,
            weight=self.config.smc_weight,
            weighted_score=smc_score * (self.config.smc_weight / 100),
            details=self._smc_details(smc_analysis),
        ))

        # 5. Liquidity Context (10 points)
        liq_score = self._score_liquidity(liquidity_context, direction)
        components.append(ScoreComponent(
            name="Liquidity Context",
            raw_score=liq_score,
            weight=self.config.liquidity_context_weight,
            weighted_score=liq_score * (self.config.liquidity_context_weight / 100),
            details=self._liquidity_details(liquidity_context),
        ))

        # 6. Volume Confirmation (10 points)
        vol_score = self._score_volume(volume_data, direction)
        components.append(ScoreComponent(
            name="Volume Confirmation",
            raw_score=vol_score,
            weight=self.config.volume_confirmation_weight,
            weighted_score=vol_score * (self.config.volume_confirmation_weight / 100),
            details=self._volume_details(volume_data),
        ))

        # 7. Market Regime Alignment (10 points)
        regime_score = self._score_regime_alignment(market_regime, direction)
        components.append(ScoreComponent(
            name="Market Regime Alignment",
            raw_score=regime_score,
            weight=self.config.market_regime_weight,
            weighted_score=regime_score * (self.config.market_regime_weight / 100),
            details=f"Regime: {market_regime}",
        ))

        # 8. Machine Learning (10 points)
        ml_score = self._score_ml(ml_prediction, direction)
        components.append(ScoreComponent(
            name="Machine Learning",
            raw_score=ml_score,
            weight=self.config.ml_weight,
            weighted_score=ml_score * (self.config.ml_weight / 100),
            details=self._ml_details(ml_prediction),
        ))

        # 9. Risk Management (10 points)
        risk_score = self._score_risk(risk_assessment)
        components.append(ScoreComponent(
            name="Risk Management",
            raw_score=risk_score,
            weight=self.config.risk_management_weight,
            weighted_score=risk_score * (self.config.risk_management_weight / 100),
            details=self._risk_details(risk_assessment),
        ))

        # Calculate base total
        base_total = sum(c.weighted_score for c in components)

        # Apply adaptive adjustments based on market regime
        if self.config.adaptive_scoring_enabled:
            base_total, adaptive_adjustments = self._apply_adaptive_adjustments(
                base_total, market_regime, components
            )

        # Trade Quality Bonus (5 points)
        quality_bonus = self._calculate_quality_bonus(
            components, risk_assessment, ml_prediction
        )
        quality_weighted = quality_bonus * (self.config.trade_quality_bonus_weight / 100)

        components.append(ScoreComponent(
            name="Trade Quality Bonus",
            raw_score=quality_bonus,
            weight=self.config.trade_quality_bonus_weight,
            weighted_score=quality_weighted,
            details=self._quality_details(quality_bonus),
        ))

        # Final score
        raw = base_total + quality_weighted
        if raw != raw:  # NaN guard
            raw = 50.0
        total_score = min(100, max(0, round(raw)))

        # Classification
        classification = self._classify_score(total_score)

        return SignalScore(
            symbol=symbol,
            direction=direction,
            total_score=total_score,
            classification=classification,
            components=components,
            market_regime=market_regime,
            adaptive_adjustments=adaptive_adjustments,
            trade_quality_bonus=quality_bonus,
            timestamp=datetime.utcnow(),
        )

    # =================================================================
    # COMPONENT SCORING METHODS
    # =================================================================

    def _score_higher_tf_trend(self, trend: Dict[str, Any], direction: str) -> float:
        """Score higher timeframe trend alignment (0-100)."""
        if not trend:
            return 50.0

        trend_direction = trend.get("direction", "neutral")
        strength = trend.get("strength", 0.5)

        if direction == "LONG":
            if trend_direction == "bullish":
                return 70 + (strength * 30)
            elif trend_direction == "bearish":
                return 30 - (strength * 30)
            else:
                return 50.0
        else:  # SHORT
            if trend_direction == "bearish":
                return 70 + (strength * 30)
            elif trend_direction == "bullish":
                return 30 - (strength * 30)
            else:
                return 50.0

    def _score_lower_tf_trend(self, trend: Dict[str, Any], direction: str) -> float:
        """Score lower timeframe trend alignment (0-100)."""
        if not trend:
            return 50.0

        trend_direction = trend.get("direction", "neutral")
        momentum = trend.get("momentum", 0.5)

        if direction == "LONG":
            if trend_direction == "bullish":
                return 60 + (momentum * 40)
            elif trend_direction == "bearish":
                return 40 - (momentum * 40)
            else:
                return 50.0
        else:
            if trend_direction == "bearish":
                return 60 + (momentum * 40)
            elif trend_direction == "bullish":
                return 40 - (momentum * 40)
            else:
                return 50.0

    def _score_technical_indicators(self, indicators: Dict[str, Any], direction: str) -> float:
        """Score technical indicator alignment (0-100)."""
        if not indicators:
            return 50.0

        scores = []

        # RSI
        if "rsi" in indicators:
            rsi = max(0, min(100, float(indicators["rsi"])))
            if direction == "LONG":
                # Longs benefit from lower RSI (oversold-to-neutral room to rise)
                scores.append(100 - rsi)
            else:
                # Shorts benefit from higher RSI (overbought-to-neutral room to fall)
                scores.append(rsi)

        # MACD
        if "macd_above_signal" in indicators:
            macd_aligned = indicators["macd_above_signal"]
            if direction == "LONG":
                scores.append(80 if macd_aligned else 20)
            else:
                scores.append(20 if macd_aligned else 80)

        # ADX
        if "adx" in indicators:
            adx = indicators["adx"]
            scores.append(min(100, adx * 4))  # Scale ADX to 0-100

        # Bollinger position
        if "bb_position" in indicators:
            bb_pos = indicators["bb_position"]
            if direction == "LONG":
                scores.append((1 - bb_pos) * 100)  # Lower is better for longs
            else:
                scores.append(bb_pos * 100)  # Higher is better for shorts

        return np.mean(scores) if scores else 50.0

    def _score_smc(self, smc: Dict[str, Any], direction: str) -> float:
        """Score Smart Money Concepts alignment (0-100)."""
        if not smc:
            return 50.0

        scores = []

        # Structure alignment
        structure = smc.get("current_structure", "UNKNOWN")
        if direction == "LONG":
            if structure == "UPTREND":
                scores.append(90)
            elif structure == "DOWNTREND":
                scores.append(20)
            else:
                scores.append(50)
        else:
            if structure == "DOWNTREND":
                scores.append(90)
            elif structure == "UPTREND":
                scores.append(20)
            else:
                scores.append(50)

        # BOS confirmation
        bos_events = smc.get("bos_events", [])
        if isinstance(bos_events, int):
            bos_events = []
        if bos_events:
            latest_bos = bos_events[-1]
            if latest_bos.get("direction") == "bullish" and direction == "LONG":
                scores.append(85)
            elif latest_bos.get("direction") == "bearish" and direction == "SHORT":
                scores.append(85)
            else:
                scores.append(40)

        # Order block presence
        obs = smc.get("order_blocks", [])
        if obs:
            relevant_obs = [
                ob for ob in obs
                if (ob.get("ob_type") == "BULLISH" and direction == "LONG") or
                   (ob.get("ob_type") == "BEARISH" and direction == "SHORT")
            ]
            if relevant_obs:
                avg_strength = np.mean([ob.get("strength", 0.5) for ob in relevant_obs])
                scores.append(60 + (avg_strength * 40))

        # FVG alignment
        fvgs = smc.get("fvgs", [])
        if isinstance(fvgs, int):
            fvgs = []
        if fvgs:
            open_fvgs = [f for f in fvgs if f.get("status") == "OPEN"]
            if open_fvgs:
                scores.append(70)

        # Liquidity sweep
        sweeps = smc.get("liquidity_sweeps", [])
        if isinstance(sweeps, int):
            sweeps = []
        if sweeps:
            latest_sweep = sweeps[-1]
            if latest_sweep.get("sweep_type") == "sell_side" and direction == "LONG":
                scores.append(80)
            elif latest_sweep.get("sweep_type") == "buy_side" and direction == "SHORT":
                scores.append(80)

        return np.mean(scores) if scores else 50.0

    def _score_liquidity(self, liquidity: Dict[str, Any], direction: str) -> float:
        """Score liquidity context (0-100)."""
        if not liquidity:
            return 50.0

        spread_percent = liquidity.get("spread_percent", 0.1)
        depth = liquidity.get("depth_usdt", 1000000)

        # Lower spread is better
        spread_score = max(0, 100 - (spread_percent * 200))

        # Higher depth is better
        depth_score = min(100, depth / 100000)

        return (spread_score + depth_score) / 2

    def _score_volume(self, volume: Dict[str, Any], direction: str) -> float:
        """Score volume confirmation (0-100)."""
        if not volume:
            return 50.0

        rel_volume = volume.get("relative_volume", 1.0)
        volume_trend = volume.get("volume_trend", "neutral")

        # Higher relative volume is better
        vol_score = min(100, rel_volume * 50)

        # Volume trend alignment
        if direction == "LONG" and volume_trend == "increasing":
            vol_score += 10
        elif direction == "SHORT" and volume_trend == "increasing":
            vol_score += 10

        return min(100, vol_score)

    def _score_regime_alignment(self, regime: str, direction: str) -> float:
        """Score market regime alignment (0-100)."""
        regime_scores = {
            ("trending_up", "LONG"): 90,
            ("trending_down", "SHORT"): 90,
            ("trending_up", "SHORT"): 30,
            ("trending_down", "LONG"): 30,
            ("ranging", "LONG"): 60,
            ("ranging", "SHORT"): 60,
            ("high_volatility", "LONG"): 50,
            ("high_volatility", "SHORT"): 50,
            ("low_volatility", "LONG"): 65,
            ("low_volatility", "SHORT"): 65,
            ("unknown", "LONG"): 50,
            ("unknown", "SHORT"): 50,
        }

        return regime_scores.get((regime, direction), 50.0)

    def _score_ml(self, ml: Optional[Dict[str, Any]], direction: str) -> float:
        """Score ML prediction alignment (0-100)."""
        if not ml:
            return 50.0

        probability = ml.get("probability_of_success", 0.5)
        confidence = ml.get("confidence", 0.0)

        # Guard against NaN from failed probability calibration
        if probability is None or (isinstance(probability, float) and (probability != probability)):
            probability = 0.5
        if confidence is None or (isinstance(confidence, float) and (confidence != confidence)):
            confidence = 0.0

        # Direction-aware ML scoring
        ml_direction = ml.get("direction", None)
        if ml_direction == direction:
            # Model was trained for this direction — probability is already success chance
            base_score = probability * 100
        else:
            # Legacy / no direction tag — probability = chance of upward move
            if direction == "LONG":
                base_score = probability * 100
            else:
                base_score = (1 - probability) * 100

        # Weight by confidence
        return base_score * (0.5 + confidence * 0.5)

    def _score_risk(self, risk: Dict[str, Any]) -> float:
        """Score risk assessment (0-100)."""
        if not risk:
            return 50.0

        if not risk.get("approved", False):
            return 0.0

        scores = []

        # Risk/Reward ratio
        rr = risk.get("risk_reward", 1.5)
        if rr >= 3.0:
            scores.append(100)
        elif rr >= 2.0:
            scores.append(90)
        elif rr >= 1.5:
            scores.append(80)
        elif rr >= 1.0:
            scores.append(60)
        else:
            scores.append(30)

        # Position size appropriateness
        position_risk = risk.get("position_risk_percent", 1.0)
        if position_risk <= 1.0:
            scores.append(100)
        elif position_risk <= 2.0:
            scores.append(80)
        elif position_risk <= 3.0:
            scores.append(60)
        else:
            scores.append(30)

        # Liquidation safety
        liquidation_distance = risk.get("liquidation_distance_percent", 50)
        if liquidation_distance >= 50:
            scores.append(100)
        elif liquidation_distance >= 30:
            scores.append(80)
        elif liquidation_distance >= 20:
            scores.append(60)
        else:
            scores.append(20)

        return np.mean(scores) if scores else 50.0

    # =================================================================
    # ADAPTIVE SCORING
    # =================================================================

    def _apply_adaptive_adjustments(
        self,
        base_total: float,
        regime: str,
        components: List[ScoreComponent],
    ) -> Tuple[float, Dict[str, float]]:
        """Apply adaptive adjustments based on market regime."""
        adjustments = {}

        if regime == "trending_up":
            # Increase trend and momentum weights
            for c in components:
                if c.name in ["Higher Timeframe Trend", "Lower Timeframe Trend"]:
                    adjustment = base_total * 0.05
                    base_total += adjustment
                    adjustments["trend_boost"] = adjustment
                    break

        elif regime == "trending_down":
            for c in components:
                if c.name in ["Higher Timeframe Trend", "Lower Timeframe Trend"]:
                    adjustment = base_total * 0.05
                    base_total += adjustment
                    adjustments["trend_boost"] = adjustment
                    break

        elif regime == "ranging":
            # Increase mean reversion and S/R weights
            for c in components:
                if c.name in ["Smart Money Concepts", "Liquidity Context"]:
                    adjustment = base_total * 0.05
                    base_total += adjustment
                    adjustments["range_boost"] = adjustment
                    break

        elif regime == "high_volatility":
            # Increase ATR weighting and risk penalties
            for c in components:
                if c.name == "Risk Management":
                    adjustment = base_total * 0.05
                    base_total += adjustment
                    adjustments["volatility_risk_adjustment"] = adjustment
                    break

            # Penalty for low scores in high volatility
            if base_total < 70:
                penalty = base_total * self.config.volatility_adjustment_factor
                base_total -= penalty
                adjustments["volatility_penalty"] = -penalty

        elif regime == "low_volatility":
            # Slightly lower threshold
            adjustment = base_total * 0.02
            base_total += adjustment
            adjustments["low_volatility_boost"] = adjustment

        return base_total, adjustments

    def _calculate_quality_bonus(
        self,
        components: List[ScoreComponent],
        risk: Dict[str, Any],
        ml: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate trade quality bonus (0-100)."""
        bonus = 0.0

        # High component scores
        avg_component = np.mean([c.raw_score for c in components])
        if avg_component > 80:
            bonus += 30
        elif avg_component > 70:
            bonus += 20
        elif avg_component > 60:
            bonus += 10

        # Excellent risk/reward
        rr = risk.get("risk_reward", 1.0)
        if rr >= 3.0:
            bonus += 25
        elif rr >= 2.5:
            bonus += 20
        elif rr >= 2.0:
            bonus += 15

        # High ML confidence
        if ml and ml.get("confidence", 0) > 0.8:
            bonus += 20
        elif ml and ml.get("confidence", 0) > 0.6:
            bonus += 10

        # Strong SMC confirmation
        smc_component = next((c for c in components if c.name == "Smart Money Concepts"), None)
        if smc_component and smc_component.raw_score > 85:
            bonus += 15

        return min(100, bonus)

    # =================================================================
    # CLASSIFICATION
    # =================================================================

    def _classify_score(self, score: int) -> str:
        """Classify score into signal quality tier."""
        if score >= self.config.institutional_grade_threshold:
            return SignalClassification.INSTITUTIONAL_GRADE.value
        elif score >= self.config.premium_threshold:
            return SignalClassification.PREMIUM.value
        elif score >= self.config.strong_threshold:
            return SignalClassification.STRONG.value
        elif score >= self.config.standard_threshold:
            return SignalClassification.STANDARD.value
        elif score >= self.config.watchlist_threshold:
            return SignalClassification.WATCHLIST.value
        else:
            return SignalClassification.REJECT.value

    # =================================================================
    # DETAIL STRINGS
    # =================================================================

    def _ht_trend_details(self, trend: Dict[str, Any]) -> str:
        return f"{trend.get('direction', 'unknown')} (strength: {trend.get('strength', 0):.2f})"

    def _lt_trend_details(self, trend: Dict[str, Any]) -> str:
        return f"{trend.get('direction', 'unknown')} (momentum: {trend.get('momentum', 0):.2f})"

    def _ti_details(self, indicators: Dict[str, Any]) -> str:
        details = []
        if "rsi" in indicators:
            details.append(f"RSI: {indicators['rsi']:.1f}")
        if "adx" in indicators:
            details.append(f"ADX: {indicators['adx']:.1f}")
        return ", ".join(details) if details else "No dominant indicators"

    def _smc_details(self, smc: Dict[str, Any]) -> str:
        details = [f"Structure: {smc.get('current_structure', 'unknown')}"]
        bos_events = smc.get("bos_events", [])
        if isinstance(bos_events, list) and bos_events:
            details.append(f"BOS: {len(bos_events)}")
        order_blocks = smc.get("order_blocks", [])
        if isinstance(order_blocks, list) and order_blocks:
            details.append(f"OBs: {len(order_blocks)}")
        return ", ".join(details)

    def _liquidity_details(self, liquidity: Dict[str, Any]) -> str:
        return f"Spread: {liquidity.get('spread_percent', 0):.3f}%, Depth: ${liquidity.get('depth_usdt', 0):,.0f}"

    def _volume_details(self, volume: Dict[str, Any]) -> str:
        return f"RelVol: {volume.get('relative_volume', 1):.2f}x, Trend: {volume.get('volume_trend', 'neutral')}"

    def _ml_details(self, ml: Optional[Dict[str, Any]]) -> str:
        if not ml:
            return "No ML prediction available"
        return f"Prob: {ml.get('probability_of_success', 0):.1%}, Conf: {ml.get('confidence', 0):.1%}"

    def _risk_details(self, risk: Dict[str, Any]) -> str:
        if not risk.get("approved", False):
            return f"REJECTED: {risk.get('rejection_reason', 'Unknown')}"
        return f"RR: {risk.get('risk_reward', 0):.2f}, Risk: {risk.get('position_risk_percent', 0):.2f}%"

    def _quality_details(self, bonus: float) -> str:
        if bonus >= 80:
            return "Exceptional trade setup quality"
        elif bonus >= 60:
            return "High quality trade setup"
        elif bonus >= 40:
            return "Good trade setup quality"
        else:
            return "Standard trade setup quality"

    # =================================================================
    # STRATEGY PROFILE OVERRIDES
    # =================================================================

    def apply_strategy_profile(
        self,
        score: SignalScore,
        profile: StrategyProfile,
    ) -> SignalScore:
        """Apply strategy profile adjustments to score."""
        adjustments = {
            StrategyProfile.SCALPING: {
                "timeframe_preference": "lower",
                "min_score": 75,
                "volatility_preference": "high",
            },
            StrategyProfile.INTRADAY: {
                "timeframe_preference": "balanced",
                "min_score": 70,
                "volatility_preference": "moderate",
            },
            StrategyProfile.SWING: {
                "timeframe_preference": "higher",
                "min_score": 80,
                "volatility_preference": "low",
            },
        }

        profile_config = adjustments.get(profile, adjustments[StrategyProfile.INTRADAY])

        # Apply profile-specific adjustments
        adjusted_score = score.total_score

        if profile == StrategyProfile.SCALPING:
            # Scalping: favor volume and momentum
            for c in score.components:
                if c.name in ["Volume Confirmation", "Lower Timeframe Trend"]:
                    adjusted_score += 2

        elif profile == StrategyProfile.SWING:
            # Swing: favor higher timeframe and structure
            for c in score.components:
                if c.name in ["Higher Timeframe Trend", "Smart Money Concepts"]:
                    adjusted_score += 2

        # Recalculate classification if score changed significantly
        if abs(adjusted_score - score.total_score) > 3:
            new_classification = self._classify_score(adjusted_score)
        else:
            new_classification = score.classification

        # Return new score with adjusted values
        return SignalScore(
            symbol=score.symbol,
            direction=score.direction,
            total_score=min(100, adjusted_score),
            classification=new_classification,
            components=score.components,
            market_regime=score.market_regime,
            adaptive_adjustments=score.adaptive_adjustments,
            trade_quality_bonus=score.trade_quality_bonus,
            timestamp=score.timestamp,
        )
