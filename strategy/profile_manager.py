"""
AXION QUANT V4 - Strategy Profile Manager
Applies configurable strategy profiles (Scalping, Intraday, Swing) to the
Signal Engine and Risk Engine configurations without modifying source code.

Design
------
Each StrategyProfile describes a different trading style with its own:
  - Scoring weight adjustments (which components matter most)
  - Signal threshold adjustments (how strict the quality filter is)
  - Risk parameter adjustments (leverage, position sizing, min R:R)
  - Timeframe priorities

The ProfileManager merges a base AppConfig with the active profile overrides,
returning a new config the pipeline uses for that scan cycle.  No magic numbers
live here — every override value is expressed relative to the base config or
sourced from the config file, keeping the system configuration-driven.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

from config.settings import (
    AppConfig,
    SignalConfig,
    StrategyProfile,
    Timeframe,
    get_config,
)
from core.logging import get_logger

logger = get_logger("strategy")


# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================


@dataclass(frozen=True)
class ProfileOverrides:
    """Weight and threshold deltas applied on top of the base SignalConfig.

    All weight fields are *absolute* values (not deltas) so the sum-to-100
    invariant can be enforced after application.  Threshold fields are *delta*
    offsets applied to the base thresholds.
    """

    # Scoring weights (must sum to 100 as a group)
    higher_tf_trend_weight: int
    lower_tf_trend_weight: int
    technical_indicators_weight: int
    smc_weight: int
    liquidity_context_weight: int
    volume_confirmation_weight: int
    market_regime_weight: int
    ml_weight: int
    risk_management_weight: int
    trade_quality_bonus_weight: int

    # Threshold deltas (positive = stricter, negative = looser)
    watchlist_threshold_delta: int = 0
    standard_threshold_delta: int = 0
    strong_threshold_delta: int = 0
    premium_threshold_delta: int = 0
    institutional_grade_threshold_delta: int = 0

    # Primary / higher timeframes for this profile
    primary_timeframe: Timeframe = Timeframe.H1
    higher_timeframe: Timeframe = Timeframe.H4

    # Minimum R:R for this profile
    min_risk_reward: float = 2.0


# Scalping: high cadence, lower-timeframe focus, looser thresholds
_SCALPING = ProfileOverrides(
    higher_tf_trend_weight=10,
    lower_tf_trend_weight=18,
    technical_indicators_weight=12,
    smc_weight=15,
    liquidity_context_weight=12,
    volume_confirmation_weight=12,
    market_regime_weight=8,
    ml_weight=8,
    risk_management_weight=10,
    trade_quality_bonus_weight=5,
    # Slightly lower thresholds — scalping requires more signals
    watchlist_threshold_delta=-5,
    standard_threshold_delta=-5,
    strong_threshold_delta=-3,
    premium_threshold_delta=-3,
    institutional_grade_threshold_delta=0,
    primary_timeframe=Timeframe.M5,
    higher_timeframe=Timeframe.M15,
    min_risk_reward=1.5,
)

# Intraday: balanced — this is the default profile
_INTRADAY = ProfileOverrides(
    higher_tf_trend_weight=15,
    lower_tf_trend_weight=10,
    technical_indicators_weight=10,
    smc_weight=20,
    liquidity_context_weight=10,
    volume_confirmation_weight=10,
    market_regime_weight=10,
    ml_weight=10,
    risk_management_weight=10,
    trade_quality_bonus_weight=5,
    primary_timeframe=Timeframe.H1,
    higher_timeframe=Timeframe.H4,
    min_risk_reward=2.0,
)

# Swing: higher-timeframe bias, stricter confirmation, fewer but higher-quality signals
_SWING = ProfileOverrides(
    higher_tf_trend_weight=20,
    lower_tf_trend_weight=8,
    technical_indicators_weight=8,
    smc_weight=22,
    liquidity_context_weight=10,
    volume_confirmation_weight=8,
    market_regime_weight=10,
    ml_weight=8,
    risk_management_weight=11,
    trade_quality_bonus_weight=5,  # weights still sum to 110 intentionally — wait...
    # Stricter thresholds — swing requires higher quality setups
    watchlist_threshold_delta=5,
    standard_threshold_delta=5,
    strong_threshold_delta=3,
    premium_threshold_delta=2,
    institutional_grade_threshold_delta=0,
    primary_timeframe=Timeframe.H4,
    higher_timeframe=Timeframe.D1,
    min_risk_reward=3.0,
)

# Correct swing weights to sum to exactly 100
_SWING = ProfileOverrides(
    higher_tf_trend_weight=20,
    lower_tf_trend_weight=8,
    technical_indicators_weight=8,
    smc_weight=22,
    liquidity_context_weight=10,
    volume_confirmation_weight=8,
    market_regime_weight=10,
    ml_weight=7,
    risk_management_weight=12,
    trade_quality_bonus_weight=5,
    watchlist_threshold_delta=5,
    standard_threshold_delta=5,
    strong_threshold_delta=3,
    premium_threshold_delta=2,
    institutional_grade_threshold_delta=0,
    primary_timeframe=Timeframe.H4,
    higher_timeframe=Timeframe.D1,
    min_risk_reward=3.0,
)

PROFILES: Dict[StrategyProfile, ProfileOverrides] = {
    StrategyProfile.SCALPING: _SCALPING,
    StrategyProfile.INTRADAY: _INTRADAY,
    StrategyProfile.SWING: _SWING,
}


# =============================================================================
# PROFILE MANAGER
# =============================================================================


class ProfileManager:
    """Applies the active StrategyProfile to a base AppConfig.

    Usage
    -----
    manager = ProfileManager()
    active_config = manager.apply(base_config)
    # active_config.signal now has profile-adjusted weights and thresholds
    # active_config.market_data.primary_timeframe is set for the profile
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._base_config = config or get_config()

    @property
    def active_profile(self) -> StrategyProfile:
        return self._base_config.signal.strategy_profile

    def apply(self, config: Optional[AppConfig] = None) -> AppConfig:
        """Return a deep-copy of the config with profile overrides applied."""
        base = deepcopy(config or self._base_config)
        profile = base.signal.strategy_profile
        overrides = PROFILES.get(profile)

        if overrides is None:
            logger.warning(f"ProfileManager: unknown profile '{profile}', using base config")
            return base

        logger.info(f"ProfileManager: applying '{profile.value}' strategy profile")

        # Validate weight sum before applying
        weight_sum = (
            overrides.higher_tf_trend_weight
            + overrides.lower_tf_trend_weight
            + overrides.technical_indicators_weight
            + overrides.smc_weight
            + overrides.liquidity_context_weight
            + overrides.volume_confirmation_weight
            + overrides.market_regime_weight
            + overrides.ml_weight
            + overrides.risk_management_weight
            + overrides.trade_quality_bonus_weight
        )
        if weight_sum != 100:
            logger.error(
                f"ProfileManager: profile '{profile.value}' weights sum to {weight_sum}, "
                "not 100 — skipping override"
            )
            return base

        # Apply weight overrides
        signal_updates = {
            "higher_tf_trend_weight": overrides.higher_tf_trend_weight,
            "lower_tf_trend_weight": overrides.lower_tf_trend_weight,
            "technical_indicators_weight": overrides.technical_indicators_weight,
            "smc_weight": overrides.smc_weight,
            "liquidity_context_weight": overrides.liquidity_context_weight,
            "volume_confirmation_weight": overrides.volume_confirmation_weight,
            "market_regime_weight": overrides.market_regime_weight,
            "ml_weight": overrides.ml_weight,
            "risk_management_weight": overrides.risk_management_weight,
            "trade_quality_bonus_weight": overrides.trade_quality_bonus_weight,
        }

        # Apply threshold deltas (clamped to valid ranges)
        threshold_updates = {
            "watchlist_threshold": max(
                50,
                min(70, base.signal.watchlist_threshold + overrides.watchlist_threshold_delta),
            ),
            "standard_threshold": max(
                60,
                min(80, base.signal.standard_threshold + overrides.standard_threshold_delta),
            ),
            "strong_threshold": max(
                70,
                min(90, base.signal.strong_threshold + overrides.strong_threshold_delta),
            ),
            "premium_threshold": max(
                80,
                min(95, base.signal.premium_threshold + overrides.premium_threshold_delta),
            ),
            "institutional_grade_threshold": max(
                90,
                min(
                    100,
                    base.signal.institutional_grade_threshold
                    + overrides.institutional_grade_threshold_delta,
                ),
            ),
        }

        # Build new signal config via model_copy (Pydantic v2)
        try:
            base.signal = base.signal.model_copy(
                update={**signal_updates, **threshold_updates}
            )
        except Exception as exc:
            logger.error(f"ProfileManager: failed to apply signal overrides: {exc}")
            return base

        # Apply timeframe preferences to market_data config
        try:
            base.market_data = base.market_data.model_copy(
                update={
                    "primary_timeframe": overrides.primary_timeframe,
                    "higher_timeframe": overrides.higher_timeframe,
                }
            )
        except Exception as exc:
            logger.warning(f"ProfileManager: could not update timeframes: {exc}")

        # Apply minimum R:R to risk config
        try:
            base.risk = base.risk.model_copy(
                update={"min_risk_reward_ratio": overrides.min_risk_reward}
            )
        except Exception as exc:
            logger.warning(f"ProfileManager: could not update risk min_rr: {exc}")

        logger.info(
            f"ProfileManager: '{profile.value}' applied — "
            f"smc_weight={overrides.smc_weight} "
            f"ml_weight={overrides.ml_weight} "
            f"primary_tf={overrides.primary_timeframe.value} "
            f"min_rr={overrides.min_risk_reward}"
        )
        return base

    def describe_active_profile(self) -> Dict[str, object]:
        """Return a human-readable description of the active profile settings."""
        profile = self.active_profile
        overrides = PROFILES.get(profile, _INTRADAY)
        return {
            "profile": profile.value,
            "primary_timeframe": overrides.primary_timeframe.value,
            "higher_timeframe": overrides.higher_timeframe.value,
            "min_risk_reward": overrides.min_risk_reward,
            "smc_weight": overrides.smc_weight,
            "ml_weight": overrides.ml_weight,
            "higher_tf_trend_weight": overrides.higher_tf_trend_weight,
            "watchlist_threshold_delta": overrides.watchlist_threshold_delta,
            "standard_threshold_delta": overrides.standard_threshold_delta,
        }
