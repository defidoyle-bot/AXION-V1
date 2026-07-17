"""
AXION QUANT V4 - Smart Money Concepts Engine
Institutional-grade market structure analysis based on validated swing structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import SMCConfig, get_config
from core.logging import get_logger

logger = get_logger("smc")


# =============================================================================
# SMC DATA MODELS
# =============================================================================

class SwingType(Enum):
    """Classification of swing points."""
    MAJOR_HIGH = auto()
    MAJOR_LOW = auto()
    MINOR_HIGH = auto()
    MINOR_LOW = auto()
    INTERNAL_HIGH = auto()
    INTERNAL_LOW = auto()
    EXTERNAL_HIGH = auto()
    EXTERNAL_LOW = auto()


class StructureType(Enum):
    """Market structure types."""
    UPTREND = auto()
    DOWNTREND = auto()
    RANGING = auto()
    UNKNOWN = auto()


class OBType(Enum):
    """Order block types."""
    BULLISH = auto()
    BEARISH = auto()


class FVGStatus(Enum):
    """Fair Value Gap status."""
    OPEN = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """A validated swing high or low."""
    index: int
    timestamp: datetime
    price: float
    swing_type: SwingType
    strength: float  # ATR multiple
    confirmed: bool
    confirmation_timestamp: Optional[datetime] = None

    def is_high(self) -> bool:
        return self.swing_type in (SwingType.MAJOR_HIGH, SwingType.MINOR_HIGH, 
                                   SwingType.INTERNAL_HIGH, SwingType.EXTERNAL_HIGH)

    def is_low(self) -> bool:
        return self.swing_type in (SwingType.MAJOR_LOW, SwingType.MINOR_LOW,
                                   SwingType.INTERNAL_LOW, SwingType.EXTERNAL_LOW)


@dataclass
class BreakOfStructure:
    """Break of Structure event."""
    index: int
    timestamp: datetime
    direction: str  # "bullish" or "bearish"
    broken_swing: SwingPoint
    break_price: float
    break_distance_atr: float
    displacement_confirmed: bool
    volume_confirmed: bool
    confidence: float


@dataclass
class ChangeOfCharacter:
    """Change of Character event."""
    index: int
    timestamp: datetime
    previous_structure: StructureType
    new_structure: StructureType
    trigger_swing: SwingPoint
    choch_swing: SwingPoint
    volume_confirmed: bool
    confidence: float


@dataclass
class OrderBlock:
    """Institutional Order Block."""
    index: int
    timestamp: datetime
    ob_type: OBType
    top: float
    bottom: float
    creation_candle: Dict[str, Any]
    mitigation_status: str  # "fresh", "mitigated", "invalidated"
    touch_count: int
    validity: bool
    strength: float
    volume_confirmed: bool
    max_age_candles: int

    @property
    def mid_price(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)


@dataclass
class FairValueGap:
    """Fair Value Gap (imbalance)."""
    index: int
    timestamp: datetime
    upper_boundary: float
    lower_boundary: float
    gap_size: float
    gap_size_atr: float
    filled_percent: float
    status: FVGStatus
    creation_timestamp: datetime
    confidence: float

    @property
    def is_bullish(self) -> bool:
        return self.gap_size > 0


@dataclass
class LiquiditySweep:
    """Liquidity sweep detection."""
    index: int
    timestamp: datetime
    sweep_type: str  # "buy_side" or "sell_side"
    liquidity_level: float
    sweep_candle: Dict[str, Any]
    recovery_confirmed: bool
    volume_confirmed: bool
    displacement_after: bool
    confidence: float


@dataclass
class SupplyDemandZone:
    """Supply or Demand zone."""
    index: int
    timestamp: datetime
    zone_type: str  # "supply" or "demand"
    top: float
    bottom: float
    creation_time: datetime
    freshness: float  # 0-1, higher is fresher
    strength: float
    tested_count: int
    max_age_candles: int


@dataclass
class PremiumDiscount:
    """Premium/Discount zone calculation."""
    equilibrium: float
    premium_zone_top: float
    premium_zone_bottom: float
    discount_zone_top: float
    discount_zone_bottom: float
    current_position: str  # "premium", "discount", "equilibrium"
    distance_from_eq_percent: float


@dataclass
class Displacement:
    """Displacement measurement."""
    index: int
    timestamp: datetime
    direction: str
    atr_multiple: float
    body_percent: float
    relative_volume: float
    momentum_score: float
    confidence: float


@dataclass
class SMCAnalysis:
    """Complete SMC analysis result."""
    symbol: str
    timeframe: str
    current_structure: StructureType
    swing_points: List[SwingPoint]
    bos_events: List[BreakOfStructure]
    choch_events: List[ChangeOfCharacter]
    order_blocks: List[OrderBlock]
    fvgs: List[FairValueGap]
    liquidity_sweeps: List[LiquiditySweep]
    supply_demand_zones: List[SupplyDemandZone]
    premium_discount: Optional[PremiumDiscount]
    displacements: List[Displacement]
    equal_highs: List[Dict[str, Any]]
    equal_lows: List[Dict[str, Any]]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_structure": self.current_structure.name,
            "swing_points": len(self.swing_points),
            "bos_events": len(self.bos_events),
            "choch_events": len(self.choch_events),
            "order_blocks": len(self.order_blocks),
            "fvgs": len(self.fvgs),
            "liquidity_sweeps": len(self.liquidity_sweeps),
            "supply_demand_zones": len(self.supply_demand_zones),
            "displacements": len(self.displacements),
            "equal_highs": len(self.equal_highs),
            "equal_lows": len(self.equal_lows),
        }


# =============================================================================
# SMC ENGINE
# =============================================================================

class SMCEngine:
    """Smart Money Concepts analysis engine."""

    def __init__(self, config: Optional[SMCConfig] = None):
        self.config = config or get_config().smc

    def analyze(self, df: pd.DataFrame, atr: pd.Series) -> SMCAnalysis:
        """Perform complete SMC analysis on price data."""
        symbol = df.get("symbol", ["UNKNOWN"]).iloc[-1] if "symbol" in df.columns else "UNKNOWN"
        timeframe = df.get("timeframe", ["UNKNOWN"]).iloc[-1] if "timeframe" in df.columns else "UNKNOWN"

        # 1. Detect swing points (foundation of all SMC)
        swing_points = self._detect_swings(df, atr)

        # 2. Determine current market structure
        current_structure = self._determine_structure(swing_points)

        # 3. Detect BOS and CHOCH
        bos_events = self._detect_bos(df, swing_points, atr)
        choch_events = self._detect_choch(df, swing_points, atr, current_structure)

        # 4. Detect Order Blocks
        order_blocks = self._detect_order_blocks(df, swing_points, atr)

        # 5. Detect Fair Value Gaps
        fvgs = self._detect_fvgs(df, atr)

        # 6. Detect Liquidity Sweeps
        liquidity_sweeps = self._detect_liquidity_sweeps(df, swing_points, atr)

        # 7. Detect Supply/Demand Zones
        supply_demand = self._detect_supply_demand(df, swing_points, atr)

        # 8. Calculate Premium/Discount
        premium_discount = self._calculate_premium_discount(df)

        # 9. Detect Displacements
        displacements = self._detect_displacements(df, atr)

        # 10. Detect Equal Highs/Lows
        equal_highs, equal_lows = self._detect_equal_levels(df)

        return SMCAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            current_structure=current_structure,
            swing_points=swing_points,
            bos_events=bos_events,
            choch_events=choch_events,
            order_blocks=order_blocks,
            fvgs=fvgs,
            liquidity_sweeps=liquidity_sweeps,
            supply_demand_zones=supply_demand,
            premium_discount=premium_discount,
            displacements=displacements,
            equal_highs=equal_highs,
            equal_lows=equal_lows,
            timestamp=datetime.now(timezone.utc),
        )

    # =================================================================
    # SWING DETECTION (Foundation)
    # =================================================================

    def _detect_swings(self, df: pd.DataFrame, atr: pd.Series) -> List[SwingPoint]:
        """Detect swing highs and lows using pivot logic."""
        pivot = self.config.pivot_length
        highs = df["high"]
        lows = df["low"]
        closes = df["close"]

        swing_points = []

        for i in range(pivot, len(df) - pivot):
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0001

            # Swing High: current high is highest in pivot window
            is_swing_high = all(highs.iloc[i] >= highs.iloc[i-j] for j in range(1, pivot+1)) and                            all(highs.iloc[i] >= highs.iloc[i+j] for j in range(1, pivot+1))

            # Swing Low: current low is lowest in pivot window
            is_swing_low = all(lows.iloc[i] <= lows.iloc[i-j] for j in range(1, pivot+1)) and                           all(lows.iloc[i] <= lows.iloc[i+j] for j in range(1, pivot+1))

            if is_swing_high:
                price_range = highs.iloc[i-pivot:i+pivot+1].max() - lows.iloc[i-pivot:i+pivot+1].min()
                strength = price_range / current_atr if current_atr > 0 else 0

                swing_type = self._classify_swing_strength(strength, "high")

                # Confirmation: price must close below swing high
                confirmed = closes.iloc[i] < highs.iloc[i]

                swing_points.append(SwingPoint(
                    index=i,
                    timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                    price=highs.iloc[i],
                    swing_type=swing_type,
                    strength=strength,
                    confirmed=confirmed,
                    confirmation_timestamp=df.index[i] if confirmed and hasattr(df.index, "__getitem__") else None,
                ))

            elif is_swing_low:
                price_range = highs.iloc[i-pivot:i+pivot+1].max() - lows.iloc[i-pivot:i+pivot+1].min()
                strength = price_range / current_atr if current_atr > 0 else 0

                swing_type = self._classify_swing_strength(strength, "low")

                # Confirmation: price must close above swing low
                confirmed = closes.iloc[i] > lows.iloc[i]

                swing_points.append(SwingPoint(
                    index=i,
                    timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                    price=lows.iloc[i],
                    swing_type=swing_type,
                    strength=strength,
                    confirmed=confirmed,
                    confirmation_timestamp=df.index[i] if confirmed and hasattr(df.index, "__getitem__") else None,
                ))

        return swing_points

    def _classify_swing_strength(self, strength: float, direction: str) -> SwingType:
        """Classify swing by strength relative to ATR."""
        if direction == "high":
            if strength >= self.config.major_swing_threshold:
                return SwingType.MAJOR_HIGH
            elif strength >= self.config.minor_swing_threshold:
                return SwingType.MINOR_HIGH
            else:
                return SwingType.INTERNAL_HIGH
        else:
            if strength >= self.config.major_swing_threshold:
                return SwingType.MAJOR_LOW
            elif strength >= self.config.minor_swing_threshold:
                return SwingType.MINOR_LOW
            else:
                return SwingType.INTERNAL_LOW

    # =================================================================
    # STRUCTURE DETERMINATION
    # =================================================================

    def _determine_structure(self, swing_points: List[SwingPoint]) -> StructureType:
        """Determine current market structure from swing points."""
        if len(swing_points) < 4:
            return StructureType.UNKNOWN

        # Get last 4 significant swings
        significant = [s for s in swing_points if s.swing_type in (
            SwingType.MAJOR_HIGH, SwingType.MAJOR_LOW, SwingType.MINOR_HIGH, SwingType.MINOR_LOW
        )]

        if len(significant) < 4:
            return StructureType.UNKNOWN

        recent = significant[-4:]

        # Higher highs and higher lows = uptrend
        hh = recent[-2].price > recent[-4].price if recent[-2].is_high() and recent[-4].is_high() else False
        hl = recent[-1].price > recent[-3].price if recent[-1].is_low() and recent[-3].is_low() else False

        # Lower highs and lower lows = downtrend
        lh = recent[-2].price < recent[-4].price if recent[-2].is_high() and recent[-4].is_high() else False
        ll = recent[-1].price < recent[-3].price if recent[-1].is_low() and recent[-3].is_low() else False

        if hh and hl:
            return StructureType.UPTREND
        elif lh and ll:
            return StructureType.DOWNTREND
        else:
            return StructureType.RANGING

    # =================================================================
    # BREAK OF STRUCTURE (BOS)
    # =================================================================

    def _detect_bos(self, df: pd.DataFrame, swing_points: List[SwingPoint], atr: pd.Series) -> List[BreakOfStructure]:
        """Detect Break of Structure events."""
        if len(swing_points) < 2:
            return []

        bos_events = []
        closes = df["close"]
        volumes = df["volume"]

        for i in range(1, len(swing_points)):
            current_swing = swing_points[i]
            previous_swing = swing_points[i-1]

            # Only check confirmed swings
            if not current_swing.confirmed or not previous_swing.confirmed:
                continue

            # Bullish BOS: current swing high breaks above previous swing high
            if current_swing.is_high() and previous_swing.is_high():
                break_distance = current_swing.price - previous_swing.price
                break_atr = break_distance / atr.iloc[current_swing.index] if atr.iloc[current_swing.index] > 0 else 0

                if break_atr >= self.config.bos_min_break_distance:
                    # Check displacement
                    displacement = False
                    if self.config.bos_require_displacement:
                        # Look for strong candle after break
                        if current_swing.index < len(df) - 1:
                            next_candle = df.iloc[current_swing.index + 1]
                            candle_body = abs(next_candle["close"] - next_candle["open"])
                            candle_atr = candle_body / atr.iloc[current_swing.index + 1] if atr.iloc[current_swing.index + 1] > 0 else 0
                            displacement = candle_atr >= self.config.bos_displacement_atr_multiple

                    # Volume confirmation
                    avg_volume = volumes.iloc[max(0, current_swing.index-20):current_swing.index].mean()
                    volume_confirmed = volumes.iloc[current_swing.index] > avg_volume * 1.2 if avg_volume > 0 else False

                    confidence = min(1.0, break_atr / self.config.bos_min_break_distance) * 0.5
                    if displacement:
                        confidence += 0.25
                    if volume_confirmed:
                        confidence += 0.25

                    bos_events.append(BreakOfStructure(
                        index=current_swing.index,
                        timestamp=current_swing.timestamp,
                        direction="bullish",
                        broken_swing=previous_swing,
                        break_price=current_swing.price,
                        break_distance_atr=break_atr,
                        displacement_confirmed=displacement,
                        volume_confirmed=volume_confirmed,
                        confidence=confidence,
                    ))

            # Bearish BOS: current swing low breaks below previous swing low
            elif current_swing.is_low() and previous_swing.is_low():
                break_distance = previous_swing.price - current_swing.price
                break_atr = break_distance / atr.iloc[current_swing.index] if atr.iloc[current_swing.index] > 0 else 0

                if break_atr >= self.config.bos_min_break_distance:
                    displacement = False
                    if self.config.bos_require_displacement:
                        if current_swing.index < len(df) - 1:
                            next_candle = df.iloc[current_swing.index + 1]
                            candle_body = abs(next_candle["close"] - next_candle["open"])
                            candle_atr = candle_body / atr.iloc[current_swing.index + 1] if atr.iloc[current_swing.index + 1] > 0 else 0
                            displacement = candle_atr >= self.config.bos_displacement_atr_multiple

                    avg_volume = volumes.iloc[max(0, current_swing.index-20):current_swing.index].mean()
                    volume_confirmed = volumes.iloc[current_swing.index] > avg_volume * 1.2 if avg_volume > 0 else False

                    confidence = min(1.0, break_atr / self.config.bos_min_break_distance) * 0.5
                    if displacement:
                        confidence += 0.25
                    if volume_confirmed:
                        confidence += 0.25

                    bos_events.append(BreakOfStructure(
                        index=current_swing.index,
                        timestamp=current_swing.timestamp,
                        direction="bearish",
                        broken_swing=previous_swing,
                        break_price=current_swing.price,
                        break_distance_atr=break_atr,
                        displacement_confirmed=displacement,
                        volume_confirmed=volume_confirmed,
                        confidence=confidence,
                    ))

        return bos_events

    # =================================================================
    # CHANGE OF CHARACTER (CHOCH)
    # =================================================================

    def _detect_choch(self, df: pd.DataFrame, swing_points: List[SwingPoint], 
                     current_structure: StructureType, atr: pd.Series) -> List[ChangeOfCharacter]:
        """Detect Change of Character events."""
        # Handle case where current_structure might be a Series due to vectorized operations
        if isinstance(current_structure, pd.Series):
            if current_structure.empty:
                return []
            current_structure = current_structure.iloc[-1]

        if current_structure == StructureType.UNKNOWN:
            return []

        choch_events = []
        volumes = df["volume"]

        # Need at least 6 significant swings for CHOCH
        significant = [s for s in swing_points if s.confirmed]
        if len(significant) < 6:
            return []

        for i in range(3, len(significant)):
            recent = significant[i-3:i+1]

            # In uptrend, CHOCH is first break below previous significant low
            if current_structure == StructureType.UPTREND:
                if recent[-1].is_low() and recent[-3].is_low():
                    if recent[-1].price < recent[-3].price:
                        # Check minimum trend bars
                        if recent[-1].index - recent[-3].index >= self.config.choch_min_trend_bars:
                            volume_confirmed = False
                            if self.config.choch_require_volume:
                                avg_vol = volumes.iloc[max(0, recent[-1].index-20):recent[-1].index].mean()
                                volume_confirmed = volumes.iloc[recent[-1].index] > avg_vol * 1.2 if avg_vol > 0 else False

                            confidence = 0.6
                            if volume_confirmed:
                                confidence += 0.2

                            choch_events.append(ChangeOfCharacter(
                                index=recent[-1].index,
                                timestamp=recent[-1].timestamp,
                                previous_structure=StructureType.UPTREND,
                                new_structure=StructureType.DOWNTREND,
                                trigger_swing=recent[-3],
                                choch_swing=recent[-1],
                                volume_confirmed=volume_confirmed,
                                confidence=confidence,
                            ))

            # In downtrend, CHOCH is first break above previous significant high
            elif current_structure == StructureType.DOWNTREND:
                if recent[-2].is_high() and recent[-4].is_high():
                    if recent[-2].price > recent[-4].price:
                        if recent[-2].index - recent[-4].index >= self.config.choch_min_trend_bars:
                            volume_confirmed = False
                            if self.config.choch_require_volume:
                                avg_vol = volumes.iloc[max(0, recent[-2].index-20):recent[-2].index].mean()
                                volume_confirmed = volumes.iloc[recent[-2].index] > avg_vol * 1.2 if avg_vol > 0 else False

                            confidence = 0.6
                            if volume_confirmed:
                                confidence += 0.2

                            choch_events.append(ChangeOfCharacter(
                                index=recent[-2].index,
                                timestamp=recent[-2].timestamp,
                                previous_structure=StructureType.DOWNTREND,
                                new_structure=StructureType.UPTREND,
                                trigger_swing=recent[-4],
                                choch_swing=recent[-2],
                                volume_confirmed=volume_confirmed,
                                confidence=confidence,
                            ))

        return choch_events

    # =================================================================
    # ORDER BLOCKS
    # =================================================================

    def _detect_order_blocks(self, df: pd.DataFrame, swing_points: List[SwingPoint], 
                            atr: pd.Series) -> List[OrderBlock]:
        """Detect institutional order blocks."""
        if len(swing_points) < 3:
            return []

        order_blocks = []

        for i in range(1, len(swing_points)):
            current = swing_points[i]
            previous = swing_points[i-1]

            if not current.confirmed:
                continue

            # Bullish OB: formed before a swing low, strong bullish candle after
            if current.is_low() and previous.is_high():
                ob_index = current.index - 1
                if ob_index < 0 or ob_index >= len(df):
                    continue

                ob_candle = df.iloc[ob_index]

                # Check for strong bullish candle after
                if current.index < len(df) - 1:
                    next_candle = df.iloc[current.index + 1]
                    body = abs(next_candle["close"] - next_candle["open"])
                    body_atr = body / atr.iloc[current.index + 1] if atr.iloc[current.index + 1] > 0 else 0

                    if body_atr >= 1.0:  # Strong displacement
                        # Volume confirmation
                        avg_vol = df["volume"].iloc[max(0, ob_index-20):ob_index].mean()
                        vol_confirmed = ob_candle["volume"] > avg_vol * 1.2 if avg_vol > 0 else False

                        strength = min(1.0, body_atr / 2.0)
                        if vol_confirmed:
                            strength += 0.2

                        order_blocks.append(OrderBlock(
                            index=ob_index,
                            timestamp=df.index[ob_index] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                            ob_type=OBType.BULLISH,
                            top=ob_candle["high"],
                            bottom=ob_candle["low"],
                            creation_candle=ob_candle.to_dict(),
                            mitigation_status="fresh",
                            touch_count=0,
                            validity=True,
                            strength=min(1.0, strength),
                            volume_confirmed=vol_confirmed,
                            max_age_candles=self.config.ob_max_age_candles,
                        ))

            # Bearish OB: formed before a swing high, strong bearish candle after
            elif current.is_high() and previous.is_low():
                ob_index = current.index - 1
                if ob_index < 0 or ob_index >= len(df):
                    continue

                ob_candle = df.iloc[ob_index]

                if current.index < len(df) - 1:
                    next_candle = df.iloc[current.index + 1]
                    body = abs(next_candle["close"] - next_candle["open"])
                    body_atr = body / atr.iloc[current.index + 1] if atr.iloc[current.index + 1] > 0 else 0

                    if body_atr >= 1.0:
                        avg_vol = df["volume"].iloc[max(0, ob_index-20):ob_index].mean()
                        vol_confirmed = ob_candle["volume"] > avg_vol * 1.2 if avg_vol > 0 else False

                        strength = min(1.0, body_atr / 2.0)
                        if vol_confirmed:
                            strength += 0.2

                        order_blocks.append(OrderBlock(
                            index=ob_index,
                            timestamp=df.index[ob_index] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                            ob_type=OBType.BEARISH,
                            top=ob_candle["high"],
                            bottom=ob_candle["low"],
                            creation_candle=ob_candle.to_dict(),
                            mitigation_status="fresh",
                            touch_count=0,
                            validity=True,
                            strength=min(1.0, strength),
                            volume_confirmed=vol_confirmed,
                            max_age_candles=self.config.ob_max_age_candles,
                        ))

        return order_blocks

    # =================================================================
    # FAIR VALUE GAPS
    # =================================================================

    def _detect_fvgs(self, df: pd.DataFrame, atr: pd.Series) -> List[FairValueGap]:
        """Detect Fair Value Gaps (3-candle imbalances)."""
        fvgs = []

        for i in range(2, len(df)):
            candle_1 = df.iloc[i-2]
            candle_2 = df.iloc[i-1]
            candle_3 = df.iloc[i]

            current_atr = atr.iloc[i] if atr.iloc[i] > 0 else 0.0001

            # Bullish FVG: candle 1 high < candle 3 low
            if candle_1["high"] < candle_3["low"]:
                gap_size = candle_3["low"] - candle_1["high"]
                gap_atr = gap_size / current_atr

                if gap_atr >= self.config.fvg_min_gap_size:
                    # Check if filled
                    filled = False
                    filled_percent = 0.0

                    if i < len(df) - 1:
                        for j in range(i+1, min(len(df), i + self.config.fvg_max_age_candles)):
                            if df.iloc[j]["low"] <= candle_1["high"]:
                                filled = True
                                filled_percent = 100.0
                                break
                            elif df.iloc[j]["low"] < candle_3["low"]:
                                filled_percent = ((candle_3["low"] - df.iloc[j]["low"]) / gap_size) * 100

                    status = FVGStatus.OPEN
                    if filled:
                        status = FVGStatus.FILLED
                    elif filled_percent > 0:
                        status = FVGStatus.PARTIALLY_FILLED

                    confidence = min(1.0, gap_atr / (self.config.fvg_min_gap_size * 2))

                    fvgs.append(FairValueGap(
                        index=i,
                        timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                        upper_boundary=candle_3["low"],
                        lower_boundary=candle_1["high"],
                        gap_size=gap_size,
                        gap_size_atr=gap_atr,
                        filled_percent=filled_percent,
                        status=status,
                        creation_timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                        confidence=confidence,
                    ))

            # Bearish FVG: candle 1 low > candle 3 high
            elif candle_1["low"] > candle_3["high"]:
                gap_size = candle_1["low"] - candle_3["high"]
                gap_atr = gap_size / current_atr

                if gap_atr >= self.config.fvg_min_gap_size:
                    filled = False
                    filled_percent = 0.0

                    if i < len(df) - 1:
                        for j in range(i+1, min(len(df), i + self.config.fvg_max_age_candles)):
                            if df.iloc[j]["high"] >= candle_1["low"]:
                                filled = True
                                filled_percent = 100.0
                                break
                            elif df.iloc[j]["high"] > candle_3["high"]:
                                filled_percent = ((df.iloc[j]["high"] - candle_3["high"]) / gap_size) * 100

                    status = FVGStatus.OPEN
                    if filled:
                        status = FVGStatus.FILLED
                    elif filled_percent > 0:
                        status = FVGStatus.PARTIALLY_FILLED

                    confidence = min(1.0, gap_atr / (self.config.fvg_min_gap_size * 2))

                    fvgs.append(FairValueGap(
                        index=i,
                        timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                        upper_boundary=candle_1["low"],
                        lower_boundary=candle_3["high"],
                        gap_size=gap_size,
                        gap_size_atr=gap_atr,
                        filled_percent=filled_percent,
                        status=status,
                        creation_timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                        confidence=confidence,
                    ))

        return fvgs

    # =================================================================
    # LIQUIDITY SWEEPS
    # =================================================================

    def _detect_liquidity_sweeps(self, df: pd.DataFrame, swing_points: List[SwingPoint], 
                                  atr: pd.Series) -> List[LiquiditySweep]:
        """Detect buy-side and sell-side liquidity sweeps."""
        sweeps = []

        if len(swing_points) < 3:
            return sweeps

        # Group swing highs and lows
        highs = [s for s in swing_points if s.is_high()]
        lows = [s for s in swing_points if s.is_low()]

        # Detect equal highs (buy-side liquidity)
        for i in range(len(highs)):
            for j in range(i+1, len(highs)):
                price_diff = abs(highs[i].price - highs[j].price)
                tolerance = highs[i].price * self.config.liquidity_equal_tolerance

                if price_diff <= tolerance:
                    # Check for sweep between these levels
                    liquidity_level = max(highs[i].price, highs[j].price)

                    for k in range(highs[j].index + 1, min(len(df), highs[j].index + self.config.liquidity_recovery_bars + 5)):
                        if k >= len(df):
                            break

                        candle = df.iloc[k]

                        # Wick penetration above equal high
                        if candle["high"] > liquidity_level:
                            wick_penetration = (candle["high"] - liquidity_level) / atr.iloc[k] if atr.iloc[k] > 0 else 0

                            if wick_penetration >= self.config.liquidity_wick_penetration:
                                # Check recovery (close back below)
                                recovery = candle["close"] < liquidity_level

                                # Volume confirmation
                                avg_vol = df["volume"].iloc[max(0, k-20):k].mean()
                                vol_confirmed = candle["volume"] > avg_vol * 1.3 if avg_vol > 0 else False

                                # Displacement after sweep
                                displacement = False
                                if k < len(df) - 1:
                                    next_candle = df.iloc[k+1]
                                    body = abs(next_candle["close"] - next_candle["open"])
                                    displacement = body / atr.iloc[k+1] > 1.0 if atr.iloc[k+1] > 0 else False

                                confidence = 0.4
                                if recovery:
                                    confidence += 0.2
                                if vol_confirmed:
                                    confidence += 0.2
                                if displacement:
                                    confidence += 0.2

                                sweeps.append(LiquiditySweep(
                                    index=k,
                                    timestamp=df.index[k] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                                    sweep_type="buy_side",
                                    liquidity_level=liquidity_level,
                                    sweep_candle=candle.to_dict(),
                                    recovery_confirmed=recovery,
                                    volume_confirmed=vol_confirmed,
                                    displacement_after=displacement,
                                    confidence=confidence,
                                ))
                                break

        # Detect equal lows (sell-side liquidity)
        for i in range(len(lows)):
            for j in range(i+1, len(lows)):
                price_diff = abs(lows[i].price - lows[j].price)
                tolerance = lows[i].price * self.config.liquidity_equal_tolerance

                if price_diff <= tolerance:
                    liquidity_level = min(lows[i].price, lows[j].price)

                    for k in range(lows[j].index + 1, min(len(df), lows[j].index + self.config.liquidity_recovery_bars + 5)):
                        if k >= len(df):
                            break

                        candle = df.iloc[k]

                        if candle["low"] < liquidity_level:
                            wick_penetration = (liquidity_level - candle["low"]) / atr.iloc[k] if atr.iloc[k] > 0 else 0

                            if wick_penetration >= self.config.liquidity_wick_penetration:
                                recovery = candle["close"] > liquidity_level

                                avg_vol = df["volume"].iloc[max(0, k-20):k].mean()
                                vol_confirmed = candle["volume"] > avg_vol * 1.3 if avg_vol > 0 else False

                                displacement = False
                                if k < len(df) - 1:
                                    next_candle = df.iloc[k+1]
                                    body = abs(next_candle["close"] - next_candle["open"])
                                    displacement = body / atr.iloc[k+1] > 1.0 if atr.iloc[k+1] > 0 else False

                                confidence = 0.4
                                if recovery:
                                    confidence += 0.2
                                if vol_confirmed:
                                    confidence += 0.2
                                if displacement:
                                    confidence += 0.2

                                sweeps.append(LiquiditySweep(
                                    index=k,
                                    timestamp=df.index[k] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                                    sweep_type="sell_side",
                                    liquidity_level=liquidity_level,
                                    sweep_candle=candle.to_dict(),
                                    recovery_confirmed=recovery,
                                    volume_confirmed=vol_confirmed,
                                    displacement_after=displacement,
                                    confidence=confidence,
                                ))
                                break

        return sweeps

    # =================================================================
    # SUPPLY/DEMAND ZONES
    # =================================================================

    def _detect_supply_demand(self, df: pd.DataFrame, swing_points: List[SwingPoint], 
                             atr: pd.Series) -> List[SupplyDemandZone]:
        """Detect supply and demand zones."""
        zones = []

        for i in range(1, len(swing_points)):
            current = swing_points[i]
            previous = swing_points[i-1]

            if not current.confirmed:
                continue

            # Demand zone: before a swing low, strong bullish move after
            if current.is_low() and previous.is_high():
                zone_start = max(0, current.index - 5)
                zone_data = df.iloc[zone_start:current.index]

                if len(zone_data) >= 2:
                    zone_top = zone_data["high"].max()
                    zone_bottom = zone_data["low"].min()
                    zone_width = zone_top - zone_bottom

                    if zone_width / atr.iloc[current.index] >= self.config.sd_zone_min_width_atr:
                        strength = min(1.0, (zone_width / atr.iloc[current.index]) / (self.config.sd_zone_min_width_atr * 2))

                        zones.append(SupplyDemandZone(
                            index=current.index,
                            timestamp=current.timestamp,
                            zone_type="demand",
                            top=zone_top,
                            bottom=zone_bottom,
                            creation_time=current.timestamp,
                            freshness=1.0,
                            strength=strength,
                            tested_count=0,
                            max_age_candles=self.config.sd_zone_max_age_candles,
                        ))

            # Supply zone: before a swing high, strong bearish move after
            elif current.is_high() and previous.is_low():
                zone_start = max(0, current.index - 5)
                zone_data = df.iloc[zone_start:current.index]

                if len(zone_data) >= 2:
                    zone_top = zone_data["high"].max()
                    zone_bottom = zone_data["low"].min()
                    zone_width = zone_top - zone_bottom

                    if zone_width / atr.iloc[current.index] >= self.config.sd_zone_min_width_atr:
                        strength = min(1.0, (zone_width / atr.iloc[current.index]) / (self.config.sd_zone_min_width_atr * 2))

                        zones.append(SupplyDemandZone(
                            index=current.index,
                            timestamp=current.timestamp,
                            zone_type="supply",
                            top=zone_top,
                            bottom=zone_bottom,
                            creation_time=current.timestamp,
                            freshness=1.0,
                            strength=strength,
                            tested_count=0,
                            max_age_candles=self.config.sd_zone_max_age_candles,
                        ))

        return zones

    # =================================================================
    # PREMIUM/DISCOUNT
    # =================================================================

    def _calculate_premium_discount(self, df: pd.DataFrame) -> PremiumDiscount:
        """Calculate premium and discount zones relative to equilibrium."""
        lookback = self.config.premium_discount_lookback

        if len(df) < lookback:
            lookback = len(df)

        recent = df.iloc[-lookback:]

        # Equilibrium = median of recent range
        equilibrium = (recent["high"].max() + recent["low"].min()) / 2
        total_range = recent["high"].max() - recent["low"].min()

        # Premium zone: upper 30% of range
        premium_zone_bottom = equilibrium + (total_range * 0.35)
        premium_zone_top = recent["high"].max()

        # Discount zone: lower 30% of range
        discount_zone_bottom = recent["low"].min()
        discount_zone_top = equilibrium - (total_range * 0.35)

        current_price = df["close"].iloc[-1]

        if current_price > premium_zone_bottom:
            position = "premium"
            distance = ((current_price - equilibrium) / (premium_zone_top - equilibrium)) * 100
        elif current_price < discount_zone_top:
            position = "discount"
            distance = ((equilibrium - current_price) / (equilibrium - discount_zone_bottom)) * 100
        else:
            position = "equilibrium"
            distance = 0.0

        return PremiumDiscount(
            equilibrium=equilibrium,
            premium_zone_top=premium_zone_top,
            premium_zone_bottom=premium_zone_bottom,
            discount_zone_top=discount_zone_top,
            discount_zone_bottom=discount_zone_bottom,
            current_position=position,
            distance_from_eq_percent=distance,
        )

    # =================================================================
    # DISPLACEMENT
    # =================================================================

    def _detect_displacements(self, df: pd.DataFrame, atr: pd.Series) -> List[Displacement]:
        """Detect displacement candles."""
        displacements = []

        for i in range(1, len(df)):
            candle = df.iloc[i]
            current_atr = atr.iloc[i] if atr.iloc[i] > 0 else 0.0001

            body = abs(candle["close"] - candle["open"])
            body_atr = body / current_atr
            body_percent = body / (candle["high"] - candle["low"]) if (candle["high"] - candle["low"]) > 0 else 0

            avg_volume = df["volume"].iloc[max(0, i-20):i].mean()
            rel_volume = candle["volume"] / avg_volume if avg_volume > 0 else 1.0

            # Check if displacement criteria met
            if body_atr >= self.config.displacement_atr_multiple and                body_percent >= self.config.displacement_body_percent and                rel_volume >= self.config.displacement_relative_volume:

                direction = "bullish" if candle["close"] > candle["open"] else "bearish"
                momentum = body_atr * rel_volume

                displacements.append(Displacement(
                    index=i,
                    timestamp=df.index[i] if hasattr(df.index, "__getitem__") else datetime.now(timezone.utc),
                    direction=direction,
                    atr_multiple=body_atr,
                    body_percent=body_percent,
                    relative_volume=rel_volume,
                    momentum_score=momentum,
                    confidence=min(1.0, momentum / 5.0),
                ))

        return displacements

    # =================================================================
    # EQUAL HIGHS/LOWS
    # =================================================================

    def _detect_equal_levels(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Detect equal highs and equal lows."""
        equal_highs = []
        equal_lows = []

        lookback = min(100, len(df))
        recent = df.iloc[-lookback:]

        # Find equal highs
        for i in range(len(recent)):
            for j in range(i+1, len(recent)):
                high_diff = abs(recent.iloc[i]["high"] - recent.iloc[j]["high"])
                tolerance = recent.iloc[i]["high"] * self.config.liquidity_equal_tolerance

                if high_diff <= tolerance:
                    equal_highs.append({
                        "index_1": i,
                        "index_2": j,
                        "price": recent.iloc[i]["high"],
                        "tolerance": tolerance,
                    })

        # Find equal lows
        for i in range(len(recent)):
            for j in range(i+1, len(recent)):
                low_diff = abs(recent.iloc[i]["low"] - recent.iloc[j]["low"])
                tolerance = recent.iloc[i]["low"] * self.config.liquidity_equal_tolerance

                if low_diff <= tolerance:
                    equal_lows.append({
                        "index_1": i,
                        "index_2": j,
                        "price": recent.iloc[i]["low"],
                        "tolerance": tolerance,
                    })

        return equal_highs, equal_lows
