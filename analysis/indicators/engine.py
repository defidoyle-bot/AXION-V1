"""
AXION QUANT V4 - Technical Indicators Engine
Comprehensive indicator calculations with caching and shared computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import IndicatorConfig, get_config
from core.logging import get_logger

logger = get_logger("indicators")


# =============================================================================
# INDICATOR RESULTS
# =============================================================================

@dataclass
class IndicatorResults:
    """Container for all calculated indicators."""

    # Trend
    ema: Dict[int, pd.Series] = field(default_factory=dict)
    sma: Dict[int, pd.Series] = field(default_factory=dict)
    vwap: Optional[pd.Series] = None
    supertrend: Optional[pd.Series] = None
    supertrend_direction: Optional[pd.Series] = None
    ichimoku: Optional[Dict[str, pd.Series]] = None

    # Momentum
    rsi: Optional[pd.Series] = None
    stoch_k: Optional[pd.Series] = None
    stoch_d: Optional[pd.Series] = None
    macd: Optional[pd.Series] = None
    macd_signal: Optional[pd.Series] = None
    macd_histogram: Optional[pd.Series] = None
    cci: Optional[pd.Series] = None
    roc: Optional[pd.Series] = None

    # Volatility
    atr: Optional[pd.Series] = None
    bb_upper: Optional[pd.Series] = None
    bb_middle: Optional[pd.Series] = None
    bb_lower: Optional[pd.Series] = None
    keltner_upper: Optional[pd.Series] = None
    keltner_middle: Optional[pd.Series] = None
    keltner_lower: Optional[pd.Series] = None
    donchian_upper: Optional[pd.Series] = None
    donchian_lower: Optional[pd.Series] = None

    # Volume
    obv: Optional[pd.Series] = None
    mfi: Optional[pd.Series] = None
    volume_profile: Optional[Dict[float, float]] = None
    relative_volume: Optional[pd.Series] = None
    vwap_deviation: Optional[pd.Series] = None

    # Trend Strength
    adx: Optional[pd.Series] = None
    adx_plus_di: Optional[pd.Series] = None
    adx_minus_di: Optional[pd.Series] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event payload."""
        result = {}

        for attr_name in [
            "vwap", "supertrend", "supertrend_direction",
            "rsi", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_histogram",
            "cci", "roc", "atr", "bb_upper", "bb_middle", "bb_lower",
            "keltner_upper", "keltner_middle", "keltner_lower",
            "donchian_upper", "donchian_lower",
            "obv", "mfi", "relative_volume", "vwap_deviation",
            "adx", "adx_plus_di", "adx_minus_di",
        ]:
            value = getattr(self, attr_name)
            if value is not None:
                if isinstance(value, pd.Series):
                    result[attr_name] = value.iloc[-1] if len(value) > 0 else None
                else:
                    result[attr_name] = value

        # Handle dicts of series
        for attr_name in ["ema", "sma"]:
            value = getattr(self, attr_name)
            if value:
                result[attr_name] = {str(k): v.iloc[-1] if len(v) > 0 else None for k, v in value.items()}

        if self.ichimoku:
            result["ichimoku"] = {k: v.iloc[-1] if len(v) > 0 else None for k, v in self.ichimoku.items()}

        return result


# =============================================================================
# INDICATOR ENGINE
# =============================================================================

class IndicatorEngine:
    """Calculates all technical indicators with shared computation."""

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or get_config().indicators
        self._cache: Dict[str, Any] = {}

    def calculate_all(self, df: pd.DataFrame) -> IndicatorResults:
        """Calculate all indicators for a DataFrame."""
        results = IndicatorResults()

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate shared base metrics first
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # =================================================================
        # TREND INDICATORS
        # =================================================================

        # EMA
        for period in self.config.ema_periods:
            results.ema[period] = self._ema(df["close"], period)

        # SMA
        for period in self.config.sma_periods:
            results.sma[period] = self._sma(df["close"], period)

        # VWAP
        results.vwap = self._vwap(df, self.config.vwap_anchor)

        # Supertrend
        results.supertrend, results.supertrend_direction = self._supertrend(
            df, self.config.supertrend_period, self.config.supertrend_multiplier
        )

        # Ichimoku Cloud
        results.ichimoku = self._ichimoku(
            df,
            self.config.ichimoku_tenkan,
            self.config.ichimoku_kijun,
            self.config.ichimoku_senkou_b,
        )

        # =================================================================
        # MOMENTUM INDICATORS
        # =================================================================

        # RSI
        results.rsi = self._rsi(df["close"], self.config.rsi_period)

        # Stochastic RSI
        results.stoch_k, results.stoch_d = self._stochastic_rsi(
            df["close"],
            self.config.stoch_k_period,
            self.config.stoch_d_period,
            self.config.stoch_smooth,
        )

        # MACD
        results.macd, results.macd_signal, results.macd_histogram = self._macd(
            df["close"],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )

        # CCI
        results.cci = self._cci(df, self.config.cci_period)

        # ROC
        results.roc = self._roc(df["close"], self.config.roc_period)

        # =================================================================
        # VOLATILITY INDICATORS
        # =================================================================

        # ATR (shared with other indicators)
        results.atr = self._atr(df, self.config.atr_period)

        # Bollinger Bands
        results.bb_upper, results.bb_middle, results.bb_lower = self._bollinger_bands(
            df["close"], self.config.bb_period, self.config.bb_std
        )

        # Keltner Channels
        results.keltner_upper, results.keltner_middle, results.keltner_lower = self._keltner_channels(
            df, self.config.keltner_period, self.config.keltner_multiplier
        )

        # Donchian Channels
        results.donchian_upper, results.donchian_lower = self._donchian_channels(
            df, self.config.donchian_period
        )

        # =================================================================
        # VOLUME INDICATORS
        # =================================================================

        # OBV
        results.obv = self._obv(df, self.config.obv_use_close)

        # MFI
        results.mfi = self._mfi(df, self.config.mfi_period)

        # Volume Profile
        results.volume_profile = self._volume_profile(df, self.config.volume_profile_bins)

        # Relative Volume
        results.relative_volume = self._relative_volume(df, self.config.relative_volume_period)

        # VWAP Deviation
        if results.vwap is not None:
            results.vwap_deviation = ((df["close"] - results.vwap) / results.vwap) * 100

        # =================================================================
        # TREND STRENGTH
        # =================================================================

        # ADX
        results.adx, results.adx_plus_di, results.adx_minus_di = self._adx(
            df, self.config.adx_period
        )

        logger.debug(f"Calculated all indicators for {len(df)} rows")
        return results

    # =================================================================
    # TREND CALCULATIONS
    # =================================================================

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def _sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    def _vwap(self, df: pd.DataFrame, anchor: str = "D") -> pd.Series:
        """Volume Weighted Average Price."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical * df["volume"]).groupby(pd.Grouper(freq=anchor)).cumsum() / df["volume"].groupby(pd.Grouper(freq=anchor)).cumsum()
        return vwap

    def _supertrend(self, df: pd.DataFrame, period: int, multiplier: float) -> Tuple[pd.Series, pd.Series]:
        """Supertrend indicator."""
        atr = self._atr(df, period)

        hl2 = (df["high"] + df["low"]) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if df["close"].iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1

                # Adjust bands
                if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i-1]:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                elif direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = supertrend.iloc[i-1]

        return supertrend, direction

    def _ichimoku(self, df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int) -> Dict[str, pd.Series]:
        """Ichimoku Cloud indicator."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        chikou_span = close.shift(-kijun)

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }

    # =================================================================
    # MOMENTUM CALCULATIONS
    # =================================================================

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _stochastic_rsi(self, series: pd.Series, k_period: int, d_period: int, smooth: int) -> Tuple[pd.Series, pd.Series]:
        """Stochastic RSI."""
        rsi = self._rsi(series, k_period)
        stoch_rsi = (rsi - rsi.rolling(k_period).min()) / (rsi.rolling(k_period).max() - rsi.rolling(k_period).min())
        k = stoch_rsi.rolling(smooth).mean()
        d = k.rolling(d_period).mean()
        return k, d

    def _macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = self._ema(series, fast)
        ema_slow = self._ema(series, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        histogram = macd - macd_signal
        return macd, macd_signal, histogram

    def _cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Commodity Channel Index."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        sma_typical = typical.rolling(period).mean()
        mean_deviation = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (typical - sma_typical) / (0.015 * mean_deviation)

    def _roc(self, series: pd.Series, period: int) -> pd.Series:
        """Rate of Change."""
        return ((series - series.shift(period)) / series.shift(period)) * 100

    # =================================================================
    # VOLATILITY CALCULATIONS
    # =================================================================

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _bollinger_bands(self, series: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        middle = self._sma(series, period)
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _keltner_channels(self, df: pd.DataFrame, period: int, multiplier: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        middle = typical.rolling(period).mean()
        atr = self._atr(df, period)
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        return upper, middle, lower

    def _donchian_channels(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Donchian Channels."""
        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        return upper, lower

    # =================================================================
    # VOLUME CALCULATIONS
    # =================================================================

    def _obv(self, df: pd.DataFrame, use_close: bool = True) -> pd.Series:
        """On-Balance Volume."""
        if use_close:
            direction = np.where(df["close"] > df["close"].shift(1), 1, 
                                np.where(df["close"] < df["close"].shift(1), -1, 0))
        else:
            direction = np.where(df["close"] > df["open"], 1, 
                                np.where(df["close"] < df["open"], -1, 0))

        obv = (direction * df["volume"]).cumsum()
        return obv

    def _mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Money Flow Index."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical * df["volume"]

        positive_flow = pd.Series(np.where(typical > typical.shift(1), raw_money_flow, 0), index=df.index)
        negative_flow = pd.Series(np.where(typical < typical.shift(1), raw_money_flow, 0), index=df.index)

        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()

        money_ratio = positive_sum / negative_sum
        return 100 - (100 / (1 + money_ratio))

    def _volume_profile(self, df: pd.DataFrame, bins: int) -> Dict[float, float]:
        """Volume Profile - volume distribution by price level."""
        price_range = df["high"].max() - df["low"].min()
        bin_size = price_range / bins

        profile = {}
        for i in range(bins):
            lower = df["low"].min() + (i * bin_size)
            upper = lower + bin_size
            mask = (df["close"] >= lower) & (df["close"] < upper)
            volume_at_level = df.loc[mask, "volume"].sum()
            profile[round((lower + upper) / 2, 8)] = volume_at_level

        return profile

    def _relative_volume(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Relative Volume (current vs average)."""
        avg_volume = df["volume"].rolling(period).mean()
        return df["volume"] / avg_volume

    # =================================================================
    # TREND STRENGTH CALCULATIONS
    # =================================================================

    def _adx(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = self._atr(df, period)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()

        return adx, plus_di, minus_di
