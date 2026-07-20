---
name: Directional scoring bias
description: Avoid hardcoding directional bias in signal scoring inputs; derive trend inputs from live market structure to keep LONG/SHORT scoring symmetric.
---

# Directional scoring bias

**Rule:** When a scoring engine evaluates directional signals (LONG vs SHORT), every trend/structure input must be derived from live market data, not hardcoded to one direction.

**Why:** The AXION QUANT `SignalHandler` originally passed hardcoded `bullish` values for both higher and lower timeframe trend components. Because the trend component has a 15% weight and the scoring formula penalizes direction mismatch, every SHORT began with a near-zero trend score while every LONG began with a near-perfect score. This created a systemic LONG bias regardless of the actual market structure.

**How to apply:**
- Use the current SMC structure (e.g., `UPTREND`/`DOWNTREND`/`RANGING`) to set the trend direction passed into the scoring engine.
- Keep the scoring formula itself symmetric so that a LONG in a bullish structure scores the same as a SHORT in a bearish structure under equivalent conditions.
- Verify RSI/indicator sub-scores are directionally opposite and bounded to the same range for both sides.
