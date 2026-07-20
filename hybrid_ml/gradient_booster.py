"""
Gradient Boosting model trained on structured trade data.

Trains a LightGBM classifier on features derived from completed trades
(price, volume, trade size, prior outcomes, market conditions) to predict
the probability of a trade being profitable.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger("hybrid_ml.gradient_booster")

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    lgb = None  # type: ignore[assignment]
    _HAS_LGB = False
    logger.warning("LightGBM not available — GradientBooster will return neutral predictions")


@dataclass
class BoosterPrediction:
    """Prediction output from the gradient booster."""
    win_probability: float
    confidence: float
    feature_importance: Dict[str, float]
    feature_values: Dict[str, float]
    model_version: str
    timestamp: datetime


@dataclass
class BoosterPerformance:
    """Performance metrics for the booster model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    total_trades: int
    timestamp: datetime


class TradeFeatureEngineer:
    """Engineers features from trade data for the gradient booster."""

    @staticmethod
    def create_trade_features(trade: Dict[str, Any]) -> Dict[str, float]:
        """Create feature vector from a single trade."""
        features = {}

        # Price features
        entry = trade.get("entry_price", 0)
        sl = trade.get("stop_loss", 0)
        features["entry_price"] = float(entry)
        features["stop_loss_distance_pct"] = float(abs(entry - sl) / entry * 100) if entry > 0 and sl > 0 else 0.0

        tp = trade.get("take_profit", [])
        features["take_profit_distance_pct"] = float(
            abs(tp[0] - entry) / entry * 100
        ) if tp and entry > 0 else 0.0

        # Trade features
        features["position_size"] = float(trade.get("position_size", 0))
        features["leverage"] = float(trade.get("leverage", 1))
        features["margin"] = float(trade.get("margin", 0))

        # Direction
        features["direction_long"] = 1.0 if trade.get("direction") == "LONG" else 0.0

        # Market conditions at entry
        features["atr_percent"] = float(trade.get("atr_percent", 0))
        features["rsi"] = float(trade.get("rsi", 50))
        features["spread"] = float(trade.get("spread", 0))

        features["volume_24h"] = float(trade.get("volume_24h", 0))
        features["market_regime_ranging"] = 1.0 if trade.get("market_regime") == "RANGING" else 0.0
        features["market_regime_uptrend"] = 1.0 if trade.get("market_regime") == "UPTREND" else 0.0
        features["market_regime_downtrend"] = 1.0 if trade.get("market_regime") == "DOWNTREND" else 0.0

        # Risk metrics
        features["risk_reward"] = float(trade.get("risk_reward", 0))
        features["score"] = float(trade.get("score", 0))

        # ── New enriched features from indicator snapshot ─────────────────
        # ML model confidence
        features["ml_probability"] = float(trade.get("ml_probability", 0.0))
        features["ml_confidence"] = float(trade.get("ml_confidence", 0.0))

        # Volatility & momentum
        features["bb_width"] = float(trade.get("bb_width", 0.0))
        features["bb_position"] = float(trade.get("bb_position", 0.5))
        features["macd_histogram"] = float(trade.get("macd_histogram", 0.0))
        features["adx"] = float(trade.get("adx", 0.0))

        return features

    @staticmethod
    def create_training_data(trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a training DataFrame from a list of completed trades."""
        rows = []
        for trade in trades:
            features = TradeFeatureEngineer.create_trade_features(trade)
            features["pnl"] = float(trade.get("pnl", 0))
            features["won"] = 1.0 if features["pnl"] > 0 else 0.0
            rows.append(features)

        df = pd.DataFrame(rows)
        return df


class GradientBooster:
    """LightGBM classifier trained on structured trade data."""

    def __init__(self, model_dir: str = "models/hybrid_ml"):
        self.model: Any = None
        self.feature_names: List[str] = []
        self.model_version = "1.0.0"
        self.last_training_time: Optional[datetime] = None
        self.performance_history: List[BoosterPerformance] = []
        self._model_path = Path(model_dir)
        self._model_path.mkdir(parents=True, exist_ok=True)
        self._is_ready = False

        # Load saved model if available
        self._load_latest()

    def _load_latest(self) -> bool:
        """Load the most recent saved booster model."""
        model_files = sorted(
            self._model_path.glob("booster_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not model_files:
            return False

        try:
            with open(model_files[0], "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data.get("feature_names", [])
            self.model_version = data.get("model_version", "1.0.0")
            self.last_training_time = data.get("timestamp")
            perf = data.get("performance")
            if perf:
                self.performance_history.append(perf)
            self._is_ready = True
            logger.info(f"GradientBooster: loaded model from {model_files[0]}")
            return True
        except Exception as e:
            logger.warning(f"GradientBooster: failed to load model: {e}")
            return False

    def train(self, trades: List[Dict[str, Any]]) -> Optional[BoosterPerformance]:
        """Train the booster on completed trades."""
        if not _HAS_LGB:
            logger.warning("GradientBooster: LightGBM not available, skipping training")
            return None

        if len(trades) < 20:
            logger.info(f"GradientBooster: insufficient trades ({len(trades)} < 20), skipping training")
            return None

        df = TradeFeatureEngineer.create_training_data(trades)
        feature_cols = [c for c in df.columns if c not in ("pnl", "won")]
        X = df[feature_cols].values
        y = df["won"].values

        self.feature_names = feature_cols

        # Train model
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.01,
            min_child_samples=5,
            objective="binary",
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)

        # Evaluate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        perf = BoosterPerformance(
            accuracy=float(accuracy_score(y, y_pred)),
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1_score=float(f1_score(y, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.5,
            total_trades=len(trades),
            timestamp=datetime.now(timezone.utc),
        )

        self.performance_history.append(perf)
        self.last_training_time = datetime.now(timezone.utc)
        self._is_ready = True
        self._save()

        logger.info(
            f"GradientBooster: trained on {len(trades)} trades — "
            f"ROC-AUC: {perf.roc_auc:.4f}, accuracy: {perf.accuracy:.2%}"
        )
        return perf

    def predict(self, trade_features: Dict[str, float]) -> BoosterPrediction:
        """Predict win probability for a potential trade."""
        if not self._is_ready or self.model is None:
            return BoosterPrediction(
                win_probability=0.5,
                confidence=0.0,
                feature_importance={},
                feature_values=trade_features,
                model_version="uninitialized",
                timestamp=datetime.now(timezone.utc),
            )

        # Build feature array in the correct order
        feature_values = []
        for col in self.feature_names:
            feature_values.append(trade_features.get(col, 0.0))
        X = np.array([feature_values])

        proba = float(self.model.predict_proba(X)[0, 1])

        # Confidence based on distance from 0.5
        confidence = float(abs(proba - 0.5) * 2)

        # Feature importance
        importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, imp in zip(self.feature_names, self.model.feature_importances_):
                if imp > 0:
                    importance[name] = float(imp)

        return BoosterPrediction(
            win_probability=proba,
            confidence=confidence,
            feature_importance=dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]),
            feature_values=trade_features,
            model_version=self.model_version,
            timestamp=datetime.now(timezone.utc),
        )

    def _save(self) -> None:
        """Save model to disk."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._model_path / f"booster_{timestamp}.pkl"

        perf = self.performance_history[-1] if self.performance_history else None
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "timestamp": self.last_training_time,
            "performance": perf,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        # Cleanup old models
        for old in sorted(self._model_path.glob("booster_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)[5:]:
            old.unlink()

        logger.info(f"GradientBooster: saved to {path}")
