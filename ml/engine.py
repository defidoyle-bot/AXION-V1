"""
AXION QUANT V4 - Machine Learning Engine
Production-grade ML advisor with time-series validation, auto-retraining, and probability calibration.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from config.settings import MLConfig, MLModelType, get_config
from core.logging import get_logger

logger = get_logger("ml")


# =============================================================================
# ML DATA MODELS
# =============================================================================

@dataclass(frozen=True, slots=True)
class MLPrediction:
    """ML prediction output - NEVER returns buy/sell commands."""
    symbol: str
    timeframe: str
    probability_of_success: float
    confidence: float
    model_used: str
    model_version: str
    feature_importance: Dict[str, float]
    prediction_explanation: str
    prediction_timestamp: datetime
    market_regime: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "probability_of_success": round(self.probability_of_success, 4),
            "confidence": round(self.confidence, 4),
            "model_used": self.model_used,
            "model_version": self.model_version,
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "prediction_explanation": self.prediction_explanation,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "market_regime": self.market_regime,
        }


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    expectancy: float
    timestamp: datetime


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Engineers features from market data, indicators, and SMC."""

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or get_config().ml

    def create_features(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        smc_data: Dict[str, Any],
        trade_history: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """Create ML features from all data sources."""
        features = pd.DataFrame(index=df.index)

        # Technical Features
        if self.config.technical_features:
            features = self._add_technical_features(features, df, indicators)

        # Price Action Features
        if self.config.price_action_features:
            features = self._add_price_action_features(features, smc_data)

        # Market Features
        if self.config.market_features:
            features = self._add_market_features(features, df, indicators)

        # Time Features
        if self.config.time_features:
            features = self._add_time_features(features, df)

        # Trade Features
        if self.config.trade_features and trade_history:
            features = self._add_trade_features(features, trade_history)

        # Drop NaN rows
        features = features.dropna()

        return features

    def _add_technical_features(
        self, features: pd.DataFrame, df: pd.DataFrame, indicators: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add technical indicator features."""
        # Convert indicators to Series if they are dicts
        processed_indicators = {}
        for k, v in indicators.items():
            if isinstance(v, dict):
                # Handle dict from results.to_dict() which might be {index: value}
                processed_indicators[k] = pd.Series(v, index=df.index)
            else:
                processed_indicators[k] = pd.Series(v, index=df.index)

        # Price relative to EMAs
        for period in [9, 21, 50, 200]:
            ema_key = f"ema_{period}"
            if ema_key in processed_indicators:
                features[f"price_above_ema_{period}"] = (df["close"] > processed_indicators[ema_key]).astype(int)
                features[f"distance_from_ema_{period}"] = (df["close"] - processed_indicators[ema_key]) / processed_indicators[ema_key]

        # RSI features
        if "rsi" in processed_indicators:
            features["rsi"] = processed_indicators["rsi"]
            features["rsi_overbought"] = (processed_indicators["rsi"] > 70).astype(int)
            features["rsi_oversold"] = (processed_indicators["rsi"] < 30).astype(int)
            features["rsi_slope"] = processed_indicators["rsi"].diff(3)

        # MACD features
        if "macd" in processed_indicators and "macd_signal" in processed_indicators:
            features["macd"] = processed_indicators["macd"]
            features["macd_signal"] = processed_indicators["macd_signal"]
            features["macd_histogram"] = processed_indicators.get("macd_histogram", processed_indicators["macd"] - processed_indicators["macd_signal"])
            features["macd_above_signal"] = (processed_indicators["macd"] > processed_indicators["macd_signal"]).astype(int)

        # ATR features
        if "atr" in processed_indicators:
            features["atr"] = processed_indicators["atr"]
            features["atr_percent"] = processed_indicators["atr"] / df["close"] * 100

        # Bollinger Bands
        if "bb_upper" in processed_indicators and "bb_lower" in processed_indicators:
            bb_width = processed_indicators["bb_upper"] - processed_indicators["bb_lower"]
            features["bb_position"] = (df["close"] - processed_indicators["bb_lower"]) / bb_width.replace(0, 0.0001)
            features["bb_squeeze"] = ((processed_indicators["bb_upper"] - processed_indicators["bb_lower"]) / df["close"] < 0.02).astype(int)

        # ADX
        if "adx" in processed_indicators:
            features["adx"] = processed_indicators["adx"]
            features["strong_trend"] = (processed_indicators["adx"] > 25).astype(int)

        # Volume
        if "relative_volume" in processed_indicators:
            features["relative_volume"] = processed_indicators["relative_volume"]

        return features

    def _add_price_action_features(self, features: pd.DataFrame, smc_data: Dict[str, Any]) -> pd.DataFrame:
        """Add SMC-based price action features."""
        # Structure features
        structure = smc_data.get("current_structure", "UNKNOWN")
        features["structure_uptrend"] = 1 if structure == "UPTREND" else 0
        features["structure_downtrend"] = 1 if structure == "DOWNTREND" else 0
        features["structure_ranging"] = 1 if structure == "RANGING" else 0

        # Swing point counts
        features["swing_high_count"] = smc_data.get("swing_high_count", 0)
        features["swing_low_count"] = smc_data.get("swing_low_count", 0)

        # BOS/CHOCH presence
        features["recent_bos"] = 1 if smc_data.get("bos_events", []) else 0
        features["recent_choch"] = 1 if smc_data.get("choch_events", []) else 0

        # Order block proximity
        obs = smc_data.get("order_blocks", [])
        features["ob_bullish_nearby"] = sum(1 for ob in obs if ob.get("ob_type") == "BULLISH") if obs else 0
        features["ob_bearish_nearby"] = sum(1 for ob in obs if ob.get("ob_type") == "BEARISH") if obs else 0

        # FVG presence
        fvgs = smc_data.get("fvgs", [])
        features["fvg_open_count"] = sum(1 for fvg in fvgs if fvg.get("status") == "OPEN") if fvgs else 0

        # Liquidity sweep
        sweeps = smc_data.get("liquidity_sweeps", [])
        features["recent_sweep"] = 1 if sweeps else 0

        # Premium/Discount
        pd_data = smc_data.get("premium_discount", {})
        features["in_premium_zone"] = 1 if pd_data.get("current_position") == "premium" else 0
        features["in_discount_zone"] = 1 if pd_data.get("current_position") == "discount" else 0

        return features

    def _add_market_features(
        self, features: pd.DataFrame, df: pd.DataFrame, indicators: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add market condition features."""
        # Volatility regime
        if "atr" in indicators:
            atr_percent = indicators["atr"] / df["close"] * 100
            features["high_volatility"] = (atr_percent > atr_percent.rolling(50).mean() * 1.5).astype(int)
            features["low_volatility"] = (atr_percent < atr_percent.rolling(50).mean() * 0.5).astype(int)

        # Price momentum
        features["returns_1"] = df["close"].pct_change(1)
        features["returns_5"] = df["close"].pct_change(5)
        features["returns_10"] = df["close"].pct_change(10)

        # Candle features
        features["body_size"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"])
        features["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"])
        features["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"])

        return features

    def _add_time_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if hasattr(df.index, "hour"):
            features["hour"] = df.index.hour
            features["day_of_week"] = df.index.dayofweek
            features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

            # Session
            features["is_asian_session"] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
            features["is_london_session"] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
            features["is_ny_session"] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)

        return features

    def _add_trade_features(self, features: pd.DataFrame, trade_history: List[Dict]) -> pd.DataFrame:
        """Add historical trade performance features."""
        if not trade_history:
            return features

        recent_trades = trade_history[-50:]
        wins = [t for t in recent_trades if t.get("pnl", 0) > 0]
        losses = [t for t in recent_trades if t.get("pnl", 0) <= 0]

        features["recent_win_rate"] = len(wins) / len(recent_trades) if recent_trades else 0.5
        features["consecutive_wins"] = self._count_consecutive(recent_trades, "win")
        features["consecutive_losses"] = self._count_consecutive(recent_trades, "loss")

        if wins and losses:
            avg_win = np.mean([t["pnl"] for t in wins])
            avg_loss = abs(np.mean([t["pnl"] for t in losses]))
            features["avg_rr"] = avg_win / avg_loss if avg_loss > 0 else 1.0
        else:
            features["avg_rr"] = 1.0

        return features

    def _count_consecutive(self, trades: List[Dict], outcome: str) -> int:
        """Count consecutive wins or losses."""
        count = 0
        for trade in reversed(trades):
            is_win = trade.get("pnl", 0) > 0
            if (outcome == "win" and is_win) or (outcome == "loss" and not is_win):
                count += 1
            else:
                break
        return count

    def create_target(self, df: pd.DataFrame, horizon: int = 10) -> pd.Series:
        """Create target variable: 1 if price goes up in horizon, 0 if down."""
        future_returns = df["close"].shift(-horizon) / df["close"] - 1
        return (future_returns > 0).astype(int)


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory for creating ML models."""

    @staticmethod
    def create_model(model_type: MLModelType, **kwargs) -> Any:
        """Create a model instance based on type."""
        if model_type == MLModelType.XGBOOST:
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 6),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    subsample=kwargs.get("subsample", 0.8),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42,
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                return ModelFactory.create_model(MLModelType.RANDOM_FOREST, **kwargs)

        elif model_type == MLModelType.LIGHTGBM:
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 6),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    subsample=kwargs.get("subsample", 0.8),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                    objective="binary",
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to RandomForest")
                return ModelFactory.create_model(MLModelType.RANDOM_FOREST, **kwargs)

        elif model_type == MLModelType.CATBOOST:
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=kwargs.get("iterations", 100),
                    depth=kwargs.get("depth", 6),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    loss_function="Logloss",
                    random_seed=42,
                    verbose=False,
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to RandomForest")
                return ModelFactory.create_model(MLModelType.RANDOM_FOREST, **kwargs)

        elif model_type == MLModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 5),
                min_samples_leaf=kwargs.get("min_samples_leaf", 2),
                random_state=42,
                n_jobs=-1,
            )

        elif model_type == MLModelType.EXTRA_TREES:
            return ExtraTreesClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 5),
                random_state=42,
                n_jobs=-1,
            )

        elif model_type == MLModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# ML ENGINE
# =============================================================================

class MLEngine:
    """Production Machine Learning Engine."""

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or get_config().ml
        self.feature_engineer = FeatureEngineer(self.config)
        self.model: Optional[Any] = None
        self.calibrated_model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_names: List[str] = []
        self.model_version = self.config.model_version
        self.last_training_time: Optional[datetime] = None
        self.performance_history: List[ModelPerformance] = []
        self._model_path = Path(self.config.model_save_path)
        self._model_path.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        smc_data: Dict[str, Any],
        trade_history: Optional[List[Dict]] = None,
    ) -> ModelPerformance:
        """Train the ML model with time-series validation."""
        logger.info("Starting ML model training...")

        # Create features and target
        features = self.feature_engineer.create_features(df, indicators, smc_data, trade_history)
        target = self.feature_engineer.create_target(df, self.config.prediction_horizon)

        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        if len(features) < 100:
            raise ValueError(f"Insufficient data for training: {len(features)} samples")

        self.feature_names = features.columns.tolist()

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.walk_forward_windows)

        best_model = None
        best_score = -np.inf
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train model
            model = ModelFactory.create_model(self.config.model_type)
            model.fit(X_train_scaled, y_train)

            # Evaluate
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba > 0.5).astype(int)

            score = roc_auc_score(y_val, val_proba)
            fold_scores.append(score)

            if score > best_score:
                best_score = score
                best_model = model
                self.scaler = scaler

            logger.info(f"Fold {fold + 1}: ROC-AUC = {score:.4f}")

        # Train final model on all data
        self.model = best_model

        # Calibrate probabilities
        if self.config.calibrate_probabilities:
            self._calibrate_model(features, target)

        # Save model
        self._save_model()

        # Record performance
        performance = self._evaluate_model(features, target)
        self.performance_history.append(performance)
        self.last_training_time = datetime.now(timezone.utc)

        logger.info(
            f"Training complete. Model: {self.config.model_type.value}, "
            f"ROC-AUC: {performance.roc_auc:.4f}"
        )

        return performance

    def _calibrate_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Calibrate probability outputs."""
        if self.model is None or self.scaler is None:
            return

        X_scaled = self.scaler.transform(features)

        method = self.config.calibration_method
        # Ensure cv is at least 2 and at most 5, and not more than the number of samples in the smallest class
        n_samples = len(features)
        cv_value = min(5, max(2, n_samples // 10))
        
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method=method, cv=cv_value
        )
        self.calibrated_model.fit(X_scaled, target)

        logger.info(f"Probability calibration complete using {method}")

    def predict(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        smc_data: Dict[str, Any],
        symbol: str,
        timeframe: str,
        trade_history: Optional[List[Dict]] = None,
    ) -> MLPrediction:
        """Generate probability prediction for a symbol."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Create features
        features = self.feature_engineer.create_features(df, indicators, smc_data, trade_history)

        if features.empty:
            return MLPrediction(
                symbol=symbol,
                timeframe=timeframe,
                probability_of_success=0.5,
                confidence=0.0,
                model_used=self.config.model_type.value,
                model_version=self.model_version,
                feature_importance={},
                prediction_explanation="Insufficient data for prediction",
                prediction_timestamp=datetime.now(timezone.utc),
                market_regime="unknown",
            )

        # Use latest data point
        latest_features = features.iloc[-1:]
        X = self.scaler.transform(latest_features) if self.scaler else latest_features.values

        # Predict
        if self.calibrated_model is not None:
            proba = self.calibrated_model.predict_proba(X)[0, 1]
        else:
            proba = self.model.predict_proba(X)[0, 1]

        # Feature importance
        feature_importance = self._get_feature_importance()

        # Market regime
        market_regime = smc_data.get("current_structure", "UNKNOWN")

        # Confidence based on probability distance from 0.5
        confidence = abs(proba - 0.5) * 2  # Scale to 0-1

        # Explanation
        explanation = self._generate_explanation(proba, feature_importance, market_regime)

        return MLPrediction(
            symbol=symbol,
            timeframe=timeframe,
            probability_of_success=round(proba, 4),
            confidence=round(confidence, 4),
            model_used=self.config.model_type.value,
            model_version=self.model_version,
            feature_importance=feature_importance,
            prediction_explanation=explanation,
            prediction_timestamp=datetime.now(timezone.utc),
            market_regime=market_regime,
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from model."""
        if self.model is None:
            return {}

        importance = {}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for name, imp in zip(self.feature_names, importances):
                if imp > self.config.feature_importance_threshold:
                    importance[name] = float(imp)

        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            for name, coef in zip(self.feature_names, coefs):
                if coef > self.config.feature_importance_threshold:
                    importance[name] = float(coef)

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    def _generate_explanation(self, proba: float, importance: Dict[str, float], regime: str) -> str:
        """Generate human-readable prediction explanation."""
        if proba > 0.7:
            direction = "strong bullish"
        elif proba > 0.6:
            direction = "moderately bullish"
        elif proba > 0.5:
            direction = "slightly bullish"
        elif proba > 0.4:
            direction = "slightly bearish"
        elif proba > 0.3:
            direction = "moderately bearish"
        else:
            direction = "strong bearish"

        top_features = list(importance.keys())[:3]
        feature_str = ", ".join(top_features) if top_features else "no dominant features"

        return f"{direction} bias ({proba:.1%} probability) based on {regime} regime. Key factors: {feature_str}."

    def _evaluate_model(self, features: pd.DataFrame, target: pd.Series) -> ModelPerformance:
        """Evaluate model performance."""
        X = self.scaler.transform(features) if self.scaler else features.values

        if self.calibrated_model is not None:
            proba = self.calibrated_model.predict_proba(X)[:, 1]
            pred = self.calibrated_model.predict(X)
        else:
            proba = self.model.predict_proba(X)[:, 1]
            pred = self.model.predict(X)

        return ModelPerformance(
            accuracy=accuracy_score(target, pred),
            precision=precision_score(target, pred, zero_division=0),
            recall=recall_score(target, pred, zero_division=0),
            f1_score=f1_score(target, pred, zero_division=0),
            roc_auc=roc_auc_score(target, proba),
            pr_auc=0.0,  # Would need precision_recall_curve
            log_loss=log_loss(target, proba),
            brier_score=brier_score_loss(target, proba),
            win_rate=0.0,  # Would need trade outcomes
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            expectancy=0.0,
            timestamp=datetime.now(timezone.utc),
        )

    def _save_model(self) -> None:
        """Save model to disk."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_file = self._model_path / f"model_{self.config.model_type.value}_{timestamp}.pkl"

        model_data = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "config": self.config,
            "timestamp": datetime.now(timezone.utc),
        }

        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_file}")
        self._cleanup_old_models()

    def _cleanup_old_models(self) -> None:
        """Remove old model files keeping only the most recent."""
        model_files = sorted(self._model_path.glob("model_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)

        for old_file in model_files[self.config.max_models_to_keep:]:
            old_file.unlink()
            logger.debug(f"Removed old model: {old_file}")

    def load_latest_model(self) -> bool:
        """Load the most recent saved model."""
        model_files = sorted(self._model_path.glob("model_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)

        if not model_files:
            logger.warning("No saved models found")
            return False

        latest = model_files[0]

        try:
            with open(latest, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.calibrated_model = model_data.get("calibrated_model")
            self.scaler = model_data.get("scaler")
            self.feature_names = model_data.get("feature_names", [])
            self.model_version = model_data.get("model_version", "unknown")

            logger.info(f"Loaded model from {latest}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def should_retrain(self) -> bool:
        """Check if model performance has degraded and retraining is needed."""
        if not self.performance_history:
            return True

        if self.last_training_time is None:
            return True

        # Check time since last training
        hours_since = (datetime.now(timezone.utc) - self.last_training_time).total_seconds() / 3600
        if hours_since >= self.config.retrain_interval_hours:
            logger.info(f"Retraining triggered: {hours_since:.1f} hours since last training")
            return True

        # Check performance degradation
        if len(self.performance_history) >= 2:
            recent = self.performance_history[-1]
            previous = self.performance_history[-2]

            if recent.roc_auc < previous.roc_auc - self.config.performance_degrade_threshold:
                logger.info(
                    f"Retraining triggered: ROC-AUC degraded from {previous.roc_auc:.4f} "
                    f"to {recent.roc_auc:.4f}"
                )
                return True

        return False
