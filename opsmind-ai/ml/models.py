"""
OpsMind AI — ML/DL Models
- Isolation Forest: Anomaly Detection
- XGBoost: Predictive Analytics (failure prediction)
- LSTM: Time-series forecasting (deep learning)
- Autoencoder: Unsupervised anomaly detection (DL)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR
#    (simulates IT ops metrics — CPU, memory, latency, error rate)
# ─────────────────────────────────────────

def generate_synthetic_metrics(n_samples: int = 2000, anomaly_frac: float = 0.05) -> pd.DataFrame:
    """Generate realistic IT operations time-series data with injected anomalies."""
    np.random.seed(42)
    n_anomaly = int(n_samples * anomaly_frac)
    n_normal  = n_samples - n_anomaly

    # Normal operations
    normal = pd.DataFrame({
        "cpu_usage":      np.random.normal(45, 12, n_normal).clip(5, 85),
        "memory_usage":   np.random.normal(60, 15, n_normal).clip(20, 90),
        "disk_io":        np.random.normal(30, 10, n_normal).clip(0, 70),
        "network_latency":np.random.normal(20, 8, n_normal).clip(5, 60),
        "error_rate":     np.random.beta(0.5, 10, n_normal) * 5,
        "request_count":  np.random.poisson(500, n_normal).astype(float),
        "response_time":  np.random.normal(150, 40, n_normal).clip(50, 400),
        "label": 0,
    })

    # Anomalous operations (spikes, resource exhaustion, high errors)
    anomaly = pd.DataFrame({
        "cpu_usage":      np.random.choice([
            np.random.normal(92, 4, n_anomaly // 2),
            np.random.normal(5, 2, n_anomaly - n_anomaly // 2),
        ], axis=0).flatten()[:n_anomaly].clip(0, 100),
        "memory_usage":   np.random.normal(95, 3, n_anomaly).clip(80, 100),
        "disk_io":        np.random.normal(90, 5, n_anomaly).clip(70, 100),
        "network_latency":np.random.normal(800, 200, n_anomaly).clip(200, 2000),
        "error_rate":     np.random.beta(5, 1, n_anomaly) * 40 + 20,
        "request_count":  np.random.choice(
            [np.random.poisson(5000, n_anomaly), np.random.poisson(10, n_anomaly)],
            axis=0
        ).flatten()[:n_anomaly].astype(float),
        "response_time":  np.random.normal(2000, 500, n_anomaly).clip(500, 10000),
        "label": 1,
    })

    df = pd.concat([normal, anomaly], ignore_index=True).sample(frac=1, random_state=42)
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="5min")
    df["service"]   = np.random.choice(
        ["auth-service", "api-gateway", "db-cluster", "ml-pipeline", "cache-layer"],
        len(df)
    )
    return df


# ─────────────────────────────────────────
# 2. ISOLATION FOREST — Unsupervised Anomaly Detection
# ─────────────────────────────────────────

class AnomalyDetector:
    """Isolation Forest-based anomaly detector for real-time metrics."""

    FEATURES = ["cpu_usage", "memory_usage", "disk_io",
                 "network_latency", "error_rate", "request_count", "response_time"]

    def __init__(self, contamination: float = 0.05):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iso_forest", IsolationForest(
                contamination=contamination,
                n_estimators=200,
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = df[self.FEATURES].fillna(0)
        self.pipeline.fit(X)
        self.is_fitted = True
        print("[AnomalyDetector] Fitted on", len(df), "samples")
        return self

    def predict(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Returns anomaly score and flag for a single metrics snapshot."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = pd.DataFrame([metrics])[self.FEATURES].fillna(0)
        raw_score = self.pipeline.named_steps["iso_forest"].score_samples(
            self.pipeline.named_steps["scaler"].transform(X)
        )[0]
        prediction = self.pipeline.predict(X)[0]  # -1 = anomaly, 1 = normal
        anomaly_score = float(np.clip(1 - (raw_score + 0.5), 0, 1))  # 0–1 scale

        severity = "normal"
        if anomaly_score > 0.85:
            severity = "critical"
        elif anomaly_score > 0.65:
            severity = "high"
        elif anomaly_score > 0.45:
            severity = "medium"

        return {
            "is_anomaly": prediction == -1,
            "anomaly_score": round(anomaly_score, 4),
            "severity": severity,
            "raw_isolation_score": round(float(raw_score), 4),
            "detected_at": datetime.utcnow().isoformat(),
        }

    def save(self, path: str = "ml/artifacts/anomaly_detector.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str = "ml/artifacts/anomaly_detector.pkl") -> "AnomalyDetector":
        instance = cls()
        instance.pipeline = joblib.load(path)
        instance.is_fitted = True
        return instance


# ─────────────────────────────────────────
# 3. XGBOOST — Predictive Failure Analytics
# ─────────────────────────────────────────

class FailurePredictor:
    """XGBoost classifier predicting system failure probability in next 30 min."""

    FEATURES = ["cpu_usage", "memory_usage", "disk_io",
                 "network_latency", "error_rate", "request_count",
                 "response_time", "cpu_trend", "error_trend"]

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cpu_trend"]   = df["cpu_usage"].rolling(3, min_periods=1).mean()
        df["error_trend"] = df["error_rate"].rolling(3, min_periods=1).mean()
        return df

    def fit(self, df: pd.DataFrame) -> "FailurePredictor":
        df = self._add_trend_features(df)
        X = df[self.FEATURES].fillna(0)
        y = df["label"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False,
        )
        self.is_fitted = True
        preds = self.model.predict(X_scaled)
        print("[FailurePredictor] AUC:", round(roc_auc_score(y, self.model.predict_proba(X_scaled)[:, 1]), 4))
        print(classification_report(y, preds, target_names=["Normal", "Failure"]))
        return self

    def predict_proba(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        row = {**metrics, "cpu_trend": metrics["cpu_usage"], "error_trend": metrics["error_rate"]}
        X = pd.DataFrame([row])[self.FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X)
        prob = float(self.model.predict_proba(X_scaled)[0][1])
        risk = "low"
        if prob > 0.75: risk = "critical"
        elif prob > 0.5: risk = "high"
        elif prob > 0.25: risk = "medium"
        return {
            "failure_probability": round(prob, 4),
            "risk_level": risk,
            "recommendation": _get_recommendation(metrics, prob),
            "predicted_at": datetime.utcnow().isoformat(),
        }

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        return dict(zip(self.FEATURES, self.model.feature_importances_.tolist()))

    def save(self, path: str = "ml/artifacts/failure_predictor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str = "ml/artifacts/failure_predictor.pkl") -> "FailurePredictor":
        instance = cls()
        data = joblib.load(path)
        instance.model  = data["model"]
        instance.scaler = data["scaler"]
        instance.is_fitted = True
        return instance


# ─────────────────────────────────────────
# 4. LSTM AUTOENCODER — Deep Learning Anomaly (TF/Keras)
# ─────────────────────────────────────────

def build_lstm_autoencoder(n_features: int = 7, seq_len: int = 10):
    """
    LSTM Autoencoder for sequential anomaly detection.
    High reconstruction error → anomaly.
    Requires TensorFlow.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        inputs  = keras.Input(shape=(seq_len, n_features))
        # Encoder
        encoded = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(inputs)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.LSTM(32, activation="tanh", return_sequences=False)(encoded)
        # Bottleneck
        bottleneck = keras.layers.Dense(16, activation="relu")(encoded)
        # Decoder
        repeated = keras.layers.RepeatVector(seq_len)(bottleneck)
        decoded  = keras.layers.LSTM(32, activation="tanh", return_sequences=True)(repeated)
        decoded  = keras.layers.Dropout(0.2)(decoded)
        decoded  = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(decoded)
        outputs  = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(decoded)

        model = keras.Model(inputs, outputs, name="lstm_autoencoder")
        model.compile(optimizer="adam", loss="mse")
        return model
    except ImportError:
        print("[LSTM] TensorFlow not available — skipping LSTM autoencoder.")
        return None


class LSTMAnomalyDetector:
    """LSTM Autoencoder wrapper for sequence-based anomaly detection."""
    SEQ_LEN = 10
    FEATURES = ["cpu_usage", "memory_usage", "disk_io",
                 "network_latency", "error_rate", "request_count", "response_time"]

    def __init__(self):
        self.model     = None
        self.scaler    = MinMaxScaler()
        self.threshold = 0.05
        self.is_fitted = False

    def _make_sequences(self, data: np.ndarray) -> np.ndarray:
        return np.array([data[i:i + self.SEQ_LEN] for i in range(len(data) - self.SEQ_LEN)])

    def fit(self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 64):
        self.model = build_lstm_autoencoder(len(self.FEATURES), self.SEQ_LEN)
        if self.model is None:
            return self
        X = self.scaler.fit_transform(df[self.FEATURES].fillna(0))
        X_seq = self._make_sequences(X)
        self.model.fit(
            X_seq, X_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )
        # Set threshold as 95th percentile of reconstruction error on normal data
        recon = self.model.predict(X_seq, verbose=0)
        errors = np.mean(np.power(X_seq - recon, 2), axis=(1, 2))
        self.threshold = float(np.percentile(errors, 95))
        self.is_fitted = True
        print(f"[LSTM Autoencoder] Trained. Anomaly threshold: {self.threshold:.6f}")
        return self

    def reconstruct_error(self, sequence: np.ndarray) -> float:
        """Return mean reconstruction error for a sequence."""
        if self.model is None:
            return 0.0
        seq_scaled = self.scaler.transform(sequence)
        seq_input  = seq_scaled[np.newaxis, :, :]
        recon = self.model.predict(seq_input, verbose=0)
        return float(np.mean(np.power(seq_input - recon, 2)))

    def save(self, dir_path: str = "ml/artifacts/lstm_autoencoder"):
        if self.model:
            self.model.save(dir_path)
            joblib.dump({"scaler": self.scaler, "threshold": self.threshold},
                        f"{dir_path}/meta.pkl")

    @classmethod
    def load(cls, dir_path: str = "ml/artifacts/lstm_autoencoder") -> "LSTMAnomalyDetector":
        import tensorflow as tf
        instance = cls()
        instance.model = tf.keras.models.load_model(dir_path)
        meta = joblib.load(f"{dir_path}/meta.pkl")
        instance.scaler    = meta["scaler"]
        instance.threshold = meta["threshold"]
        instance.is_fitted = True
        return instance


# ─────────────────────────────────────────
# 5. HELPER
# ─────────────────────────────────────────

def _get_recommendation(metrics: Dict[str, float], prob: float) -> str:
    recs = []
    if metrics.get("cpu_usage", 0) > 85:
        recs.append("Scale up compute or kill high-CPU processes immediately.")
    if metrics.get("memory_usage", 0) > 90:
        recs.append("Trigger memory cleanup / increase heap allocation.")
    if metrics.get("error_rate", 0) > 15:
        recs.append("Inspect error logs — possible cascading failure upstream.")
    if metrics.get("network_latency", 0) > 500:
        recs.append("Check network routing — high latency detected.")
    if not recs:
        recs.append("Monitor closely. No single dominant root cause identified.")
    return " | ".join(recs)


# ─────────────────────────────────────────
# 6. MODEL TRAINING ENTRYPOINT
# ─────────────────────────────────────────

def train_all_models(save: bool = True):
    """Train and optionally save all ML models."""
    print("=" * 55)
    print("  OpsMind AI — ML Training Pipeline")
    print("=" * 55)

    df = generate_synthetic_metrics(n_samples=3000)
    print(f"[Data] Generated {len(df)} samples | Anomaly rate: {df['label'].mean():.2%}")

    # Anomaly Detector
    ad = AnomalyDetector()
    ad.fit(df)
    if save:
        ad.save()

    # Failure Predictor
    fp = FailurePredictor()
    fp.fit(df)
    if save:
        fp.save()

    # LSTM Autoencoder (if TF available)
    lstm = LSTMAnomalyDetector()
    lstm.fit(df, epochs=10)   # Reduced for speed; increase for better accuracy
    if save and lstm.is_fitted:
        lstm.save()

    print("\n✅ All models trained and saved.")
    return ad, fp, lstm


if __name__ == "__main__":
    train_all_models()
