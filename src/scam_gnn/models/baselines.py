from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from scam_gnn.dependencies import MissingOptionalDependency, optional_import


@dataclass(slots=True)
class ModelOutput:
    model_name: str
    scores: np.ndarray
    train_seconds: float
    metadata: dict[str, object]


class NumpyLogisticRegression:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 400, l2: float = 1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-6
        x_scaled = (x - self.mean_) / self.std_
        self.weights_ = np.zeros(x.shape[1], dtype=np.float32)
        self.bias_ = 0.0
        for _ in range(self.max_iter):
            logits = x_scaled @ self.weights_ + self.bias_
            predictions = self._sigmoid(logits)
            error = predictions - y
            grad_w = (x_scaled.T @ error) / len(x_scaled) + self.l2 * self.weights_
            grad_b = float(error.mean())
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        x_scaled = (x - self.mean_) / self.std_
        return self._sigmoid(x_scaled @ self.weights_ + self.bias_)


class NumpyMLPClassifier:
    def __init__(self, hidden_dim: int = 32, learning_rate: float = 0.05, max_iter: int = 300, l2: float = 1e-4):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w1_: np.ndarray | None = None
        self.b1_: np.ndarray | None = None
        self.w2_: np.ndarray | None = None
        self.b2_: float = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int = 7) -> "NumpyMLPClassifier":
        rng = np.random.default_rng(seed)
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-6
        x_scaled = (x - self.mean_) / self.std_
        input_dim = x_scaled.shape[1]
        self.w1_ = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, self.hidden_dim)).astype(np.float32)
        self.b1_ = np.zeros(self.hidden_dim, dtype=np.float32)
        self.w2_ = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim), size=(self.hidden_dim,)).astype(np.float32)
        self.b2_ = 0.0
        for _ in range(self.max_iter):
            z1 = x_scaled @ self.w1_ + self.b1_
            a1 = np.maximum(z1, 0.0)
            logits = a1 @ self.w2_ + self.b2_
            predictions = self._sigmoid(logits)
            error = predictions - y
            grad_w2 = (a1.T @ error) / len(x_scaled) + self.l2 * self.w2_
            grad_b2 = float(error.mean())
            hidden_error = (error[:, None] * self.w2_[None, :]) * (z1 > 0.0)
            grad_w1 = (x_scaled.T @ hidden_error) / len(x_scaled) + self.l2 * self.w1_
            grad_b1 = hidden_error.mean(axis=0)
            self.w2_ -= self.learning_rate * grad_w2
            self.b2_ -= self.learning_rate * grad_b2
            self.w1_ -= self.learning_rate * grad_w1
            self.b1_ -= self.learning_rate * grad_b1
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in [self.mean_, self.std_, self.w1_, self.b1_, self.w2_]):
            raise RuntimeError("Model must be fitted before prediction.")
        x_scaled = (x - self.mean_) / self.std_
        z1 = x_scaled @ self.w1_ + self.b1_
        a1 = np.maximum(z1, 0.0)
        logits = a1 @ self.w2_ + self.b2_
        return self._sigmoid(logits)


def run_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> ModelOutput:
    started = time.perf_counter()
    model = NumpyLogisticRegression().fit(x_train, y_train)
    return ModelOutput(
        model_name="logistic_regression",
        scores=model.predict_proba(x_test),
        train_seconds=time.perf_counter() - started,
        metadata={"implementation": "numpy"},
    )


def run_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> ModelOutput:
    started = time.perf_counter()
    model = NumpyMLPClassifier().fit(x_train, y_train, seed=seed)
    return ModelOutput(
        model_name="mlp",
        scores=model.predict_proba(x_test),
        train_seconds=time.perf_counter() - started,
        metadata={"implementation": "numpy"},
    )


def run_tree_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> ModelOutput:
    sklearn_ensemble = optional_import("sklearn.ensemble")
    if sklearn_ensemble is None:
        raise MissingOptionalDependency("Tree ensemble baseline requires scikit-learn.")
    started = time.perf_counter()
    # Prefer the serial sklearn booster here because histogram boosting can hit
    # multiprocessing permission issues in restricted Windows environments.
    model = sklearn_ensemble.GradientBoostingClassifier(random_state=seed)
    model.fit(x_train, y_train)
    return ModelOutput(
        model_name="tree_ensemble",
        scores=model.predict_proba(x_test)[:, 1],
        train_seconds=time.perf_counter() - started,
        metadata={"implementation": "sklearn"},
    )
