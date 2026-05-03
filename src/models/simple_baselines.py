from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PersistenceBaseline:
    last_target_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "PersistenceBaseline":
        if len(y) == 0:
            raise ValueError("PersistenceBaseline requires non-empty y.")
        self.last_target_ = np.asarray(y[-1], dtype=np.float32)
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.last_target_ is None:
            raise RuntimeError("PersistenceBaseline must be fit before predict.")
        return np.repeat(self.last_target_[None, :], int(n), axis=0)


@dataclass
class RollingMeanBaseline:
    window: int = 256
    mean_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "RollingMeanBaseline":
        if len(y) == 0:
            raise ValueError("RollingMeanBaseline requires non-empty y.")
        tail = np.asarray(y[-max(int(self.window), 1) :], dtype=np.float32)
        self.mean_ = tail.mean(axis=0)
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("RollingMeanBaseline must be fit before predict.")
        return np.repeat(self.mean_[None, :], int(n), axis=0)


@dataclass
class RidgeBaseline:
    alpha: float = 1.0
    coef_: Optional[np.ndarray] = None
    x_mean_: Optional[np.ndarray] = None
    x_std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        if len(x) == 0:
            raise ValueError("RidgeBaseline requires non-empty x.")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x_mean_ = x.mean(axis=0)
        self.x_std_ = x.std(axis=0)
        self.x_std_[self.x_std_ < 1e-8] = 1.0
        xz = (x - self.x_mean_) / self.x_std_
        x_aug = np.concatenate([np.ones((len(xz), 1)), xz], axis=1)
        reg = np.eye(x_aug.shape[1]) * float(self.alpha)
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("RidgeBaseline must be fit before predict.")
        x = np.asarray(x, dtype=np.float64)
        xz = (x - self.x_mean_) / self.x_std_
        x_aug = np.concatenate([np.ones((len(xz), 1)), xz], axis=1)
        return (x_aug @ self.coef_).astype(np.float32)
