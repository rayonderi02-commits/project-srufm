"""SVM classifier wrapper used by the training and inference pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


class SVMModel:
    """
    Support Vector Machine (SVM) is a powerful supervised machine learning algorithm
    used primarily for classification and regression tasks. It works by finding the optimal hyperplane
    (decision boundary) that maximizes the margin—the distance—between different classes in a high-dimensional
    space, effectively handling both linear and non-linear data.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        random_state: int = 42,
        class_weight: str | dict | None = None,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.class_weight = class_weight
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=random_state,
            class_weight=class_weight,
        )
        self._is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMModel":
        """Fit the SVM on feature vectors and integer labels."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got shape {X_train.shape}.")
        if len(np.unique(y_train)) < 2:
            raise ValueError("SVM training requires at least two classes.")

        self.model.fit(X_train, y_train)
        self._is_trained = True
        return self

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict | None = None,
        cv: int = 3,
    ) -> "SVMModel":
        """Tune SVM hyperparameters with cross-validation, then keep the best model."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if param_grid is None:
            param_grid = {
                "C": [0.5, 1.0, 3.0, 10.0],
                "gamma": ["scale", 0.01, 0.003, 0.001],
            }
        class_counts = np.bincount(y_train)
        usable_counts = class_counts[class_counts > 0]
        n_splits = min(cv, int(usable_counts.min())) if len(usable_counts) else 0
        if n_splits < 2:
            return self.train(X_train, y_train)

        base = SVC(
            kernel=self.kernel,
            probability=True,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )
        search = GridSearchCV(
            base,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state),
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.C = self.model.C
        self.gamma = self.model.gamma
        self.best_params_ = search.best_params_
        self.best_cv_score_ = float(search.best_score_)
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels."""
        self._require_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each row in X."""
        self._require_trained()
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def load(self, path: str | Path) -> "SVMModel":
        """Load a persisted model into this instance."""
        loaded = joblib.load(path)
        if isinstance(loaded, SVMModel):
            self.__dict__.update(loaded.__dict__)
        elif isinstance(loaded, SVC):
            self.model = loaded
            self._is_trained = True
        else:
            raise TypeError(f"Unsupported SVM artifact type: {type(loaded)!r}")
        return self

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("SVMModel has not been trained.")
