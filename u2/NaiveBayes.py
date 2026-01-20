import numpy as np
import pandas as pd


class NaiveBayes:
    """Implementación de Naive Bayes con distribución Gaussiana"""

    def __init__(self):
        self.class_prior = None  # P(y)
        self.mean = None  # medias por clase
        self.std = None  # desviaciones std por clase
        self.num_classes = None
        self.classes_ = None  # Para mapear clases string a índices
        self.class_to_idx = None
        self.idx_to_class = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena el modelo"""
        # Convertir a array si es necesario
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Mapear clases a índices
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes_)}

        # Convertir y a índices numéricos
        y_idx = np.array([self.class_to_idx[c] for c in y])

        n_features = X.shape[1]

        self.class_prior = np.zeros(self.num_classes)
        self.mean = np.zeros((self.num_classes, n_features))
        self.std = np.zeros((self.num_classes, n_features))

        for c in range(self.num_classes):
            X_c = X[y_idx == c]
            self.class_prior[c] = X_c.shape[0] / X.shape[0]
            self.mean[c, :] = X_c.mean(axis=0)
            self.std[c, :] = X_c.std(axis=0) + 1e-6

        print(f"Modelo entrenado con {self.num_classes} clases")

    def _gaussian_log_likelihood(self, class_idx: int, x: np.ndarray) -> float:
        """Calcula la log-verosimilitud Gaussiana"""
        mean = self.mean[class_idx]
        std = self.std[class_idx]
        exponent = -0.5 * ((x - mean) / std) ** 2
        log_prob = exponent - np.log(std) - 0.5 * np.log(2 * np.pi)
        return log_prob.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice las clases para X"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        y_pred = []
        for x in X:
            log_posteriors = []
            for c in range(self.num_classes):
                prior = np.log(self.class_prior[c])
                likelihood = self._gaussian_log_likelihood(c, x)
                log_posteriors.append(prior + likelihood)
            y_pred.append(self.idx_to_class[np.argmax(log_posteriors)])
        return np.array(y_pred)
