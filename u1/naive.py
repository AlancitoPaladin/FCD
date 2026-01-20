"""
Codificar el algoritmo de naive-bayes con el mnist

Separar las características de la etiqueta

Contar las veces que aparece cada etiqueta

Calcular el promedio de aparición de clases

Calcular las probabilidades de características se asume distribución gaussiana

Acceder al número de clases y por cada una calcular el promedio y desviación estándar

Función para calcular la probabilidad posterior de cada clase

Aplicarlo a test

"""

import numpy as np

from classifier import Classifier
from utils import Utils as ut


class NaiveBayes(Classifier):
    def __init__(self):
        self.class_prior = None  # P(y)
        self.mean = None  # medias por clase
        self.std = None  # desviaciones std por clase
        self.num_classes = None

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        n_features = X.shape[1]

        self.class_prior = np.zeros(self.num_classes)
        self.mean = np.zeros((self.num_classes, n_features))
        self.std = np.zeros((self.num_classes, n_features))

        for c in range(self.num_classes):
            X_c = X[y == c]  # todas las imágenes de clase
            self.class_prior[c] = X_c.shape[0] / X.shape[0]
            self.mean[c, :] = X_c.mean(axis=0)
            self.std[c, :] = X_c.std(axis=0) + 1e-6  # evitar división por cero

    def _gaussian_log_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        std = self.std[class_idx]
        exponent = -0.5 * ((x - mean) / std) ** 2
        log_prob = exponent - np.log(std) - 0.5 * np.log(2 * np.pi)
        return log_prob.sum()

    def predict(self, X):
        y_pred = []
        for x in X:
            log_posteriors = []
            for c in range(self.num_classes):
                prior = np.log(self.class_prior[c])
                likelihood = self._gaussian_log_likelihood(c, x)
                log_posteriors.append(prior + likelihood)
            y_pred.append(np.argmax(log_posteriors))
        return np.array(y_pred)


if __name__ == "__main__":
    # Cargar datos
    train_images, train_labels = ut.load_dataset("../mnist/mnist_train.csv")
    test_images, test_labels = ut.load_dataset("../mnist/mnist_test.csv")

    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    train_images = (train_images > 0).astype(int)
    test_images = (test_images > 0).astype(int)

    # Entrenar
    nb = NaiveBayes()
    nb.fit(train_images, train_labels)

    # Probar en test
    y_pred = nb.predict(test_images)

    # Evaluar
    acc = np.mean(y_pred == test_labels)
    print("Accuracy:", acc)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_labels, y_pred)
    print("Confusion matrix:\n", cm)
