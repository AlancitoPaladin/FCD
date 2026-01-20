"""

Cargar dataset

Separar los datos de la etiqueta

Hacer una selección aleatoria

Separar en 3 partes: train, validation, testing para evitar el sobreajuste (overfitting)
... en otro caso se puede separar solo en: training y testing.

La proporción de elección es elegible.

Validación cruzada (cross validation) -- wikipedia

Hacer el muestreo por medio de una semilla, para obtener el indice aleatoriamente.
Y generar el vector en base a la muestra obtenida

Calcular las primeras 5 distancias y ver cual es la etiqueta que mas veces aparece.

Calcular la distancia euclidiana entre un punto y todas las de entrenamiento y solo tomar la primeras 5 o 10

Hacer el contador de cuántas veces aparece la misma etiqueta.
"""

from collections import Counter

import numpy as np

from classifier import Classifier
from utils import Utils as u  # solo para cargar dataset


class KnnVectorizedOptimized(Classifier):
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Aplanar y convertir a float32 para eficiencia
        self.X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
        self.y_train = y_train

    def predict(self, X_test):
        X_test = X_test.reshape(len(X_test), -1).astype(np.float32)

        predictions = []

        # Distancias vectorizadas (Euclidiana)
        X_train_sq = np.sum(self.X_train ** 2, axis=1)
        X_test_sq = np.sum(X_test ** 2, axis=1)[:, np.newaxis]
        cross = np.dot(X_test, self.X_train.T)
        dists = np.sqrt(X_test_sq + X_train_sq - 2 * cross)

        for dist_row in dists:
            # Usar argpartition para encontrar los k más cercanos sin ordenar todo
            nearest_idxs = np.argpartition(dist_row, self.k)[:self.k]
            nearest_labels = self.y_train[nearest_idxs]
            pred = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X_test, y_test, batch_size=1000):
        """
        Calcula exactitud (accuracy) en batches para evitar problemas de memoria.
        """
        n = len(X_test)
        correct = 0
        for i in range(0, n, batch_size):
            X_batch = X_test[i:i + batch_size]
            y_batch = y_test[i:i + batch_size]
            y_pred = self.predict(X_batch)
            correct += np.sum(y_pred == y_batch)
        return correct / n


if __name__ == "__main__":
    train_path = "../mnist/mnist_train.csv"
    test_path = "../mnist/mnist_test.csv"


    # Cargar MNIST
    X_train, y_train = u.load_dataset(train_path)
    X_test, y_test = u.load_dataset(test_path)

    knn = KnnVectorizedOptimized(k=3)
    knn.fit(X_train, y_train)

    # Predicción de un solo dígito
    print("Predicción primer dígito:", knn.predict(X_test[:1])[0])

    # Exactitud en todo MNIST usando batches
    acc = knn.score(X_test, y_test, batch_size=1000)
    print("Exactitud en todo MNIST:", acc)
