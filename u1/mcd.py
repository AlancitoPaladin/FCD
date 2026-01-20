import numpy as np
from utils import Utils
from classifier import Classifier

class MinimumDistanceClassifier(Utils, Classifier):
    """
    Clasificador de distancia mínima (Minimum Distance Classifier).
    Calcula los centroides de cada clase y clasifica nuevos datos basándose en
    la distancia euclidiana mínima a estos centroides.
    """

    def __init__(self):
        """Inicializa el clasificador."""
        super().__init__()
        self.centroids = {}  # Diccionario para almacenar los centroides por clase
        self.classes = []  # Lista de clases únicas

    def fit(self, X, y):
        """
        Entrena el clasificador calculando los centroides para cada clase.

        Args:
            X: Matriz de características (n_muestras, n_características)
            y: Vector de etiquetas (n_muestras)
        """
        # Obtener clases únicas
        self.classes = np.unique(y)

        # Calcular el centroide para cada clase
        for c in self.classes:
            # Seleccionar características de la clase
            X_c = X[y == c]
            # Calcular el vector promedio (centroide)
            centroid = np.mean(X_c, axis=0)
            # Almacenar el centroide
            self.centroids[c] = centroid

        return self

    def predict(self, X):
        """
        Predice la clase de cada muestra en X basándose en la distancia mínima al centroide.

        Args:
            X: Matriz de características a clasificar (n_muestras, n_características)

        Returns:
            Predicciones de clase para cada muestra
        """
        predictions = []

        for x in X:
            # Calcular distancias euclidianas a todos los centroides
            distances = {}
            for c in self.classes:
                centroid = self.centroids[c]
                # Distancia euclidiana
                distance = self.euclidean_distance(x, centroid)
                distances[c] = distance

            # Encontrar la clase con la distancia mínima
            min_class = min(distances, key=distances.get)
            predictions.append(min_class)

        return np.array(predictions)
