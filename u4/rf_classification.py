"""
Clasificador Random Forest con paradigma orientado a objetos.
Realiza clasificación en el dataset iris.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


class RandomForestClassification:
    """
    Clase para manejar la clasificación con Random Forest.
    """

    def __init__(self, n_trees=500, test_size=0.3, random_state=123):
        """
        Inicializa el clasificador Random Forest.

        Args:
            n_trees (int): Número de árboles en el bosque
            test_size (float): Proporción de datos para prueba
            random_state (int): Semilla para reproducibilidad
        """
        self.n_trees = n_trees
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=random_state
        )
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.predictions = None
        self.confusion_matrix = None
        self.accuracy = None

    def load_data(self, filepath):
        """Carga el dataset desde un archivo CSV."""
        self.data = pd.read_csv(filepath)
        print(f"Dataset cargado: {self.data.shape}")

    def prepare_data(self, target_column='Species'):
        """
        Prepara los datos separando características y etiquetas.

        Args:
            target_column (str): Nombre de la columna objetivo
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        self.train_data, self.test_data, self.train_labels, self.test_labels = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        print(f"Datos de entrenamiento: {self.train_data.shape}")
        print(f"Datos de prueba: {self.test_data.shape}")

    def train(self):
        """Entrena el modelo Random Forest."""
        self.model.fit(self.train_data, self.train_labels)
        print("Modelo entrenado correctamente")

    def predict(self):
        """Realiza predicciones en los datos de prueba."""
        self.predictions = self.model.predict(self.test_data)
        print("Predicciones realizadas")

    def evaluate(self):
        """
        Calcula la matriz de confusión y la precisión.
        """
        self.confusion_matrix = confusion_matrix(
            self.test_labels,
            self.predictions
        )
        self.accuracy = accuracy_score(self.test_labels, self.predictions)

        print("\n=== RESULTADOS ===")
        print("Matriz de confusión:")
        print(self.confusion_matrix)
        print(f"\nPrecisión (Accuracy): {self.accuracy:.4f}")

    def plot_feature_importance(self):
        """Visualiza la importancia de las características."""
        feature_importance = self.model.feature_importances_
        feature_names = self.train_data.columns

        # Ordena por importancia
        indices = np.argsort(feature_importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de las Características")
        plt.bar(range(len(feature_importance)),
                feature_importance[indices])
        plt.xticks(range(len(feature_importance)),
                   feature_names[indices], rotation=45)
        plt.xlabel("Características")
        plt.ylabel("Importancia")
        plt.tight_layout()
        plt.show()

    def run(self, filepath, target_column='Species'):
        """
        Ejecuta el pipeline completo de clasificación.

        Args:
            filepath (str): Ruta del archivo de datos
            target_column (str): Nombre de la columna objetivo
        """
        self.load_data(filepath)
        self.prepare_data(target_column)
        self.train()
        self.predict()
        self.evaluate()
        self.plot_feature_importance()


# Uso del programa
if __name__ == "__main__":
    # Crear instancia del clasificador
    rf_classifier = RandomForestClassification(
        n_trees=500,
        test_size=0.3,
        random_state=123
    )

    # Ejecutar el pipeline completo
    # Nota: Cambiar la ruta según donde tengas el dataset iris
    try:
        rf_classifier.run("iris.csv", target_column="Species")
    except FileNotFoundError:
        print("Dataset no encontrado. Usando dataset de sklearn...")
        from sklearn.datasets import load_iris

        # Cargar iris de sklearn
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['Species'] = iris.target_names[iris.target]

        # Guardar y ejecutar
        df.to_csv("iris.csv", index=False)
        rf_classifier.run("iris.csv", target_column="Species")
