"""
Selección de características con Random Forest mediante Monte Carlo.
Implementación con paradigma orientado a objetos y procesamiento multihilo.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class DataGenerator:
    """Genera datos sintéticos para clasificación multiclase."""

    def __init__(self, n_features=21, random_state=123):
        """
        Inicializa el generador de datos.

        Args:
            n_features (int): Número de características
            random_state (int): Semilla para reproducibilidad
        """
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)

    def make_features(self, n_samples):
        """
        Genera una matriz de características con distribución normal.

        Args:
            n_samples (int): Número de muestras
x
        Returns:
            ndarray: Matriz de características
        """
        mean = np.random.uniform(-1, 1)
        std = np.random.uniform(0.5, 2)
        return np.random.normal(mean, std, size=(n_samples, self.n_features))

    def generate_dataset(self, class_sizes):
        """
        Genera dataset multiclase balanceado.

        Args:
            class_sizes (list): Tamaño de cada clase

        Returns:
            DataFrame: Dataset con características y etiquetas
        """
        X_list = []
        y_list = []

        for class_idx, size in enumerate(class_sizes):
            X_class = self.make_features(size)
            X_list.append(X_class)
            y_list.extend([f"Class{class_idx + 1}"] * size)

        X = np.vstack(X_list)
        y = np.array(y_list)

        # Crear DataFrame
        feature_names = [f"X{i + 1}" for i in range(self.n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y

        return df


class MonteCarloIterationWorker:
    """
    Worker que ejecuta una iteración individual del proceso Monte Carlo.
    """

    def __init__(self, data, iteration_num, sample_fraction,
                 n_trees, random_state, target_column='Class'):
        """
        Inicializa el worker.

        Args:
            data (DataFrame): Dataset completo
            iteration_num (int): Número de iteración
            sample_fraction (float): Fracción de muestreo
            n_trees (int): Número de árboles
            random_state (int): Semilla
            target_column (str): Columna objetivo
        """
        self.data = data
        self.iteration_num = iteration_num
        self.sample_fraction = sample_fraction
        self.n_trees = n_trees
        self.random_state = random_state
        self.target_column = target_column

    def stratified_sample(self):
        """Realiza muestreo estratificado aleatorio."""
        indices = []

        for class_label in self.data[self.target_column].unique():
            class_indices = self.data[
                self.data[self.target_column] == class_label
                ].index.tolist()
            sample_size = int(len(class_indices) * self.sample_fraction)
            sampled = np.random.choice(
                class_indices, sample_size, replace=False
            )
            indices.extend(sampled)

        return self.data.iloc[indices]

    def execute(self):
        """
        Ejecuta una iteración del proceso Monte Carlo.

        Returns:
            tuple: (iteration_num, feature_importances)
        """
        # Muestreo estratificado
        train_data = self.stratified_sample()

        # Separar características y etiquetas
        X_train = train_data.drop(columns=[self.target_column])
        y_train = train_data[self.target_column]

        # Entrenar Random Forest
        model = RandomForestClassifier(
            n_estimators=self.n_trees,
            random_state=self.random_state + self.iteration_num
        )
        model.fit(X_train, y_train)

        return self.iteration_num, model.feature_importances_


class MonteCarloFeatureSelector:
    """
    Selector de características mediante Monte Carlo con Random Forest
    con soporte para procesamiento multihilo.
    """

    def __init__(self, n_iterations=50, sample_fraction=0.7,
                 n_trees=100, random_state=4561, n_workers=4):
        """
        Inicializa el selector de características.

        Args:
            n_iterations (int): Número de iteraciones Monte Carlo
            sample_fraction (float): Fracción de datos en cada iteración
            n_trees (int): Número de árboles en Random Forest
            random_state (int): Semilla para reproducibilidad
            n_workers (int): Número de hilos de trabajo
        """
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.n_trees = n_trees
        self.random_state = random_state
        self.n_workers = n_workers
        self.importance_results = None
        self.feature_importance = None
        self.final_ranking = None
        self.lock = Lock()

    def _execute_iteration(self, iteration_num, data, target_column):
        """
        Método privado para ejecutar una iteración.

        Args:
            iteration_num (int): Número de iteración
            data (DataFrame): Dataset
            target_column (str): Columna objetivo

        Returns:
            tuple: (iteration_num, importancias)
        """
        worker = MonteCarloIterationWorker(
            data=data,
            iteration_num=iteration_num,
            sample_fraction=self.sample_fraction,
            n_trees=self.n_trees,
            random_state=self.random_state,
            target_column=target_column
        )
        return worker.execute()

    def run_sequential(self, data, target_column='Class'):
        """
        Ejecuta el proceso Monte Carlo de forma secuencial (sin hilos).

        Args:
            data (DataFrame): Dataset con características y etiquetas
            target_column (str): Nombre de la columna objetivo
        """
        np.random.seed(self.random_state)

        n_features = len(data.columns) - 1
        self.importance_results = np.zeros((self.n_iterations, n_features))

        feature_names = [col for col in data.columns if col != target_column]

        print(f"Ejecutando {self.n_iterations} iteraciones (MODO SECUENCIAL)...")

        for i in range(self.n_iterations):
            _, importances = self._execute_iteration(i, data, target_column)
            self.importance_results[i, :] = importances

            if (i + 1) % 10 == 0:
                print(f"  Iteración {i + 1}/{self.n_iterations} completada")

        self._finalize_results(feature_names)

    def run_parallel(self, data, target_column='Class'):
        """
        Ejecuta el proceso Monte Carlo con procesamiento multihilo.

        Args:
            data (DataFrame): Dataset con características y etiquetas
            target_column (str): Nombre de la columna objetivo
        """
        np.random.seed(self.random_state)

        n_features = len(data.columns) - 1
        self.importance_results = np.zeros((self.n_iterations, n_features))

        feature_names = [col for col in data.columns if col != target_column]

        print(f"Ejecutando {self.n_iterations} iteraciones")
        print(f"  Modo: MULTIHILO con {self.n_workers} workers")
        print(f"  Iniciando procesos paralelos...\n")

        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Enviar todas las iteraciones al pool
            futures = {
                executor.submit(
                    self._execute_iteration, i, data, target_column
                ): i for i in range(self.n_iterations)
            }

            # Procesar resultados conforme se completan
            for future in as_completed(futures):
                iter_num, importances = future.result()

                with self.lock:
                    self.importance_results[iter_num, :] = importances
                    completed += 1

                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"  ✓ {completed}/{self.n_iterations} completadas "
                              f"({elapsed:.2f}s)")

        elapsed = time.time() - start_time
        print(f"\n  Tiempo total: {elapsed:.2f} segundos")
        self._finalize_results(feature_names)

    def _finalize_results(self, feature_names):
        """
        Finaliza los resultados calculando importancias.

        Args:
            feature_names (list): Nombres de características
        """
        # Calcular promedio de importancias
        self.feature_importance = np.mean(self.importance_results, axis=0)

        # Crear ranking ordenado
        self.final_ranking = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False)

        print("✓ Análisis completado")

    def run(self, data, target_column='Class', use_parallel=True):
        """
        Ejecuta el proceso Monte Carlo.

        Args:
            data (DataFrame): Dataset con características y etiquetas
            target_column (str): Nombre de la columna objetivo
            use_parallel (bool): Usar procesamiento multihilo
        """
        if use_parallel:
            self.run_parallel(data, target_column)
        else:
            self.run_sequential(data, target_column)

    def get_results(self):
        """
        Obtiene los resultados del análisis.

        Returns:
            DataFrame: Ranking de características ordenado
        """
        return self.final_ranking

    def plot_importance(self, top_n=None):
        """
        Visualiza la importancia de las características.

        Args:
            top_n (int): Mostrar solo las top N características
        """
        if self.final_ranking is None:
            print("Error: ejecuta run() primero")
            return

        data_to_plot = (
            self.final_ranking.head(top_n)
            if top_n else self.final_ranking
        )

        plt.figure(figsize=(12, 6))
        plt.barh(data_to_plot['Feature'], data_to_plot['Importance'])
        plt.xlabel('Mean Decrease Gini (promedio sobre iteraciones)')
        plt.ylabel('Características')
        plt.title('Importancia de Características - Monte Carlo Feature Selection')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def print_results(self):
        """Imprime los resultados en consola."""
        print("\n=== RANKING DE IMPORTANCIA DE CARACTERÍSTICAS ===")
        print(self.final_ranking.to_string(index=False))


class FeatureSelectionPipeline:
    """
    Pipeline completo para selección de características.
    """

    def __init__(self, class_sizes=None, n_features=21,
                 n_iterations=50, sample_fraction=0.7, n_workers=4):
        """
        Inicializa el pipeline.

        Args:
            class_sizes (list): Tamaños de cada clase
            n_features (int): Número de características
            n_iterations (int): Iteraciones Monte Carlo
            sample_fraction (float): Fracción de muestreo
            n_workers (int): Número de hilos de trabajo
        """
        if class_sizes is None:
            class_sizes = [513, 1223, 2731]
        self.class_sizes = class_sizes
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.n_workers = n_workers
        self.data = None
        self.selector = None

    def run(self, use_parallel=True):
        """
        Ejecuta el pipeline completo.

        Args:
            use_parallel (bool): Usar procesamiento multihilo
        """
        print("=" * 70)
        print("PIPELINE DE SELECCIÓN DE CARACTERÍSTICAS CON RANDOM FOREST")
        print("=" * 70)

        # Generar datos
        print("\n1. Generando dataset sintético...")
        generator = DataGenerator(n_features=self.n_features, random_state=123)
        self.data = generator.generate_dataset(self.class_sizes)

        print(f"   Dataset generado: {self.data.shape}")
        print(f"   Distribución de clases:")
        print(self.data['Class'].value_counts().sort_index())

        # Seleccionar características
        print("\n2. Ejecutando selección de características...")
        self.selector = MonteCarloFeatureSelector(
            n_iterations=self.n_iterations,
            sample_fraction=self.sample_fraction,
            random_state=4561,
            n_workers=self.n_workers
        )
        self.selector.run(self.data, use_parallel=use_parallel)

        # Mostrar resultados
        print("\n3. Resultados:")
        self.selector.print_results()

        # Visualizar
        print("\n4. Generando gráfico...")
        self.selector.plot_importance(top_n=15)


# Uso del programa
if __name__ == "__main__":
    # Ejecutar con procesamiento multihilo (recomendado)
    pipeline = FeatureSelectionPipeline(
        class_sizes=[513, 1223, 2731],
        n_features=21,
        n_iterations=50,
        sample_fraction=0.7,
        n_workers=4  # Número de hilos
    )
    pipeline.run(use_parallel=True)

    # Para comparar, descomenta la siguiente línea para modo secuencial:
    # pipeline.run(use_parallel=False)
