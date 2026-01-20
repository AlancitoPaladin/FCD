import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCAAnalyzer:
    """
    Análisis completo de PCA con múltiples criterios para determinar
    el número óptimo de componentes.
    Basado en la metodología MATLAB del análisis de dataset yeast.
    """

    def __init__(self, data=None, random_state=42):
        """
        Inicializa el analizador PCA.

        Args:
            data: Array de datos (n_samples, n_features)
            random_state: Semilla para reproducibilidad
        """
        self.data = data
        self.random_state = random_state
        np.random.seed(random_state)

        # Atributos calculados
        self.n_samples = None
        self.n_features = None
        self.data_centered = None
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.variance_percentage = None
        self.optimal_components = None

    def load_data(self, filepath):
        """
        Carga datos desde un archivo CSV.

        Args:
            filepath: Ruta al archivo CSV

        Returns:
            Array de datos cargados
        """
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.data = df[numeric_cols].values
        print(f"Datos cargados: {self.data.shape}")
        return self.data

    def center_data(self):
        """
        Centra los datos restando la media.
        Equivalente a: datac = data - repmat(sum(data)/n,n,1)
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Usa load_data() primero.")

        self.n_samples, self.n_features = self.data.shape
        self.mean = np.mean(self.data, axis=0)
        self.data_centered = self.data - self.mean

        print(f"✓ Datos centrados")
        print(f"  Muestras (n): {self.n_samples}")
        print(f"  Características (p): {self.n_features}")

        return self.data_centered

    def compute_covariance_and_eigen(self):
        """
        Calcula la matriz de covarianza y descomposición en eigenvalores/eigenvectores.
        Equivalente a: covm = cov(datac); [eigvec, eigval] = eig(covm);
        """
        if self.data_centered is None:
            raise ValueError("Centra los datos primero con center_data().")

        print("\nCalculando matriz de covarianza...")
        self.cov_matrix = np.cov(self.data_centered.T)
        print(f"✓ Matriz de covarianza: {self.cov_matrix.shape}")

        print("Calculando eigenvalores y eigenvectores...")
        eigval, eigvec = np.linalg.eig(self.cov_matrix)

        # Ordenar en orden descendente (flipud equivalente)
        idx = np.argsort(eigval)[::-1]
        self.eigenvalues = np.real(eigval[idx])
        self.eigenvectors = np.real(eigvec[:, idx])

        print(f"Eigenvalores ordenados: {len(self.eigenvalues)}")

        return self.eigenvalues, self.eigenvectors

    def compute_variance_percentage(self):
        """
        Calcula el porcentaje acumulativo de varianza explicada.
        Equivalente a: pervar = 100*cumsum(eigval)/sum(eigval);
        """
        if self.eigenvalues is None:
            raise ValueError("Calcula eigenvalores primero con compute_covariance_and_eigen().")

        cumsum_eigval = np.cumsum(self.eigenvalues)
        self.variance_percentage = 100 * cumsum_eigval / np.sum(self.eigenvalues)

        print("\nPorcentaje de varianza acumulada:")
        for i in range(min(10, len(self.variance_percentage))):
            bar = "█" * int(self.variance_percentage[i] / 2)
            print(f"  PC{i + 1}: {bar:<50} {self.variance_percentage[i]:6.2f}%")

        return self.variance_percentage

    def criterion_kaiser(self):
        """
        Criterio de Kaiser: mantiene eigenvalores mayores que la media.
        Equivalente a: avgeig = mean(eigval); ind = find(eigval > avgeig);

        Returns:
            Número de componentes según criterio Kaiser
        """
        avg_eigenvalue = np.mean(self.eigenvalues)
        n_components_kaiser = np.sum(self.eigenvalues > avg_eigenvalue)

        print(f"\nCRITERIO DE KAISER")
        print(f"  Media de eigenvalores: {avg_eigenvalue:.4f}")
        print(f"  Componentes a mantener: {n_components_kaiser}")
        print(f"  Varianza explicada: {self.variance_percentage[n_components_kaiser - 1]:.2f}%")

        return n_components_kaiser

    def criterion_scree(self, threshold=95):
        """
        Criterio de Scree: selecciona componentes que expliquen
        un porcentaje mínimo de varianza (típicamente 95%).

        Args:
            threshold: Porcentaje de varianza acumulada (default: 95%)

        Returns:
            Número de componentes según criterio Scree
        """
        n_components_scree = np.argmax(self.variance_percentage >= threshold) + 1

        print(f"\nCRITERIO DE SCREE")
        print(f"  Umbral de varianza: {threshold}%")
        print(f"  Componentes a mantener: {n_components_scree}")
        print(f"  Varianza explicada: {self.variance_percentage[n_components_scree - 1]:.2f}%")

        return n_components_scree

    def criterion_average_variance(self):
        """
        Criterio de proporcionalidad de varianza: mantiene componentes
        cuya varianza sea mayor que la varianza promedio esperada.
        Equivalente al método 'g' del código MATLAB.

        Returns:
            Número de componentes según este criterio
        """
        # g = 1/p + 1/(p-1) + ... + 1/2 (media armónica)
        p = self.n_features
        g = np.sum(1 / np.arange(1, p + 1)) / p

        # Proporciones de varianza
        prop_var = self.eigenvalues / np.sum(self.eigenvalues)

        # Encontrar índices donde prop_var > g
        ind = np.where(prop_var > g)[0]
        n_components_avg = len(ind) if len(ind) > 0 else 1

        print(f"\n⚖️  CRITERIO DE VARIANZA PROMEDIO")
        print(f"  Umbral (media armónica): {g:.4f}")
        print(f"  Componentes que superan umbral: {n_components_avg}")
        if n_components_avg > 0:
            print(f"  Varianza explicada: {self.variance_percentage[n_components_avg - 1]:.2f}%")

        return n_components_avg

    def recommend_components(self):
        """
        Realiza todas las pruebas y recomienda el número óptimo de componentes.

        Returns:
            Número recomendado de componentes
        """
        print("\n" + "=" * 70)
        print("ANÁLISIS DE CRITERIOS DE DIMENSIONALIDAD")
        print("=" * 70)

        n_kaiser = self.criterion_kaiser()
        n_scree = self.criterion_scree(threshold=95)
        n_avg = self.criterion_average_variance()

        # Tomar la mediana como recomendación
        candidates = [n_kaiser, n_scree, n_avg]
        self.optimal_components = int(np.median(candidates))

        print(f"\n{'=' * 70}")
        print(f"✓ RECOMENDACIÓN: Mantener {self.optimal_components} componentes")
        print(f"{'=' * 70}\n")

        return self.optimal_components

    def project_data(self, n_components=None):
        """
        Proyecta los datos en el espacio de componentes principales.
        Equivalente a: Xp = datac*P; (donde P = eigvec[:, 1:n_components])

        Args:
            n_components: Número de componentes (si None, usa recomendación)

        Returns:
            Datos proyectados
        """
        if n_components is None:
            if self.optimal_components is None:
                n_components = self.recommend_components()
            else:
                n_components = self.optimal_components

        P = self.eigenvectors[:, :n_components]
        X_projected = np.dot(self.data_centered, P)

        print(f"✓ Datos proyectados a {n_components} componentes: {X_projected.shape}")

        return X_projected

    def plot_scree(self):
        """
        Grafica el Scree Plot.
        Equivalente a: figure, plot(1:length(eigval),eigval,'ko-')
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.eigenvalues) + 1), self.eigenvalues, 'ko-', linewidth=2, markersize=6)
        plt.title('Scree Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Eigenvalue Index (k)', fontsize=12)
        plt.ylabel('Eigenvalue', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt

    def plot_variance_cumulative(self):
        """
        Grafica el porcentaje acumulativo de varianza.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.variance_percentage) + 1),
                 self.variance_percentage, 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=95, color='r', linestyle='--', label='95% varianza')
        plt.title('Varianza Acumulada', fontsize=14, fontweight='bold')
        plt.xlabel('Número de Componentes', fontsize=12)
        plt.ylabel('Varianza Acumulada (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt

    def plot_3d_projection(self, X_projected=None):
        """
        Grafica la proyección en 3D.
        Equivalente a: figure,plot3(Xp(:,1),Xp(:,2),Xp(:,3),'k*')

        Args:
            X_projected: Datos proyectados (si None, calcula automáticamente)
        """
        if X_projected is None:
            X_projected = self.project_data(n_components=3)

        if X_projected.shape[1] < 3:
            print("Se necesitan al menos 3 componentes para graficar en 3D")
            return None

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2],
                   c='black', marker='*', s=200, alpha=0.6)
        ax.set_xlabel('PC 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PC 2', fontsize=11, fontweight='bold')
        ax.set_zlabel('PC 3', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        return fig

    def plot_matrix(self, X_projected=None):
        """
        Grafica la matriz de pares (scatter plots).
        Equivalente a: figure,plotmatrix(Xp)

        Args:
            X_projected: Datos proyectados
        """
        if X_projected is None:
            X_projected = self.project_data()

        n_components = X_projected.shape[1]
        fig, axes = plt.subplots(n_components, n_components, figsize=(12, 12))

        for i in range(n_components):
            for j in range(n_components):
                ax = axes[i, j] if n_components > 1 else axes

                if i == j:
                    # Diagonal: histograma
                    ax.hist(X_projected[:, i], bins=20, color='lightblue', edgecolor='black')
                else:
                    # Scatter plot
                    ax.scatter(X_projected[:, j], X_projected[:, i],
                               c='black', s=30, alpha=0.5)

                if j == 0:
                    ax.set_ylabel(f'PC{i + 1}', fontweight='bold')
                else:
                    ax.set_ylabel('')

                if i == n_components - 1:
                    ax.set_xlabel(f'PC{j + 1}', fontweight='bold')
                else:
                    ax.set_xlabel('')

        plt.tight_layout()
        return fig

    def summary(self):
        """Imprime un resumen completo del análisis."""
        print("\n" + "=" * 70)
        print("RESUMEN DEL ANÁLISIS PCA")
        print("=" * 70)
        print(f"Datos originales: {self.n_samples} muestras × {self.n_features} características")
        print(f"Reducción recomendada: {self.optimal_components} componentes")
        print(f"Reducción dimensional: {(1 - self.optimal_components / self.n_features) * 100:.1f}%")
        print(f"Varianza explicada: {self.variance_percentage[self.optimal_components - 1]:.2f}%")
        print("=" * 70 + "\n")


# Ejemplo de uso completo
if __name__ == "__main__":
    # Crear analizador
    analyzer = PCAAnalyzer()

    # Cargar datos
    analyzer.load_data("../u4/features_pca.csv")

    # Proceso de análisis
    analyzer.center_data()
    analyzer.compute_covariance_and_eigen()
    analyzer.compute_variance_percentage()

    # Determinar número óptimo de componentes
    n_comp = analyzer.recommend_components()

    # Proyectar datos
    X_projected = analyzer.project_data(n_components=n_comp)

    # Mostrar resumen
    analyzer.summary()

    # Visualizaciones
    print("Generando gráficas...")
    analyzer.plot_scree()
    analyzer.plot_variance_cumulative()
    analyzer.plot_3d_projection(X_projected)
    analyzer.plot_matrix(X_projected)

    plt.show()
