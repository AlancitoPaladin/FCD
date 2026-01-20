from pathlib import Path

import numpy as np
import pandas as pd


class PCAManual:
    """
    Implementación manual de PCA (Principal Component Analysis) clásico.
    Sin usar sklearn, solo NumPy para operaciones matriciales.
    """

    def __init__(self, n_components=None, random_state=42):
        """
        Inicializa el PCA manual.

        Args:
            n_components: Número de componentes principales a extraer
            random_state: Semilla para reproducibilidad
        """
        self.n_components = n_components
        self.random_state = random_state
        np.random.seed(random_state)

        # Atributos que se calculan durante el ajuste
        self.mean = None
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Ajusta el PCA a los datos de entrada.

        Args:
            X: Array de datos (n_samples, n_features)

        Returns:
            self
        """
        n_samples, n_features = X.shape

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Paso 1: Estandarización (restar la media)
        print(f"Datos originales: {X.shape}")
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        print(f"Media de características: {self.mean[:5]}... (primeros 5)")

        # Paso 2: Calcular matriz de covarianza
        print("\nCalculando matriz de covarianza...")
        self.cov_matrix = np.cov(X_centered.T)  # (n_features, n_features)
        print(f"Matriz de covarianza: {self.cov_matrix.shape}")

        # Paso 3: Descomposición en valores y vectores propios (eigendecomposition)
        print("\nCalculando eigenvalores y eigenvectores...")
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)

        # Paso 4: Ordenar por eigenvalores descendentes
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

        # Paso 5: Seleccionar los primeros n_components
        self.components = self.eigenvectors[:, :self.n_components]
        self.eigenvalues = self.eigenvalues[:self.n_components]

        # Paso 6: Calcular varianza explicada
        self.explained_variance = self.eigenvalues
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance

        print(f"Componentes principales extraídos: {self.n_components}")
        print(f"Varianza explicada: {np.sum(self.explained_variance_ratio) * 100:.2f}%")

        return self

    def fit_batch(self, X, batch_size=50):
        """
        Ajusta PCA procesando datos por lotes para ahorrar memoria.
        """
        n_samples, n_features = X.shape

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        print(f"Datos originales: {X.shape}")
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # SVD es más eficiente que covarianza para datos grandes
        print("\nCalculando SVD en modo económico...")
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.eigenvalues = (S ** 2) / (n_samples - 1)
        self.eigenvectors = Vt.T
        self.components = self.eigenvectors[:, :self.n_components]
        self.eigenvalues = self.eigenvalues[:self.n_components]

        self.explained_variance = self.eigenvalues
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance

        print(f"Componentes principales extraídos: {self.n_components}")
        print(f"Varianza explicada: {np.sum(self.explained_variance_ratio) * 100:.2f}%")

        return self

    def transform(self, X):
        """
        Proyecta los datos en el espacio de componentes principales.

        Args:
            X: Array de datos (n_samples, n_features)

        Returns:
            Datos proyectados (n_samples, n_components)
        """
        if self.components is None:
            raise ValueError("PCA no ha sido ajustado. Llama a fit() primero.")

        # Centra los datos usando la media aprendida
        X_centered = X - self.mean

        # Proyecta en los componentes principales
        X_transformed = np.dot(X_centered, self.components)

        return X_transformed

    def fit_transform(self, X):
        """
        Ajusta el PCA y transforma los datos en una sola operación.

        Args:
            X: Array de datos

        Returns:
            Datos transformados
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Proyecta los datos transformados de vuelta al espacio original.

        Args:
            X_transformed: Datos en el espacio PCA

        Returns:
            Datos reconstruidos en el espacio original
        """
        if self.components is None:
            raise ValueError("PCA no ha sido ajustado.")

        # Reconstruye: X_original ≈ X_transformed · components^T + mean
        X_reconstructed = np.dot(X_transformed, self.components.T)
        X_reconstructed += self.mean

        return X_reconstructed

    def get_explained_variance(self):
        """Retorna la varianza explicada por cada componente."""
        return self.explained_variance_ratio

    def get_cumulative_variance(self):
        """Retorna la varianza explicada acumulada."""
        return np.cumsum(self.explained_variance_ratio)

    def print_summary(self):
        """Imprime un resumen del análisis PCA."""
        print("\n" + "=" * 60)
        print("RESUMEN DEL ANÁLISIS PCA")
        print("=" * 60)
        print(f"Número de componentes: {self.n_components}")
        print(f"Dimensionalidad original: {self.components.shape[0]}")
        print(f"Dimensionalidad reducida: {self.components.shape[1]}")
        print(f"Reducción: {(1 - self.n_components / self.components.shape[0]) * 100:.1f}%")
        print(f"\nVarianza explicada acumulada:")

        cumsum = np.cumsum(self.explained_variance_ratio)
        for i, var in enumerate(cumsum[:min(10, self.n_components)]):
            print(f"  PC{i + 1}: {var * 100:.2f}%")

        if self.n_components > 10:
            print(f"  ... ({self.n_components - 10} componentes más)")

        print("=" * 60 + "\n")


class VGG16FeaturesProcessor:
    """
    Procesa características VGG16 aplicando PCA manual.
    """

    def __init__(self, features_path="../u1/features.csv"):
        """
        Inicializa el procesador.

        Args:
            features_path: Ruta al archivo CSV con características VGG16
        """
        self.features_path = features_path
        self.df = None
        self.X = None
        self.feature_names = None

    def load_features(self):
        """Carga las características del archivo CSV."""
        print(f"Cargando características desde: {self.features_path}")
        self.df = pd.read_csv(self.features_path)
        print(f"Forma del dataset: {self.df.shape}")
        print(f"Primeras columnas: {self.df.columns[:5].tolist()}")

        # Identifica columnas numéricas (características)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols
        self.X = self.df[numeric_cols].values

        print(f"Número de características: {len(self.feature_names)}")
        print(f"Número de muestras: {self.X.shape[0]}")

        return self.X

    def apply_pca(self, n_components=None, variance_ratio=0.95):
        """
        Aplica PCA manual a las características.

        Args:
            n_components: Número de componentes (si None, usa variance_ratio)
            variance_ratio: Ratio de varianza a explicar (si n_components es None)

        Returns:
            Datos transformados por PCA
        """
        if self.X is None:
            self.load_features()

        # Si no se especifica n_components, calcula basado en varianza
        if n_components is None:
            pca_temp = PCAManual(n_components=self.X.shape[1])
            pca_temp.fit(self.X)
            cumsum = pca_temp.get_cumulative_variance()
            n_components = np.argmax(cumsum >= variance_ratio) + 1
            print(f"Componentes necesarios para {variance_ratio * 100}% varianza: {n_components}")

        # Ajusta y transforma
        pca = PCAManual(n_components=n_components)
        X_pca = pca.fit_transform(self.X)

        pca.print_summary()

        return pca, X_pca

    def save_pca_features(self, X_pca, output_path=None):
        """Guarda las características transformadas por PCA."""
        if output_path is None:
            output_path = str(Path(self.features_path).stem) + "_pca.csv"

        # Crea columnas con nombres descriptivos
        n_components = X_pca.shape[1]
        pca_columns = [f"PC{i + 1}" for i in range(n_components)]

        df_pca = pd.DataFrame(X_pca, columns=pca_columns)

        # Preserva las columnas no numéricas (si las hay)
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            for col in non_numeric_cols:
                df_pca[col] = self.df[col].values

        df_pca.to_csv(output_path, index=False)
        print(f"Características PCA guardadas en: {output_path}")

        return output_path

    def plot_variance(self, pca, max_components=50):
        """
        Imprime la varianza explicada por componente.

        Args:
            pca: Objeto PCAManual ajustado
            max_components: Máximo número de componentes a mostrar
        """
        variance_ratio = pca.get_explained_variance()
        cumsum = pca.get_cumulative_variance()

        print("\n" + "=" * 60)
        print("VARIANZA EXPLICADA POR COMPONENTE")
        print("=" * 60)

        for i in range(min(max_components, len(variance_ratio))):
            bar = "█" * int(variance_ratio[i] * 100)
            print(f"PC{i + 1:3d}: {bar:<50} {variance_ratio[i] * 100:6.2f}% | Acum: {cumsum[i] * 100:6.2f}%")

        print("=" * 60 + "\n")


# Ejemplo de uso
if __name__ == "__main__":
    # Procesa características VGG16
    processor = VGG16FeaturesProcessor("../u1/features.csv")

    # Carga características
    X = processor.load_features()

    # Aplica PCA manual
    pca, X_pca = processor.apply_pca(variance_ratio=0.95)

    # Muestra varianza explicada
    processor.plot_variance(pca, max_components=30)

    # Guarda características transformadas
    processor.save_pca_features(X_pca)

    # Ejemplo adicional: reconstrucción
    print("\nEjemplo de reconstrucción:")
    X_reconstructed = pca.inverse_transform(X_pca)
    error_reconstruccion = np.mean((X - X_reconstructed) ** 2)
    print(f"Error cuadrado medio de reconstrucción: {error_reconstruccion:.4f}")
