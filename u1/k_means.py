"""
K-means para segmentación de imágenes
Adapta cada píxel de la imagen a un cluster y asigna colores distintivos
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from classifier import Classifier


class KmeansImageSegmentation(Classifier):
    def __init__(self, k=3, max_iter=20, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.clusters = None
        self.cluster_colors = None
        self.original_shape = None

    def _initialize_centroids(self, X):
        """Inicializa centroides seleccionando k puntos aleatorios"""
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices].astype(np.float64)

    def _initialize_centroids_plus(self, X):
        """K-means++ para mejor inicialización"""
        centroids = []
        # Primer centroide aleatorio
        centroids.append(X[np.random.randint(0, len(X))])

        for _ in range(1, self.k):
            # Calcular distancias al centroide más cercano
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])
            # Seleccionar siguiente centroide con probabilidad proporcional a distancia²
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.random()
            idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[idx])

        return np.array(centroids, dtype=np.float64)

    def _assign_clusters(self, X):
        """Asigna cada punto al centroide más cercano"""
        # Vectorized distance calculation for efficiency
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, clusters):
        """Actualiza centroides al promedio de puntos asignados"""
        new_centroids = []
        for c in range(self.k):
            points = X[clusters == c]
            if len(points) > 0:
                new_centroids.append(points.mean(axis=0))
            else:
                # Si no hay puntos, reinicializar aleatoriamente
                new_centroids.append(X[np.random.randint(0, len(X))])
        return np.array(new_centroids, dtype=np.float64)

    def _generate_cluster_colors(self):
        """Genera colores distintivos para cada cluster"""
        predefined_colors = np.array([
            [255, 0, 0],  # Rojo
            [0, 255, 0],  # Verde
            [0, 0, 255],  # Azul
            [255, 255, 0],  # Amarillo
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cian
        ])

        if self.k <= len(predefined_colors):
            self.cluster_colors = predefined_colors[:self.k]
        else:
            # Generar colores aleatorios si necesitamos más de 10
            extra_colors = np.random.randint(0, 256, (self.k - len(predefined_colors), 3))
            self.cluster_colors = np.vstack([predefined_colors, extra_colors])

    def _generate_cluster_colors_from_centroids(self):
        """Usar los propios centroides como colores (más natural)"""
        self.cluster_colors = np.clip(self.centroids, 0, 255).astype(np.uint8)

    def fit(self, image_data, use_plus_plus=True, use_centroid_colors=False):
        """
        Entrena K-means en los datos de la imagen

        Args:
            image_data: Array de píxeles (height*width, 3) para RGB
            use_plus_plus: Si usar K-means++ para inicialización
            use_centroid_colors: Si usar centroides como colores o colores distintivos
        """
        X = image_data.astype(np.float64)

        # Inicializar centroides
        if use_plus_plus:
            self.centroids = self._initialize_centroids_plus(X)
        else:
            self.centroids = self._initialize_centroids(X)

        print(f"Iniciando K-means con k={self.k}...")

        for iteration in range(self.max_iter):
            # Asignar clusters
            clusters = self._assign_clusters(X)

            # Actualizar centroides
            new_centroids = self._update_centroids(X, clusters)

            # Verificar convergencia
            if np.allclose(self.centroids, new_centroids, rtol=self.tol):
                print(f"Convergió en {iteration + 1} iteraciones")
                break

            self.centroids = new_centroids

            if iteration % 5 == 0:
                print(f"Iteración {iteration + 1}/{self.max_iter}")

        self.clusters = clusters

        # Generar colores para clusters
        if use_centroid_colors:
            self._generate_cluster_colors_from_centroids()
        else:
            self._generate_cluster_colors()

        print("Entrenamiento completado!")

    def predict(self, image_data):
        """Predice clusters para nuevos datos de imagen"""
        if self.centroids is None:
            raise ValueError("El modelo debe ser entrenado primero")
        return self._assign_clusters(image_data.astype(np.float64))

    def segment_image(self, image_path, use_centroid_colors=False):
        """
        Segmenta una imagen completa

        Args:
            image_path: Ruta a la imagen
            use_centroid_colors: Si usar centroides como colores

        Returns:
            original_image: Imagen original
            segmented_image: Imagen segmentada
            clusters_reshaped: Array de clusters con forma de imagen
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        print(f"Procesando imagen de tamaño: {image.shape}")

        # Guardar forma original
        self.original_shape = image.shape
        height, width = image.shape[:2]

        pixels = image.reshape(-1, 3)

        # Entrenar K-means
        self.fit(pixels, use_centroid_colors=use_centroid_colors)

        segmented_pixels = self.cluster_colors[self.clusters]
        segmented_image = segmented_pixels.reshape(height, width, 3)
        clusters_reshaped = self.clusters.reshape(height, width)

        return image, segmented_image, clusters_reshaped

    def create_cluster_mask(self, cluster_id):
        """Crea una máscara binaria para un cluster específico"""
        if self.clusters is None or self.original_shape is None:
            raise ValueError("Debe segmentar una imagen primero")

        mask = (self.clusters == cluster_id).reshape(self.original_shape[:2])
        return mask.astype(np.uint8) * 255

    def get_cluster_statistics(self):
        """Obtiene estadísticas de los clusters"""
        if self.clusters is None:
            return None

        stats = {}
        total_pixels = len(self.clusters)

        for i in range(self.k):
            cluster_pixels = np.sum(self.clusters == i)
            percentage = (cluster_pixels / total_pixels) * 100
            color = self.cluster_colors[i] if self.cluster_colors is not None else self.centroids[i]

            stats[f'Cluster_{i}'] = {
                'pixels': cluster_pixels,
                'percentage': percentage,
                'centroid': self.centroids[i],
                'color': color
            }

        return stats


def visualize_segmentation(original, segmented, k):
    """Visualiza solo la imagen original y la segmentada"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Imagen original
    axes[0].imshow(original)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')

    # Imagen segmentada
    axes[1].imshow(segmented.astype(np.uint8))
    axes[1].set_title(f'Imagen Segmentada')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "../images/example.jpg"

    segmentator = KmeansImageSegmentation(k=4, max_iter=20)

    print("Segmentando imagen...")
    original, segmented, clusters = segmentator.segment_image(image_path,
                                                              use_centroid_colors=False)
    visualize_segmentation(original, segmented, clusters)
    plt.tight_layout()
    plt.show()
