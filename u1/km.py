import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from classifier import Classifier
from utils import Utils as ut


class Kmeans(Classifier):
    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol  # Added tolerance for convergence
        self.centroids = None
        self.clusters = None
        self.inertia_ = None  # Track within-cluster sum of squares

    def _initialize_centroids(self, X):
        """Initialize centroids using random selection"""
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]

    def _initialize_centroids_plus(self, X):
        """K-means++ initialization for better centroid placement"""
        centroids = []
        # Choose first centroid randomly
        centroids.append(X[np.random.randint(0, len(X))])

        for _ in range(1, self.k):
            # Calculate distances to nearest centroid
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.random()
            idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, clusters):
        """Update centroids to the mean of assigned points"""
        new_centroids = []
        for c in range(self.k):
            points = X[clusters == c]
            if len(points) > 0:
                new_centroids.append(points.mean(axis=0))
            else:
                # If no points assigned, reinitialize randomly
                new_centroids.append(X[np.random.randint(0, len(X))])
        return np.array(new_centroids)

    def _calculate_inertia(self, X, clusters):
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for c in range(self.k):
            points = X[clusters == c]
            if len(points) > 0:
                inertia += np.sum((points - self.centroids[c]) ** 2)
        return inertia

    def fit(self, X, use_plus_plus=True):
        """Fit K-means to the data"""
        if use_plus_plus:
            self.centroids = self._initialize_centroids_plus(X)
        else:
            self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            # Assign points to clusters
            clusters = self._assign_clusters(X)

            # Update centroids
            new_centroids = self._update_centroids(X, clusters)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids, rtol=self.tol):
                print(f"Converged after {iteration + 1} iterations")
                break

            self.centroids = new_centroids

        self.clusters = clusters
        self.inertia_ = self._calculate_inertia(X, clusters)

    def predict(self, X):
        """Predict cluster labels for new data"""
        return self._assign_clusters(X)


# ---------- Enhanced Hu moments functions ----------
def hu_moments_with_cv2(img_flat):
    """
    Compute 7 Hu moments using OpenCV
    img_flat: flattened image array (784 elements for 28x28)
    Returns: 7 Hu moments (log-scaled)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not available. Install with: pip install opencv-python")

    img = img_flat.reshape(28, 28)

    # Ensure proper data type and range
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Calculate moments
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()

    # Log scale for numerical stability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu_log


def hu_moments_with_skimage(img_flat):
    """
    Compute 7 Hu moments using scikit-image
    Alternative when OpenCV is not available
    """
    try:
        from skimage.measure import moments, moments_central, moments_normalized, moments_hu
    except ImportError:
        raise ImportError("scikit-image not available. Install with: pip install scikit-image")

    img = img_flat.reshape(28, 28).astype(float)

    # Calculate central moments and then Hu moments
    m = moments(img)
    cr = m[1, 0] / m[0, 0]  # centroid row
    cc = m[0, 1] / m[0, 0]  # centroid col

    # Central moments
    mu = moments_central(img, cr, cc)

    # Normalized central moments
    nu = moments_normalized(mu)

    # Hu moments
    hu = moments_hu(nu)

    # Log scale
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu_log


def extract_hu_features(images, backend="cv2", verbose=True):
    """
    Extract Hu moments features from a batch of images
    """
    start_time = time.time()

    if backend == "cv2":
        try:
            hu_features = np.array([hu_moments_with_cv2(img) for img in images])
        except ImportError as e:
            print(f"Warning: {e}")
            print("Falling back to skimage...")
            backend = "skimage"

    if backend == "skimage":
        hu_features = np.array([hu_moments_with_skimage(img) for img in images])

    extraction_time = time.time() - start_time

    if verbose:
        print(f"Extracted Hu moments using {backend}")
        print(f"Features shape: {hu_features.shape}")
        print(f"Extraction time: {extraction_time:.2f}s")

    return hu_features, extraction_time


def evaluate_clustering(features, labels_true=None, k_range=range(2, 21)):
    """
    Evaluate clustering performance for different k values
    """
    inertias = []
    silhouette_scores = []

    try:
        from sklearn.metrics import silhouette_score
        use_silhouette = True
    except ImportError:
        use_silhouette = False
        print("Warning: scikit-learn not available for silhouette analysis")

    for k in k_range:
        km = Kmeans(k=k, max_iter=100)
        km.fit(features, use_plus_plus=True)
        inertias.append(km.inertia_)

        if use_silhouette and len(np.unique(km.clusters)) > 1:
            score = silhouette_score(features, km.clusters)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)

    return k_range, inertias, silhouette_scores


def plot_clustering_analysis(k_range, inertias, silhouette_scores):
    """
    Plot elbow curve and silhouette analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)

    # Silhouette analysis
    if max(silhouette_scores) > 0:
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Silhouette analysis\nnot available',
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()


def plot_clusters_pca(features, clusters, title="Clusters Visualization"):
    """
    Visualize clusters in 2D using PCA
    """
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=clusters, cmap='tab10', s=15, alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

    return pca.explained_variance_ratio_


if __name__ == "__main__":
    print("Loading MNIST dataset...")
    train_images, train_labels = ut.load_dataset("../mnist/mnist_test.csv")
    train_images = train_images.reshape(len(train_images), -1)

    print(f"Dataset shape: {train_images.shape}")

    # Extract Hu moments features
    print("\nExtracting Hu moments features...")
    hu_features, extraction_time = extract_hu_features(train_images, backend="cv2")

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    hu_scaled = scaler.fit_transform(hu_features)

    # Analyze optimal k
    print("\nAnalyzing optimal number of clusters...")
    k_range, inertias, silhouette_scores = evaluate_clustering(hu_scaled, k_range=range(2, 16))
    plot_clustering_analysis(k_range, inertias, silhouette_scores)

    # Perform final clustering with optimal k
    optimal_k = 10  # You can adjust this based on the analysis
    print(f"\nPerforming K-means clustering with k={optimal_k}...")

    km = Kmeans(k=optimal_k, max_iter=100)
    start_time = time.time()
    km.fit(hu_scaled, use_plus_plus=True)
    clustering_time = time.time() - start_time

    print(f"Clustering completed in {clustering_time:.2f}s")
    print(f"Final inertia: {km.inertia_:.2f}")

    # Visualize results
    print("\nVisualizing clustering results...")
    explained_var = plot_clusters_pca(hu_scaled, km.clusters,
                                      "K-means Clustering using Hu Moments")

    print(f"\nSummary:")
    print(f"- Feature extraction: {extraction_time:.2f}s")
    print(f"- Clustering time: {clustering_time:.2f}s")
    print(f"- PCA explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var[:2]):.2%}")
    print(f"- Number of clusters: {optimal_k}")
    print(f"- Final inertia: {km.inertia_:.2f}")