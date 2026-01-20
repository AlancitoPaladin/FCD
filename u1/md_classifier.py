import os

import matplotlib.pyplot as plt
import numpy as np

from mcd import MinimumDistanceClassifier

# -----------------------------
# Cargar MNIST desde .npz
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "mnist", "mnist.npz")

with np.load(dataset_path) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)


# -----------------------------
# Función para procesar imágenes
# -----------------------------
def process_images_raw(images, n_samples=None):
    if n_samples is None:
        n_samples = len(images)
    else:
        n_samples = min(n_samples, len(images))

    # Aplanar y normalizar entre 0 y 1
    return images[:n_samples].reshape(n_samples, -1) / 255.0


# -----------------------------
# Preparar datos
# -----------------------------
n_train = 60000
n_test = 10000

print(f"Procesando {n_train} imágenes de entrenamiento (Raw Pixels)...")
X_train = process_images_raw(x_train, n_train)
y_train = y_train[:n_train]

print(f"Procesando {n_test} imágenes de prueba...")
X_test = process_images_raw(x_test, n_test)
y_test = y_test[:n_test]

# -----------------------------
# Entrenar Minimum Distance Classifier
# -----------------------------
mdc = MinimumDistanceClassifier()
mdc.fit(X_train, y_train)

accuracy = mdc.score(X_test, y_test)
print(f"Precisión del clasificador: {accuracy:.4f}")

# -----------------------------
# Probar una muestra aleatoria
# -----------------------------
test_idx = 424
test_image = x_test[test_idx]
test_features = x_test[test_idx].reshape(1, -1) / 255.0
pred_class = mdc.predict(test_features)[0]
actual_class = y_test[test_idx]

print(f"Dígito de prueba - Clase real: {actual_class}, Clase predicha: {pred_class}")

plt.figure(figsize=(5, 5))
plt.imshow(test_image, cmap='gray')
plt.title(f'Clase real {actual_class} | Predicción {pred_class}')
plt.axis('off')
plt.show()

# Distancias a todos los centroides
print("\nDistancias del dígito de prueba a los centroides:")
for c in mdc.classes:
    centroid = mdc.centroids[c]
    distance = np.sqrt(np.sum((test_features[0] - centroid) ** 2))
    print(f"Clase {c}: {distance:.4f}")
