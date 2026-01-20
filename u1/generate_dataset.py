import cv2
import numpy as np
import pandas as pd
from utils import Utils as u

def hu_moments_opencv(image, eps=1e-12):
    """
    Calcula los momentos de Hu de una imagen usando OpenCV.
    """
    # Binarizar con Otsu
    _, thresh = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Momentos
    moments = cv2.moments(thresh)

    # Hu moments
    hu = cv2.HuMoments(moments).flatten()

    # Escala logarítmica para estabilidad
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)

    return hu_log

def generate_hu_dataset(csv_path, output_path, n_samples=None):
    images, labels = u.load_dataset(csv_path)
    if n_samples is not None:
        images, labels = images[:n_samples], labels[:n_samples]

    hu_features = []
    for i, img in enumerate(images):
        hu = hu_moments_opencv(img)
        hu_features.append(hu)

        if (i + 1) % 5000 == 0:
            print(f"Procesadas {i+1} imágenes...")

    # Guardar en DataFrame
    columns = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    df = pd.DataFrame(hu_features, columns=columns)
    df.insert(0, "label", labels)

    df.to_csv(output_path, index=False)
    print(f"Dataset guardado en: {output_path}")

if __name__ == "__main__":
    train_path = "../mnist/mnist_train.csv"
    test_path = "../mnist/mnist_test.csv"

    # Train
    generate_hu_dataset(train_path, "../mnist_hu/mnist_hu_train.csv")

    # Test
    generate_hu_dataset(test_path, "../mnist_hu/mnist_hu_test.csv")
