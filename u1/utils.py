import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image


class Utils:
    @staticmethod
    def load_signs(dataset_path, per_class=15):
        """
        Carga un número igual de imágenes por clase.
        per_class: cuántas imágenes tomar de cada carpeta (clase)
        """
        X = []
        y = []

        for label in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, label)
            if not os.path.isdir(folder_path):
                continue

            # Obtener todas las imágenes de la clase
            all_imgs = os.listdir(folder_path)
            # Mezclar para tomar una muestra aleatoria
            np.random.shuffle(all_imgs)
            selected_imgs = all_imgs[:per_class]

            for fname in selected_imgs:
                img_path = os.path.join(folder_path, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    # Redimensionar a 224x224 para VGG16
                    X.append(img)
                    y.append(label)

        y = np.array(y)
        return X, y

    @staticmethod
    def load_dataset(file_path):
        """
        Loads a CSV file from the MNIST dataset and converts it into image matrices.

        Args:
            file_path: Path to the CSV file

        Returns:
            images: Numpy array with images (n_samples, 28, 28)
            labels: Numpy array with labels (n_samples)
        """
        # Load the dataset using pandas
        data = pd.read_csv(file_path)
        labels = data['label'].values
        pixels = data.drop('label', axis=1).values
        images = pixels.reshape(-1, 28, 28).astype(np.uint8)
        return images, labels

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def score(self, X, y):
        """
        Calcula la precisión del clasificador.

        Args:
            X: Matriz de características (n_muestras, n_características)
            y: Etiquetas verdaderas (n_muestras)

        Returns:
            Precisión del clasificador (0-1)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    @staticmethod
    def load_image(image):
        test_image = Image.open(image).convert("L")
        test_image = np.array(test_image)

        return test_image

    @staticmethod
    def load_image_resized(image_path):
        """
        Carga una imagen, la redimensiona a 224x224 y devuelve un array numpy (224,224,3)
        listo para usar en VGG16.
        """
        test_image = Image.open(image_path).convert('RGB')
        new_image = test_image.resize((224, 224))
        new_image = np.array(new_image)
        return new_image
