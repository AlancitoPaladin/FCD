"""

Cargar el modelo pre-entrenado de VGG-16.

Forzar el tamaño de la entrada(resize).

¿Cómo cortar las capas del modelo?

"""
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16

from utils import Utils as ut


# from tensorflow.keras.models import Model

class VisualGeometryGroup:
    def __init__(self, model):
        self.model = model

    def extract_features(self, image):
        new_image = np.array(image)
        new_image = np.expand_dims(new_image, axis=0)
        img = preprocess_input(new_image)
        features = self.model.predict(img, verbose=0)

        return features.flatten()


if __name__ == '__main__':
    # model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = VGG16(weights='imagenet', include_top=False)  # Remueve 3 capas por default
    vgg = VisualGeometryGroup(model)

    # cut_model = Model(inputs=model.input, outputs=model.layers[-6].output) # para cortar el modelo a las capas necesarias

    """Prueba con una sola imagen del modelo"""
    # image_path = "../dataset/red_bull.jpg"
    # img = ut.load_image_resized(image_path)

    data_path = '../dataset'
    X, y = ut.load_signs(dataset_path=data_path, per_class=15)

    df = pd.DataFrame()

    features_list = []
    for img in X:
        features = vgg.extract_features(img)  # img ya debe ser 224x224x3
        features_list.append(features)

    # Crear DataFrame y nombrar columnas
    df = pd.DataFrame(features_list)
    df.columns = [f"feat_{i}" for i in range(df.shape[1])]  # feat_0, feat_1, ...
    df["label"] = y  # agregar etiquetas

    # Guardar en CSV
    df.to_csv("features.csv", index=False)
    print("Features guardadas correctamente en features.csv")


#    print(features)
#   print(features.shape)
