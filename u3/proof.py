

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- Parte 1: Cargar Modelo y Realizar Segmentación ---

# 1. Cargar un modelo YOLOv8-Seg preentrenado (en el dataset COCO)
# 'yolov8n-seg.pt' es la versión "nano" (más pequeña y rápida)
print("Cargando modelo YOLOv8-Seg...")
model = YOLO('yolov8n-seg.pt')
print("Modelo cargado.")

# 2. Ruta a tu imagen de prueba
# Asegúrate de que esta imagen esté en la misma carpeta que tu script,
# o especifica la ruta completa.
image_path = '../images/red_bull.jpg'  # Reemplaza con el nombre de tu imagen

# 3. Realizar la detección y segmentación en la imagen
print(f"Procesando imagen: {image_path}...")
results = model(image_path)  # Los resultados contienen detecciones y máscaras
print("Procesamiento terminado.")

# --- Parte 2: Visualizar Resultados (Opcional, pero útil para entender) ---

# Imprimir los nombres de las clases detectadas y sus IDs
# El dataset COCO tiene 80 clases, por ejemplo:
# 0: person, 1: bicycle, ..., 39: bottle, 40: cup, ..., 73: book, 76: scissors
names = model.names
print("\nClases detectadas:")
for r in results:
    if r.boxes:  # Asegurarse de que haya detecciones
        for box in r.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"- {names[class_id]} (Confianza: {conf:.2f})")

# Muestra la imagen con las segmentaciones superpuestas
# Esta es la forma más fácil de ver lo que YOLOv8-Seg ha detectado.
# La imagen resultante se abrirá en una ventana.
annotated_frame = results[0].plot()
cv2.imshow("YOLOv8 Segmentation Results", annotated_frame)
cv2.waitKey(0)  # Espera hasta que se presione una tecla
cv2.destroyAllWindows()

# --- Parte 3: Extraer Máscara y Calcular Color Dominante (Parte clave para tu proyecto) ---

print("\nExtrayendo información de las máscaras...")

# Leer la imagen original con OpenCV para trabajar con sus píxeles
img_cv = cv2.imread(image_path)
if img_cv is None:
    print(f"Error: No se pudo cargar la imagen en {image_path}")
    exit()

# Iterar sobre los resultados para obtener las máscaras
if results[0].masks:  # Asegurarse de que haya máscaras
    for i, mask_data in enumerate(results[0].masks):
        # La máscara es un tensor, necesitamos convertirla a un array de numpy
        # y redimensionarla a las dimensiones originales de la imagen
        mask = mask_data.data[0].cpu().numpy()
        original_shape = results[0].orig_shape  # (altura, ancho)
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convertir a booleano: True donde hay objeto, False donde no
        mask_boolean = mask_resized > 0.5

        # Aplicar la máscara a la imagen original para obtener solo los píxeles del objeto
        pixels_of_object = img_cv[mask_boolean]

        # Calcular el color dominante usando la media (simplificado)
        # Para un cálculo más robusto del color dominante (ej. con K-Means),
        # se necesitaría un paso adicional. Aquí solo promediamos.
        if len(pixels_of_object) > 0:
            dominant_color_bgr = np.mean(pixels_of_object, axis=0).astype(int)
            dominant_color_rgb = (dominant_color_bgr[2], dominant_color_bgr[1],
                                  dominant_color_bgr[0])  # Convertir BGR a RGB

            class_id = int(results[0].boxes.cls[i])
            object_name = names[class_id]

            print(f"Objeto {i + 1} ({object_name}):")
            print(f"  Color dominante (RGB): {dominant_color_rgb}")

            # (Opcional) Visualizar solo el objeto segmentado
            # Crea una imagen en blanco, y coloca los píxeles del objeto
            isolated_object_img = np.zeros_like(img_cv)
            isolated_object_img[mask_boolean] = img_cv[mask_boolean]
            cv2.imshow(f"Objeto {i + 1} - {object_name} Aislado", isolated_object_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Objeto {i + 1}: No hay píxeles para calcular el color dominante (máscara vacía?).")
else:
    print("No se detectaron máscaras en la imagen.")

print("\nEjemplo terminado.")