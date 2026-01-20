import kagglehub
import shutil
import os

# Encontrar donde KaggleHub guardó los archivos
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
print(f"Archivos están en: {path}")

# Ver qué archivos hay
import glob
files = glob.glob(os.path.join(path, "*.csv"))
print("Archivos disponibles:")
for f in files:
    print(f"  - {f}")

# Crear carpeta en tu proyecto
os.makedirs("../mnist", exist_ok=True)

# Mover archivos
for file in files:
    filename = os.path.basename(file)
    destination = os.path.join("../mnist", filename)
    shutil.copy2(file, destination)
    print(f"Movido: {filename} -> {destination}")