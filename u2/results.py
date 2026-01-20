import csv
import os

from FrequencyTable import DatasetAnalyzer

# Construir ruta absoluta al CSV
BASE_DIR = os.path.dirname(__file__)  # carpeta donde está este script
path = os.path.join(BASE_DIR, "..", "mnist_hu", "mnist_hu_test.csv")

if not os.path.isfile(path):
    raise FileNotFoundError(f"No se encontró el CSV en: {path}")

data: dict = {}
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    data = {h: [] for h in headers}
    for row in reader:
        for h in headers:
            v = row[h]
            try:
                data[h].append(float(v))
            except (ValueError, TypeError):
                data[h].append(v)

analyzer = DatasetAnalyzer()
analyzer.set_data(data)
analyzer.analyze_all()

print("Generando reporte de resumen...")
analyzer.summary_report()

print("\nGenerando gráficos individuales...")

# ALTERNATIVA: Usar el nuevo método para graficar múltiples columnas en una sola figura
print("\nGenerando gráfico combinado...")
analyzer.plot_multiple(["H1", "H2", "H3", "H4", "H5", "H6", "H7"],
                       kind="hist",
                       figsize=(16, 12))
