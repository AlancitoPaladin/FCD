"""
Pipeline orientado a objetos para procesamiento y clasificación de MalMem
"""

import warnings

import numpy as np
import pandas as pd

from MalMemPipeline import MalMemPipeline

warnings.filterwarnings('ignore')

# Punto de entrada
if __name__ == "__main__":
    # Crear pipeline
    pipeline = MalMemPipeline(
        input_file="../fuscated/Obfuscated-MalMem2022.csv",
        output_dir="../results"
    )

    # Ejecutar preprocesamiento
    pipeline.run_preprocessing()

    # Ejecutar clasificación
    results = pipeline.run_classification()

    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name}: Accuracy = {metrics['accuracy']:.4f}")
    print("=" * 60 + "\n")
