from pathlib import Path

import pandas as pd


class DatasetSaver:
    """Clase para guardar datasets"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: pd.DataFrame, filename: str):
        """Guarda un dataset en CSV"""
        filepath = self.output_dir / filename
        data.to_csv(filepath, index=False)
        print(f"Dataset guardado: {filepath}")
