from pathlib import Path

import pandas as pd


class DataPreprocessor:
    """Clase para preprocesar los datos"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.removed_columns = []

    def remove_zero_columns(self, start_col: int = 1, end_col: int = 55) -> pd.DataFrame:
        """Elimina columnas que contienen solo ceros"""
        cols_to_check = self.data.columns[start_col:end_col]
        zero_cols = []

        for col in cols_to_check:
            if (self.data[col] == 0).all():
                zero_cols.append(col)

        self.removed_columns = zero_cols
        self.data = self.data.drop(columns=zero_cols)
        print(f"Columnas eliminadas: {len(zero_cols)}")
        print(f"Nueva dimensión: {self.data.shape}")
        return self.data

    def rename_columns(self, n_features: int = 52) -> pd.DataFrame:
        """Renombra las columnas con nombres estandarizados"""
        old_names = list(self.data.columns)
        new_names = ["Category"] + [f"f{i}" for i in range(1, n_features + 1)] + ["Class"]

        self.name_mapping = pd.DataFrame({'old': old_names, 'new': new_names})
        self.data.columns = new_names
        print("Columnas renombradas")
        return self.data

    def get_processed_data(self) -> pd.DataFrame:
        """Retorna los datos procesados"""
        return self.data

    def save_name_mapping(self, output_path: Path):
        """Guarda el mapeo de nombres"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.name_mapping.to_csv(output_path, sep='\t', index=False)
        print(f"Mapeo guardado en: {output_path}")
