import pandas as pd


class DataLoader:
    """Clase para cargar y realizar operaciones iniciales sobre los datos"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self) -> pd.DataFrame:
        """Carga el archivo CSV"""
        self.data = pd.read_csv(self.filepath)
        print(f"Datos cargados: {self.data.shape}")
        return self.data

    def get_summary(self) -> pd.DataFrame:
        """Retorna resumen estadístico de los datos"""
        return self.data.describe()
