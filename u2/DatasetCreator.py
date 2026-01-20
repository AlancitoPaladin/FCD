import pandas as pd


class DatasetCreator:
    """Clase para crear diferentes versiones del dataset (2, 4 y 16 clases)"""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def create_twoclass(self) -> pd.DataFrame:
        """Crea dataset de 2 clases (sin Category)"""
        twoclass = self.data.iloc[:, 1:54].copy()
        print(f"Dataset de 2 clases creado: {twoclass.shape}")
        return twoclass

    def create_fourclass(self) -> pd.DataFrame:
        """Crea dataset de 4 clases"""
        # Separar por clase
        buenos = self.data[self.data['Class'] == 'Benign']
        malos = self.data[self.data['Class'] == 'Malware']

        # Procesar categorías de malware
        malos_col1 = malos['Category'].astype(str)
        split_malos = malos_col1.str.split('-', expand=True)
        first_elements = split_malos[0].values

        # Crear etiquetas
        eti_buenos = ['Benign'] * len(buenos)
        etifour = eti_buenos + list(first_elements)

        # Crear dataset
        fourclass = self.data.iloc[:, 1:53].copy()
        fourclass['Class'] = etifour

        print(f"Dataset de 4 clases creado: {fourclass.shape}")
        print(f"Clases: {fourclass['Class'].unique()}")
        return fourclass

    def create_sixteenclass(self) -> pd.DataFrame:
        """Crea dataset de 16 clases"""
        # Separar por clase
        buenos = self.data[self.data['Class'] == 'Benign']
        malos = self.data[self.data['Class'] == 'Malware']

        # Procesar categorías de malware
        malos_col1 = malos['Category'].astype(str)
        split_malos = malos_col1.str.split('-', expand=True)
        first_elements = split_malos[0].values
        second_elements = split_malos[1].values

        # Crear etiquetas
        eti_buenos = ['Benign'] * len(buenos)
        etisixteen = eti_buenos + [f"{f}-{s}" for f, s in zip(first_elements, second_elements)]

        # Crear dataset
        sixteenclass = self.data.iloc[:, 1:53].copy()
        sixteenclass['Class'] = etisixteen

        print(f"Dataset de 16 clases creado: {sixteenclass.shape}")
        print(f"Número de clases: {sixteenclass['Class'].nunique()}")
        return sixteenclass
