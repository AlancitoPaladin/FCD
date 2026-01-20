from pathlib import Path

import numpy as np
import pandas as pd


class GaussianNoiseGenerator:
    """
    Generador de datos sintéticos mediante la adición de ruido gaussiano.
    Útil para aumentar datasets desbalanceados.
    """

    def __init__(self, dataset_path, random_state=42):
        """
        Inicializa el generador.
        
        Args:
            dataset_path: Ruta al archivo CSV
            random_state: Semilla para reproducibilidad
        """
        self.dataset_path = dataset_path
        self.random_state = random_state
        np.random.seed(random_state)
        self.df = None
        self.numeric_columns = None

    def load_dataset(self):
        """Carga el dataset CSV."""
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset cargado: {self.df.shape}")
        print(f"Columnas: {self.df.columns.tolist()}")
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Columnas numéricas: {self.numeric_columns}")
        return self.df

    def analyze_balance(self):
        """Analiza el desbalance del dataset si hay columna de clase."""
        # Intenta encontrar la columna de etiqueta (clase)
        possible_label_cols = ['label', 'class', 'target', 'Category', 'clase', 'etiqueta']
        label_col = None

        for col in possible_label_cols:
            if col in self.df.columns:
                label_col = col
                break

        if label_col is None and len(self.df.columns) > 0:
            # Usa la última columna como etiqueta por defecto
            label_col = self.df.columns[-1]

        if label_col:
            print(f"\nDesbalance en columna '{label_col}':")
            print(self.df[label_col].value_counts())
            return label_col

        return None

    def add_gaussian_noise(self, data, std_dev=0.01, num_samples=1):
        """
        Añade ruido gaussiano a los datos.
        
        Args:
            data: Array o fila de datos
            std_dev: Desviación estándar del ruido (porcentaje de la desv. est. original)
            num_samples: Número de muestras sintéticas a generar
            
        Returns:
            Array con datos sintéticos con ruido
        """
        synthetic_data = []

        for _ in range(num_samples):
            # Calcula ruido gaussiano con media 0
            noise = np.random.normal(loc=0, scale=std_dev, size=len(data))
            # Añade ruido a los datos originales
            noisy_data = data + noise
            synthetic_data.append(noisy_data)

        return np.array(synthetic_data)

    def generate_synthetic_data(self, std_dev=0.05, balance=True, label_column=None):
        """
        Genera datos sintéticos para balancear el dataset.
        
        Args:
            std_dev: Desviación estándar del ruido gaussiano
            balance: Si True, equilibra las clases minoritarias
            label_column: Nombre de la columna de etiqueta
            
        Returns:
            DataFrame con datos originales + sintéticos
        """
        if self.df is None:
            self.load_dataset()

        synthetic_df = self.df.copy()

        if balance and label_column is None:
            label_column = self.analyze_balance()

        if balance and label_column and label_column in self.df.columns:
            # Obtiene la distribución de clases
            class_counts = self.df[label_column].value_counts()
            max_count = class_counts.max()

            print(f"\nGenerando datos sintéticos para balancear...")
            print(f"Clase mayoritaria: {max_count} muestras")

            for class_label, count in class_counts.items():
                if count < max_count:
                    # Número de muestras sintéticas a generar
                    num_synthetic = max_count - count

                    # Obtiene las muestras de la clase minoritaria
                    class_data = self.df[self.df[label_column] == class_label]

                    print(f"Clase '{class_label}': {count} → {max_count} muestras " +
                          f"(+{num_synthetic} sintéticas)")

                    # Genera datos sintéticos
                    synthetic_samples = []
                    for idx, (_, row) in enumerate(class_data.iterrows()):
                        numeric_values = row[self.numeric_columns].values.astype(float)

                        # Ajusta std_dev según el rango de los datos
                        if len(self.numeric_columns) > 0:
                            data_std = numeric_values.std()
                            adjusted_std = std_dev * data_std if data_std > 0 else std_dev
                        else:
                            adjusted_std = std_dev

                        # Genera muestras por interpolación
                        samples_per_row = max(1, num_synthetic // len(class_data))

                        for _ in range(samples_per_row):
                            noisy = self.add_gaussian_noise(
                                numeric_values,
                                std_dev=adjusted_std,
                                num_samples=1
                            )[0]

                            # Crea fila sintética
                            new_row = row.copy()
                            for i, col in enumerate(self.numeric_columns):
                                new_row[col] = noisy[i]
                            synthetic_samples.append(new_row)

                    # Añade muestras sintéticas al dataset
                    synthetic_df = pd.concat(
                        [synthetic_df, pd.DataFrame(synthetic_samples)],
                        ignore_index=True
                    )
        else:
            print("\nGenerando datos sintéticos para todos los datos...")
            # Genera sintéticos para todo el dataset
            synthetic_samples = []

            for idx, (_, row) in enumerate(self.df.iterrows()):
                numeric_values = row[self.numeric_columns].values.astype(float)

                # Añade 3 muestras sintéticas por cada muestra original
                noisy_samples = self.add_gaussian_noise(
                    numeric_values,
                    std_dev=std_dev,
                    num_samples=3
                )

                for noisy in noisy_samples:
                    new_row = row.copy()
                    for i, col in enumerate(self.numeric_columns):
                        new_row[col] = noisy[i]
                    synthetic_samples.append(new_row)

            synthetic_df = pd.concat(
                [synthetic_df, pd.DataFrame(synthetic_samples)],
                ignore_index=True
            )

        print(f"\nDataset original: {self.df.shape[0]} muestras")
        print(f"Dataset aumentado: {synthetic_df.shape[0]} muestras")

        return synthetic_df

    def save_dataset(self, output_df, output_path=None):
        """Guarda el dataset aumentado."""
        if output_path is None:
            base_path = Path(self.dataset_path).stem
            output_path = f"{base_path}_augmented.csv"

        output_df.to_csv(output_path, index=False)
        print(f"\nDataset guardado en: {output_path}")
        return output_path


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al dataset desbalanceado
    dataset_path = "../fuscated/Obfuscated-MalMem2022.csv"

    # Crea el generador
    generator = GaussianNoiseGenerator(dataset_path)

    # Carga y analiza el dataset
    generator.load_dataset()
    label_col = generator.analyze_balance()

    # Genera datos sintéticos con ruido gaussiano
    augmented_df = generator.generate_synthetic_data(
        std_dev=0.05,  # Desviación estándar del ruido (5%)
        balance=True,
        label_column=label_col
    )

    # Guarda el nuevo dataset
    generator.save_dataset(augmented_df)
