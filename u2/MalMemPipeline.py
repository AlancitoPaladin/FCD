from pathlib import Path

from sklearn.model_selection import train_test_split

from DataLoader import DataLoader
from DatasetCreator import DatasetCreator
from DatasetSaver import DatasetSaver
from ModelEvaluator import ModelEvaluator
from DataPreprocessor import DataPreprocessor
from NaiveBayes import NaiveBayes


class MalMemPipeline:
    """Pipeline completo para procesar y clasificar MalMem"""

    def __init__(self, input_file: str, output_dir: str = "./fuscated/Obfuscated-MalMem2022.csv"):
        self.input_file = input_file
        self.output_dir = Path(output_dir).expanduser()
        self.data = None
        self.datasets = {}

    def run_preprocessing(self):
        """Ejecuta el preprocesamiento completo"""
        print("\n" + "=" * 60)
        print("INICIANDO PREPROCESAMIENTO")
        print("=" * 60 + "\n")

        # Cargar datos
        loader = DataLoader(self.input_file)
        self.data = loader.load()

        # Preprocesar
        preprocessor = DataPreprocessor(self.data)
        self.data = preprocessor.remove_zero_columns()
        self.data = preprocessor.rename_columns()

        # Guardar mapeo de nombres
        preprocessor.save_name_mapping(self.output_dir / "NAMES.txt")

        # Crear diferentes versiones del dataset
        creator = DatasetCreator(self.data)
        self.datasets['twoclass'] = creator.create_twoclass()
        self.datasets['fourclass'] = creator.create_fourclass()
        self.datasets['sixteenclass'] = creator.create_sixteenclass()

        # Guardar datasets
        saver = DatasetSaver(self.output_dir)
        for name, dataset in self.datasets.items():
            saver.save(dataset, f"{name}.csv")

        print("\n✓ Preprocesamiento completado\n")

    def run_classification(self, test_size: float = 0.3, random_state: int = 1237):
        """Ejecuta la clasificación con Naive Bayes"""
        print("\n" + "=" * 60)
        print("INICIANDO CLASIFICACIÓN")
        print("=" * 60 + "\n")

        results = {}

        for name, dataset in self.datasets.items():
            print(f"\n--- Procesando {name} ---")

            # Separar X e y
            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            print(f"Train: {X_train.shape}, Test: {X_test.shape}")

            # Entrenar
            model = NaiveBayes()
            model.fit(X_train, y_train)

            # Predecir
            y_pred = model.predict(X_test)

            # Evaluar
            metrics = ModelEvaluator.evaluate(y_test, y_pred, name)
            results[name] = metrics

        print("\n✓ Clasificación completada\n")
        return results
