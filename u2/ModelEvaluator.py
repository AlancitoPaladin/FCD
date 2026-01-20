import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class ModelEvaluator:
    """Clase para evaluar modelos"""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "Dataset"):
        """Evalúa las predicciones y muestra métricas"""
        print(f"\n{'=' * 60}")
        print(f"Evaluación: {dataset_name}")
        print(f"{'=' * 60}")

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {acc:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nMatriz de Confusión:\n{cm}")

        # Reporte de clasificación
        print(f"\nReporte de Clasificación:")
        print(classification_report(y_true, y_pred))

        return {
            'accuracy': acc,
            'confusion_matrix': cm
        }
