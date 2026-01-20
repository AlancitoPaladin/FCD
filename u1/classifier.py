from abc import ABC, abstractmethod

class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass