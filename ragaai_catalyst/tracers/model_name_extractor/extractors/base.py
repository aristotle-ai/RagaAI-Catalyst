from abc import ABC, abstractmethod

class ModelNameExtractor(ABC):
    @abstractmethod
    def extract(self, span: dict) -> str:
        pass