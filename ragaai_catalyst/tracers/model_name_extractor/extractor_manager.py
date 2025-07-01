from typing import List
from ragaai_catalyst.tracers.model_name_extractor.extractors.base import ModelNameExtractor

class ModelNameExtractorManager:
    def __init__(self, extractors: List[ModelNameExtractor]):
        self.extractors = extractors

    def extract_model_name(self, span: dict) -> str:
        for extractor in self.extractors:
            model_name = extractor.extract(span)
            if model_name:
                return model_name
        return ""