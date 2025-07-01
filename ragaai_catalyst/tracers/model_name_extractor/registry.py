from ragaai_catalyst.tracers.model_name_extractor.extractors.aiq_framework_model_name_extractor import \
    AiqFrameworkModelNameExtractor
from ragaai_catalyst.tracers.model_name_extractor.extractors.default_model_name_extractor import DefaultModelNameExtractor
from ragaai_catalyst.tracers.model_name_extractor.extractor_manager import ModelNameExtractorManager

extractor_manager = ModelNameExtractorManager([
    DefaultModelNameExtractor(),
    AiqFrameworkModelNameExtractor(),
    # Add more extractors here
])