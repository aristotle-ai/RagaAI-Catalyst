import json
import logging

from ragaai_catalyst.tracers.model_name_extractor.decorators import require_llm_span
from ragaai_catalyst.tracers.model_name_extractor.extractors.base import ModelNameExtractor

logger = logging.getLogger("RagaAICatalyst")


class DefaultModelNameExtractor(ModelNameExtractor):
    @require_llm_span
    def extract(self, span: dict) -> str:
        """
        Extracts 'ls_model_name' from JSON string stored in span['attributes']['metadata'].

        Args:
            span (dict): The span dictionary.

        Returns:
            str: The 'ls_model_name' value if present; otherwise, empty string.
        """
        try:
            if not isinstance(span, dict):
                logger.warning("Input span is not a dictionary, got type: %s", type(span).__name__)
                return ""

            attributes = span.get("attributes")
            if not isinstance(attributes, dict):
                logger.warning("Missing or invalid 'attributes' in span, got type: %s", type(attributes).__name__)
                return ""

            raw_metadata = attributes.get("metadata")
            if not isinstance(raw_metadata, str) or not raw_metadata.strip():
                logger.info("'metadata' is missing or not a non-empty string.")
                return ""

            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as e:
                logger.warning("Failed to decode 'metadata' JSON: %s", e, exc_info=True)
                return ""

            if not isinstance(metadata, dict):
                logger.warning("Parsed 'metadata' is not a dictionary.")
                return ""

            ls_model_name = metadata.get("ls_model_name")
            if not isinstance(ls_model_name, str) or not ls_model_name.strip():
                logger.info("'ls_model_name' key missing or not a string in metadata.")
                return ""

            return ls_model_name.strip()

        except Exception as e:
            logger.error("Unexpected error extracting ls_model_name: %s", e, exc_info=True)
            return ""

