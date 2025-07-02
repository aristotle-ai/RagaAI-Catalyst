import json
import logging

from ragaai_catalyst.tracers.model_name_extractor.decorators import require_llm_span
from ragaai_catalyst.tracers.model_name_extractor.extractors.base import ModelNameExtractor

logger = logging.getLogger("RagaAICatalyst")


class AiqFrameworkModelNameExtractor(ModelNameExtractor):
    @require_llm_span
    def extract(self, span: dict) -> str:
        """
        Extracts the model_name from aiq.metadata JSON string inside the span's attributes.
        Gracefully handles nulls, malformed JSON, and structural inconsistencies.

        Args:
            span (Dict): A span dictionary containing "attributes" with "aiq.metadata".

        Returns:
            str: Extracted model name if found, else empty string.
        """
        try:
            if not isinstance(span, dict):
                logger.warning("Span is not a dictionary: type=%s", type(span).__name__)
                return ""

            attributes = span.get("attributes")
            if not isinstance(attributes, dict):
                logger.warning("Missing or invalid 'attributes' in span: type=%s", type(attributes).__name__)
                return ""

            raw_metadata = attributes.get("aiq.metadata")
            if not isinstance(raw_metadata, str) or not raw_metadata.strip():
                logger.info("'aiq.metadata' is missing or not a non-empty string.")
                return ""

            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as e:
                logger.warning("Failed to decode 'aiq.metadata': %s", e, exc_info=True)
                return ""

            if not isinstance(metadata, dict):
                logger.warning("Parsed 'aiq.metadata' is not a dictionary.")
                return ""

            chat_responses = metadata.get("chat_responses", [])
            if not isinstance(chat_responses, list):
                logger.warning("'chat_responses' is not a list.")
                return ""

            for idx, response in enumerate(chat_responses):
                message = response.get("message")
                if not isinstance(message, dict):
                    logger.debug("Skipping chat_responses[%d]: missing or invalid 'message'.", idx)
                    continue

                response_meta = message.get("response_metadata")
                if not isinstance(response_meta, dict):
                    logger.debug("Skipping chat_responses[%d]: missing or invalid 'response_metadata'.", idx)
                    continue

                model_name = response_meta.get("model_name")
                if isinstance(model_name, str) and model_name.strip():
                    logger.debug("Model name found in chat_responses[%d]: %s", idx, model_name)
                    return model_name

            logger.info("No model_name found in any chat_responses.")
            return ""

        except Exception as e:
            logger.error("Unexpected error extracting model name: %s", e, exc_info=True)
            return ""
