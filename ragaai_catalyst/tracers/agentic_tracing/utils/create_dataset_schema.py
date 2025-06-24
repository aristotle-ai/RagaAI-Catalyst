import os
import json
import requests
import logging
from typing import Optional
from ragaai_catalyst import RagaAICatalyst

IGNORED_KEYS = {"log_source", "recorded_on"}
logger = logging.getLogger(__name__)

def create_dataset_schema_with_trace(
        project_name: str,
        dataset_name: str,
        base_url: Optional[str] = None,
        user_details: Optional[dict] = None,
        timeout: int = 120) -> requests.Response:
    schema_mapping = {}

    metadata = (
        user_details.get("trace_user_detail", {}).get("metadata", {})
        if user_details else {}
    )
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key in IGNORED_KEYS:
                continue
            schema_mapping[key] = {"columnType": "metadata"}

    payload = {
        "datasetName": dataset_name,
        "traceFolderUrl": None,
    }
    if schema_mapping:
        payload["schemaMapping"] = schema_mapping

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Name": project_name,
    }

    # Use provided base_url or fall back to default
    if base_url is None:
        logger.warning("base_url is not provided, using default: %s", RagaAICatalyst.BASE_URL)
        base_url = RagaAICatalyst.BASE_URL

    response = requests.post(
        f"{base_url}/v1/llm/dataset/logs",
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout
    )
    if not response.ok:
        print(f"Request failed: {response.status_code} - {response.text}")

    return response