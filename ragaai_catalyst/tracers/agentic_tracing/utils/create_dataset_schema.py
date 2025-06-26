import os
import json
import requests
import logging
import time
from typing import Optional
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing.upload.session_manager import session_manager

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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Name": project_name,
    }

    if schema_mapping:
        payload = json.dumps({
            "datasetName": dataset_name,
            "traceFolderUrl": None,
            "schemaMapping": schema_mapping
        })
    else:
        payload = json.dumps({
            "datasetName": dataset_name,
            "traceFolderUrl": None,
        })

    try:
        # Use provided base_url or fall back to default
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v1/llm/dataset/logs"

        response = session_manager.make_request_with_retry(
            "POST", endpoint, headers=headers, data=payload, timeout=timeout
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
        )

        if response.status_code in [200, 201]:
            logger.info(f"Dataset schema created successfully: {response.status_code}")
            return response
        elif response.status_code == 401:
            logger.warning("Received 401 error during dataset schema creation. Attempting to refresh token.")
            RagaAICatalyst.get_token(force_refresh=True)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": project_name,
            }
            response = session_manager.make_request_with_retry(
                "POST", endpoint, headers=headers, data=payload, timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            if response.status_code in [200, 201]:
                logger.info(f"Dataset schema created successfully after 401: {response.status_code}")
                return response
            else:
                logger.error(f"Failed to create dataset schema after 401: {response.status_code}")
                return None
        else:
            logger.error(f"Failed to create dataset schema: {response.status_code}")
            return None
    except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
        session_manager.handle_request_exceptions(e, "creating dataset schema")
        return None
    except RequestException as e:
        logger.error(f"Failed to create dataset schema: {e}")
        return None