import logging
import os
import time
import re
import json
from urllib.parse import urlparse, urlunparse
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

from ragaai_catalyst.ragaai_catalyst import RagaAICatalyst
from .session_manager import session_manager

logger = logging.getLogger(__name__)


def upload_code(
    hash_id, zip_path, project_name, dataset_name, base_url=None, timeout=120
):
    code_hashes_list = _fetch_dataset_code_hashes(
        project_name, dataset_name, base_url, timeout=timeout
    )

    # Handle None case during exceptions - do not proceed
    if code_hashes_list is None:
        logger.error("Failed to fetch existing code hashes, cannot proceed with upload")
        return None

    if hash_id not in code_hashes_list:
        presigned_url = _fetch_presigned_url(
            project_name, dataset_name, base_url, timeout=timeout
        )
        # Handle None case for presigned URL
        if presigned_url is None:
            logger.error("Failed to fetch presigned URL, cannot proceed with upload")
            return None

        upload_result = _put_zip_presigned_url(project_name, presigned_url, zip_path, timeout=timeout)

        # Handle upload failure
        if upload_result is False or (isinstance(upload_result, tuple) and upload_result[1] not in [200, 201]):
            logger.error("Failed to upload zip file")
            return None

        response = _insert_code(
            dataset_name,
            hash_id,
            presigned_url,
            project_name,
            base_url,
            timeout=timeout,
        )
        # Handle None response from insert_code
        if response is None:
            logger.error("Failed to insert code metadata")
            return None
        return response
    else:
        return "Code already exists"


def _fetch_dataset_code_hashes(project_name, dataset_name, base_url=None, timeout=120):
    payload = {}
    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Name": project_name,
    }

    try:
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v2/llm/dataset/code?datasetName={dataset_name}"
        response = session_manager.make_request_with_retry(
            "GET", endpoint, headers=headers, data=payload, timeout=timeout
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
        )

        if response.status_code in [200, 201]:
            return response.json()["data"]["codeHashes"]
        elif response.status_code == 401:
            logger.warning("Received 401 error. Attempting to refresh token.")
            RagaAICatalyst.get_token(force_refresh=True)
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": project_name,
            }
            response = session_manager.make_request_with_retry(
                "GET", endpoint, headers=headers, data=payload, timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            if response.status_code in [200, 201]:
                return response.json()["data"]["codeHashes"]
            else:
                logger.error(f"Failed to fetch code hashes: {response.json()['message']}")
                return None
        else:
            logger.error(f"Error while fetching dataset code hashes: {response.json()['message']}")
            return None
    except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
        session_manager.handle_request_exceptions(e, "fetching dataset code hashes")
        return None
    except RequestException as e:
        logger.error(f"Failed to fetch dataset code hashes: {e}")
        return None


def update_presigned_url(presigned_url, base_url):
    """Replaces the domain (and port, if applicable) of the presigned URL with that of the base URL."""
    # To Do: If Proxy URL has domain name how do we handle such cases? Engineering Dependency.

    presigned_parts = urlparse(presigned_url)
    base_parts = urlparse(base_url)
    # Check if base_url contains localhost or an IP address
    if re.match(r"^(localhost|\d{1,3}(\.\d{1,3}){3})$", base_parts.hostname):
        new_netloc = base_parts.hostname  # Extract domain from base_url
        if base_parts.port:  # Add port if present in base_url
            new_netloc += f":{base_parts.port}"
        updated_parts = presigned_parts._replace(netloc=new_netloc)
        return urlunparse(updated_parts)
    return presigned_url


def _fetch_presigned_url(project_name, dataset_name, base_url=None, timeout=120):
    payload = json.dumps(
        {"datasetName": dataset_name, "numFiles": 1, "contentType": "application/zip"}
    )

    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "Content-Type": "application/json",
        "X-Project-Name": project_name,
    }

    try:
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        # Changed to POST from GET
        endpoint = f"{url_base}/v1/llm/presigned-url"
        response = session_manager.make_request_with_retry(
            "POST", endpoint, headers=headers, data=payload, timeout=timeout
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
        )

        if response.status_code in [200, 201]:
            presigned_url = response.json()["data"]["presignedUrls"][0]
            presigned_url = update_presigned_url(presigned_url, url_base)
            return presigned_url
        else:
            # If POST fails, try GET
            response = session_manager.make_request_with_retry(
                "POST", endpoint, headers=headers, data=payload, timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            if response.status_code in [200, 201]:
                presigned_url = response.json()["data"]["presignedUrls"][0]
                presigned_url = update_presigned_url(presigned_url, url_base)
                return presigned_url
            elif response.status_code == 401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                RagaAICatalyst.get_token(force_refresh=True)
                headers = {
                    "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                    "Content-Type": "application/json",
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
                    presigned_url = response.json()["data"]["presignedUrls"][0]
                    presigned_url = update_presigned_url(presigned_url, url_base)
                    return presigned_url
                else:
                    logger.error(
                        f"Failed to fetch presigned URL for code upload after 401: {response.json()['message']}"
                    )
            else:
                logger.error(
                    f"Failed to fetch presigned URL for code upload: {response.json()['message']}"
                )
                return None
    except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
        session_manager.handle_request_exceptions(e, "fetching presigned URL for code upload")
        return None
    except RequestException as e:
        logger.error(f"Failed to fetch presigned URL for code upload: {e}")
        return None


def _put_zip_presigned_url(project_name, presignedUrl, filename, timeout=120):
    headers = {
        "X-Project-Name": project_name,
        "Content-Type": "application/zip",
    }

    if "blob.core.windows.net" in presignedUrl:  # Azure
        headers["x-ms-blob-type"] = "BlockBlob"
    logger.info("Uploading code to presigned URL...")
    try:
        with open(filename, "rb") as f:
            payload = f.read()

        start_time = time.time()
        response = session_manager.make_request_with_retry(
            "PUT", presignedUrl, headers=headers, data=payload, timeout=timeout
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [PUT] {presignedUrl} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
        )
        if response.status_code not in [200, 201]:
            return response, response.status_code
        return True
    except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
        session_manager.handle_request_exceptions(e, "uploading zip to presigned URL")
        return False
    except RequestException as e:
        logger.error(f"Failed to upload zip: {e}")
        return False

def _insert_code(
    dataset_name, hash_id, presigned_url, project_name, base_url=None, timeout=120
):
    payload = json.dumps(
        {
            "datasetName": dataset_name,
            "codeHash": hash_id,
            "presignedUrl": presigned_url,
        }
    )

    headers = {
        "X-Project-Name": project_name,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
    }

    try:
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v2/llm/dataset/code"
        response = session_manager.make_request_with_retry(
            "POST", endpoint, headers=headers, data=payload, timeout=timeout
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
        )
        if response.status_code in [200, 201]:
            return response.json()["message"]

        elif response.status_code == 401:
            logger.warning("Received 401 error during inserting code. Attempting to refresh token.")
            token = RagaAICatalyst.get_token(force_refresh=True)
            headers = {
                "X-Project-Name": project_name,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
            response = session_manager.make_request_with_retry(
                "POST", endpoint, headers=headers, data=payload, timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            if response.status_code in [200, 201]:
                logger.info(f"Code inserted successfully after 401: {response.json()['message']}")
                return response.json()["message"]
            else:
                logger.error(f"Failed to insert code after 401: {response.json()['message']}")
                return None
        else:
            logger.error(f"Failed to insert code: {response.json()['message']}")
            return None
    except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
        session_manager.handle_request_exceptions(e, "inserting code")
        return None
    except RequestException as e:
        logger.error(f"Failed to insert code: {e}")
        return None
