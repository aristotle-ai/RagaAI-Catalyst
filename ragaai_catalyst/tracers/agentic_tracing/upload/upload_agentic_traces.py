import json
import logging
import os
import re
import time
from urllib.parse import urlparse, urlunparse
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected
from .session_manager import session_manager

logger = logging.getLogger(__name__)

from ragaai_catalyst.ragaai_catalyst import RagaAICatalyst


class UploadAgenticTraces:
    def __init__(
        self,
        json_file_path,
        project_name,
        project_id,
        dataset_name,
        user_detail,
        base_url,
        timeout=120,
    ):
        self.json_file_path = json_file_path
        self.project_name = project_name
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.user_detail = user_detail
        self.base_url = base_url
        self.timeout = timeout

    def _get_presigned_url(self):
        payload = json.dumps(
            {
                "datasetName": self.dataset_name,
                "numFiles": 1,
            }
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }

        try:
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/presigned-url"
            # Changed to POST from GET
            response = session_manager.make_request_with_retry(
                "POST", endpoint, headers=headers, data=payload, timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )

            if response.status_code == 200:
                presignedURLs = response.json()["data"]["presignedUrls"][0]
                presignedurl = self.update_presigned_url(presignedURLs, self.base_url)
                return presignedurl
            elif response.status_code == 401:
                logger.warning("Received 401 error while getting presign url. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "X-Project-Name": self.project_name,
                }
                response = session_manager.make_request_with_retry(
                    "POST", endpoint, headers=headers, data=payload, timeout=self.timeout
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )
                if response.status_code == 200:
                    presignedURLs = response.json()["data"]["presignedUrls"][0]
                    presignedurl = self.update_presigned_url(presignedURLs, self.base_url)
                    return presignedurl
                else:
                    logger.error(
                        f"Error while getting presigned url after token refresh: {response.json()['message']}"
                    )
                    return None
            else:
                # If POST fails, try GET
                logger.warning(
                    f"POST request failed for getting presign url with status{response.status_code}.Falling back to GET request.")
                response = session_manager.make_request_with_retry(
                    "GET", endpoint, headers=headers, data=payload, timeout=self.timeout
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )
                if response.status_code == 200:
                    presignedURLs = response.json()["data"]["presignedUrls"][0]
                    presignedurl = self.update_presigned_url(
                        presignedURLs, self.base_url
                    )
                    return presignedurl
                elif response.status_code == 401:
                    logger.warning("Received 401 error while getting presign url. Attempting to refresh token.")
                    token = RagaAICatalyst.get_token(force_refresh=True)
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {token}",
                        "X-Project-Name": self.project_name,
                    }
                    response = session_manager.make_request_with_retry(
                        "POST",
                        endpoint,
                        headers=headers,
                        data=payload,
                        timeout=self.timeout,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                    )
                    if response.status_code == 200:
                        presignedURLs = response.json()["data"]["presignedUrls"][0]
                        presignedurl = self.update_presigned_url(
                            presignedURLs, self.base_url
                        )
                        return presignedurl
                    else:
                        logger.error(
                            f"Error while getting presigned url after token refresh: {response.json()['message']}"
                        )
                        return None
                else:
                    logger.error(
                        f"Error while getting presigned url: {response.json()['message']}"
                    )
                    return None
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "getting presigned URL")
            return None
        except RequestException as e:
            logger.error(f"Error while getting presigned url: {e}")
            return None

    def update_presigned_url(self, presigned_url, base_url):
        """Replaces the domain (and port, if applicable) of the presigned URL
        with that of the base URL only if the base URL contains 'localhost' or an IP address."""
        # To Do: If Proxy URL has domain name how do we handle such cases

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

    def _put_presigned_url(self, presignedUrl, filename):
        headers = {
            "Content-Type": "application/json",
        }

        if "blob.core.windows.net" in presignedUrl:  # Azure
            headers["x-ms-blob-type"] = "BlockBlob"
        logger.info("Uploading agentic traces to presigned URL...")
        try:
            with open(filename) as f:
                payload = f.read().replace("\n", "").replace("\r", "").encode()
        except Exception as e:
            logger.error(f"Error while reading file: {e}")
            return False
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "PUT", presignedUrl, headers=headers, data=payload, timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [PUT] {presignedUrl} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            if response.status_code != 200 or response.status_code != 201:
                return response, response.status_code
            return True
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "uploading trace to presigned URL")
            return False
        except RequestException as e:
            logger.error(f"Error while uploading trace to presigned url: {e}")
            return False

    def insert_traces(self, presignedUrl):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "Content-Type": "application/json",
            "X-Project-Name": self.project_name,
        }
        payload = json.dumps(
            {
                "datasetName": self.dataset_name,
                "presignedUrl": presignedUrl,
                "datasetSpans": self._get_dataset_spans(),  # Extra key for agentic traces
            }
        )
        try:
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/insert/trace"
            response = session_manager.make_request_with_retry(
                "POST", endpoint, headers=headers, data=payload, timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            if response.status_code in [200, 201]:
                logger.info(f"Traces inserted successfully: {response.json()['message']}")
                return True
            elif response.status_code == 401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "X-Project-Name": self.project_name,
                }
                response = session_manager.make_request_with_retry(
                    "POST",
                    endpoint,
                    headers=headers,
                    data=payload,
                    timeout=self.timeout,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )
                if response.status_code in [200, 201]:
                    logger.info(f"Traces inserted successfully: {response.json()['message']}")
                    return True
                else:
                    logger.error(f"Error while inserting traces after 401: {response.json()['message']}")
                    return False
            else:
                logger.error(f"Error while inserting traces: {response.json()['message']}")
                return False
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "inserting traces")
            return False
        except RequestException as e:
            logger.error(f"Error while inserting traces: {e}")
            return False

    def _get_dataset_spans(self):
        try:
            with open(self.json_file_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None
        try:
            spans = data["data"][0]["spans"]
            dataset_spans = []
            for span in spans:
                try:
                    dataset_spans.append(
                        {
                            "spanId": span.get("context", {}).get("span_id", ""),
                            "spanName": span.get("name", ""),
                            "spanHash": span.get("hash_id", ""),
                            "spanType": span.get("attributes", {}).get(
                                "openinference.span.kind", ""
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing span: {e}")
                    continue
            return dataset_spans
        except Exception as e:
            logger.error(f"Error while reading dataset spans: {e}")
            return None

    def upload_agentic_traces(self):
        try:
            presigned_url = self._get_presigned_url()
            if presigned_url is None:
                logger.warning("Warning: Failed to obtain presigned URL")
                return False

            # Upload the file using the presigned URL
            upload_result = self._put_presigned_url(presigned_url, self.json_file_path)
            if not upload_result:
                logger.error("Error: Failed to upload file to presigned URL")
                return False
            elif isinstance(upload_result, tuple):
                response, status_code = upload_result
                if status_code not in [200, 201]:
                    logger.error(
                        f"Error: Uploading agentic traces failed with status code {status_code}: {response.text if hasattr(response, 'text') else 'Unknown error'}")
                    return False
            # Insert trace records
            insert_success = self.insert_traces(presigned_url)
            if not insert_success:
                print("Error: Failed to insert trace records")
                return False

            logger.info("Successfully uploaded agentic traces")
            return True
        except FileNotFoundError:
            logger.error(f"Error: Trace file not found at {self.json_file_path}")
            return False
        except ConnectionError as e:
            logger.error(f"Error: Network connection failed while uploading traces: {e}")
            return False
        except Exception as e:
            logger.error(f"Error while uploading agentic traces: {e}")
            return False
