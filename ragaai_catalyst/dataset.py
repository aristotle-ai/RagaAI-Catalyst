import os
import csv
import json
import tempfile
import time
from ragaai_catalyst.session_manager import session_manager
from .utils import response_checker
from typing import Union
import logging
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected
from .ragaai_catalyst import RagaAICatalyst
import pandas as pd

logger = logging.getLogger(__name__)
get_token = RagaAICatalyst.get_token

# Job status constants
JOB_STATUS_FAILED = "failed"
JOB_STATUS_IN_PROGRESS = "in_progress"
JOB_STATUS_COMPLETED = "success"

class Dataset:
    BASE_URL = None
    TIMEOUT = 30

    def __init__(self, project_name):
        self.project_name = project_name
        self.num_projects = 99999
        Dataset.BASE_URL = RagaAICatalyst.BASE_URL
        self.jobId = None
        self.project_id = None
        
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "GET",
                f"{Dataset.BASE_URL}/v2/llm/projects?size={self.num_projects}",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] /v2/llm/projects | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                logger.debug("Projects list retrieved successfully")
                project_list = [
                    project["name"] for project in response.json()["data"]["content"]
                ]

                if project_name not in project_list:
                    logger.error("Project not found. Please enter a valid project name")
                    return

                # Safe assignment now that we know project exists
                matching_projects = [
                    project["id"] for project in response.json()["data"]["content"] 
                    if project["name"] == project_name
                ]
                if matching_projects:
                    self.project_id = matching_projects[0]
                else:
                    logger.error(f"Project '{project_name}' not found in project list")
                    return
                
            elif response.status_code == 401:
                logger.warning("Received 401 error during fetching project list. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers = {
                    "Authorization": f"Bearer {token}",
                }
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "GET", f"{Dataset.BASE_URL}/v2/llm/projects?size={self.num_projects}", 
                    headers=headers, timeout=self.TIMEOUT
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v2/llm/projects (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Project list fetched successfully after 401 token refresh")
                    project_list = [
                        project["name"] for project in response.json()["data"]["content"]
                    ]
                    if project_name not in project_list:
                        logger.error("Project not found. Please enter a valid project name")
                        return
                    
                    # Safe assignment now that we know project exists
                    matching_projects = [
                        project["id"] for project in response.json()["data"]["content"] 
                        if project["name"] == project_name
                    ]
                    if matching_projects:
                        self.project_id = matching_projects[0]
                    else:
                        logger.error(f"Project '{project_name}' not found in project list")
                        return
                else:
                    logger.error("Failed to fetch project list after 401 token refresh")
                    return
            else:
                logger.error(f"HTTP {response.status_code} error when fetching project list")
                return

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "fetching project list")
            logger.error(f"Failed to retrieve projects list, Dataset will have limited functionality")
        except RequestException as e:
            logger.error(f"Error while fetching project list: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing project list: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during dataset initialization: {str(e)}")

    def list_datasets(self):
        """
        Retrieves a list of datasets for a given project.

        Returns:
            list: A list of dataset names.

        Raises:
            None.
        """
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot list datasets")
            return []

        def make_request():
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {"size": 99999, "page": "0", "projectId": str(self.project_id), "search": ""}
            try:
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "POST",
                    f"{Dataset.BASE_URL}/v2/llm/dataset",
                    headers=headers,
                    json=json_data,
                    timeout=Dataset.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                return response
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "listing datasets")
                return None
            except RequestException as e:
                logger.error(f"Error while listing datasets: {e}")
                return None

        try:
            response = make_request()
            if response is None:
                return []
                
            response_checker(response, "Dataset.list_datasets")
            
            if response.status_code in [200, 201]:
                datasets = response.json()["data"]["content"]
                dataset_list = [dataset["name"] for dataset in datasets]
                return dataset_list
            elif response.status_code == 401:
                logger.warning("Received 401 error during listing datasets. Attempting to refresh token.")
                get_token(force_refresh=True)  # Fetch a new token and set it in the environment
                response = make_request()  # Retry the request
                if response and response.status_code in [200, 201]:
                    logger.info("Datasets listed successfully after 401 token refresh")
                    datasets = response.json()["data"]["content"]
                    dataset_list = [dataset["name"] for dataset in datasets]
                    return dataset_list
                else:
                    logger.error("Failed to list datasets after 401 token refresh")
                    return []
            else:
                logger.error(f"HTTP {response.status_code} error when listing datasets")
                return []
        except Exception as e:
            logger.error(f"Error in list_datasets: {e}")
            return []

    def get_schema_mapping(self):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot get schema mapping")
            return []
        
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Id": str(self.project_id),
        }
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "GET",
                f"{Dataset.BASE_URL}/v1/llm/schema-elements",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] /v1/llm/schema-elements | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                response_data = response.json()["data"]["schemaElements"]
                if not response.json()['success']:
                    logger.error('Unable to fetch Schema Elements for the CSV')
                return response_data
            elif response.status_code == 401:
                logger.warning("Received 401 error during getting schema mapping. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", f"{Dataset.BASE_URL}/v1/llm/schema-elements", headers=new_headers, timeout=Dataset.TIMEOUT)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v1/llm/schema-elements (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Schema mapping fetched successfully after 401 token refresh")
                    response_data = response.json()["data"]["schemaElements"]
                    if not response.json()['success']:
                        logger.error('Unable to fetch Schema Elements for the CSV after 401 token refresh')
                    return response_data
                else:
                    logger.error("Failed to fetch schema mapping after 401 token refresh")
                    return []
            else:
                logger.error(f"HTTP {response.status_code} error when getting schema mapping")
                return []
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "getting schema mapping")
            return []
        except RequestException as e:
            logger.error(f"Error while getting schema mapping: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing schema mapping: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_schema_mapping: {str(e)}")
            return []

    ###################### CSV Upload APIs ###################

    def get_dataset_columns(self, dataset_name):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot get dataset columns")
            return []
            
        list_dataset = self.list_datasets()
        if dataset_name not in list_dataset:
            logger.error(f"Dataset {dataset_name} does not exists. Please enter a valid dataset name")
            return []

        headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
        json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "POST",
                f"{Dataset.BASE_URL}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=Dataset.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [POST] /v2/llm/dataset | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                datasets = response.json()["data"]["content"]
                # Safe assignment to avoid IndexError
                matching_datasets = [dataset["id"] for dataset in datasets if dataset["name"]==dataset_name]
                if matching_datasets:
                    dataset_id = matching_datasets[0]
                else:
                    logger.error(f"Dataset '{dataset_name}' not found in dataset list")
                    return []
            elif response.status_code == 401:
                logger.warning("Received 401 error during getting dataset list for columns. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("POST", f"{Dataset.BASE_URL}/v2/llm/dataset", headers=new_headers, json=json_data, timeout=Dataset.TIMEOUT)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Dataset list for columns fetched successfully after 401 token refresh")
                    datasets = response.json()["data"]["content"]
                    # Safe assignment to avoid IndexError
                    matching_datasets = [dataset["id"] for dataset in datasets if dataset["name"]==dataset_name]
                    if matching_datasets:
                        dataset_id = matching_datasets[0]
                    else:
                        logger.error(f"Dataset '{dataset_name}' not found in dataset list")
                        return []
                else:
                    logger.error("Failed to fetch dataset list for columns after 401 token refresh")
                    return []
            else:
                logger.error(f"HTTP {response.status_code} error when getting dataset list for columns")
                return []
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "getting dataset list for columns")
            return []
        except RequestException as e:
            logger.error(f"Error while getting dataset list for columns: {e}")
            return []
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing dataset list for columns: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_dataset_columns while fetching dataset_id for {dataset_name}: {str(e)}")
            return []

        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "GET",
                f"{Dataset.BASE_URL}/v2/llm/dataset/{dataset_id}?initialCols=0",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] /v2/llm/dataset/{dataset_id} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                dataset_columns = response.json()["data"]["datasetColumnsResponses"]
                dataset_columns = [item["displayName"] for item in dataset_columns]
                dataset_columns = [data for data in dataset_columns if not data.startswith('_')]
                if not response.json()['success']:
                    logger.error('Unable to fetch details of for the CSV')
                return dataset_columns
            elif response.status_code == 401:
                logger.warning(f"Received 401 error during getting dataset {dataset_id} columns. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", f"{Dataset.BASE_URL}/v2/llm/dataset/{dataset_id}?initialCols=0", headers=new_headers, timeout=Dataset.TIMEOUT)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v2/llm/dataset/{dataset_id} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info(f"Dataset {dataset_id} columns fetched successfully after 401 token refresh")
                    dataset_columns = response.json()["data"]["datasetColumnsResponses"]
                    dataset_columns = [item["displayName"] for item in dataset_columns]
                    dataset_columns = [data for data in dataset_columns if not data.startswith('_')]
                    if not response.json()['success']:
                        logger.error('Unable to fetch details of for the CSV')
                    return dataset_columns
                else:
                    logger.error(f"Failed to fetch dataset {dataset_id} columns after 401 token refresh")
                    return []
            else:
                logger.error(f"HTTP {response.status_code} error when getting dataset {dataset_id} columns")
                return []
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"getting dataset {dataset_id} columns")
            return []
        except RequestException as e:
            logger.error(f"Error while getting dataset {dataset_id} columns: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing dataset {dataset_id} columns: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_dataset_columns for dataset_id {dataset_id}: {str(e)}")
            return []

    def create_from_csv(self, csv_path, dataset_name, schema_mapping):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot create from CSV")
            return
            
        list_dataset = self.list_datasets()
        if dataset_name in list_dataset:
            logger.error(f"Dataset name {dataset_name} already exists. Please enter a unique dataset name")
            return

        #### get presigned URL
        def get_presignedUrl():
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            try:
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "GET",
                    f"{Dataset.BASE_URL}/v2/llm/dataset/csv/presigned-url",
                    headers=headers,
                    timeout=Dataset.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v2/llm/dataset/csv/presigned-url | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    return response.json()
                elif response.status_code == 401:
                    logger.warning("Received 401 error during getting presigned URL. Attempting to refresh token.")
                    token = RagaAICatalyst.get_token(force_refresh=True)
                    new_headers = headers.copy()
                    new_headers["Authorization"] = f"Bearer {token}"
                    
                    start_time = time.time()
                    response = session_manager.make_request_with_retry("GET", f"{Dataset.BASE_URL}/v2/llm/dataset/csv/presigned-url", headers=new_headers, timeout=Dataset.TIMEOUT)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [GET] /v2/llm/dataset/csv/presigned-url (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                    
                    if response.status_code in [200, 201]:
                        logger.info("Presigned URL fetched successfully after 401 token refresh")
                        return response.json()
                    else:
                        logger.error("Failed to fetch presigned URL after 401 token refresh")
                        return None
                else:
                    logger.error(f"HTTP {response.status_code} error when getting presigned URL")
                    return None
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "getting presigned URL")
                return None
            except RequestException as e:
                logger.error(f"Error while getting presigned URL: {e}")
                return None

        try:
            presignedUrl = get_presignedUrl()
            if presignedUrl and presignedUrl.get('success'):
                # Safely extract URL and filename with fallback
                data_dict = presignedUrl.get('data', {})
                url = data_dict.get('presignedUrl')
                filename = data_dict.get('fileName')
                
                # Validate that url and filename are not None or empty
                if not url or not filename:
                    logger.error('Presigned URL or filename is empty/None')
                    return
            else:
                logger.error('Unable to fetch presignedUrl')
                return
        except Exception as e:
            logger.error(f"Error in get_presignedUrl: {e}")
            return

        #### put csv to presigned URL
        def put_csv_to_presignedUrl(url):
            if not url:
                logger.error("URL is None or empty, cannot upload CSV to presigned URL")
                return None

            if not os.path.exists(csv_path):
                logger.error(f"CSV file does not exist: {csv_path}")
                return None

            headers = {
                'Content-Type': 'text/csv',
                'x-ms-blob-type': 'BlockBlob',
            }
            try:
                file_size = os.path.getsize(csv_path)
                logger.debug(f"Uploading CSV file: {csv_path} (size: {file_size} bytes)")

                with open(csv_path, 'rb') as file:
                    start_time = time.time()
                    response = session_manager.make_request_with_retry(
                        "PUT",
                        url,
                        headers=headers,
                        data=file,
                        timeout=Dataset.TIMEOUT,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [PUT] presigned-url | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                    if response.status_code in [200, 201]:
                        logger.debug(f"Successfully uploaded {file_size} bytes to presigned URL")
                    else:
                        logger.error(f"Upload failed with status {response.status_code}: {response.text[:200]}")

                    return response
            except FileNotFoundError as e:
                logger.error(f"CSV file not found: {csv_path}")
                return None
            except PermissionError as e:
                logger.error(f"Permission denied accessing CSV file: {csv_path}")
                return None
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "putting CSV to presigned URL")
                return None
            except RequestException as e:
                logger.error(f"HTTP request error while putting CSV to presigned URL: {e}")
                return None

        try:
            put_csv_response = put_csv_to_presignedUrl(url)
            if put_csv_response is None:
                logger.error('Failed to upload CSV to presigned URL: No response received')
                return
            elif put_csv_response.status_code not in (200, 201):
                logger.error(f'Failed to upload CSV to presigned URL: HTTP {put_csv_response.status_code}')
                return
            else:
                logger.info('CSV file successfully uploaded to presigned URL')
        except Exception as e:
            logger.error(f"Error during CSV upload to presigned URL: {e}")
            return

        ## Upload csv to elastic
        def upload_csv_to_elastic(data):
            header = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id)
            }
            try:
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "POST",
                    f"{Dataset.BASE_URL}/v2/llm/dataset/csv",
                    headers=header,
                    json=data,
                    timeout=Dataset.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset/csv | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    return response.json()
                elif response.status_code == 400:
                    logger.error(response.json().get("message", "Bad request"))
                    return None
                elif response.status_code == 401:
                    logger.warning("Received 401 error during uploading CSV to elastic. Attempting to refresh token.")
                    token = RagaAICatalyst.get_token(force_refresh=True)
                    new_header = header.copy()
                    new_header["Authorization"] = f"Bearer {token}"
                    
                    start_time = time.time()
                    response = session_manager.make_request_with_retry("POST", f"{Dataset.BASE_URL}/v2/llm/dataset/csv", headers=new_header, json=data, timeout=Dataset.TIMEOUT)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [POST] /v2/llm/dataset/csv (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                    
                    if response.status_code in [200, 201]:
                        logger.info("CSV uploaded to elastic successfully after 401 token refresh")
                        return response.json()
                    else:
                        logger.error("Failed to upload CSV to elastic after 401 token refresh")
                        return None
                else:
                    logger.error(f"HTTP {response.status_code} error when uploading CSV to elastic")
                    return None
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "uploading CSV to elastic")
                return None
            except RequestException as e:
                logger.error(f"Error while uploading CSV to elastic: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in upload_csv_to_elastic: {str(e)}")
                return None

        def generate_schema(mapping):
            result = {}
            for column, schema_element in mapping.items():
                if isinstance(schema_element, dict):
                    result[column] = schema_element
                else:
                    result[column] = {"columnType": schema_element}
            return result

        try:
            schema_mapping = generate_schema(schema_mapping)
            
            # Double-check that filename is still valid before using it
            if not filename:
                logger.error('Filename is None or empty when creating data payload')
                return
                
            data = {
                "projectId": str(self.project_id),
                "datasetName": dataset_name,
                "fileName": filename,
                "schemaMapping": schema_mapping,
                "opType": "insert",
                "description": ""
            }
            upload_csv_response = upload_csv_to_elastic(data)
            if upload_csv_response and upload_csv_response.get('success'):
                logger.info(f"Dataset creation successful: {upload_csv_response['message']}")
                self.jobId = upload_csv_response['data']['jobId']
                logger.info(f"Job ID for dataset creation: {self.jobId}")
            else:
                logger.error('Dataset creation failed: Unable to upload CSV metadata to server')
        except (KeyError, json.JSONDecodeError, IndexError, IOError, UnicodeError) as e:
            logger.error(f"Error parsing data in create_from_csv: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in create_from_csv: {str(e)}")

    def add_rows(self, csv_path, dataset_name):
        """
        Add rows to an existing dataset from a CSV file.

        Args:
            csv_path (str): Path to the CSV file to be added
            dataset_name (str): Name of the existing dataset to add rows to

        Raises:
            ValueError: If dataset does not exist or columns are incompatible
        """
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot add rows")
            return
            
        # Get existing dataset columns
        existing_columns = self.get_dataset_columns(dataset_name)

        # Read the CSV file to check columns
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            csv_columns = df.columns.tolist()
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            return

        # Check column compatibility
        for column in existing_columns:
            if column not in csv_columns:
                df[column] = None  

        # Get presigned URL for the CSV
        def get_presignedUrl():
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            try:
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "GET",
                    f"{Dataset.BASE_URL}/v2/llm/dataset/csv/presigned-url",
                    headers=headers,
                    timeout=Dataset.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v2/llm/dataset/csv/presigned-url | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    return response.json()
                elif response.status_code == 401:
                    logger.warning("Received 401 error during getting presigned URL for add rows. Attempting to refresh token.")
                    token = RagaAICatalyst.get_token(force_refresh=True)
                    new_headers = headers.copy()
                    new_headers["Authorization"] = f"Bearer {token}"
                    
                    start_time = time.time()
                    response = session_manager.make_request_with_retry("GET", f"{Dataset.BASE_URL}/v2/llm/dataset/csv/presigned-url", headers=new_headers, timeout=Dataset.TIMEOUT)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [GET] /v2/llm/dataset/csv/presigned-url (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                    
                    if response.status_code in [200, 201]:
                        logger.info("Presigned URL for add rows fetched successfully after 401 token refresh")
                        return response.json()
                    else:
                        logger.error("Failed to fetch presigned URL for add rows after 401 token refresh")
                        return None
                else:
                    logger.error(f"HTTP {response.status_code} error when getting presigned URL for add rows")
                    return None
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "getting presigned URL for add rows")
                return None
            except RequestException as e:
                logger.error(f"Error while getting presigned URL for add rows: {e}")
                return None

        try:
            presignedUrl = get_presignedUrl()
            if presignedUrl and presignedUrl.get('success'):
                # Safely extract URL and filename with fallback
                data_dict = presignedUrl.get('data', {})
                url = data_dict.get('presignedUrl')
                filename = data_dict.get('fileName')
                
                # Validate that url and filename are not None or empty
                if not url or not filename:
                    logger.error('Presigned URL or filename is empty/None for add rows')
                    return
            else:
                logger.error('Unable to fetch presignedUrl for add rows')
                return
        except Exception as e:
            logger.error(f"Error in get_presignedUrl for add rows: {e}")
            return

        # Upload CSV to presigned URL
        def put_csv_to_presignedUrl(url):
            if not url:
                logger.error("URL is None or empty, cannot upload CSV to presigned URL for add rows")
                return None

            if not os.path.exists(csv_path):
                logger.error(f"CSV file does not exist for add rows: {csv_path}")
                return None

            headers = {
                'Content-Type': 'text/csv',
                'x-ms-blob-type': 'BlockBlob',
            }
            try:
                file_size = os.path.getsize(csv_path)
                logger.debug(f"Uploading CSV file for add rows: {csv_path} (size: {file_size} bytes)")

                with open(csv_path, 'rb') as file:
                    start_time = time.time()
                    response = session_manager.make_request_with_retry(
                        "PUT",
                        url,
                        headers=headers,
                        data=file,
                        timeout=Dataset.TIMEOUT,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [PUT] presigned-url for add rows | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                    if response.status_code in [200, 201]:
                        logger.debug(f"Successfully uploaded {file_size} bytes to presigned URL for add rows")
                    else:
                        logger.error(f"Add rows upload failed with status {response.status_code}: {response.text[:200]}")

                    return response
            except FileNotFoundError as e:
                logger.error(f"CSV file not found for add rows: {csv_path}")
                return None
            except PermissionError as e:
                logger.error(f"Permission denied accessing CSV file for add rows: {csv_path}")
                return None
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "putting CSV to presigned URL for add rows")
                return None
            except RequestException as e:
                logger.error(f"HTTP request error while putting CSV to presigned URL for add rows: {e}")
                return None

        try:
            put_csv_response = put_csv_to_presignedUrl(url)
            if put_csv_response is None:
                logger.error('Failed to upload CSV to presigned URL for add rows: No response received')
                return
            elif put_csv_response.status_code not in (200, 201):
                logger.error(f'Failed to upload CSV to presigned URL for add rows: HTTP {put_csv_response.status_code}')
                return
            else:
                logger.info('CSV file successfully uploaded to presigned URL')
        except Exception as e:
            logger.error(f"Error during CSV upload to presigned URL for add rows: {e}")
            return

        # Prepare schema mapping (assuming same mapping as original dataset)
        def generate_schema_mapping(dataset_name):
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {
                "size": 12, 
                "page": "0", 
                "projectId": str(self.project_id), 
                "search": ""
            }
            try:
                # First get dataset details
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "POST",
                    f"{Dataset.BASE_URL}/v2/llm/dataset",
                    headers=headers,
                    json=json_data,
                    timeout=Dataset.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset for schema mapping | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    datasets = response.json()["data"]["content"]
                    # Safe assignment to avoid IndexError
                    matching_datasets = [dataset["id"] for dataset in datasets if dataset["name"]==dataset_name]
                    if matching_datasets:
                        dataset_id = matching_datasets[0]
                    else:
                        logger.error(f"Dataset '{dataset_name}' not found for schema mapping")
                        return {}
                elif response.status_code == 401:
                    logger.warning("Received 401 error during getting dataset for schema mapping. Attempting to refresh token.")
                    token = RagaAICatalyst.get_token(force_refresh=True)
                    new_headers = headers.copy()
                    new_headers["Authorization"] = f"Bearer {token}"
                    
                    start_time = time.time()
                    response = session_manager.make_request_with_retry("POST", f"{Dataset.BASE_URL}/v2/llm/dataset", headers=new_headers, json=json_data, timeout=Dataset.TIMEOUT)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [POST] /v2/llm/dataset for schema mapping (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                    
                    if response.status_code in [200, 201]:
                        logger.info("Dataset for schema mapping fetched successfully after 401 token refresh")
                        datasets = response.json()["data"]["content"]
                        # Safe assignment to avoid IndexError
                        matching_datasets = [dataset["id"] for dataset in datasets if dataset["name"]==dataset_name]
                        if matching_datasets:
                            dataset_id = matching_datasets[0]
                        else:
                            logger.error(f"Dataset '{dataset_name}' not found for schema mapping")
                            return {}
                    else:
                        logger.error("Failed to fetch dataset for schema mapping after 401 token refresh")
                        return {}

                # Get dataset details to extract schema mapping
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "GET",
                    f"{Dataset.BASE_URL}/v2/llm/dataset/{dataset_id}?initialCols=0",
                    headers=headers,
                    timeout=Dataset.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v2/llm/dataset/{dataset_id} for schema mapping | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    # Extract schema mapping
                    schema_mapping = {}
                    for col in response.json()["data"]["datasetColumnsResponses"]:
                        schema_mapping[col["displayName"]] = {"columnType": col["columnType"]}
                    return schema_mapping
                elif response.status_code == 401:
                    logger.warning(f"Received 401 error during getting dataset {dataset_id} details for schema mapping. Attempting to refresh token.")
                    token = RagaAICatalyst.get_token(force_refresh=True)
                    new_headers = headers.copy()
                    new_headers["Authorization"] = f"Bearer {token}"
                    
                    start_time = time.time()
                    response = session_manager.make_request_with_retry("GET", f"{Dataset.BASE_URL}/v2/llm/dataset/{dataset_id}?initialCols=0", headers=new_headers, timeout=Dataset.TIMEOUT)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [GET] /v2/llm/dataset/{dataset_id} for schema mapping (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"Dataset {dataset_id} details for schema mapping fetched successfully after 401 token refresh")
                        schema_mapping = {}
                        for col in response.json()["data"]["datasetColumnsResponses"]:
                            schema_mapping[col["displayName"]] = {"columnType": col["columnType"]}
                        return schema_mapping
                    else:
                        logger.error(f"Failed to fetch dataset {dataset_id} details for schema mapping after 401 token refresh")
                        return {}
                else:
                    logger.error(f"HTTP {response.status_code} error when getting dataset {dataset_id} details for schema mapping")
                    return {}
                
            except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                session_manager.handle_request_exceptions(e, "getting schema mapping for add rows")
                return {}
            except RequestException as e:
                logger.error(f"Error while getting schema mapping for add rows: {e}")
                return {}
            except Exception as e:
                logger.error(f"Unexpected error in generate_schema_mapping for add rows: {str(e)}")
                return {}

        # Upload CSV to elastic
        try:
            schema_mapping = generate_schema_mapping(dataset_name)
            
            # Double-check that filename is still valid before using it
            if not filename:
                logger.error('Filename is None or empty when creating data payload for add rows')
                return
            
            data = {
                "projectId": str(self.project_id),
                "datasetName": dataset_name,
                "fileName": filename,
                "schemaMapping": schema_mapping,
                "opType": "update",  # Use update for adding rows
                "description": "Adding new rows to dataset"
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id)
            }
            
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "POST",
                f"{Dataset.BASE_URL}/v2/llm/dataset/csv",
                headers=headers,
                json=data,
                timeout=Dataset.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [POST] /v2/llm/dataset/csv for add rows | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                # Check response
                response_data = response.json()
                if response_data.get('success', False):
                    logger.info(f"Add rows operation successful: {response_data['message']}")
                    self.jobId = response_data['data']['jobId']
                    logger.info(f"Job ID for add rows operation: {self.jobId}")
                else:
                    logger.error(f"Add rows operation failed: {response_data.get('message', 'Failed to add rows')}")
            elif response.status_code == 400:
                logger.error(response.json().get("message", "Failed to add rows"))
            elif response.status_code == 401:
                logger.warning("Received 401 error during adding rows. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("POST", f"{Dataset.BASE_URL}/v2/llm/dataset/csv", headers=new_headers, json=data, timeout=Dataset.TIMEOUT)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset/csv for add rows (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Rows added successfully after 401 token refresh")
                    response_data = response.json()
                    if response_data.get('success', False):
                        logger.info(f"{response_data['message']}")
                        self.jobId = response_data['data']['jobId']
                    else:
                        logger.error(response_data.get('message', 'Failed to add rows'))
                else:
                    logger.error("Failed to add rows after 401 token refresh")
            else:
                logger.error(f"HTTP {response.status_code} error when adding rows")
        
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "uploading CSV for add rows")
        except RequestException as e:
            logger.error(f"Error while uploading CSV for add rows: {e}")
        except (KeyError, json.JSONDecodeError, IndexError, IOError, UnicodeError) as e:
            logger.error(f"Error parsing data in add_rows: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in add_rows: {str(e)}")

    def add_columns(self, text_fields, dataset_name, column_name, provider, model, variables={}):
        """
        Add a column to a dataset with dynamically fetched model parameters
        
        Args:
            project_id (int): Project ID
            dataset_id (int): Dataset ID
            column_name (str): Name of the new column
            provider (str): Name of the model provider
            model (str): Name of the model
        """
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot add columns")
            return
            
        # First, get model parameters

        # Validate text_fields input
        if not isinstance(text_fields, list):
            logger.error("text_fields must be a list of dictionaries")
            return
        
        for field in text_fields:
            if not isinstance(field, dict) or 'role' not in field or 'content' not in field:
                logger.error("Each text field must be a dictionary with 'role' and 'content' keys")
                return
            
        # First, get the dataset ID
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Id": str(self.project_id),
        }
        json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
        
        try:
            # Get dataset list
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "POST",
                f"{Dataset.BASE_URL}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=Dataset.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [POST] /v2/llm/dataset for add columns | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                datasets = response.json()["data"]["content"]
                # Find dataset ID
                dataset_id = next((dataset["id"] for dataset in datasets if dataset["name"] == dataset_name), None)
                
                if dataset_id is None:
                    logger.error(f"Dataset {dataset_name} not found")
                    return
            elif response.status_code == 401:
                logger.warning("Received 401 error during getting dataset list for add columns. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("POST", f"{Dataset.BASE_URL}/v2/llm/dataset", headers=new_headers, json=json_data, timeout=Dataset.TIMEOUT)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset for add columns (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Dataset list for add columns fetched successfully after 401 token refresh")
                    datasets = response.json()["data"]["content"]
                    dataset_id = next((dataset["id"] for dataset in datasets if dataset["name"] == dataset_name), None)
                    
                    if dataset_id is None:
                        logger.error(f"Dataset {dataset_name} not found")
                        return
                else:
                    logger.error("Failed to fetch dataset list for add columns after 401 token refresh")
                    return
            else:
                logger.error(f"HTTP {response.status_code} error when getting dataset list for add columns")
                return

            parameters_url= f"{Dataset.BASE_URL}/playground/providers/models/parameters/list"
            
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            
            # Fetch model parameters
            parameters_payload = {
                "providerName": provider,
                "modelName": model
            }
        
            # Get model parameters
            start_time = time.time()
            params_response = session_manager.make_request_with_retry(
                "POST",
                parameters_url, 
                headers=headers, 
                json=parameters_payload, 
                timeout=30
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [POST] /playground/providers/models/parameters/list | Status: {params_response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if params_response.status_code in [200, 201]:
                # Extract parameters
                all_parameters = params_response.json().get('data', [])
            elif params_response.status_code == 401:
                logger.warning("Received 401 error during getting model parameters. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                params_response = session_manager.make_request_with_retry("POST", parameters_url, headers=new_headers, json=parameters_payload, timeout=30)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /playground/providers/models/parameters/list (retry) | Status: {params_response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if params_response.status_code in [200, 201]:
                    logger.info("Model parameters fetched successfully after 401 token refresh")
                    all_parameters = params_response.json().get('data', [])
                else:
                    logger.error("Failed to fetch model parameters after 401 token refresh")
                    return
            else:
                logger.error(f"HTTP {params_response.status_code} error when getting model parameters")
                return
            
            # Filter and transform parameters for add-column API
            formatted_parameters = []
            for param in all_parameters:
                value = param.get('value')
                param_type = param.get('type')

                if value is None:
                    formatted_param = {
                        "name": param.get('name'),
                        "value": None,  # Pass None if the value is null
                        "type": param.get('type')
                    }
                else:
                    # Improved type handling
                    if param_type == "float":
                        value = float(value)  # Ensure value is converted to float
                    elif param_type == "int":
                        value = int(value)  # Ensure value is converted to int
                    elif param_type == "bool":
                        value = bool(value)  # Ensure value is converted to bool
                    elif param_type == "string":
                        value = str(value)  # Ensure value is converted to string
                    else:
                        logger.error(f"Unsupported parameter type: {param_type}")  # Handle unsupported types

                    formatted_param = {
                        "name": param.get('name'),
                        "value": value,
                        "type": param.get('type')
                    }
                formatted_parameters.append(formatted_param)
            dataset_id = next((dataset["id"] for dataset in datasets if dataset["name"] == dataset_name), None)

            # Prepare payload for add column API
            add_column_payload = {
                "rowFilterList": [],
                "columnName": column_name,
                "datasetId": dataset_id,
                "variables": variables,
                "promptTemplate": {
                    "textFields": text_fields,
                    "modelSpecs": {
                        "model": f"{provider}/{model}",
                        "parameters": formatted_parameters
                    }
                }
            }
            if variables:
                variable_specs = []
                for key, values in variables.items():
                    variable_specs.append({
                        "name": key,
                        "type": "string",
                        "schema": "query"
                    })
                add_column_payload["promptTemplate"]["variableSpecs"] = variable_specs
            
            # Make API call to add column
            add_column_url = f"{Dataset.BASE_URL}/v2/llm/dataset/add-column"
            
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "POST",
                add_column_url, 
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                    "X-Project-Id": str(self.project_id)
                }, 
                json=add_column_payload,
                timeout=30
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [POST] /v2/llm/dataset/add-column | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            # Check response
            if response.status_code in [200, 201]:
                response_data = response.json()
                if response_data.get('success', False):
                    logger.info(f"Column '{column_name}' added successfully to dataset '{dataset_name}'")
                    self.jobId = response_data['data']['jobId']
                else:
                    logger.error(response_data.get('message', 'Failed to add column'))
            elif response.status_code == 401:
                logger.warning("Received 401 error during adding column. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "POST", add_column_url, 
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f"Bearer {token}",
                        "X-Project-Id": str(self.project_id)
                    }, 
                    json=add_column_payload, timeout=30
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset/add-column (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Column added successfully after 401 token refresh")
                    response_data = response.json()
                    if response_data.get('success', False):
                        logger.info(f"Column '{column_name}' added successfully to dataset '{dataset_name}'")
                        self.jobId = response_data['data']['jobId']
                    else:
                        logger.error(response_data.get('message', 'Failed to add column'))
                else:
                    logger.error("Failed to add column after 401 token refresh")
            else:
                logger.error(f"HTTP {response.status_code} error when adding column")
        
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "adding column")
        except RequestException as e:
            logger.error(f"Error adding column: {e}")
        except Exception as e:
            logger.error(f"Unexpected error adding column: {e}")

    def get_status(self):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot get status")
            return JOB_STATUS_FAILED
            
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "GET",
                f'{Dataset.BASE_URL}/job/status', 
                headers=headers, 
                timeout=30)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] /job/status | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                if response.json()["success"]:
                    status_json = [item["status"] for item in response.json()["data"]["content"] if item["id"]==self.jobId]
                    status_json = status_json[0]
                    if status_json == "Failed":
                        logger.info("Job failed. No results to fetch.")
                        return JOB_STATUS_FAILED
                    elif status_json == "In Progress":
                        logger.info(f"Job in progress. Please wait while the job completes.\nVisit Job Status: {Dataset.BASE_URL.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to track")
                        return JOB_STATUS_IN_PROGRESS
                    elif status_json == "Completed":
                        logger.info(f"Job completed. Fetching results.\nVisit Job Status: {Dataset.BASE_URL.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to check")
                        return JOB_STATUS_COMPLETED
                    else:
                        logger.error(f"Unknown status received: {status_json}")
                        return JOB_STATUS_FAILED
                else:
                    logger.error("Request was not successful when getting job status")
                    return JOB_STATUS_FAILED
            elif response.status_code == 401:
                logger.warning("Received 401 error during getting job status. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", f'{Dataset.BASE_URL}/job/status', headers=new_headers, timeout=30)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /job/status (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Job status fetched successfully after 401 token refresh")
                    if response.json()["success"]:
                        status_json = [item["status"] for item in response.json()["data"]["content"] if item["id"]==self.jobId]
                        status_json = status_json[0]
                        if status_json == "Failed":
                            logger.info("Job failed. No results to fetch.")
                            return JOB_STATUS_FAILED
                        elif status_json == "In Progress":
                            logger.info(f"Job in progress. Please wait while the job completes.\nVisit Job Status: {Dataset.BASE_URL.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to track")
                            return JOB_STATUS_IN_PROGRESS
                        elif status_json == "Completed":
                            logger.info(f"Job completed. Fetching results.\nVisit Job Status: {Dataset.BASE_URL.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to check")
                            return JOB_STATUS_COMPLETED
                        else:
                            logger.error(f"Unknown status received: {status_json}")
                            return JOB_STATUS_FAILED
                    else:
                        logger.error("Request was not successful when getting job status")
                        return JOB_STATUS_FAILED
                else:
                    logger.error("Failed to get job status after 401 token refresh")
                    return JOB_STATUS_FAILED
            else:
                logger.error(f"HTTP {response.status_code} error when getting job status")
                return JOB_STATUS_FAILED
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "getting job status")
            return JOB_STATUS_FAILED
        except RequestException as e:
            logger.error(f"Error getting job status: {e}")
            return JOB_STATUS_FAILED
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return JOB_STATUS_FAILED

    def _jsonl_to_csv(self, jsonl_file, csv_file):
        """Convert a JSONL file to a CSV file."""
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                data = [json.loads(line) for line in infile]
            
            if not data:
                logger.info("Empty JSONL file.")
                return
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Converted {jsonl_file} to {csv_file}")
        except (IOError, UnicodeError) as e:
            logger.error(f"Error with file operations in _jsonl_to_csv: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing JSON in _jsonl_to_csv: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in _jsonl_to_csv: {str(e)}")

    def create_from_jsonl(self, jsonl_path, dataset_name, schema_mapping):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            self._jsonl_to_csv(jsonl_path, tmp_csv_path)
            self.create_from_csv(tmp_csv_path, dataset_name, schema_mapping)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting JSONL to CSV: {e}")
            pass
        except Exception as e:
            logger.error(f"Unexpected error in create_from_jsonl: {str(e)}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")

    def add_rows_from_jsonl(self, jsonl_path, dataset_name):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            self._jsonl_to_csv(jsonl_path, tmp_csv_path)
            self.add_rows(tmp_csv_path, dataset_name)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting JSONL to CSV: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in add_rows_from_jsonl: {str(e)}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")

    def create_from_df(self, df, dataset_name, schema_mapping):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            df.to_csv(tmp_csv_path, index=False)
            self.create_from_csv(tmp_csv_path, dataset_name, schema_mapping)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting DataFrame to CSV: {e}")
            pass
        except Exception as e:
            logger.error(f"Unexpected error in create_from_df: {str(e)}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")

    def add_rows_from_df(self, df, dataset_name):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            df.to_csv(tmp_csv_path, index=False)
            self.add_rows(tmp_csv_path, dataset_name)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting DataFrame to CSV: {e}")
            pass
        except Exception as e:
            logger.error(f"Unexpected error in add_rows_from_df: {str(e)}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")
    
    def delete_dataset(self, dataset_name):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot delete dataset")
            return
            
        try:
            def make_request():
                headers = {
                    'Content-Type': 'application/json',
                    "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                    "X-Project-Id": str(self.project_id),
                }
                json_data = {"size": 99999, "page": "0", "projectId": str(self.project_id), "search": ""}
                try:
                    start_time = time.time()
                    response = session_manager.make_request_with_retry(
                        "POST",
                        f"{Dataset.BASE_URL}/v2/llm/dataset",
                        headers=headers,
                        json=json_data,
                        timeout=Dataset.TIMEOUT,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"API Call: [POST] /v2/llm/dataset for delete | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                    return response
                except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
                    session_manager.handle_request_exceptions(e, "listing datasets for delete")
                    return None
                except RequestException as e:
                    logger.error(f"Error while listing datasets for delete: {e}")
                    return None

            response = make_request()
            if response is None:
                return

            if response.status_code in [200, 201]:
                datasets = response.json()["data"]["content"]
                dataset_list = [dataset["name"] for dataset in datasets]
                if dataset_name not in dataset_list:
                    logger.error(f"Dataset '{dataset_name}' does not exists. Please enter a existing dataset name")
                    return
                
                # Get dataset id safely
                matching_datasets = [dataset["id"] for dataset in datasets if dataset["name"] == dataset_name]
                if matching_datasets:
                    dataset_id = matching_datasets[0]
                else:
                    logger.error(f"Dataset '{dataset_name}' not found in delete operation")
                    return
            elif response.status_code == 401:
                logger.warning("Received 401 error during listing datasets for delete. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                
                headers = {
                    'Content-Type': 'application/json',
                    "Authorization": f"Bearer {token}",
                    "X-Project-Id": str(self.project_id),
                }
                json_data = {"size": 99999, "page": "0", "projectId": str(self.project_id), "search": ""}
                
                start_time = time.time()
                response = session_manager.make_request_with_retry("POST", f"{Dataset.BASE_URL}/v2/llm/dataset", headers=headers, json=json_data, timeout=Dataset.TIMEOUT)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [POST] /v2/llm/dataset for delete (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Dataset list for delete fetched successfully after 401 token refresh")
                    datasets = response.json()["data"]["content"]
                    dataset_list = [dataset["name"] for dataset in datasets]
                    if dataset_name not in dataset_list:
                        logger.error(f"Dataset '{dataset_name}' does not exists. Please enter a existing dataset name")
                        return
                    
                    # Get dataset id safely
                    matching_datasets = [dataset["id"] for dataset in datasets if dataset["name"] == dataset_name]
                    if matching_datasets:
                        dataset_id = matching_datasets[0]
                    else:
                        logger.error(f"Dataset '{dataset_name}' not found in delete operation")
                        return
                else:
                    logger.error("Failed to list datasets for delete after 401 token refresh")
                    return
            else:
                logger.error(f"HTTP {response.status_code} error when listing datasets for delete")
                return
            
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "DELETE",
                f"{Dataset.BASE_URL}/v1/llm/dataset/{int(dataset_id)}",
                headers={
                    'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                    "X-Project-Id": str(self.project_id)
                },
                timeout=Dataset.TIMEOUT
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [DELETE] /v1/llm/dataset/{dataset_id} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in [200, 201]:
                if response.json()["success"]:
                    logger.info(f"Dataset '{dataset_name}' deleted successfully")
                else:
                    logger.error("Request was not successful when deleting dataset")
            elif response.status_code == 401:
                logger.warning("Received 401 error during deleting dataset. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "DELETE", f"{Dataset.BASE_URL}/v1/llm/dataset/{int(dataset_id)}",
                    headers={
                        'Authorization': f"Bearer {token}",
                        "X-Project-Id": str(self.project_id)
                    }, timeout=Dataset.TIMEOUT
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [DELETE] /v1/llm/dataset/{dataset_id} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    logger.info("Dataset deleted successfully after 401 token refresh")
                    if response.json()["success"]:
                        logger.info(f"Dataset '{dataset_name}' deleted successfully")
                    else:
                        logger.error("Request was not successful when deleting dataset")
                else:
                    logger.error("Failed to delete dataset after 401 token refresh")
            else:
                logger.error(f"HTTP {response.status_code} error when deleting dataset")
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "deleting dataset")
        except RequestException as e:
            logger.error(f"Error deleting dataset: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred when deleting dataset: {e}")
        