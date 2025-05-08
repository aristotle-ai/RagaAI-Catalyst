import requests
import json
import os
import time
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

class UploadTraces:
    def __init__(self, 
                 json_file_path,
                 project_name,
                 project_id,
                 dataset_name,
                 user_detail,
                 base_url):
        self.json_file_path = json_file_path
        self.project_name = project_name
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.user_detail = user_detail
        self.base_url = base_url
        self.timeout = 10

    def _create_dataset_schema_with_trace(self, additional_metadata_keys=None, additional_pipeline_keys=None):
        SCHEMA_MAPPING_NEW = {
            "trace_id": {"columnType": "traceId"},
            "trace_uri": {"columnType": "traceUri"},
            "prompt": {"columnType": "prompt"},
            "response":{"columnType": "response"},
            "context": {"columnType": "context"},
            "llm_model": {"columnType":"pipeline"},
            "recorded_on": {"columnType": "timestamp"},
            "embed_model": {"columnType":"pipeline"},
            "log_source": {"columnType": "metadata"},
            "vector_store":{"columnType":"pipeline"},
            "feedback": {"columnType":"feedBack"},
            "model_name": {"columnType": "metadata"},
            "total_cost": {"columnType": "metadata", "dataType": "numerical"},
            "total_latency": {"columnType": "metadata", "dataType": "numerical"},
            "error": {"columnType": "metadata"}
        }

        if additional_metadata_keys:
            for key in additional_metadata_keys:
                if key == "model_name":
                    SCHEMA_MAPPING_NEW['response']["modelName"] = additional_metadata_keys[key]
                elif key == "error":
                    pass
                else:
                    SCHEMA_MAPPING_NEW[key] = {"columnType": key, "parentColumn": "response"}

        if self.user_detail and self.user_detail["trace_user_detail"]["metadata"]:
            for key in self.user_detail["trace_user_detail"]["metadata"]:
                if key not in SCHEMA_MAPPING_NEW:
                    SCHEMA_MAPPING_NEW[key] = {"columnType": "metadata"}

        if additional_pipeline_keys:
            for key in additional_pipeline_keys:
                SCHEMA_MAPPING_NEW[key] = {"columnType": "pipeline"}
                
        def make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            payload = json.dumps({
                "datasetName": self.dataset_name,
                "schemaMapping": SCHEMA_MAPPING_NEW,
                "traceFolderUrl": None,
            })
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/dataset/logs"
            response = requests.request("POST",
                endpoint,
                headers=headers,
                data=payload,
                timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000  
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms") 

            return response

        response = make_request()

        if response.status_code == 401:
            # get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return response.status_code
        return response.status_code

    def _get_presigned_url(self):
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "numFiles": 1,
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }
        try:
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/presigned-url"
            # Changed to POST from GET
            response = requests.request("POST", 
                                        endpoint, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            if response.status_code == 200:
                presignedUrls = response.json()["data"]["presignedUrls"][0]
                return presignedUrls
            else:
                response = requests.request("GET", 
                                        endpoint, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                if response.status_code == 200:
                    presignedUrls = response.json()["data"]["presignedUrls"][0]
                    return presignedUrls

                logger.error(f"Failed to fetch presigned URL: {response.json()['message']}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error while getting presigned url: {e}")
            return None


    def _put_presigned_url(self, presignedUrl, filename):
        headers = {
                "Content-Type": "application/json",
            }

        if "blob.core.windows.net" in presignedUrl:  # Azure
            headers["x-ms-blob-type"] = "BlockBlob"
        # print(f"Uploading traces...")
        with open(filename) as f:
            payload = f.read().replace("\n", "").replace("\r", "").encode()
            
        try:
            start_time = time.time()
            response = requests.request("PUT", 
                                        presignedUrl, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [PUT] {presignedUrl} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            if response.status_code != 200 or response.status_code != 201:
                return response, response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Error while uploading to presigned url: {e}")
            return None


    def _insert_traces(self, presignedUrl):
        headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "Content-Type": "application/json",
                "X-Project-Name": self.project_name,
            }
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "presignedUrl": presignedUrl,
            })
        try:
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/insert/trace"
            response = requests.request("POST", 
                                        endpoint, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            if response.status_code != 200:
                print(f"Error inserting traces: {response.json()['message']}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error while inserting traces: {e}")
            return None

    def upload_traces(self, additional_metadata_keys=None, additional_pipeline_keys=None):
        try:
            self._create_dataset_schema_with_trace(additional_metadata_keys, additional_pipeline_keys)
            presignedUrl = self._get_presigned_url()
            if presignedUrl is None:
                return
            self._put_presigned_url(presignedUrl, self.json_file_path)
            self._insert_traces(presignedUrl)
            # print("Traces uploaded")
        except Exception as e:
            print(f"Error while uploading rag traces: {e}")