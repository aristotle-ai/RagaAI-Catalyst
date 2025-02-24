import os
import json
import requests
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import RagaAICatalyst

def create_dataset_schema_with_trace_rag(dataset_name, project_name, additional_metadata_keys=None, additional_pipeline_keys=None):
    SCHEMA_MAPPING_NEW = {
        "trace_id": {"columnType": "traceId"},
        "trace_uri": {"columnType": "traceUri"},
        "prompt": {"columnType": "prompt"},
        "response":{"columnType": "response"},
        "context": {"columnType": "context"},
        "llm_model": {"columnType":"pipeline"},
        "recorded_on": {"columnType": "metadata"},
        "embed_model": {"columnType":"pipeline"},
        "log_source": {"columnType": "metadata"},
        "vector_store":{"columnType":"pipeline"},
        "feedback": {"columnType":"feedBack"}
    }

    if additional_metadata_keys:
        for key in additional_metadata_keys:
            if key == "model_name":
                SCHEMA_MAPPING_NEW['response']["modelName"] = additional_metadata_keys[key]
            else:
                SCHEMA_MAPPING_NEW[key] = {"columnType": key, "parentColumn": "response"}

    if additional_pipeline_keys:
        for key in additional_pipeline_keys:
            SCHEMA_MAPPING_NEW[key] = {"columnType": "pipeline"}
            
    def make_request():
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": project_name,
        }
        payload = json.dumps({
            "datasetName": dataset_name,
            "schemaMapping": SCHEMA_MAPPING_NEW,
            "traceFolderUrl": None,
        })
        response = requests.request("POST",
            f"{RagaAICatalyst.BASE_URL}/v1/llm/dataset/logs",
            headers=headers,
            data=payload,
            timeout=30
        )

        return response

    response = make_request()

    if response.status_code == 401:
        # get_token()  # Fetch a new token and set it in the environment
        response = make_request()  # Retry the request
    if response.status_code != 200:
        return response.status_code
    return response.status_code
