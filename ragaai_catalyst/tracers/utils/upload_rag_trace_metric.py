import os
import requests
import json

from ...ragaai_catalyst import RagaAICatalyst
from ...dataset import Dataset


def get_user_trace_metrics(project_name, dataset_name):
    try:
        list_datasets = Dataset(project_name=project_name).list_datasets()
        if not list_datasets:
            return []
        elif dataset_name not in list_datasets:
            return []
        else:
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": project_name,
            }
            response = requests.request("GET", 
                                        f"{RagaAICatalyst.BASE_URL}/v1/llm/trace/metrics?datasetName={dataset_name}", 
                                        headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Error fetching traces metrics: {response.json()['message']}")
                return None
            
            return response.json()["data"]["columns"]
    except Exception as e:
        print(f"Error fetching traces metrics: {e}")
        return None

def get_trace_metrics_from_trace(traces):
    metrics = []

    if "metrics" in traces.keys():
        if len(traces["metrics"]) > 0:
            metrics.extend(traces["metrics"])
    return metrics


def _change_metrics_format_for_payload(metrics):
    formatted_metrics = []
    for metric in metrics:
        if any(m["name"] == metric.get("displayName") or m['name'] == metric.get("name") for m in formatted_metrics):
            continue
        metric_display_name = metric["name"]
        if metric.get("displayName"):
            metric_display_name = metric['displayName']
        formatted_metrics.append({
            "name": metric_display_name,
            "displayName": metric_display_name,
            "config": {"source": "user"},
        })
    return formatted_metrics

def upload_rag_trace_metric(json_file_path, dataset_name, project_name):
    try:
        with open(json_file_path, "r") as f:
            traces = json.load(f)
        metrics = get_trace_metrics_from_trace(traces[0])
        metrics = _change_metrics_format_for_payload(metrics)

        user_trace_metrics = get_user_trace_metrics(project_name, dataset_name)
        if user_trace_metrics:
            user_trace_metrics_list = [metric["displayName"] for metric in user_trace_metrics]

        if user_trace_metrics:
            for metric in metrics:
                if metric["displayName"] in user_trace_metrics_list:
                    metricConfig = next((user_metric["metricConfig"] for user_metric in user_trace_metrics if
                                         user_metric["displayName"] == metric["displayName"]), None)
                    if not metricConfig or metricConfig.get("Metric Source", {}).get("value") != "user":
                        raise ValueError(
                            f"Metrics {metric['displayName']} already exist in dataset {dataset_name} of project {project_name}.")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": project_name,
        }
        payload = json.dumps({
            "datasetName": dataset_name,
            "metrics": metrics
        })
        print(payload)
        print(headers)
        response = requests.request("POST",
                                    f"{RagaAICatalyst.BASE_URL}/v1/llm/trace/metrics",
                                    headers=headers,
                                    data=payload,
                                    timeout=10)
        if response.status_code != 200:
            raise ValueError(f"Error inserting agentic trace metrics")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error submitting traces: {e}")
        return None

    return response
