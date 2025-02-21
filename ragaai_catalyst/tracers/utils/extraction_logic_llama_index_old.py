import json
from typing import Dict, Any, Optional


def extract_llama_index_data(data):
    # import pdb; pdb.set_trace()
    """
    Transform llama_index trace data into standardized format
    """
    data = data[0]

    # Extract top-level metadata
    trace_data = {
        "trace_id": data.get("trace_id"),
        "project_id": data.get("project_id"),
        "session_id": data.get("session_id"),
        "trace_type": data.get("trace_type"),
        "pipeline": data.get("pipeline"),
        "metadata": data.get("metadata"),
        "prompt_length": 0,
        "data": {
            "query": None,
            "context": None,
            "response": None,
            "system_prompt": None,
            "expected_response": None
        }
    }

    # Process traces
    if "traces" in data:
        for entry in data["traces"]:
            event_type = entry.get("event_type")

            # query
            if event_type == "query" or event_type == "retrieve":
                if "query_str" in entry["payload"]:
                    trace_data["data"]["prompt"] = entry["payload"]["query_str"]
            if event_type == "llm":
                if "completion" in entry["payload"]:
                    trace_data["data"]["prompt"] = entry["payload"]["formatted_prompt"]

            # context
            if event_type == "retrieve":
                if "nodes" in entry["payload"]:
                    nodes = entry["payload"]["nodes"]
                    context_texts = []
                    for node in nodes:
                        if "node" in node and "text" in node["node"]:
                            context_texts.append(node["node"]["text"].replace('\n', ' '))
                    # Join all context texts with a space
                    trace_data["data"]["context"] = context_texts  # " ".join(context_texts[::-1])

            # response and system_prompt
            elif event_type == "llm":
                if "response" in entry["payload"]:
                    response = entry["payload"]["response"]
                    trace_data["data"]["response"] = response['message']['content']
                elif "completion" in entry["payload"]:
                    trace_data["data"]["response"] = entry["payload"]["completion"]["text"]

                if "messages" in entry["payload"]:
                    messages = entry["payload"]["messages"]
                    system_messages = [item['content'] for item in messages if item['role'] == 'system']
                    if system_messages:
                        trace_data["data"]["system_prompt"] = system_messages[0]

        if data["traces"][0]['expected_response']:
            trace_data["data"]["expected_response"] = data["traces"][0]['expected_response']
        trace_data["prompt_length"] = len(trace_data["data"]["prompt"])

    return trace_data