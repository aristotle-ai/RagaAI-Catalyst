import json
from typing import Dict, Any, Optional


def extract_llama_index_data(data):
    """
    Transform llama_index trace data into standardized format
    """
    data = data[0]
    with open("/Users/ritikagoel/workspace/llamaindex_sub_query_issue/RagaAI-Catalyst/llama_final_result1.json", 'w') as f:
        json.dump(data, f, indent=4,default=str)        
    # Extract top-level metadata
    trace_data = {
        "project_id": data.get("project_id"),
        "trace_id": data.get("trace_id"),
        "session_id": data.get("session_id"),
        "trace_type": data.get("trace_type"),
        "pipeline": data.get("pipeline"),
        "metadata":data.get("metadata") ,
        "prompt_length": 0,  
        "data": {
            "prompt": None,
            "context": None,
            "response": None,
            "system_prompt": None
        }
    }


    def get_prompt(data):
        for span in data:
            if span["event_type"]=="QueryStartEvent":
                prompt = span.get("query", "")
                return prompt
            if span["event_type"]=="QueryEndEvent":
                prompt = span.get("query", "")
                return prompt
            if span["event_type"]=="RetrievalStartEvent":
                prompt = span.get("str_or_query_bundle", "")
                return prompt


    def get_context(data):
        for span in data:
            if span["event_type"]=="RetrievalEndEvent":
                context = span.get("text", "")
                return context
    
    def get_response(data):
        responses = []
        for span in data:
            if span["event_type"] == "QueryEndEvent":
                response = span.get("response", "")
                responses.append(response)
            elif span["event_type"] == "LLMCompletionEndEvent":
                response = span.get("response", "")
                responses.append(response)
        return responses
            # if span["event_type"]=="LLMCompletionEndEvent":
            #     response = span.get("response", "")
            #     return response
    
    
            # if span["event_type"]=="LLMPredictEndEvent":
            #     response = span.get("output", "")
            #     return response
            # if span["event_type"]=="SynthesizeEndEvent":
            #     response = span.get("response", "")
            #     return response

    def get_system_prompt(data):
        for span in data:
            if span["event_type"]=="LLMChatStartEvent":
                response = span.get("messages", "")
                response = response[0]
                return response
            if span["event_type"]=="LLMCompletionStartEvent":
                response = span.get("prompt", "")
                return response

    # Process traces
    if "traces" in data:
        with open("/Users/ritikagoel/workspace/llamaindex_sub_query_issue/RagaAI-Catalyst/llama_final_result.json", 'w') as f:
            json.dump(data, f, indent=2,default=str)        
        prompt = get_prompt(data["traces"])
        context = get_context(data["traces"])
        response = get_response(data["traces"])
        system_prompt = get_system_prompt(data["traces"])

    trace_data["data"]["prompt"] = prompt
    trace_data["data"]["context"] = context
    trace_data["data"]["response"] = response
    trace_data["data"]["system_prompt"] = system_prompt
    return [trace_data]