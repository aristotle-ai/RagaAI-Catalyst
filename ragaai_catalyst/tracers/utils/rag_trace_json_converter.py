import json
from litellm import model_cost
import logging
import os
import re
from datetime import datetime
import tiktoken

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)

def rag_trace_json_converter(input_trace, custom_model_cost, trace_id, user_details, tracer_type):
    trace_aggregate = {}
    
    def get_prompt(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] in ["ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI"]:
                    return span["attributes"].get("llm.input_messages.1.message.content")

                elif span["name"] == "LLMChain":
                    return json.loads(span["attributes"].get("input.value", "{}")).get("question")

                elif span["name"] == "RetrievalQA":
                    return span["attributes"].get("input.value")
                
                elif span["name"] == "VectorStoreRetriever":
                    return span["attributes"].get("input.value")
                
        return None
    
    def get_response(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] in ["ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI"]:
                    return span["attributes"].get("llm.output_messages.0.message.content")

                elif span["name"] == "LLMChain":
                    return json.loads(span["attributes"].get("output.value", ""))

                elif span["name"] == "RetrievalQA":
                    return span["attributes"].get("output.value")

        return None
    
    def get_context(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "VectorStoreRetriever":
                    return span["attributes"].get("retrieval.documents.1.document.content")
        return None
        
    prompt = get_prompt(input_trace)
    response = get_response(input_trace)
    context = get_context(input_trace)
    
    if tracer_type == "rag/langchain":
        trace_aggregate["tracer_type"] = "langchain"
    else:
        trace_aggregate["tracer_type"] = "llamaindex"

    trace_aggregate['trace_id'] = trace_id
    trace_aggregate['session_id'] = None
    trace_aggregate["metadata"] = user_details.get("trace_user_detail", {}).get("metadata")

    #dummy data need to be fetched
    trace_aggregate["pipeline"] = {
        'llm_model': 'gpt-4o-mini', 
        'vector_store': 'faiss',
        'embed_model': 'text-embedding-ada-002'
        }
    
    trace_aggregate["data"] = {}
    trace_aggregate["data"]["prompt"] = prompt
    trace_aggregate["data"]["response"] = response
    trace_aggregate["data"]["context"] = context
    
    if tracer_type == "rag/langchain":
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost, prompt, response)
    else:
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost)
    
    trace_aggregate["metadata"].update(additional_metadata)
    return trace_aggregate, additional_metadata

def get_additional_metadata(spans, custom_model_cost, model_cost_dict, prompt="", response=""):
    additional_metadata = {}
    additional_metadata["cost"] = 0.0
    additional_metadata["tokens"] = {}
    try:
        for span in spans:
            if span["name"] in ["ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI"]:
                start_time = datetime.fromisoformat(span.get("start_time", "")[:-1])  # Remove 'Z' and parse
                end_time = datetime.fromisoformat(span.get("end_time", "")[:-1])    # Remove 'Z' and parse
                additional_metadata["latency"] = (end_time - start_time).total_seconds()
                additional_metadata["model_name"] = span["attributes"].get("llm.model_name", "").replace("models/", "")
                additional_metadata["model"] = additional_metadata["model_name"]
                try:
                    additional_metadata["tokens"]["prompt"] = span["attributes"]["llm.token_count.prompt"]

                except:
                    logger.warning("Warning: prompt token not found. using tiktoken to get tokens.")
                    additional_metadata["tokens"]["prompt"] = num_tokens_from_messages(
                        model=additional_metadata["model_name"],
                        message=prompt
                    )
                try:
                    additional_metadata["tokens"]["completion"] = span["attributes"]["llm.token_count.completion"]
                except:
                    logger.warning("Warning: completion token not found. using tiktoken to get tokens.")
                    additional_metadata["tokens"]["completion"] = num_tokens_from_messages(
                        model=additional_metadata["model_name"],
                        message=response
                    )
                additional_metadata["tokens"]["total"] = additional_metadata["tokens"]["prompt"] + additional_metadata["tokens"]["completion"]

    except Exception as e:
        logger.error(f"Error getting additional metadata: {str(e)}")
    
    try:
        if custom_model_cost.get(additional_metadata.get('model_name')):
            model_cost_data = custom_model_cost[additional_metadata.get('model_name')]
        else:
            model_cost_data = model_cost_dict.get(additional_metadata.get('model_name'))
        if 'tokens' in additional_metadata and all(k in additional_metadata['tokens'] for k in ['prompt', 'completion']):
            prompt_cost = additional_metadata["tokens"]["prompt"]*model_cost_data.get("input_cost_per_token", 0)
            completion_cost = additional_metadata["tokens"]["completion"]*model_cost_data.get("output_cost_per_token", 0)
            additional_metadata["cost"] = prompt_cost + completion_cost 
            additional_metadata["total_cost"] = additional_metadata["cost"]
            additional_metadata["total_latency"] = additional_metadata["latency"]
            additional_metadata["prompt_tokens"] = additional_metadata["tokens"]["prompt"]
            additional_metadata["completion_tokens"] = additional_metadata["tokens"]["completion"]
    except Exception as e:
        logger.error(f"Error getting model cost data: {str(e)}")
    
    try:
        additional_metadata.pop("tokens", None)
    except Exception as e:
        logger.error(f"Error removing tokens from additional metadata: {str(e)}")

    return additional_metadata

def num_tokens_from_messages(model, message):
    # GPT models
    if re.match(r'^gpt-', model):
        """Check if the model is any GPT model (pattern: ^gpt-)
        This matches any model name that starts with 'gpt-'
        """
        def num_tokens_from_string(string: str, encoding_name: str) -> int:
            """Returns the number of tokens in a text string."""
            encoding = tiktoken.get_encoding(encoding_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        
        if re.match(r'^gpt-4o.*', model):
            """Check for GPT-4 Optimized models (pattern: ^gpt-4o.*)
            Examples that match:
            - gpt-4o
            - gpt-4o-mini
            - gpt-4o-2024-08-06
            The .* allows for any characters after 'gpt-4o'
            """
            encoding_name = "o200k_base"
            return num_tokens_from_string(message, encoding_name)
        
        elif re.match(r'^gpt-(4|3\.5).*', model):
            """Check for GPT-4 and GPT-3.5 models (pattern: ^gpt-(4|3\.5).*)
            Uses cl100k_base encoding for GPT-4 and GPT-3.5 models
            Examples that match:
            - gpt-4
            - gpt-4-turbo
            - gpt-4-2024-08-06
            - gpt-3.5-turbo
            - gpt-3.5-turbo-16k
            """
            encoding_name = "cl100k_base"
            return num_tokens_from_string(message, encoding_name)
        
        else:
            """Default case for any other GPT models
            Uses o200k_base encoding as the default tokenizer
            """
            return num_tokens_from_string(message, encoding_name="o200k_base")
        

    # Gemini models 
    elif re.match(r'^gemini-', model):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=GOOGLE_API_KEY)

        response = client.models.count_tokens(
                model=model,
                contents=message,
            )
        return response.total_tokens