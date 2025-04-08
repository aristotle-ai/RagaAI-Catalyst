import json
from litellm import model_cost
import logging
import os
import tiktoken
from datetime import datetime

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)

def rag_trace_json_converter(input_trace, custom_model_cost, trace_id, user_details, tracer_type):
    trace_aggregate = {}

    def get_prompt(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "ChatOpenAI":
                    return span["attributes"]["llm.input_messages.1.message.content"]

                elif span["name"] == "LLMChain":
                     return json.loads(span["attributes"]["input.value"])["question"]

                elif span["name"] == "RetrievalQA":
                     return span["attributes"]["input.value"]
                 
                elif span["name"] == "VectorStoreRetriever":
                     return span["attributes"]["input.value"]
        elif tracer_type == "rag/llamaindex":
            for span in input_trace:
                if span["name"] == "BaseQueryEngine.query":
                    return span["attributes"]["input.value"]
                elif "query_bundle" in span["attributes"].get("input.value", ""):
                    try:
                        query_data = json.loads(span["attributes"]["input.value"])
                        if "query_bundle" in query_data:
                            return query_data["query_bundle"]["query_str"]
                    except json.JSONDecodeError:
                        logger.error("Failed to parse query_bundle JSON")
            return None
    
    def get_response(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "ChatOpenAI":
                     return span["attributes"]["llm.output_messages.0.message.content"]
                elif span["name"] == "LLMChain":
                     return json.loads(span["attributes"]["output.value"])
                elif span["name"] == "RetrievalQA":
                    return span["attributes"]["output.value"]
        elif tracer_type == "rag/llamaindex":
            for span in input_trace:
                if span["name"] == "BaseQueryEngine.query":
                    return span["attributes"]["output.value"]
            return None
    
    def get_context(input_trace):
        if tracer_type == "rag/langchain":
            for span in input_trace:
                if span["name"] == "VectorStoreRetriever":
                    return span["attributes"]["retrieval.documents.1.document.content"]
        elif tracer_type == "rag/llamaindex":
            for span in input_trace:
                if span["name"] == "BaseRetriever.retrieve":
                    return span["attributes"]["retrieval.documents.1.document.content"]
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
    trace_aggregate["metadata"] = user_details["trace_user_detail"]["metadata"]

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
        additional_metadata = get_additional_metadata(input_trace, custom_model_cost, model_cost)
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
            if span["name"] in ["ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI","OpenAI.chat"]:
                start_time = datetime.fromisoformat(span.get("start_time", "")[:-1])  # Remove 'Z' and parse
                end_time = datetime.fromisoformat(span.get("end_time", "")[:-1])    # Remove 'Z' and parse
                additional_metadata["latency"] = (end_time - start_time).total_seconds()
                additional_metadata["model_name"] = span["attributes"].get("llm.model_name", "").replace("models/", "")
                additional_metadata["model"] = additional_metadata["model_name"]
                try:
                    additional_metadata["tokens"]["prompt"] = span["attributes"]["llm.token_count.prompt"]

                except:
                    logging.warning("Warning: prompt token not found. using tiktoken to get tokens.")
                    additional_metadata["tokens"]["prompt"] = num_tokens_from_messages(
                        model=additional_metadata["model_name"],
                        prompt_messages=[{"role": "user", "content": prompt}]
                    )
                try:
                    additional_metadata["tokens"]["completion"] = span["attributes"]["llm.token_count.completion"]
                except:
                    logging.warning("Warning: completion token not found. using tiktoken to get tokens.")
                    additional_metadata["tokens"]["completion"] = num_tokens_from_messages(
                        model=additional_metadata["model_name"],
                        response_message={"role": "assistant", "content": response}
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

def num_tokens_from_messages(model, prompt_messages=None, response_message=None):
    # GPT models
    if model.startswith("gpt-"):
        
        """Calculate the number of tokens used by either prompt messages or response message.
        
        Args:
            model: The model name to use for token calculation
            prompt_messages: List of prompt messages (mutually exclusive with response_message)
            response_message: Response message from the assistant (mutually exclusive with prompt_messages)
        
        Returns:
            int: Number of tokens in either the prompt or completion
        
        Raises:
            ValueError: If both prompt_messages and response_message are provided or if neither is provided
        """
        if (prompt_messages is not None and response_message is not None) or (prompt_messages is None and response_message is None):
            raise ValueError("Exactly one of prompt_messages or response_message must be provided")

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logging.warning("Warning: model not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")
        
        if model in {
            "gpt-3.5-turbo-0125",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18"
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif "gpt-3.5-turbo" in model:
            logging.warning("Warning: gpt-3.5-turbo may update over time. Using gpt-3.5-turbo-0125.")
            return num_tokens_from_messages(model="gpt-3.5-turbo-0125", 
                                        prompt_messages=prompt_messages, response_message=response_message)
        elif "gpt-4o-mini" in model:
            logging.warning("Warning: gpt-4o-mini may update over time. Using gpt-4o-mini-2024-07-18.")
            return num_tokens_from_messages(model="gpt-4o-mini-2024-07-18",
                                        prompt_messages=prompt_messages, response_message=response_message)
        elif "gpt-4o" in model:
            logging.warning("Warning: gpt-4o may update over time. Using gpt-4o-2024-08-06.")
            return num_tokens_from_messages(model="gpt-4o-2024-08-06",
                                        prompt_messages=prompt_messages, response_message=response_message)
        elif "gpt-4" in model:
            logging.warning("Warning: gpt-4 may update over time. Using gpt-4-0613.")
            return num_tokens_from_messages(model="gpt-4-0613",
                                        prompt_messages=prompt_messages, response_message=response_message)
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}."""
            )
        
        total_tokens = 0
        
        if prompt_messages is not None:
            for message in prompt_messages:
                num_tokens = tokens_per_message
                for key, value in message.items():
                    token_count = len(encoding.encode(str(value)))
                    num_tokens += token_count
                    if key == "name":
                        num_tokens += tokens_per_name
                total_tokens += num_tokens
            return total_tokens
        
        else:  # response_message is not None
            num_tokens = tokens_per_message
            if isinstance(response_message, dict):
                message = response_message
            else:
                message = {"role": "assistant", "content": response_message}
                
            for key, value in message.items():
                token_count = len(encoding.encode(str(value)))
                num_tokens += token_count
                if key == "name":
                    num_tokens += tokens_per_name
            
            return num_tokens + 3  # Adding the assistant message prefix tokens

    # Gemini models 
    elif model.startswith("gemini-"):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=GOOGLE_API_KEY)

        if prompt_messages is not None:
            response = client.models.count_tokens(
                model=model,
                contents=prompt_messages,
            )
            return response.total_tokens
        else:
            response = client.models.count_tokens(
                model=model,
                contents=response_message,
            )
            return response.total_tokens        