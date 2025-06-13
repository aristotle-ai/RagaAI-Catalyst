import os
import json
import pytest
import subprocess
import re
import sys
from pathlib import Path
from importlib import resources


# Add the parent directory to sys.path to import modules from there
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_llamaindex_rag():
    script_path = os.path.join(
        Path(__file__).resolve().parent,
        "llamaindex_rag.py"
    )
    print(script_path)
    
    result = subprocess.run(
        [sys.executable, script_path],
        check=True,
        capture_output=True,
        text=True
    )
    return result.stdout + result.stderr

def extract_trace_path(output):
    """Extract the trace file path from the script output"""
    # Pattern to match the trace file path in the output
    pattern = re.compile(r"Trace saved to (.*\.json)")
    match = pattern.search(output)
    
    if match:
        return match.group(1)
    
    # Fallback pattern if the above doesn't match
    pattern = re.compile(r"Submitting new upload task for file: (.*\.json)")
    match = pattern.search(output)
    
    if match:
        return match.group(1)
    
    raise ValueError("Could not find trace file path in output")


def test_trace_total_cost():
    output = run_llamaindex_rag()
    trace_file_path = extract_trace_path(output)
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    # trace_file_path = os.path.join(
    #     Path(__file__).resolve().parent, 
    #     "rag_agent_traces.json"
    # )
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    total_cost = trace_data["metadata"]["total_cost"]

    assert trace_data["metadata"]["cost"]["total_cost"] == total_cost, \
        f"Expected total_cost in metadata.cost to be {total_cost}, got {trace_data['metadata']['cost']['total_cost']}"
    
def test_span_cost_consistency():
    # output = run_llamaindex_rag()
    # trace_file_path = extract_trace_path(output)
    # with open(trace_file_path, 'r') as f:
    #     trace_data = json.load(f)
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    total_cost = trace_data["metadata"]["total_cost"]
    openai_spans = [span for span in trace_data["data"][0]["spans"] if span["name"] in ["ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI", "OpenAI", "ChatOpenAI_LangchainOpenAI", "ChatOpenAI_ChatModels",
                                "ChatVertexAI", "VertexAI", "ChatLiteLLM", "ChatBedrock", "AzureChatOpenAI", "ChatAnthropicVertex"]]
    if openai_spans:
        openai_span = openai_spans[0]
        if "llm.cost" in openai_span["attributes"]:
            openai_cost = openai_span["attributes"]["llm.cost"]["total_cost"]
            assert abs(openai_cost - total_cost) < 0.00001, \
                f"Expected OpenAI span cost to be close to {total_cost}, got {openai_cost}"

def test_cost_consistency():
    # output = run_llamaindex_rag()
    # trace_file_path = extract_trace_path(output)
    # with open(trace_file_path, 'r') as f:
    #     trace_data = json.load(f)

    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    total_cost = trace_data["metadata"]["total_cost"]
    input_cost = trace_data["metadata"]["cost"]["input_cost"]
    output_cost = trace_data["metadata"]["cost"]["output_cost"]
    calculated_cost = round(input_cost + output_cost, 5)  # Round to 5 decimal places
    
    assert abs(calculated_cost - total_cost) < 0.00001, \
        f"Total cost {total_cost} should approximately equal the sum of input ({input_cost}) and output ({output_cost}) costs"
    
def test_llm_non_zero_prompt_tokens():
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)

    assert "tokens" in trace_data["metadata"], "Expected tokens data to be present in metadata"
    prompt_tokens = trace_data["metadata"]["tokens"]["prompt_tokens"]

    assert prompt_tokens > 0, "Expected non-zero prompt tokens from LiteLLM response"

def test_llm_non_zero_completion_tokens():
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)

    assert "tokens" in trace_data["metadata"], "Expected tokens data to be present in metadata"
    completion_tokens = trace_data["metadata"]["tokens"]["completion_tokens"]

    assert completion_tokens > 0, "Expected non-zero prompt tokens from LiteLLM response"


def test_export_all_trace_columns():
    """
    Test that exports all columns from the trace file without any filtering.
    This function loads the rag_agent_traces.json file and prints out the complete
    structure to allow for full inspection of all data elements.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    

    for key in trace_data.keys():
        print(f"  - {key}")
    
    
    for key in trace_data.get("metadata", {}).keys():
        print(f"  - {key}")
    
    spans = trace_data.get("data", [{}])[0].get("spans", [])
    # print(f"Found {len(spans  spans")
    
    for i, span in enumerate(spans):
        attributes = span.get("attributes", {})
        for attr_key in attributes.keys():
            attr_value = attributes[attr_key]
            # For display purposes, truncate very long values
            if isinstance(attr_value, str) and len(attr_value) > 100:
                attr_value = attr_value[:100] + "... [truncated]"
            print(f"    - {attr_key}: {attr_value}")

    tokens = trace_data.get("metadata", {}).get("tokens", {})
    costs = trace_data.get("metadata", {}).get("cost", {})
    
    for k, v in tokens.items():
        print(f"  - {k}: {v}")
    
    for k, v in costs.items():
        print(f"  - {k}: {v}")
    
    assert "id" in trace_data, "Trace data should have an 'id' field"
    assert "metadata" in trace_data, "Trace data should have a 'metadata' field"
    assert "data" in trace_data, "Trace data should have a 'data' field"
    assert len(spans) > 0, "Trace data should contain at least one span"


def test_exclude_vital_columns():
    """
    Test that verifies vital columns are excluded while masking.
    This test checks that fields like model_name, cost, latency, span_id, trace_id, etc.
    are not present in the exported trace data.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Define the vital columns that should be excluded
    vital_columns = [
        "model_name",
        "cost",
        "latency",
        "span_id",
        "trace_id"
    ]
    
    # Check that each vital column is not present in the trace data
    for column in vital_columns:
        assert column not in trace_data, f"Expected {column} to be excluded from trace data"

def test_export_trace_id():
    """
    Test that exports top-level keys from the trace file and checks for 'id' field.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Print which test is running
    print("\nTesting for 'id' field:")
    # Only print if the key exists
    if "id" in trace_data:
        print(f"  - 'id' field found: {trace_data['id'][:10]}...")
    else:
        print("  - 'id' field NOT found")
    
    # Assert that we have loaded data successfully
    assert "id" in trace_data, "Trace data should have an 'id' field"

def test_export_trace_metadata():
    """
    Test that exports top-level keys from the trace file and checks for 'metadata' field.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Print which test is running
    print("\nTesting for 'metadata' field:")
    # Only print if the key exists
    if "metadata" in trace_data:
        print(f"  - 'metadata' field found: {str(trace_data['metadata'])[:10]}...")
    else:
        print("  - 'metadata' field NOT found")
    
    # Assert that we have loaded data successfully
    assert "metadata" in trace_data, "Trace data should have a 'metadata' field"

def test_export_trace_data():
    """
    Test that exports top-level keys from the trace file and checks for 'data' field.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Print which test is running
    print("\nTesting for 'data' field:")
    # Only print if the key exists
    if "data" in trace_data:
        print(f"  - 'data' field found: {str(trace_data['data'])[:10]}...")
    else:
        print("  - 'data' field NOT found")
    
    # Assert that we have loaded data successfully
    assert "data" in trace_data, "Trace data should have a 'data' field"
    

def test_prompt_value():
    """
    Test that verifies the prompt extracted from the trace matches the expected value.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Create an instance of the class containing rag_trace_json_converter
    class TraceConverter:
        def __init__(self):
            pass
        
        def extract_prompt(self, input_trace):
            try:
                tracer_type = input_trace.get("tracer_type")
                spans = input_trace.get("data", [{}])[0].get("spans", [])
                if tracer_type == "langchain":
                    return self._extract_langchain_prompt(spans)
                elif tracer_type == "llamaindex":
                    return self._extract_llamaindex_prompt(spans)
                return ""
            except Exception as e:
                return ""

        def _extract_langchain_prompt(self, spans):
            for span in spans:
                attributes = span.get("attributes", {})
                prompt = self._get_input_message_prompt(attributes)
                if prompt:
                    return prompt
                prompt = self._get_llm_prompts(attributes)
                if prompt:
                    return prompt
                prompt = self._get_specific_span_prompt(span)
                if prompt:
                    return prompt
            return ""

        def _get_input_message_prompt(self, attributes):
            for key, value in attributes.items():
                if key.startswith("llm.input_messages.") and key.endswith(".message.role") and value == "user":
                    message_num = key.split(".")[2]
                    content_key = f"llm.input_messages.{message_num}.message.content"
                    return attributes.get(content_key, "")
            return ""

        def _get_llm_prompts(self, attributes):
            for key, value in attributes.items():
                if key.startswith("llm.prompts") and isinstance(value, list):
                    for message in value:
                        if isinstance(message, str) and "Human:" in message:
                            return message.split("Human:")[1].strip()
            return ""

        def _get_specific_span_prompt(self, span):
            if not span.get("name", "").startswith(("LLMChain", "RetrievalQA", "VectorStoreRetriever")):
                return ""
            input_value = span.get("attributes", {}).get("input.value", "")
            if span["name"] == "LLMChain":
                try:
                    return json.loads(input_value).get("question", "")
                except json.JSONDecodeError:
                    return ""
            return input_value

        def _extract_llamaindex_prompt(self, spans):
            for span in spans:
                if span.get("name", "").startswith("BaseQueryEngine.query"):
                    return span.get("attributes", {}).get("input.value", "")
                attributes = span.get("attributes", {})
                if "query_bundle" in attributes.get("input.value", ""):
                    try:
                        query_data = json.loads(attributes["input.value"])
                        return query_data.get("query_bundle", {}).get("query_str", "")
                    except json.JSONDecodeError:
                        pass
            return ""

        def rag_trace_json_converter(self, input_trace):
            prompt = self.extract_prompt(input_trace)
            return prompt

    converter = TraceConverter()
    prompt = converter.rag_trace_json_converter(trace_data)
    
    # Assert that the prompt matches the expected value
    expected_prompt = "what is this paper about?"
    assert prompt == expected_prompt, f"Expected prompt to be '{expected_prompt}', but got '{prompt}'"

def test_response_value():
    """
    Test that verifies the response extracted from the trace matches the expected value.
    """
    # Load the trace file
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Create an instance of the class containing extract_response
    class ResponseExtractor:
        def __init__(self):
            pass
            
        def extract_response(self, input_trace):
            spans = input_trace.get("data", [{}])[0].get("spans", [])
            tracer_type = input_trace.get("tracer_type")
            if tracer_type == "langchain":
                for span in spans:
                    attributes = span.get("attributes", {})

                    # Check llm.output_messages
                    for key, value in attributes.items():
                        if key.startswith("llm.output_messages.") and key.endswith(".message.content"):
                            return value

                    for key, value in attributes.items():
                        if key.startswith("output.value"):
                            try:
                                output_json = json.loads(value)
                                generations = output_json.get("generations", [])
                                if generations and isinstance(generations[0], list) and generations[0]:
                                    return generations[0][0].get("text", "")
                            except json.JSONDecodeError:
                                continue

                    # Fallback to specific spans
                    if span.get("name") in ["LLMChain", "RetrievalQA", "VectorStoreRetriever"]:
                        output_value = attributes.get("output.value", "")
                        if span["name"] == "LLMChain":
                            try:
                                return json.loads(output_value) if output_value else ""
                            except json.JSONDecodeError:
                                continue
                        return output_value

            elif tracer_type == "llamaindex":
                for span in spans:
                    if span.get("name", "").startswith("BaseQueryEngine.query"):
                        return span.get("attributes", {}).get("output.value", "")

            print(f"No response found in {tracer_type} trace")
            return ""


    extractor = ResponseExtractor()
    response = extractor.extract_response(trace_data)
    
    # Assert that the response is not empty
    print(response)
    assert response, "Response should not be empty"
    assert isinstance(response, str), f"Response should be a string, got {type(response)}"

def test_context_value():
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    class ContextExtractor:
        def __init__(self):
            pass
            
        def extract_context(self, input_trace):
            try:
                spans = input_trace.get("data", [{}])[0].get("spans", [])
                tracer_type = input_trace.get("tracer_type")
                for span in spans:
                    if tracer_type == "langchain" and span.get("name").startswith("VectorStoreRetriever"):
                        return span.get("attributes", {}).get("retrieval.documents.1.document.content", "")
                    if tracer_type == "llamaindex" and span.get("name").startswith("BaseRetriever.retrieve"):
                        return span.get("attributes", {}).get("retrieval.documents.1.document.content", "")
                    if span.get("name").startswith("CustomContextSpan"):
                        return span.get("attributes", {}).get("input.value", "")

                print(f"No context found in {tracer_type} trace")
                return ""

            except Exception as e:
                print(f"Error extracting context: {str(e)}")
                return ""

    extractor = ContextExtractor()
    context = extractor.extract_context(trace_data)
    
    # Assert that the context is not empty
    print(context)
    assert context, "Context should not be empty"
    assert isinstance(context, str), f"Context should be a string, got {type(context)}"
    
    # Assert that the context matches the expected value
    expected_context = "An Introduction to Artificial Intelligence\nArtificial Intelligence (AI) is the simulation of human intelligence in machines that are designed to\nthink and act like humans. These machines can perform tasks such as learning, reasoning,\nproblem-solving, and understanding language.\nKey Areas of AI:\n1. Machine Learning: Algorithms that allow computers to learn from and make predictions based on\ndata.\n2. Natural Language Processing (NLP): Enabling machines to understand and interpret human\nlanguage.\n3. Robotics: The design and use of robots that can perform tasks autonomously.\n4. Computer Vision: Giving machines the ability to interpret and make decisions based on visual\ninputs.\n5. Neural Networks: Modeled after the human brain, used in deep learning to recognize patterns.\nAI Applications:\n1. Healthcare: AI is transforming the medical field by enabling better diagnostics and personalized\nmedicine.\n2. Autonomous Vehicles: AI powers self-driving cars and enhances transportation safety.\n3. Finance: AI assists in fraud detection, algorithmic trading, and customer service through chatbots.\n4. Education: Personalized learning experiences and AI-powered tutors are improving the education\nsystem.\nChallenges in AI:\n1. Ethical Considerations: The impact of AI on employment and privacy is a major concern."
    assert context == expected_context, f"Expected context to be '{expected_context}', but got '{context}'"

if __name__ == "__main__":
    test_trace_total_cost()
    test_span_cost_consistency()
    test_cost_consistency()
    test_llm_non_zero_prompt_tokens()
    test_llm_non_zero_completion_tokens()
    test_export_all_trace_columns()
    test_exclude_vital_columns()
    test_export_trace_id()
    test_export_trace_metadata()
    test_export_trace_data()
    test_prompt_value()
    test_response_value()
    test_context_value()

    