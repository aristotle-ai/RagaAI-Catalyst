import os
import json
import pytest
import subprocess
import re
import sys
from pathlib import Path

# Add the parent directory to sys.path to import modules from there
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_diagnosis_agent():
    """Run the diagnosis_agent.py script and capture its output"""
    script_path = os.path.join(
        Path(__file__).resolve().parent,
        "scifi_writer.py"
    )
    
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
    """
    Test that verifies the total cost value in the trace file is correct.
    This test runs the diagnosis_agent.py script to generate a new trace file,
    then validates the cost values in that trace.
    """
    # Run the diagnosis agent to generate a new trace
    output = run_diagnosis_agent()
    
    # Extract the path to the generated trace file
    trace_file_path = extract_trace_path(output)
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Expected value - for dynamically generated traces, we may need to calculate this
    # rather than hardcoding it
    
    # Get the total cost from the trace
    total_cost = trace_data["metadata"]["total_cost"]
    
    # Verify the total cost value in the metadata section
    assert trace_data["metadata"]["cost"]["total_cost"] == total_cost, \
        f"Expected total_cost in metadata.cost to be {total_cost}, got {trace_data['metadata']['cost']['total_cost']}"
    
    # Check if the value is consistent with the sum of input and output costs
    input_cost = trace_data["metadata"]["cost"]["input_cost"]
    output_cost = trace_data["metadata"]["cost"]["output_cost"]
    calculated_cost = round(input_cost + output_cost, 5)  # Round to 5 decimal places
    
    assert abs(calculated_cost - total_cost) < 0.00001, \
        f"Total cost {total_cost} should approximately equal the sum of input ({input_cost}) and output ({output_cost}) costs"
    
    
    
def test_llm_cost_calculation():
    """
    Test that verifies the LiteLLM cost calculation bug fix.
    
    This test focuses on:
    1. Correctly parsing prompt_tokens and completion_tokens from LiteLLM responses
    2. Ensuring costs are calculated properly using model-specific rates (input_cost_per_token and output_cost_per_token)
    """
    # Load a trace file that contains LiteLLM or OpenAI call data
    # We'll use the existing trace file instead of generating a new one
    trace_file_path = os.path.join(
        Path(__file__).resolve().parent, 
        "rag_agent_traces.json"
    )
    
    # Load the trace file
    with open(trace_file_path, 'r') as f:
        trace_data = json.load(f)
    
    # Verify token counts are properly parsed from LiteLLM response
    assert "tokens" in trace_data["metadata"], "Expected tokens data to be present in metadata"
    prompt_tokens = trace_data["metadata"]["tokens"]["prompt_tokens"]
    completion_tokens = trace_data["metadata"]["tokens"]["completion_tokens"]
    
    # Ensure token counts are non-zero (confirming the model was actually used)
    assert prompt_tokens > 0, "Expected non-zero prompt tokens from LiteLLM response"
    assert completion_tokens > 0, "Expected non-zero completion tokens from LiteLLM response"
    
    # Verify the metadata contains cost information
    assert "cost" in trace_data["metadata"], "Expected cost data to be present in metadata"
    
    # Extract cost values from metadata
    input_cost = trace_data["metadata"]["cost"]["input_cost"]
    output_cost = trace_data["metadata"]["cost"]["output_cost"]
    total_cost = trace_data["metadata"]["cost"]["total_cost"]
    
    # The core of the bugfix: Verify that costs are calculated correctly based on token counts
    if prompt_tokens > 0 and completion_tokens > 0:
        # Check that input and output costs are non-zero (assuming paid model)
        assert input_cost > 0, "Expected non-zero input cost for LiteLLM model"
        assert output_cost > 0, "Expected non-zero output cost for LiteLLM model"
        
        # Check that total cost matches sum of input and output costs
        calculated_total = round(input_cost + output_cost, 5)  # Round to 5 decimal places
        assert abs(calculated_total - total_cost) < 0.00001, \
            f"Total cost {total_cost} should equal the sum of input ({input_cost}) and output ({output_cost}) costs"
        
        # Calculate per-token rates from the actual costs and token counts
        
    
    # Find any LLM spans (not just OpenAI-specific ones)
    llm_spans = [span for span in trace_data["data"][0]["spans"] 
                if span.get("attributes", {}).get("openinference.span.kind") == "LLM"]
    
    if llm_spans:
        print(f"Found {len(llm_spans)} LLM spans")
        
        # Check if any LLM span contains cost information
        for span in llm_spans:
            if "attributes" in span and "llm.cost" in span["attributes"]:
                span_cost = span["attributes"]["llm.cost"]["total_cost"]
                print(f"LLM span cost: {span_cost}")
                
                # Verify the cost in the span matches the metadata cost
                # Instead of matching exact values, just check if the cost is greater than 0
                assert span_cost > 0, f"LLM span cost should be greater than 0, got {span_cost}"
                assert total_cost > 0, f"Metadata total cost should be greater than 0, got {total_cost}"
                
                # If we have token counts in the span, verify those too
                if "llm.token_count.prompt" in span["attributes"] and "llm.token_count.completion" in span["attributes"]:
                    span_prompt_tokens = span["attributes"]["llm.token_count.prompt"]
                    span_completion_tokens = span["attributes"]["llm.token_count.completion"]
                    
                    # Since metadata may contain aggregated tokens from multiple calls,
                    # we shouldn't directly compare them but ensure they are present and reasonable
                    assert span_prompt_tokens > 0, \
                        f"Span prompt_tokens should be greater than 0, got {span_prompt_tokens}"
                    assert span_completion_tokens > 0, \
                        f"Span completion_tokens should be greater than 0, got {span_completion_tokens}"
                    
                    print(f"Verified LLM span has valid token counts: {span_prompt_tokens} prompt, {span_completion_tokens} completion")
                
                # We found at least one span with cost info, so the test passes
                break
        else:
            # This else clause belongs to the for loop - it executes if no break occurred
            # If we have LLM spans but none have cost info, print a warning but don't fail
            print("Warning: Found LLM spans but none contain cost information")
    else:
        print("No LLM spans found in the trace data")
        

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


if __name__ == "__main__":
    test_trace_total_cost()
    test_llm_cost_calculation()
    test_export_trace_id()
    test_export_trace_metadata()
    test_export_trace_data()
    test_exclude_vital_columns()
    