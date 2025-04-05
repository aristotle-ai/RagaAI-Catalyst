import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from examples.test_utils.get_trace_data import (
    run_command,
    extract_information,
    load_trace_data
)

from examples.test_utils.get_components import (
    get_component_structure_and_sequence
)

@pytest.mark.parametrize("model, provider, async_llm, syntax", [
    ("gpt-4o-mini", "openai", False, "chat"),
    ("gemini-1.5-flash", "google_genai", False, "chat"),
    # ("gemini-1.5-flash", "google_vertexai", False, "chat"),
    # ("gpt-3.5-turbo", "azure", False, "chat"),
    # ("gemini-1.5-flash", "anthropic", False, "chat"),
])
def test_research_assistant(model: str, provider: str, async_llm: bool, syntax: str):
    # Build the command to run research_assistant.py with the provided arguments
    command = f'python research_assistant.py --model {model} --provider {provider} --async_llm {async_llm} --syntax {syntax}'
    cwd = os.path.dirname(os.path.abspath(__file__))  # Use the current directory
    output = run_command(command, cwd=cwd)
    
    # Extract trace file location from logs
    locations = extract_information(output)

    # Load and validate the trace data
    data = load_trace_data(locations)

    # Get component structure and sequence
    component_sequence = get_component_structure_and_sequence(data)

    # Print component sequence
    print("Component sequence:", component_sequence)

    # Validate component sequence
    assert len(component_sequence) >= 2, f"Expected at least 2 components, got {len(component_sequence)}"