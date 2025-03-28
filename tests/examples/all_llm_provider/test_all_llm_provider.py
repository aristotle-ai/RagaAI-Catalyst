import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.test_utils.get_trace_data import (
    run_command,
    extract_information,
    load_trace_data
)

from examples.test_utils.get_components import (
    get_component_structure_and_sequence
)

@pytest.mark.parametrize("provider, model, async_mode", [
    # OpenAI
    ("openai", "gpt-4o-mini", True),
    ("openai", "gpt-4o-mini", False),
    
    # # Anthropic
    # ("anthropic", "claude-3-opus-20240229", True),
    # ("anthropic", "claude-3-opus-20240229", False),
    
    # # Groq
    # ("groq", "llama3-8b-8192", True),
    # ("groq", "llama3-8b-8192", False),
    
    # LiteLLM
    ("litellm", "gpt-4o-mini", True),
    ("litellm", "gpt-4o-mini", False),
    
    # Azure
    ("azure", "azure-gpt-4o-mini", True),
    ("azure", "azure-gpt-4o-mini", False),
    
    # Google
    ("google", "gemini-1.5-flash", True),
    ("google", "gemini-1.5-flash", False),
    
    # Chat Google
    ("chat_google", "gemini-1.5-flash", True),
    ("chat_google", "gemini-1.5-flash", False),
])

def test_all_llm_provider(provider: str, model: str, async_mode: bool):
    # Build the command to run all_llm_provider.py with the provided arguments
    command = f'python all_llm_provider.py --model {model} --provider {provider} --async_llm {async_mode}'
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
    assert len(component_sequence) == 1, f"Expected 1 component, got {len(component_sequence)}"


    