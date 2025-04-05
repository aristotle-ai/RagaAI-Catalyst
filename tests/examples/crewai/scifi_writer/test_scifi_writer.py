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

@pytest.mark.parametrize("info", [
    ("testing-scifi-writer"),
])
def test_scifi_writer(info: str):
    # Build the command to run scifi_writer.py with the provided arguments
    command = f'python scifi_writer.py --info {info}'
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

    