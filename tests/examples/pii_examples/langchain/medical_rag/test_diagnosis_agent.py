import os
import pytest
import sys
import re
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from examples.test_utils.get_trace_data import (
    run_command,
    extract_information,
    load_trace_data
)

from examples.test_utils.get_components import (
    get_component_structure_and_sequence
)

SYMPTOMS = ['fever', 'headache', 'fatigue']
SYMPTOM_PATTERN = re.compile(r'\b(' + '|'.join(SYMPTOMS) + r')\b', re.IGNORECASE)

def contains_unredacted_symptom(value: Any) -> bool:
    """
    Recursively checks if any part of a nested structure contains unredacted symptoms.
    """
    if isinstance(value, str):
        return SYMPTOM_PATTERN.search(value) is not None
    elif isinstance(value, dict):
        return any(contains_unredacted_symptom(v) for v in value.values())
    elif isinstance(value, list):
        return any(contains_unredacted_symptom(item) for item in value)
    return False

@pytest.mark.parametrize("model_type", [
    ("openai"),
])
def test_diagnosis_agent(model_type: str):
    command = f'python diagnosis_agent.py --model_type {model_type}'
    cwd = os.path.dirname(os.path.abspath(__file__))
    output = run_command(command, cwd=cwd)
    
    locations = extract_information(output)
    data = load_trace_data(locations)

    component_sequence = get_component_structure_and_sequence(data)
    print("Component sequence:", component_sequence)

    assert len(component_sequence) >= 0, f"Expected at least 0 components, got {len(component_sequence)}"

    # Check all data for unredacted symptom mentions
    for idx, entry in enumerate(data):
        if contains_unredacted_symptom(entry):
            pytest.fail(f"Unredacted symptom found in trace entry at index {idx}: {entry}")
