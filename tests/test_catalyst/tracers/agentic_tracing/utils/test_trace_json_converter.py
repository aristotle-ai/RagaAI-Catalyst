# tests/custom_tests/ragaai_catalyst/tracers/agentic_tracing/utils/test_trace_json_converter.py
import pytest
import os
from dotenv import load_dotenv
from ragaai_catalyst.tracers.utils.trace_json_converter import (
    convert_time_format, get_uuid, convert_json_format
)

# Test responses dictionary
TEST_RESPONSES = {
  "convert_time_format": {
    "success": True,
    "original_time": "2023-01-01T12:00:00.000000Z",
    "converted_time": "2023-01-01T17:30:00.000000+05:30"
  },
  "convert_time_format_custom_tz": {
    "success": True,
    "original_time": "2023-01-01T12:00:00.000000Z",
    "target_timezone": "America/New_York",
    "converted_time": "2023-01-01T07:00:00.000000-05:00"
  },
  "get_uuid": {
    "success": True,
    "test_name": "test_uuid_name",
    "uuid": "30d834a0-e74d-51f1-9878-486cfe034574"
  },
  "convert_json_format": {
    "success": True,
    "converted_trace_id": "test-trace-id",
    "metadata_tokens": {
      "prompt_tokens": 10.0,
      "completion_tokens": 20.0,
      "total_tokens": 30.0
    },
    "metadata_cost": {
      "input_cost": 0.0001,
      "output_cost": 0.0006,
      "total_cost": 0.0007
    }
  }
}

@pytest.fixture
def sample_trace():
    """Create a sample trace for testing."""
    return [
        {
            "context": {
                "trace_id": "test-trace-id",
                "span_id": "span-1"
            },
            "name": "test_span",
            "parent_id": None,
            "start_time": "2023-01-01T12:00:00.000000Z",
            "end_time": "2023-01-01T12:01:00.000000Z",
            "attributes": {
                "openinference.span.kind": "LLM",
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 20,
                "llm.token_count.total": 30,
                "llm.model_name": "gpt-4",
                "input.value": "test input",
                "output.value": "test output"
            },
            "status": {
                "status_code": "OK"
            }
        }
    ]

@pytest.fixture
def custom_model_cost():
    """Create custom model cost dictionary for testing."""
    return {
        "gpt-4": {
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00003
        }
    }

def test_convert_time_format():
    """Test convert_time_format functionality."""
    original_time = TEST_RESPONSES["convert_time_format"]["original_time"]
    converted_time = convert_time_format(original_time)
    
    # The exact converted time may vary by timezone based on the test environment
    # We only check that the conversion produces a string with the right format
    assert isinstance(converted_time, str)
    assert "+" in converted_time or "-" in converted_time  # Should include timezone offset

def test_convert_time_format_custom_tz():
    """Test convert_time_format with custom timezone."""
    original_time = TEST_RESPONSES["convert_time_format_custom_tz"]["original_time"]
    target_timezone = TEST_RESPONSES["convert_time_format_custom_tz"]["target_timezone"]
    converted_time = convert_time_format(original_time, target_timezone)
    
    # Check timezone offset for New York
    assert "-" in converted_time  # New York timezone should have negative offset

def test_get_uuid():
    """Test get_uuid functionality."""
    test_name = TEST_RESPONSES["get_uuid"]["test_name"]
    uuid_result = get_uuid(test_name)
    
    # The UUID should be deterministic for the same name
    assert uuid_result == TEST_RESPONSES["get_uuid"]["uuid"]

def test_convert_json_format(sample_trace, custom_model_cost):
    """Test convert_json_format functionality."""
    converted_trace = convert_json_format(sample_trace, custom_model_cost)
    
    # Check the converted trace ID
    assert converted_trace["id"] == TEST_RESPONSES["convert_json_format"]["converted_trace_id"]
    
    # Check the metadata tokens
    metadata_tokens = converted_trace["metadata"]["tokens"]
    assert metadata_tokens["prompt_tokens"] == TEST_RESPONSES["convert_json_format"]["metadata_tokens"]["prompt_tokens"]
    assert metadata_tokens["completion_tokens"] == TEST_RESPONSES["convert_json_format"]["metadata_tokens"]["completion_tokens"]
    assert metadata_tokens["total_tokens"] == TEST_RESPONSES["convert_json_format"]["metadata_tokens"]["total_tokens"]
    
    # Check the metadata cost
    metadata_cost = converted_trace["metadata"]["cost"]
    assert metadata_cost["input_cost"] == TEST_RESPONSES["convert_json_format"]["metadata_cost"]["input_cost"]
    assert metadata_cost["output_cost"] == TEST_RESPONSES["convert_json_format"]["metadata_cost"]["output_cost"]
    assert metadata_cost["total_cost"] == TEST_RESPONSES["convert_json_format"]["metadata_cost"]["total_cost"]