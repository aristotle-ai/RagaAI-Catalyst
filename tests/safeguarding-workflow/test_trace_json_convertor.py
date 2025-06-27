import pytest
import logging
import json
from unittest.mock import Mock, patch
from ragaai_catalyst.tracers.utils.trace_json_converter import convert_json_format, get_spans

@pytest.fixture
def mock_input_trace():
    """Create a mock input trace for testing"""
    return [
        {
            "name": "test_span",
            "context": {"trace_id": "test_trace_id"},
            "parent_id": None,
            "start_time": "2023-01-01T00:00:00.000000Z",
            "end_time": "2023-01-01T00:01:00.000000Z",
            "attributes": {
                "openinference.span.kind": "llm",
                "llm.model_name": "test-model",
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 20
            }
        }
    ]

def test_convert_json_format_get_spans_error(mock_input_trace, caplog):
    """Test convert_json_format logs error when get_spans fails"""
    caplog.set_level(logging.ERROR)
    
    # Patch get_spans to raise an exception
    with patch('ragaai_catalyst.tracers.utils.trace_json_converter.get_spans', 
               side_effect=Exception("Test error")):
        
        # Call convert_json_format
        result = convert_json_format(
            mock_input_trace, 
            custom_model_cost={}, 
            user_context="", 
            user_gt="",
            external_id=None
        )
        
        # Check that error was logged
        assert "Error in get_spans function: Test error" in caplog.text