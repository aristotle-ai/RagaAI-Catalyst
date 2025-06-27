import pytest
import logging
import json
from unittest.mock import Mock, patch
from opentelemetry.sdk.trace.export import SpanExportResult
from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import RAGATraceExporter

@pytest.fixture
def trace_exporter():
    """Create a RAGATraceExporter for testing"""
    return RAGATraceExporter(
        tracer_type="test",
        files_to_zip=[],
        project_name="test_project",
        project_id="123",
        dataset_name="test_dataset",
        user_details={},
        base_url="http://example.com",
        custom_model_cost={},
        timeout=30
    )

@pytest.fixture
def mock_span():
    """Create a mock span for testing"""
    mock = Mock()
    mock.to_json.return_value = json.dumps({
        "context": {"trace_id": "test_trace_id"},
        "parent_id": None,
        "name": "test_span"
    })
    return mock

@pytest.fixture
def mock_span_no_trace_id():
    """Create a mock span with no trace_id for testing"""
    mock = Mock()
    mock.to_json.return_value = json.dumps({
        "context": {},
        "name": "test_span"
    })
    return mock

def test_export_trace_id_none(trace_exporter, mock_span_no_trace_id, caplog):
    """Test export logs error when trace ID is None"""
    caplog.set_level(logging.ERROR)
    
    # Call export with a span that has no trace_id
    result = trace_exporter.export([mock_span_no_trace_id])
    
    # Check that error was logged
    assert "Trace ID is None" in caplog.text
    
    # Verify result is still success
    assert result == SpanExportResult.SUCCESS

def test_export_error_processing_complete_trace(trace_exporter, mock_span, caplog):
    """Test export logs error when processing complete trace fails"""
    caplog.set_level(logging.ERROR)
    
    # Set up process_complete_trace to raise an exception
    trace_exporter.process_complete_trace = Mock(side_effect=Exception("Test error"))
    
    # Call export with a span
    result = trace_exporter.export([mock_span])
    
    # Check that error was logged
    assert "Error processing complete trace: Test error" in caplog.text
    
    # Verify result is still success
    assert result == SpanExportResult.SUCCESS


def test_export_error_deleting_trace(trace_exporter, mock_span, caplog):
    """Test export logs error when deleting trace fails"""
    caplog.set_level(logging.ERROR)
    
    # Create a dictionary with a mocked __delitem__ that raises an exception
    original_trace_spans = trace_exporter.trace_spans
    
    class MockDict(dict):
        def __delitem__(self, key):
            raise Exception("Test delete error")
    
    # Replace the trace_spans with our mocked dictionary
    trace_exporter.trace_spans = MockDict()
    
    try:
        # Call export with a span
        result = trace_exporter.export([mock_span])
        
        # Check that error was logged
        assert "Error deleting trace: Test delete error" in caplog.text
        
        # Verify result is still success
        assert result == SpanExportResult.SUCCESS
    finally:
        # Restore the original dictionary
        trace_exporter.trace_spans = original_trace_spans