import pytest
import logging
from unittest.mock import Mock, patch
from opentelemetry.sdk.trace.export import SpanExportResult
from ragaai_catalyst.tracers.exporters.dynamic_trace_exporter import DynamicTraceExporter

@pytest.fixture
def mock_exporter():
    """Create a mock RAGATraceExporter for testing"""
    mock = Mock()
    mock.export.return_value = SpanExportResult.SUCCESS
    mock.shutdown.return_value = None
    return mock

@pytest.fixture
def dynamic_exporter(mock_exporter):
    """Create a DynamicTraceExporter with mocked underlying exporter"""
    exporter = DynamicTraceExporter(
        tracer_type="test",
        files_to_zip=[],
        project_name="test_project",
        project_id="123",
        dataset_name="test_dataset",
        user_details={},
        base_url="http://example.com",
        custom_model_cost={},
        timeout=30,
        post_processor=None,
        max_upload_workers=5,
        user_context="",
        user_gt="",
        external_id=None
    )
    exporter._exporter = mock_exporter
    return exporter

def test_export_error_updating_properties(dynamic_exporter, caplog):
    """Test export logs error when updating properties fails"""
    caplog.set_level(logging.ERROR)
    
    # Set up the _update_exporter_properties method to raise an exception
    dynamic_exporter._update_exporter_properties = Mock(side_effect=Exception("Test error"))
    
    # Call export with some dummy spans
    spans = [Mock()]
    dynamic_exporter.export(spans)
    
    # Check that error was logged
    assert "Error updating exporter properties: Test error" in caplog.text

def test_export_error_exporting_trace(dynamic_exporter, caplog, mock_exporter):
    """Test export logs error when exporting trace fails"""
    caplog.set_level(logging.ERROR)
    
    # Set up the mock_exporter to raise an exception
    mock_exporter.export.side_effect = Exception("Export error")
    
    # Call export with some dummy spans
    spans = [Mock()]
    dynamic_exporter.export(spans)
    
    # Check that error was logged
    assert "Error exporting trace: Export error" in caplog.text

def test_shutdown_error_updating_properties(dynamic_exporter, caplog):
    """Test shutdown logs error when updating properties fails"""
    caplog.set_level(logging.ERROR)
    
    # Set up the _update_exporter_properties method to raise an exception
    dynamic_exporter._update_exporter_properties = Mock(side_effect=Exception("Test error"))
    
    # Call shutdown
    dynamic_exporter.shutdown()
    
    # Check that error was logged
    assert "Error updating exporter properties: Test error" in caplog.text

def test_shutdown_error_shutting_down(dynamic_exporter, caplog, mock_exporter):
    """Test shutdown logs error when shutting down fails"""
    caplog.set_level(logging.ERROR)
    
    # Set up the mock_exporter to raise an exception
    mock_exporter.shutdown.side_effect = Exception("Shutdown error")
    
    # Call shutdown
    dynamic_exporter.shutdown()
    
    # Check that error was logged
    assert "Error shutting down exporter: Shutdown error" in caplog.text