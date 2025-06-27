import pytest
import os
import logging
import dotenv
dotenv.load_dotenv()
from ragaai_catalyst.tracers.agentic_tracing.utils.span_attributes import SpanAttributes

def test_execute_metrics_invalid_metric_type(caplog):
    """Test handling of invalid metric type in execute_metrics"""
    caplog.set_level(logging.ERROR)  # Ensure we're capturing ERROR level logs
    
    span = SpanAttributes("test_span")
    
    # Call execute_metrics with a list containing a non-dict value
    span.execute_metrics(
        name=[123],  # Numeric value instead of dict
        model="test-model",
        provider="test-provider"
    )
    
    # Check that error is logged
    assert "Expected dict, got" in caplog.text

def test_execute_metrics_missing_name(caplog):
    """Test handling of missing 'name' in metric"""
    span = SpanAttributes("test_span")
    
    # Call execute_metrics with metric missing 'name' key
    span.execute_metrics(
        name=[{"not_name": "value"}],  # Missing 'name' key
        model="test-model",
        provider="test-provider"
    )
    
    # Check that error is logged
    assert "Metric must contain 'name'" in caplog.text
    # Function should continue without raising exception

def test_add_gt_unsupported_type(caplog):
    """Test handling of unsupported type for gt"""
    span = SpanAttributes("test_span")
    
    # Create a custom class that is not supported
    class UnsupportedType:
        pass
    
    # Add gt with unsupported type
    span.add_gt(UnsupportedType())
    
    # Check that error is logged
    assert "Unsupported type for gt" in caplog.text
    # gt should not be set
    assert span.gt is None