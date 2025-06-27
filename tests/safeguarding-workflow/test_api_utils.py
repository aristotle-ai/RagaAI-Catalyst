import pytest
import os
import logging
import dotenv
dotenv.load_dotenv()
from ragaai_catalyst.tracers.agentic_tracing.utils.api_utils import fetch_analysis_trace

def test_fetch_analysis_trace_error(caplog):
    """Test handling of error when fetching analysis trace"""
    # Use invalid base URL and trace ID
    base_url = "https://invalid-url-that-doesnt-exist.com"
    trace_id = "nonexistent_trace_id"
    
    result = fetch_analysis_trace(base_url, trace_id)
    
    # Check that error is logged
    assert "Error fetching analysis trace" in caplog.text
    # Function should return None instead of raising exception
    assert result is None

def test_fetch_analysis_trace_invalid_response(caplog):
    """Test handling of invalid response when fetching analysis trace"""
    # Use valid but incorrect URL (will return 404)
    base_url = "https://www.google.com"
    trace_id = "nonexistent_trace_id"
    
    result = fetch_analysis_trace(base_url, trace_id)
    
    # Check that error is logged
    assert "Error fetching analysis trace" in caplog.text
    # Function should return None instead of raising exception
    assert result is None