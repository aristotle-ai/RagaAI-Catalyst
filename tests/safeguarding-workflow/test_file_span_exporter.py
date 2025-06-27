import pytest
import logging
import os
import tempfile
import json
from unittest.mock import Mock, patch
from ragaai_catalyst.tracers.exporters.file_span_exporter import FileSpanExporter

@pytest.fixture
def mock_raga_client():
    """Create a mock RagaExporter for testing"""
    mock = Mock()
    mock.check_and_upload_files = Mock()
    return mock

@pytest.fixture
def file_span_exporter(mock_raga_client):
    """Create a FileSpanExporter with mocked raga_client"""
    return FileSpanExporter(
        project_name="test_project",
        session_id="test_session",
        metadata={},
        pipeline={},
        raga_client=mock_raga_client
    )

@pytest.fixture
def mock_span():
    """Create a mock span for testing"""
    mock = Mock()
    mock.to_json.return_value = json.dumps({
        "context": {"trace_id": "test_trace_id"},
        "name": "test_span"
    })
    return mock

@pytest.mark.asyncio
async def test_upload_traces_no_token(file_span_exporter, caplog, monkeypatch):
    """Test _upload_traces logs error when token is missing"""
    caplog.set_level(logging.ERROR)
    
    # Ensure RAGAAI_CATALYST_TOKEN is not set
    monkeypatch.delenv("RAGAAI_CATALYST_TOKEN", raising=False)
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"{}")
        temp_path = temp_file.name
    
    try:
        # Call _upload_traces
        result = await file_span_exporter._upload_traces(json_file_path=temp_path)
        
        # Check that error was logged
        assert "RAGAAI_CATALYST_TOKEN not found. Cannot upload traces" in caplog.text
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)