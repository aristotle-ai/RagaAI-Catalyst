import pytest
import logging
import os
import requests
from unittest.mock import Mock, patch, MagicMock
from ragaai_catalyst.tracers.tracer import Tracer

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing"""
    monkeypatch.setenv("RAGAAI_CATALYST_ACCESS_KEY", "test_access_key")
    monkeypatch.setenv("RAGAAI_CATALYST_SECRET_KEY", "test_secret_key")
    monkeypatch.setenv("RAGAAI_CATALYST_TOKEN", "test_token")
    monkeypatch.setenv("RAGAAI_CATALYST_BASE_URL", "http://test.example.com/api")

@pytest.fixture
def mock_response():
    """Create a mock response with projects data"""
    mock = Mock(spec=requests.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "data": {
            "content": [
                {"id": "123", "name": "test_project"},
                {"id": "456", "name": "other_project"}
            ]
        },
        "success": True
    }
    return mock

@pytest.fixture
def mock_aioresponse():
    """Create a mock aiohttp response object"""
    mock = MagicMock()
    mock.status = 200
    mock.json = MagicMock(return_value={"data": {}, "success": True})
    return mock

def test_register_masking_function_invalid_type(mock_env_vars, mock_response, caplog):
    """Test register_masking_function logs error with invalid function type"""
    caplog.set_level(logging.ERROR)
    
    with patch('requests.get', return_value=mock_response):
        # Initialize Tracer
        tracer = Tracer(
            project_name="test_project",
            dataset_name="test_dataset",
            tracer_type="langchain"
        )
        
        # Call register_masking_function with invalid function
        tracer.register_masking_function("not_a_function")
        
        # Check that error was logged
        assert "masking_func must be a callable" in caplog.text


def test_upload_traces_no_token(mock_env_vars, mock_response, caplog, monkeypatch):
    """Test _upload_traces logs error when token is missing"""
    caplog.set_level(logging.ERROR)
    
    with patch('requests.get', return_value=mock_response):
        # Initialize Tracer
        tracer = Tracer(
            project_name="test_project",
            dataset_name="test_dataset",
            tracer_type="langchain"
        )
        
        # Ensure RAGAAI_CATALYST_TOKEN is not set
        monkeypatch.delenv("RAGAAI_CATALYST_TOKEN", raising=False)
        
        # Mock aiohttp.ClientSession for async calls
        mock_session = MagicMock()
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Need to run in an event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Call _upload_traces
            try:
                loop.run_until_complete(tracer._upload_traces())
            except Exception:
                pass  # Ignore any exceptions
            
            # Check that error was logged
            assert "RAGAAI_CATALYST_TOKEN not found. Cannot upload traces" in caplog.text
            
            loop.close()

def test_update_dynamic_exporter_invalid_tracer_type(mock_env_vars, mock_response, caplog):
    """Test update_dynamic_exporter logs error with invalid tracer_type"""
    caplog.set_level(logging.ERROR)
    
    with patch('requests.get', return_value=mock_response):
        # Initialize Tracer with non-agentic tracer_type
        tracer = Tracer(
            project_name="test_project",
            dataset_name="test_dataset",
            tracer_type="langchain"  # Not agentic
        )
        
        # Call update_dynamic_exporter
        tracer.update_dynamic_exporter(files_to_zip=[])
        
        # Check that error was logged
        assert "This method is only available for agentic tracers" in caplog.text

def test_update_file_list_invalid_tracer_type(mock_env_vars, mock_response, caplog):
    """Test update_file_list logs error with invalid tracer_type"""
    caplog.set_level(logging.ERROR)
    
    with patch('requests.get', return_value=mock_response):
        # Initialize Tracer with non-agentic tracer_type
        tracer = Tracer(
            project_name="test_project",
            dataset_name="test_dataset",
            tracer_type="langchain"  # Not agentic
        )
        
        # Call update_file_list
        tracer.update_file_list()
        
        # Check that error was logged
        assert "This method is only available for agentic tracers" in caplog.text


def test_register_post_processor_invalid_type(mock_env_vars, mock_response, caplog):
    """Test register_post_processor logs error with invalid function type"""
    caplog.set_level(logging.ERROR)
    
    with patch('requests.get', return_value=mock_response):
        # Initialize Tracer
        tracer = Tracer(
            project_name="test_project",
            dataset_name="test_dataset",
            tracer_type="langchain"
        )
        
        try:
            # Call register_post_processor with invalid function
            tracer.register_post_processor("not_a_function")
        except TypeError:
            # Expect TypeError to be raised
            pass
        
        # Check that error was logged
        assert "post_processor_func must be a callable" in caplog.text
