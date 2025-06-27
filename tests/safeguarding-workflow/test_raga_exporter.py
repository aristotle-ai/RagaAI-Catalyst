import pytest
import logging
import os
import json
from unittest.mock import Mock, patch, MagicMock
import requests
import aiohttp
from ragaai_catalyst.tracers.exporters.raga_exporter import RagaExporter

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing"""
    monkeypatch.setenv("RAGAAI_CATALYST_ACCESS_KEY", "test_access_key")
    monkeypatch.setenv("RAGAAI_CATALYST_SECRET_KEY", "test_secret_key")
    monkeypatch.setenv("RAGAAI_CATALYST_TOKEN", "test_token")
    monkeypatch.setenv("RAGAAI_CATALYST_BASE_URL", "http://test.example.com/api")

@pytest.fixture
def mock_response():
    """Create a mock response object"""
    mock = Mock(spec=requests.Response)
    mock.status_code = 200
    mock.json.return_value = {"data": {}, "success": True}
    return mock

@pytest.fixture
def mock_aioresponse():
    """Create a mock aiohttp response object"""
    mock = MagicMock()
    mock.status = 200
    mock.json = MagicMock(return_value={"data": {}, "success": True})
    return mock

def test_init_missing_credentials(monkeypatch, caplog):
    """Test constructor logs error when credentials are missing"""
    caplog.set_level(logging.ERROR)
    
    # Ensure credentials are not set
    monkeypatch.delenv("RAGAAI_CATALYST_ACCESS_KEY", raising=False)
    monkeypatch.delenv("RAGAAI_CATALYST_SECRET_KEY", raising=False)
    
    # Mock get_token and _create_schema to avoid real API calls
    with patch('ragaai_catalyst.tracers.exporters.raga_exporter.get_token') as mock_get_token, \
         patch.object(RagaExporter, '_create_schema') as mock_create_schema:
        mock_create_schema.return_value = 200
        
        # Initialize RagaExporter
        exporter = RagaExporter(project_name="test_project", dataset_name="test_dataset")
        
        # Check that error was logged
        assert "RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables must be set" in caplog.text

def test_create_schema_error(mock_env_vars, caplog):
    """Test _create_schema logs error when schema creation fails"""
    caplog.set_level(logging.ERROR)
    
    # Mock _create_schema to return error status
    with patch.object(RagaExporter, '_create_schema') as mock_create_schema:
        mock_create_schema.return_value = 400
        
        # Initialize RagaExporter
        exporter = RagaExporter(project_name="test_project", dataset_name="test_dataset")
        
        # Check that error was logged
        assert "Failed to create schema" in caplog.text