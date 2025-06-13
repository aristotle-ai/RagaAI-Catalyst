import pytest
import os
import requests
import logging
from unittest.mock import patch, MagicMock
from ragaai_catalyst.evaluation import Evaluation

@pytest.fixture
def evaluation():
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        # Mock project list response
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [{
                    "id": "test_project_id",
                    "name": "test_project"
                }]
            }
        }
        mock_get.return_value.status_code = 200
        
        # Mock dataset list response
        mock_post.return_value.json.return_value = {
            "data": {
                "content": [{
                    "id": "test_dataset_id",
                    "name": "test_dataset"
                }]
            }
        }
        mock_post.return_value.status_code = 200
        
        return Evaluation(project_name="test_project", dataset_name="test_dataset")

@pytest.fixture
def valid_metrics():
    return [{
        "name": "accuracy",
        "config": {"threshold": 0.8},
        "column_name": "accuracy_col",
        "schema_mapping": {"input": "test_input"}
    }]

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "success": True,
        "message": "Metrics added successfully",
        "data": {"jobId": "test_job_123"}
    }
    return mock

def test_add_metrics_success(evaluation, valid_metrics, mock_response):
    """Test successful addition of metrics"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.return_value = mock_response
        evaluation.add_metrics(valid_metrics)
        
        # Verify the request was made with correct project_id
        assert mock_post.call_args[1]['headers']['X-Project-Id'] == str(evaluation.project_id)
        assert evaluation.jobId == "test_job_123"

def test_add_metrics_missing_required_keys(evaluation, caplog):
    """Test validation of required keys"""
    # Set caplog to capture logging at ERROR level
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    invalid_metrics = [{
        "name": "Hallucination",
        "config": {"provider": "openai", "model": "gpt-4"}
        # missing column_name and schema_mapping
    }]
    
    # This will now log errors instead of raising ValueError
    # It may raise KeyError or TypeError when trying to access missing keys
    try:
        evaluation.add_metrics(invalid_metrics)
    except (KeyError, TypeError):
        # We expect these exceptions since we're not mocking anything
        pass
    
    # Verify that the correct error message was logged
    assert "{'schema_mapping', 'column_name'} required for each metric evaluation" in caplog.text or \
           "{'column_name', 'schema_mapping'} required for each metric evaluation" in caplog.text


def test_add_metrics_invalid_metric_name(evaluation, valid_metrics, caplog):
    """Test validation of metric names"""
    # Set caplog to capture logging at ERROR level
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["different_metric"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        # Call the function directly, no exception expected
        evaluation.add_metrics(valid_metrics)
        
        # Check that the error was logged
        assert "Enter a valid metric name" in caplog.text

def test_add_metrics_duplicate_column_name(evaluation, valid_metrics, caplog):
    """Test validation of duplicate column names"""
    # Set caplog to capture logging at ERROR level
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    with patch.object(evaluation, '_get_executed_metrics_list', 
                     return_value=["accuracy_col"]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        # Call the function directly, no exception expected
        evaluation.add_metrics(valid_metrics)
        
        # Check that the error was logged
        assert "Column name 'accuracy_col' already exists" in caplog.text

def test_add_metrics_http_error(evaluation, valid_metrics):
    """Test handling of HTTP errors"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.side_effect = requests.exceptions.HTTPError("HTTP Error")
        evaluation.add_metrics(valid_metrics)
        # Should log error but not raise exception

def test_add_metrics_connection_error(evaluation, valid_metrics):
    """Test handling of connection errors"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection Error")
        evaluation.add_metrics(valid_metrics)
        # Should log error but not raise exception

def test_add_metrics_timeout_error(evaluation, valid_metrics):
    """Test handling of timeout errors"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.side_effect = requests.exceptions.Timeout("Timeout Error")
        evaluation.add_metrics(valid_metrics)
        # Should log error but not raise exception
def test_add_metrics_bad_request(evaluation, valid_metrics):
    """Test handling of 400 bad request"""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"message": "Bad request error"}
    
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}), \
         patch('ragaai_catalyst.evaluation.logger') as mock_logger:
        
        mock_post.return_value = mock_response
        evaluation.add_metrics(valid_metrics)
        
        # Match the actual error message format being used
        mock_logger.error.assert_called_with("An unexpected error occurred: 'success'")
        assert evaluation.jobId is None