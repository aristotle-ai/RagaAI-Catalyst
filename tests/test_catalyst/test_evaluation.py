from unittest.mock import patch
import time
import pytest
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from datetime import datetime 
from typing import Dict, List
from ragaai_catalyst.evaluation import Evaluation
from ragaai_catalyst.ragaai_catalyst import RagaAICatalyst
import requests
from unittest.mock import patch, MagicMock

# Simplified model configurations
MODEL_CONFIGS = [
    {"provider": "openai", "model": "gpt-4"},  # Only one OpenAI model
    {"provider": "gemini", "model": "gemini-1.5-flash"}  # Only one Gemini model
]

# Common metrics to test
CORE_METRICS = [
    'Hallucination',
    'Faithfulness',
    'Response Correctness',
    'Context Relevancy'
]

CHAT_METRICS = [
    'Agent Quality',
    'User Chat Quality'
]

@pytest.fixture
def base_url():
    return os.getenv("RAGAAI_CATALYST_BASE_URL")

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    }

@pytest.fixture
def evaluation(base_url, access_keys):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Evaluation(
        project_name="prompt_metric_dataset_sk", 
        dataset_name="dataset_19feb_1"
    )

@pytest.fixture
def chat_evaluation(base_url, access_keys):
    """Create evaluation instance for chat metrics"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Evaluation(
        project_name="prompt_metric_dataset_sk", 
        dataset_name="dataset_19feb_1"
    )

# Basic initialization tests
def test_evaluation_initialization(evaluation):
    """Test if evaluation is initialized correctly"""
    assert evaluation.project_name == "prompt_metric_dataset_sk"
    assert evaluation.dataset_name == "dataset_19feb_1"

def test_project_does_not_exist():
    """Test initialization with non-existent project"""
    with pytest.raises(ValueError, match="Project not found"):
        Evaluation(project_name="non_existent_project", dataset_name="dataset")

def test_dataset_does_not_exist():
    """Test initialization with non-existent dataset"""
    with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
        # Mock successful project retrieval
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [
                    {"name": "valid_project", "id": "123"}
                ]
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Mock dataset retrieval with no matching dataset
        mock_post.return_value.json.return_value = {
            "data": {
                "content": [
                    {"name": "other_dataset", "id": "456"}
                ]
            }
        }
        mock_post.return_value.raise_for_status = MagicMock()
        
        with pytest.raises(ValueError, match="Dataset not found"):
            Evaluation(project_name="valid_project", dataset_name="nonexistent_dataset")

# Parameterized validation tests
@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_metric_validation_checks(evaluation, provider_config):
    """Test all validation checks in one parameterized test"""
    schema_mapping = {
        'Query': 'Prompt',
        'Response': 'Response',
        'Context': 'Context',
    }
    
    # Test missing schema_mapping
    with pytest.raises(ValueError):
        evaluation.add_metrics([{
            "name": "Hallucination",
            "config": provider_config,
            "column_name": "test_column"
        }])
    
    # Test missing column_name
    with pytest.raises(ValueError):
        evaluation.add_metrics([{
            "name": "Hallucination",
            "config": provider_config,
            "schema_mapping": schema_mapping
        }])
    
    # Test missing metric name
    with pytest.raises(ValueError):
        evaluation.add_metrics([{
            "config": provider_config,
            "column_name": "test_column",
            "schema_mapping": schema_mapping
        }])
# Additional test cases for test_evaluation.py

# Test Initialization and Configuration
def test_evaluation_timeout_configuration(evaluation):
    """Test if timeout is configured correctly"""
    assert evaluation.timeout == 20
    assert evaluation.num_projects == 99999

def test_evaluation_base_url_configuration(evaluation, base_url):
    """Test if base URL is configured correctly"""
    assert evaluation.base_url == f"{base_url}"

# Test List Metrics Functionality
def test_list_metrics_success(evaluation):
    """Test successful metrics listing"""
    mock_metrics = ["Hallucination", "Faithfulness", "Response Correctness"]
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "data": {"metrics": [{"name": m} for m in mock_metrics]}
        }
        mock_get.return_value.status_code = 200
        
        result = evaluation.list_metrics()
        assert result == mock_metrics



# Test Metric Mapping Functionality
def test_get_mapping_valid_schema(evaluation):
    """Test mapping generation with valid schema"""
    metric_name = "Hallucination"
    metrics_schema = [{
        "name": "Hallucination",
        "config": {
            "requiredFields": [
                {"name": "Query"},
                {"name": "Response"}
            ]
        }
    }]
    schema_mapping = {
        "user_query": "Query",
        "model_response": "Response"
    }
    
    with patch.object(evaluation, '_get_dataset_schema', return_value=[
        {"displayName": "user_query"},
        {"displayName": "model_response"}
    ]):
        result = evaluation._get_mapping(metric_name, metrics_schema, schema_mapping)
        assert len(result) == 2
        assert result[0]["schemaName"] == "Query"
        assert result[0]["variableName"] == "user_query"

def test_get_mapping_invalid_column(evaluation):
    """Test mapping generation with invalid column"""
    metric_name = "Hallucination"
    metrics_schema = [{
        "name": "Hallucination",
        "config": {
            "requiredFields": [
                {"name": "Query"}
            ]
        }
    }]
    schema_mapping = {
        "invalid_column": "Query"
    }
    
    with patch.object(evaluation, '_get_dataset_schema', return_value=[]):
        with pytest.raises(ValueError, match="Column 'invalid_column' is not present"):
            evaluation._get_mapping(metric_name, metrics_schema, schema_mapping)

# Test Status Checking
def test_get_status_completed(evaluation):
    """Test status check for completed job"""
    evaluation.jobId = "test_job"
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {
                "content": [{"id": "test_job", "status": "Completed"}]
            }
        }
        assert evaluation.get_status() == "success"

def test_get_status_failed(evaluation):
    """Test status check for failed job"""
    evaluation.jobId = "test_job"
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {
                "content": [{"id": "test_job", "status": "Failed"}]
            }
        }
        assert evaluation.get_status() == "failed"

# Test Status Checking - Additional Tests
def test_get_status_in_progress(evaluation):
    """Test status check for in-progress job"""
    evaluation.jobId = "test_job"
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {
                "content": [{"id": "test_job", "status": "In Progress"}]
            }
        }
        assert evaluation.get_status() == "in_progress"

def test_get_status_request_exception(evaluation):
    """Test status check with request exception"""
    evaluation.jobId = "test_job"
    with patch('requests.get') as mock_get:
        # Simulate a request exception
        mock_get.side_effect = requests.exceptions.RequestException("Test error")
        with patch('ragaai_catalyst.evaluation.logger') as mock_logger:
            assert evaluation.get_status() == "failed"
            mock_logger.error.assert_called()

def test_get_status_http_error(evaluation):
    """Test status check with HTTP error"""
    evaluation.jobId = "test_job"
    with patch('requests.get') as mock_get:
        # Simulate an HTTP error
        mock_get.side_effect = requests.exceptions.HTTPError("HTTP Error")
        with patch('ragaai_catalyst.evaluation.logger') as mock_logger:
            assert evaluation.get_status() == "failed"
            mock_logger.error.assert_called()

# Test _update_base_json Method
def test_update_base_json(evaluation):
    """Test update_base_json method"""
    metrics = [{
        "name": "Hallucination",
        "config": {
            "provider": "openai",
            "model": "gpt-4"
        },
        "column_name": "hallucination_score",
        "schema_mapping": {
            "query": "Query",
            "response": "Response"
        }
    }]
    
    metrics_schema_response = [{
        "name": "Hallucination",
        "config": {
            "requiredFields": [
                {"name": "Query"},
                {"name": "Response"}
            ]
        }
    }]
    
    with patch.object(evaluation, '_get_metrics_schema_response', return_value=metrics_schema_response), \
         patch.object(evaluation, '_get_mapping', return_value=[{"schemaName": "Query", "variableName": "query"}, {"schemaName": "Response", "variableName": "response"}]):
        
        result = evaluation._update_base_json(metrics)
        
        assert "datasetId" in result
        assert "metricParams" in result
        assert len(result["metricParams"]) == 1
        assert result["metricParams"][0]["metricSpec"]["name"] == "Hallucination"
        assert result["metricParams"][0]["metricSpec"]["displayName"] == "hallucination_score"

def test_update_base_json_invalid_provider(evaluation):
    """Test update_base_json with invalid provider"""
    metrics = [{
        "name": "Hallucination",
        "config": {
            "provider": "invalid_provider",  # Invalid provider
            "model": "gpt-4"
        },
        "column_name": "hallucination_score",
        "schema_mapping": {
            "query": "Query",
            "response": "Response"
        }
    }]
    
    metrics_schema_response = [{
        "name": "Hallucination",
        "config": {
            "requiredFields": [
                {"name": "Query"},
                {"name": "Response"}
            ]
        }
    }]
    
    with patch.object(evaluation, '_get_metrics_schema_response', return_value=metrics_schema_response):
        with pytest.raises(ValueError, match="Enter a valid provider name"):
            evaluation._update_base_json(metrics)

def test_update_base_json_with_threshold(evaluation):
    """Test update_base_json with threshold configuration"""
    metrics = [{
        "name": "Hallucination",
        "config": {
            "provider": "openai",
            "model": "gpt-4",
            "threshold": {"gte": 0.8}  # Threshold configuration
        },
        "column_name": "hallucination_score",
        "schema_mapping": {
            "query": "Query",
            "response": "Response"
        }
    }]
    
    metrics_schema_response = [{
        "name": "Hallucination",
        "config": {
            "requiredFields": [
                {"name": "Query"},
                {"name": "Response"}
            ]
        }
    }]
    
    with patch.object(evaluation, '_get_metrics_schema_response', return_value=metrics_schema_response), \
         patch.object(evaluation, '_get_mapping', return_value=[{"schemaName": "Query", "variableName": "query"}, {"schemaName": "Response", "variableName": "response"}]):
        
        result = evaluation._update_base_json(metrics)
        
        # Check threshold is properly set in the configuration
        assert result["metricParams"][0]["metricSpec"]["config"]["params"]["threshold"] == {"gte": 0.8}

def test_update_base_json_invalid_threshold(evaluation):
    """Test update_base_json with invalid threshold configuration"""
    metrics = [{
        "name": "Hallucination",
        "config": {
            "provider": "openai",
            "model": "gpt-4",
            "threshold": {"gte": 0.8, "lte": 0.5}  # Invalid: Multiple thresholds
        },
        "column_name": "hallucination_score",
        "schema_mapping": {
            "query": "Query",
            "response": "Response"
        }
    }]
    
    metrics_schema_response = [{
        "name": "Hallucination",
        "config": {
            "requiredFields": [
                {"name": "Query"},
                {"name": "Response"}
            ]
        }
    }]
    
    # We need to mock _get_metrics_schema_response properly
    # The original mock was still allowing real network calls
    with patch.object(evaluation, '_get_metrics_schema_response') as mock_schema_response:
        # Set the return value directly, rather than allowing the function to run
        mock_schema_response.return_value = metrics_schema_response
        
        with pytest.raises(ValueError, match="'threshold' can only take one argument"):
            evaluation._update_base_json(metrics)

# Test Append Metrics
def test_append_metrics_invalid_input(evaluation):
    """Test append metrics with invalid input"""
    with pytest.raises(ValueError, match="display_name should be a string"):
        evaluation.append_metrics(123)

def test_append_metrics_error(evaluation):
    """Test append metrics with error"""
    with patch('requests.request') as mock_request:
        mock_request.side_effect = requests.exceptions.RequestException()
        evaluation.append_metrics("new_metric")
        assert evaluation.jobId is None

def test_list_metrics_http_error(evaluation):
    """Test metrics listing with HTTP error"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.HTTPError()
        with patch('ragaai_catalyst.evaluation.logger') as mock_logger:
            result = evaluation.list_metrics()
            assert result is None  # The method returns None on HTTPError
            mock_logger.error.assert_called()  # Verify error was logged

def test_get_dataset_schema_error(evaluation):
    """Test schema retrieval with error"""
    # Mock _get_dataset_id_based_on_dataset_type to return a test ID
    with patch.object(evaluation, '_get_dataset_id_based_on_dataset_type', return_value="test_dataset_id"), \
         patch('requests.post') as mock_post, \
         patch('ragaai_catalyst.evaluation.logger') as mock_logger:
        
        # Simulate a request exception
        mock_post.side_effect = requests.exceptions.RequestException("Test error")
        
        result = evaluation._get_dataset_schema()
        
        # Verify the results
        assert result == {}  # Should return empty dict on error
        mock_logger.error.assert_called()  # Verify error was logged
        assert "An error occurred: Test error" in str(mock_logger.error.call_args[0][0])

def test_get_dataset_schema_http_error(evaluation):
    """Test schema retrieval with HTTP error"""
    with patch.object(evaluation, '_get_dataset_id_based_on_dataset_type', return_value="test_dataset_id"), \
         patch('requests.post') as mock_post, \
         patch('ragaai_catalyst.evaluation.logger') as mock_logger:
        
        # Simulate an HTTP error
        mock_post.side_effect = requests.exceptions.HTTPError("HTTP Error")
        
        result = evaluation._get_dataset_schema()
        
        # Verify the results
        assert result == {}
        mock_logger.error.assert_called()
        assert "HTTP error occurred: HTTP Error" in str(mock_logger.error.call_args[0][0])

def test_get_dataset_schema_connection_error(evaluation):
    """Test schema retrieval with connection error"""
    with patch.object(evaluation, '_get_dataset_id_based_on_dataset_type', return_value="test_dataset_id"), \
         patch('requests.post') as mock_post, \
         patch('ragaai_catalyst.evaluation.logger') as mock_logger:
        
        # Simulate a connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection Error")
        
        result = evaluation._get_dataset_schema()
        
        # Verify the results
        assert result == {}
        mock_logger.error.assert_called()
        assert "Connection error occurred: Connection Error" in str(mock_logger.error.call_args[0][0])

def test_get_dataset_schema_timeout_error(evaluation):
    """Test schema retrieval with timeout error"""
    with patch.object(evaluation, '_get_dataset_id_based_on_dataset_type', return_value="test_dataset_id"), \
         patch('requests.post') as mock_post, \
         patch('ragaai_catalyst.evaluation.logger') as mock_logger:
        
        # Simulate a timeout error
        mock_post.side_effect = requests.exceptions.Timeout("Timeout Error")
        
        result = evaluation._get_dataset_schema()
        
        # Verify the results
        assert result == {}
        mock_logger.error.assert_called()
        assert "Timeout error occurred: Timeout Error" in str(mock_logger.error.call_args[0][0])

def test_get_dataset_id_error(evaluation):
    """Test error in getting dataset ID"""
    with patch.object(evaluation, '_get_dataset_id_based_on_dataset_type') as mock_get_id:
        # Simulate an error in getting dataset ID
        mock_get_id.side_effect = requests.exceptions.RequestException("Dataset ID Error")
        
        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            evaluation._get_dataset_schema()
        
        assert "Dataset ID Error" in str(exc_info.value)
# Fix for test_get_dataset_schema_success
def test_get_dataset_schema_success(evaluation):
    """Test successful schema retrieval"""
    mock_columns = [{"displayName": "col1"}, {"displayName": "col2"}, {"displayName": "col3"}]
    
    # Need to mock _get_dataset_id_based_on_dataset_type first
    with patch.object(evaluation, '_get_dataset_id_based_on_dataset_type', return_value="test_dataset_id"), \
         patch('requests.post') as mock_post:
        
        mock_post.return_value.json.return_value = {
            "data": {
                "columns": mock_columns
            }
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        result = evaluation._get_dataset_schema()
        assert result == mock_columns

# Test _get_metrics_schema_response
def test_get_metrics_schema_response_success(evaluation):
    """Test successful retrieval of metrics schema"""
    mock_metrics = [
        {"name": "Hallucination", "config": {"requiredFields": []}},
        {"name": "Faithfulness", "config": {"requiredFields": []}}
    ]
    
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "data": {"metrics": mock_metrics}
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        result = evaluation._get_metrics_schema_response()
        assert result == mock_metrics

def test_get_metrics_schema_response_error(evaluation):
    """Test metrics schema retrieval with error"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("Test error")
        with patch('ragaai_catalyst.evaluation.logger') as mock_logger:
            result = evaluation._get_metrics_schema_response()
            # The method returns None for RequestException
            assert result is None
            mock_logger.error.assert_called()  # Error should be logged
            assert "An error occurred: Test error" in str(mock_logger.error.call_args[0][0]) # Error should be logged

# Test _get_variablename_from_user_schema_mapping
def test_get_variablename_from_user_schema_mapping_success(evaluation):
    """Test successful variable name mapping"""
    schema_name = "query"
    metric_name = "Hallucination"
    schema_mapping = {"user_query": "Query"}
    metric_to_evaluate = "prompt"
    
    # Mock _get_dataset_schema to return user schema with user_query
    with patch.object(evaluation, '_get_dataset_schema', return_value=[
        {"displayName": "user_query"},
        {"displayName": "response"}
    ]):
        result = evaluation._get_variablename_from_user_schema_mapping(schema_name, metric_name, schema_mapping, metric_to_evaluate)
        assert result == "user_query"

def test_get_variablename_from_user_schema_mapping_column_not_found(evaluation):
    """Test variable name mapping with column not in dataset"""
    schema_name = "query"
    metric_name = "Hallucination"
    schema_mapping = {"missing_column": "Query"}
    metric_to_evaluate = "prompt"
    
    # Mock _get_dataset_schema to return user schema without missing_column
    with patch.object(evaluation, '_get_dataset_schema', return_value=[
        {"displayName": "user_query"},
        {"displayName": "response"}
    ]):
        with pytest.raises(ValueError, match="Column 'missing_column' is not present"):
            evaluation._get_variablename_from_user_schema_mapping(schema_name, metric_name, schema_mapping, metric_to_evaluate)

def test_get_variablename_from_user_schema_mapping_not_mapped(evaluation):
    """Test variable name mapping with unmapped schema"""
    schema_name = "unmapped_schema"
    metric_name = "Hallucination"
    schema_mapping = {"user_query": "Query"}
    metric_to_evaluate = "prompt"
    
    # Mock _get_dataset_schema to return user schema
    with patch.object(evaluation, '_get_dataset_schema', return_value=[
        {"displayName": "user_query"},
        {"displayName": "response"}
    ]):
        with pytest.raises(ValueError, match="Map 'unmapped_schema' column in schema_mapping"):
            evaluation._get_variablename_from_user_schema_mapping(schema_name, metric_name, schema_mapping, metric_to_evaluate)

# Test _get_executed_metrics_list
def test_get_executed_metrics_list_success(evaluation):
    """Test successful retrieval of executed metrics list"""
    mock_columns = [
        {"displayName": "metric1"},
        {"displayName": "_system_column"},  # Should be filtered out
        {"displayName": "metric2"}
    ]
    
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "data": {
                "datasetColumnsResponses": mock_columns
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        result = evaluation._get_executed_metrics_list()
        assert result == ["metric1", "metric2"]
        assert "_system_column" not in result

def test_get_executed_metrics_list_http_error(evaluation):
    """Test executed metrics list retrieval with HTTP error"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.HTTPError("HTTP Error")
        with patch('ragaai_catalyst.evaluation.logger') as mock_logger:
            result = evaluation._get_executed_metrics_list()
            # The method returns None for HTTPError
            assert result is None
            mock_logger.error.assert_called()
            assert "HTTP error occurred: HTTP Error" in str(mock_logger.error.call_args[0][0])
# Test add_metrics
def test_add_metrics_duplicate_column(evaluation):
    """Test add_metrics with duplicate column name"""
    metrics = [{
        "name": "Hallucination",
        "config": {
            "provider": "openai",
            "model": "gpt-4"
        },
        "column_name": "existing_column",  # Column already exists
        "schema_mapping": {
            "query": "Query",
            "response": "Response"
        }
    }]
    
    # Mock that the column already exists
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=["existing_column"]), \
         patch.object(evaluation, 'list_metrics', return_value=["Hallucination"]):
        
        with pytest.raises(ValueError, match="Column name 'existing_column' already exists"):
            evaluation.add_metrics(metrics)

def test_add_metrics_invalid_metric_name(evaluation):
    """Test add_metrics with invalid metric name"""
    metrics = [{
        "name": "InvalidMetric",  # Not in the available metrics
        "config": {
            "provider": "openai",
            "model": "gpt-4"
        },
        "column_name": "new_column",
        "schema_mapping": {
            "query": "Query",
            "response": "Response"
        }
    }]
    
    # Available metrics don't include "InvalidMetric"
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["Hallucination", "Faithfulness"]):
        
        with pytest.raises(ValueError, match="Enter a valid metric name"):
            evaluation.add_metrics(metrics)

def test_add_metrics_with_gte_threshold(evaluation):
    """Test add_metrics with gte threshold configuration as shown in README"""
    metrics = [{
        "name": "Faithfulness", 
        "config": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "threshold": {"gte": 0.323}
        }, 
        "column_name": "Faithfulness_gte", 
        "schema_mapping": {
            "Query": "prompt",
            "response": "response",
            "Context": "context",
            "expectedResponse": "expected_response"
        }
    }]
    
    # Mock required functions to test threshold handling
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["Faithfulness"]), \
         patch.object(evaluation, '_update_base_json') as mock_update, \
         patch('requests.post') as mock_post, \
         patch('builtins.print') as mock_print:
        
        # Configure update_base_json to return a valid response while allowing inspection
        mock_update.side_effect = lambda m: {
            "datasetId": evaluation.dataset_id,
            "metricParams": [{
                "metricSpec": {
                    "name": m[0]["name"],
                    "displayName": m[0]["column_name"],
                    "config": {
                        "params": {
                            "threshold": {"gte": 0.323}
                        }
                    }
                }
            }]
        }
        
        mock_post.return_value.json.return_value = {
            "success": True,
            "message": "Metrics added successfully",
            "data": {"jobId": "test_job_id"}
        }
        mock_post.return_value.status_code = 200
        
        evaluation.add_metrics(metrics)
        
        # Verify the threshold was properly passed to update_base_json
        mock_update.assert_called_once()
        # Check if the threshold appears in the call arguments
        call_args = mock_update.call_args[0][0]
        assert call_args[0]["config"]["threshold"] == {"gte": 0.323}
        assert evaluation.jobId == "test_job_id"

def test_add_metrics_with_lte_threshold(evaluation):
    """Test add_metrics with lte threshold configuration as shown in README"""
    metrics = [{
        "name": "Hallucination", 
        "config": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "threshold": {"lte": 0.323}
        }, 
        "column_name": "Hallucination_lte", 
        "schema_mapping": {
            "Query": "prompt",
            "response": "response"
        }
    }]
    
    # Mock required functions to test threshold handling
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["Hallucination"]), \
         patch.object(evaluation, '_update_base_json') as mock_update, \
         patch('requests.post') as mock_post:
        
        # Configure update_base_json to return a valid response while allowing inspection
        mock_update.side_effect = lambda m: {
            "datasetId": evaluation.dataset_id,
            "metricParams": [{
                "metricSpec": {
                    "name": m[0]["name"],
                    "displayName": m[0]["column_name"],
                    "config": {
                        "params": {
                            "threshold": {"lte": 0.323}
                        }
                    }
                }
            }]
        }
        
        mock_post.return_value.json.return_value = {
            "success": True,
            "message": "Metrics added successfully",
            "data": {"jobId": "test_job_id"}
        }
        mock_post.return_value.status_code = 200
        
        evaluation.add_metrics(metrics)
        
        # Verify the threshold was properly passed to update_base_json
        mock_update.assert_called_once()
        # Check if the threshold appears in the call arguments
        call_args = mock_update.call_args[0][0]
        assert call_args[0]["config"]["threshold"] == {"lte": 0.323}
        assert evaluation.jobId == "test_job_id"

def test_add_metrics_with_eq_threshold(evaluation):
    """Test add_metrics with eq threshold configuration as shown in README"""
    metrics = [{
        "name": "Hallucination", 
        "config": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "threshold": {"eq": 0.323}
        }, 
        "column_name": "Hallucination_eq", 
        "schema_mapping": {
            "Query": "prompt",
            "response": "response"
        }
    }]
    
    # Mock required functions to test threshold handling
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["Hallucination"]), \
         patch.object(evaluation, '_update_base_json') as mock_update, \
         patch('requests.post') as mock_post:
        
        # Configure update_base_json to return a valid response while allowing inspection
        mock_update.side_effect = lambda m: {
            "datasetId": evaluation.dataset_id,
            "metricParams": [{
                "metricSpec": {
                    "name": m[0]["name"],
                    "displayName": m[0]["column_name"],
                    "config": {
                        "params": {
                            "threshold": {"eq": 0.323}
                        }
                    }
                }
            }]
        }
        
        mock_post.return_value.json.return_value = {
            "success": True,
            "message": "Metrics added successfully",
            "data": {"jobId": "test_job_id"}
        }
        mock_post.return_value.status_code = 200
        
        evaluation.add_metrics(metrics)
        
        # Verify the threshold was properly passed to update_base_json
        mock_update.assert_called_once()
        # Check if the threshold appears in the call arguments
        call_args = mock_update.call_args[0][0]
        assert call_args[0]["config"]["threshold"] == {"eq": 0.323}
        assert evaluation.jobId == "test_job_id"

def test_add_multiple_metrics(evaluation):
    """Test adding multiple metrics at once with different thresholds as shown in README"""
    metrics = [
        {
            "name": "Faithfulness", 
            "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.323}}, 
            "column_name": "Faithfulness_gte", 
            "schema_mapping": {"Query": "prompt", "response": "response"}
        },
        {
            "name": "Hallucination", 
            "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"lte": 0.323}}, 
            "column_name": "Hallucination_lte", 
            "schema_mapping": {"Query": "prompt", "response": "response"}
        },
        {
            "name": "Hallucination", 
            "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"eq": 0.323}}, 
            "column_name": "Hallucination_eq", 
            "schema_mapping": {"Query": "prompt", "response": "response"}
        }
    ]
    
    # Mock required functions
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["Faithfulness", "Hallucination"]), \
         patch.object(evaluation, '_update_base_json') as mock_update, \
         patch('requests.post') as mock_post:
        
        # Configure update_base_json to return a valid response
        mock_update.return_value = {
            "datasetId": evaluation.dataset_id,
            "metricParams": []
        }
        
        mock_post.return_value.json.return_value = {
            "success": True,
            "message": "Metrics added successfully",
            "data": {"jobId": "test_job_id"}
        }
        mock_post.return_value.status_code = 200
        
        evaluation.add_metrics(metrics)
        
        # Verify the update_base_json was called with all three metrics
        mock_update.assert_called_once()
        call_args = mock_update.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0]["name"] == "Faithfulness"
        assert call_args[1]["name"] == "Hallucination"
        assert call_args[2]["name"] == "Hallucination"
        assert call_args[0]["config"]["threshold"] == {"gte": 0.323}
        assert call_args[1]["config"]["threshold"] == {"lte": 0.323}
        assert call_args[2]["config"]["threshold"] == {"eq": 0.323}
        assert evaluation.jobId == "test_job_id"





def test_get_results_error(evaluation):
    """Test results retrieval with error"""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("Test error")
        with patch('ragaai_catalyst.evaluation.logger') as mock_logger:
            result = evaluation.get_results()
            assert isinstance(result, pd.DataFrame)
            assert result.empty
            mock_logger.error.assert_called()  # Verify error was logged
