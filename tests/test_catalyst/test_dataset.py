import pytest
import os
import dotenv
import tempfile
import json
dotenv.load_dotenv()
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from unittest.mock import patch, Mock, MagicMock, mock_open, ANY
import requests
from ragaai_catalyst import Dataset, RagaAICatalyst
from ragaai_catalyst.dataset import JOB_STATUS_FAILED, JOB_STATUS_IN_PROGRESS, JOB_STATUS_COMPLETED

# Path to test data files
csv_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_test_dataset.csv"))
jsonl_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_test_dataset.jsonl"))

# Constants for testing
TEST_PROJECT_NAME = "test_project"
TEST_DATASET_NAME = "test_dataset"
TEST_SCHEMA_MAPPING = {
    'Query': 'prompt',
    'Response': 'response',
    'Context': 'context',
    'ExpectedResponse': 'expected_response'
}


@pytest.fixture
def base_url():
    return os.getenv("RAGAAI_CATALYST_BASE_URL", "https://api.catalyst.raga.ai")

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY", "test_access_key"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY", "test_secret_key")
    }

@pytest.fixture
def mock_project_response():
    """Mock response for projects list API"""
    return {
        "data": {
            "content": [
                {"id": "123", "name": TEST_PROJECT_NAME},
                {"id": "456", "name": "other_project"}
            ]
        }
    }

@pytest.fixture
def mock_dataset_response():
    """Mock response for datasets list API"""
    return {
        "success": True,
        "message": "Datasets retrieved successfully",
        "data": {
            "content": [
                {"id": "789", "name": TEST_DATASET_NAME, "datasetType": "prompt"},
                {"id": "101", "name": "other_dataset", "datasetType": "chat"}
            ]
        }
    }

@pytest.fixture
def mock_schema_elements_response():
    """Mock response for schema elements API"""
    return {
        "success": True,
        "message": "Schema elements retrieved successfully",
        "data": {
            "content": [
                {
                    "id": "schema-1",
                    "name": "query",
                    "dataType": "string",
                    "mapTo": "prompt"
                },
                {
                    "id": "schema-2",
                    "name": "response",
                    "dataType": "string",
                    "mapTo": "response"
                }
            ]
        }
    }

@pytest.fixture
def dataset():
    """Create dataset instance with mocked initialization"""
    with patch('requests.get') as mock_get:
        # Mock the project list API call in the initialization
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [
                    {"id": "123", "name": TEST_PROJECT_NAME},
                    {"id": "456", "name": "other_project"}
                ]
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        return Dataset(project_name=TEST_PROJECT_NAME)

# Core functionality tests as mentioned in the README

def test_list_datasets(dataset, mock_dataset_response):
    """Test listing datasets - core functionality from README"""
    with patch('requests.post') as mock_post:
        # Configure mock to return a successful response
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        result = dataset.list_datasets()
        
        # Verify expected behavior
        assert isinstance(result, list), "Should return a list"
        assert len(result) == 2, "Should contain 2 datasets"
        assert TEST_DATASET_NAME in result, f"Should include {TEST_DATASET_NAME}"
        assert "other_dataset" in result, "Should include other_dataset"
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        assert "/v2/llm/dataset" in mock_post.call_args[0][0], "Should call dataset endpoint"
        assert "X-Project-Id" in mock_post.call_args[1]["headers"], "Should include project ID header"

def test_list_datasets_error(dataset):
    """Test listing datasets with API error"""
    with patch('requests.post') as mock_post:
        # Configure the mock to simulate a request exception
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        # Verify expected behavior
        with pytest.raises(requests.exceptions.RequestException, match="API Error"):
            dataset.list_datasets()

def test_get_schema_mapping(dataset, mock_schema_elements_response):
    """Test get_schema_mapping - core functionality from README"""
    with patch('requests.get') as mock_get:
        # Configure mock to return a successful response with schemaElements key
        modified_response = {
            "success": True,
            "data": {
                "schemaElements": ["prompt", "response", "context", "expected_response"]
            }
        }
        mock_get.return_value.json.return_value = modified_response
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        schema_elements = dataset.get_schema_mapping()
        
        # Verify expected behavior
        assert isinstance(schema_elements, list), "Should return a list of schema elements"
        assert len(schema_elements) == 4, "Should return 4 schema elements as per mock response"
        assert "prompt" in schema_elements, "Should include 'prompt' schema element"
        assert "response" in schema_elements, "Should include 'response' schema element"
        
        # Verify the get request was called with correct parameters
        mock_get.assert_called_once()
        assert "/v1/llm/schema-elements" in mock_get.call_args[0][0], "Should call the schema elements endpoint"

def test_get_schema_mapping_error(dataset):
    """Test get_schema_mapping with API error"""
    with patch('requests.get') as mock_get:
        # Configure the mock to simulate a request exception
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        # Verify expected behavior
        with pytest.raises(requests.exceptions.RequestException, match="API Error"):
            dataset.get_schema_mapping()

def test_get_schema_mapping_unsuccessful(dataset):
    """Test get_schema_mapping with unsuccessful API response"""
    with patch('requests.get') as mock_get:
        # Configure mock to return unsuccessful response
        # Include the 'data' key with empty schemaElements since the code tries to access it first
        mock_get.return_value.json.return_value = {
            "success": False,
            "message": "Failed to retrieve schema elements",
            "data": {
                "schemaElements": []
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        with pytest.raises(ValueError, match="Unable to fetch Schema Elements"):
            dataset.get_schema_mapping()

def test_get_dataset_columns(dataset, mock_dataset_response):
    """Test getting dataset columns"""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Mock response for dataset listing to verify existence
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Mock response for dataset details
        dataset_details_response = {
            "success": True,
            "message": "Dataset details retrieved successfully",
            "data": {
                "datasetColumnsResponses": [
                    {"displayName": "query", "columnType": "prompt"},
                    {"displayName": "response", "columnType": "response"},
                    {"displayName": "_system", "columnType": "system"}
                ]
            }
        }
        mock_get.return_value.json.return_value = dataset_details_response
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        columns = dataset.get_dataset_columns(TEST_DATASET_NAME)
        
        # Verify expected behavior
        assert isinstance(columns, list), "Should return a list of column names"
        assert len(columns) == 2, "Should return 2 columns (excluding system column)"
        assert "query" in columns, "Should include 'query' column"
        assert "response" in columns, "Should include 'response' column"
        assert "_system" not in columns, "Should not include system columns"

def test_incorrect_dataset(dataset, mock_dataset_response):
    """Test get_dataset_columns with nonexistent dataset"""
    with patch('requests.post') as mock_post:
        # Configure the mock to return a predefined dataset list
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Verify expected behavior
        with pytest.raises(ValueError, match="Dataset .* does not exists"):
            dataset.get_dataset_columns(dataset_name="nonexistent_dataset")


# Tests for creating datasets from CSV files


def test_create_from_csv_dataset_exists(dataset, mock_dataset_response):
    """Test creating a dataset with a name that already exists"""
    with patch('requests.post') as mock_post:
        # Mock list_datasets to return a list including the target dataset
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        with pytest.raises(ValueError, match="Dataset name .* already exists"):
            dataset.create_from_csv(
                csv_path=csv_path,
                dataset_name=TEST_DATASET_NAME,  # Name already in mock_dataset_response
                schema_mapping=TEST_SCHEMA_MAPPING
            )

def test_create_from_csv_presigned_url_failure(dataset, mock_dataset_response):
    """Test handling of failure to get presigned URL"""
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get:
        
        # Mock list_datasets to return a list not including the target dataset
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Mock presigned URL failure
        mock_get.return_value.json.return_value = {
            "success": False, 
            "message": "Failed to get URL"
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        with pytest.raises(ValueError, match="Unable to fetch presignedUrl"):
            dataset.create_from_csv(
                csv_path=csv_path,
                dataset_name="new_dataset",
                schema_mapping=TEST_SCHEMA_MAPPING
            )

def test_upload_csv_missing_file(dataset):
    """Test creating a dataset with a non-existent CSV file"""
    with patch('os.path.exists') as mock_exists:
        # Mock file existence check to return False
        mock_exists.return_value = False
        
        # Verify expected behavior
        with pytest.raises(ValueError):
            dataset.create_from_csv(
                csv_path="/nonexistent/path.csv",
                dataset_name="new_dataset", 
                schema_mapping=TEST_SCHEMA_MAPPING
            )

# Tests for schema mapping validation

def test_create_from_csv_empty_schema_mapping(dataset, mock_dataset_response):
    """Test creating a dataset with an empty schema mapping"""
    with patch('requests.post') as mock_post:
        # Mock list_datasets to return a list not including the target dataset
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Call the method under test with empty schema mapping
        with pytest.raises(ValueError):
            dataset.create_from_csv(
                csv_path=csv_path,
                dataset_name="new_dataset",
                schema_mapping={} # Empty dict
            )

# Tests for add_rows functionality

def test_add_rows_success(dataset, mock_dataset_response):
    """Test adding rows to an existing dataset - functionality from README"""
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get, \
         patch('requests.put') as mock_put, \
         patch('builtins.open', new_callable=MagicMock), \
         patch('pandas.read_csv') as mock_read_csv, \
         patch.object(dataset, 'list_datasets', return_value=[TEST_DATASET_NAME]), \
         patch.object(dataset, 'get_dataset_columns', return_value=["query", "response"]):
        
        # Mock dataset list and dataset columns API calls
        mock_post.return_value.json.side_effect = [
            mock_dataset_response,  # For list_datasets check
            {
                "success": True, 
                "message": "Rows added successfully", 
                "data": {"jobId": "job-123"}  # For CSV upload with jobId in the data field
            }
        ]
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.content = b'{}'  # Add this line to fix KeyError: 'content'

        # Mock dataset details API call
        mock_get.return_value.json.side_effect = [
            # First for presigned URL
            {
                "success": True, 
                "message": "Presigned URL generated",
                "data": {"presignedUrl": "https://example.com/upload-url", "fileName": "test-file.csv"}
            },
            # Then for dataset columns
            {
                "success": True,
                "message": "Dataset columns retrieved",
                "data": {"datasetColumnsResponses": [
                    {"displayName": "query", "columnType": "prompt"},
                    {"displayName": "response", "columnType": "response"}
                ]}
            }
        ]
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Mock CSV file read
        mock_df = pd.DataFrame({
            "query": ["What is AI?", "How does ML work?"],
            "response": ["AI is...", "ML works by..."]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock PUT request for uploading CSV
        mock_put.return_value.status_code = 200
        mock_put.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        dataset.add_rows(
            csv_path=csv_path,
            dataset_name=TEST_DATASET_NAME
        )

def test_add_rows_dataset_not_found(dataset, mock_dataset_response):
    """Test adding rows to a non-existent dataset"""
    with patch('requests.post') as mock_post:
        # Mock list_datasets to return a list not including the target dataset
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        with pytest.raises(ValueError, match="Dataset .* does not exists"):
            dataset.add_rows(
                csv_path=csv_path,
                dataset_name="nonexistent_dataset"
            )

def test_create_from_jsonl_success(dataset, mock_dataset_response):
    """Test creating a dataset from a JSONL file - functionality from README"""
    test_csv_path = "/tmp/test_converted.csv"
    with patch.object(dataset, '_jsonl_to_csv') as mock_convert, \
         patch.object(dataset, 'create_from_csv') as mock_create_csv, \
         patch('os.path.exists') as mock_exists, \
         patch('os.remove') as mock_remove:
        
        # Mock the conversion and file operations
        mock_convert.return_value = test_csv_path
        mock_create_csv.return_value = None
        mock_exists.return_value = True
        
        # Call the method under test
        dataset.create_from_jsonl(
            jsonl_path=jsonl_path,
            dataset_name="new_jsonl_dataset",
            schema_mapping=TEST_SCHEMA_MAPPING
        )
        
        # Verify the conversion and creation were called correctly
        mock_convert.assert_called_once_with(jsonl_path, ANY)
        mock_create_csv.assert_called_once()
        
        # Get the call arguments
        expected_path = os.path.join(tempfile.gettempdir(), f"new_jsonl_dataset.csv")
        assert mock_create_csv.call_args[0][0] == expected_path  # First positional arg (csv_path)
        assert mock_create_csv.call_args[0][1] == "new_jsonl_dataset"  # Second positional arg (dataset_name)
        assert mock_create_csv.call_args[0][2] == TEST_SCHEMA_MAPPING  # Third positional arg (schema_mapping)
        
        mock_remove.assert_called_once()

def test_create_from_df_success(dataset, mock_dataset_response):
    """Test creating a dataset from a DataFrame - functionality from README"""
    test_csv_path = "/tmp/test_converted.csv"
    with patch.object(dataset, 'create_from_csv') as mock_create_csv, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists') as mock_exists, \
         patch('os.remove') as mock_remove, \
         patch('tempfile.NamedTemporaryFile') as mock_temp:
        
        # Create a test DataFrame
        test_df = pd.DataFrame({
            "query": ["What is AI?", "How does ML work?"],
            "response": ["AI is...", "ML works by..."]
        })
        
        # Setup temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = test_csv_path
        mock_temp.return_value.__enter__.return_value = mock_temp_file
        
        # Mock the file operations
        mock_create_csv.return_value = None
        mock_to_csv.return_value = None
        mock_exists.return_value = True
        
        # Call the method under test
        dataset.create_from_df(
            df=test_df,
            dataset_name="new_df_dataset",
            schema_mapping=TEST_SCHEMA_MAPPING
        )
        
        # Verify the conversion and creation were called correctly
        mock_to_csv.assert_called_once()
        mock_create_csv.assert_called_once()
        
        # Check positional arguments
        expected_path = os.path.join(tempfile.gettempdir(), f"new_df_dataset.csv")
        assert mock_create_csv.call_args[0][0] == expected_path  # First positional arg (csv_path)
        assert mock_create_csv.call_args[0][1] == "new_df_dataset"  # Second positional arg (dataset_name)
        assert mock_create_csv.call_args[0][2] == TEST_SCHEMA_MAPPING  # Third positional arg (schema_mapping)
        
        mock_remove.assert_called_once()

def test_add_rows_from_jsonl_success(dataset):
    """Test adding rows from a JSONL file - functionality from README"""
    test_csv_path = "/tmp/test_converted.csv"
    with patch.object(dataset, '_jsonl_to_csv') as mock_convert, \
         patch.object(dataset, 'add_rows') as mock_add_rows, \
         patch('os.path.exists') as mock_exists, \
         patch('os.remove') as mock_remove:
        
        # Mock the conversion and file operations
        mock_convert.return_value = test_csv_path
        mock_add_rows.return_value = None
        mock_exists.return_value = True
        
        # Call the method under test
        dataset.add_rows_from_jsonl(
            jsonl_path="/path/to/test.jsonl",
            dataset_name="existing_dataset"
        )
        
        # Verify the conversion and addition were called correctly
        mock_convert.assert_called_once_with("/path/to/test.jsonl", ANY)
        mock_add_rows.assert_called_once()
        
        # Check positional arguments
        expected_path = os.path.join(tempfile.gettempdir(), f"existing_dataset.csv")
        assert mock_add_rows.call_args[0][0] == expected_path  # First positional arg (csv_path)
        assert mock_add_rows.call_args[0][1] == "existing_dataset"  # Second positional arg (dataset_name)
        
        mock_remove.assert_called_once()

def test_add_rows_from_df_success(dataset):
    """Test adding rows from a DataFrame - functionality from README"""
    test_csv_path = "/tmp/test_converted.csv"
    with patch.object(dataset, 'add_rows') as mock_add_rows, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('os.path.exists') as mock_exists, \
         patch('os.remove') as mock_remove, \
         patch('tempfile.NamedTemporaryFile') as mock_temp:
        
        # Create a test DataFrame
        test_df = pd.DataFrame({
            "query": ["What is AI?", "How does ML work?"],
            "response": ["AI is...", "ML works by..."]
        })
        
        # Setup temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_temp.csv"
        mock_temp.return_value.__enter__.return_value = mock_temp_file
        
        # Mock the file operations
        mock_add_rows.return_value = None
        mock_to_csv.return_value = None
        mock_exists.return_value = True
        
        # Call the method under test
        dataset.add_rows_from_df(
            df=test_df,
            dataset_name="existing_dataset"
        )
        
        # Verify the conversion and addition were called correctly
        mock_to_csv.assert_called_once()
        mock_add_rows.assert_called_once()
        
        # Check positional arguments
        expected_path = os.path.join(tempfile.gettempdir(), f"existing_dataset.csv")
        assert mock_add_rows.call_args[0][0] == expected_path  # First positional arg (csv_path)
        assert mock_add_rows.call_args[0][1] == "existing_dataset"  # Second positional arg (dataset_name)
        
        mock_remove.assert_called_once()

# Tests for add_columns functionality

def test_add_columns_success(dataset, mock_dataset_response):
    """Test adding a column to an existing dataset - functionality from README"""
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get:
        
        # Mock dataset list API call
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Mock model parameters API call
        parameters_response = {
            "data": [
                {"name": "temperature", "value": 0.7, "type": "float"},
                {"name": "max_tokens", "value": 100, "type": "int"},
                {"name": "top_p", "value": 1.0, "type": "float"}
            ]
        }
        
        # Mock column addition API call
        column_add_response = {
            "success": True,
            "message": "Column added successfully",
            "data": {"jobId": "job-456"}
        }
        
        # Configure mocks to return different responses based on URL
        def side_effect_post(*args, **kwargs):
            url = args[0]
            if "/playground/providers/models/parameters/list" in url:
                mock_resp = MagicMock()
                mock_resp.json.return_value = parameters_response
                mock_resp.status_code = 200
                return mock_resp
            elif "/v2/llm/dataset/add-column" in url:
                mock_resp = MagicMock()
                mock_resp.json.return_value = column_add_response
                mock_resp.status_code = 200
                return mock_resp
            else:
                mock_resp = MagicMock()
                mock_resp.json.return_value = mock_dataset_response
                mock_resp.status_code = 200
                return mock_resp
                
        mock_post.side_effect = side_effect_post
        
        # Test data for add_columns
        text_fields = [
            {"role": "system", "content": "you are an evaluator"},
            {"role": "user", "content": "are any of the {{context1}} {{feedback1}} related to broken hand"}
        ]
        column_name = "test_column"
        provider = "openai"
        model = "gpt-4o-mini"
        variables = {"context1": "context", "feedback1": "feedback"}
        
        # Call the method under test
        dataset.add_columns(
            text_fields=text_fields,
            dataset_name=TEST_DATASET_NAME,
            column_name=column_name,
            provider=provider,
            model=model,
            variables=variables
        )
        
        # Verify API call parameters
        add_column_call = None
        for call in mock_post.call_args_list:
            if "/v2/llm/dataset/add-column" in call[0][0]:
                add_column_call = call
                break
        
        assert add_column_call is not None, "Should call add column API"
        payload = add_column_call[1]["json"]
        assert payload["columnName"] == column_name, "Should use correct column name"
        assert payload["variables"] == variables, "Should include variables"
        assert "promptTemplate" in payload, "Should include prompt template"
        assert payload["promptTemplate"]["textFields"] == text_fields, "Should include text fields"
        assert f"{provider}/{model}" in payload["promptTemplate"]["modelSpecs"]["model"], "Should include provider and model"

def test_add_columns_invalid_text_fields(dataset):
    """Test add_columns with invalid text_fields parameter"""
    # Test with non-list text_fields
    with pytest.raises(ValueError, match="text_fields must be a list"):
        dataset.add_columns(
            text_fields="not a list",
            dataset_name=TEST_DATASET_NAME,
            column_name="test_column",
            provider="openai",
            model="gpt-4"
        )
    
    # Test with invalid text field format
    with pytest.raises(ValueError, match="Each text field must be a dictionary"):
        dataset.add_columns(
            text_fields=["not a dict"],
            dataset_name=TEST_DATASET_NAME,
            column_name="test_column",
            provider="openai",
            model="gpt-4"
        )

def test_add_columns_dataset_not_found(dataset, mock_dataset_response):
    """Test add_columns with non-existent dataset"""
    with patch('requests.post') as mock_post:
        # Configure mock to return dataset list without the target dataset
        mock_post.return_value.json.return_value = mock_dataset_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        
        # Test data for add_columns
        text_fields = [
            {"role": "system", "content": "you are an evaluator"},
            {"role": "user", "content": "are any of the contexts related to broken hand"}
        ]
        
        # Call method with non-existent dataset
        with pytest.raises(ValueError, match="Dataset .* not found"):
            dataset.add_columns(
                text_fields=text_fields,
                dataset_name="nonexistent_dataset",
                column_name="test_column",
                provider="openai",
                model="gpt-4"
            )

# Tests for internal utility functions

def test_jsonl_to_csv_empty_file(dataset, tmp_path):
    """Test the _jsonl_to_csv method with an empty JSONL file"""
    # Create empty JSONL file
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("")
    csv_output = tmp_path / "output.csv"
    
    # Mock print function to check it's called
    with patch('builtins.print') as mock_print:
        dataset._jsonl_to_csv(str(empty_jsonl), str(csv_output))
        
        # Verify print is called with expected message
        mock_print.assert_called_once_with("Empty JSONL file.")
        
        # Verify CSV file was not created or is empty
        assert not csv_output.exists() or csv_output.stat().st_size == 0

def test_jsonl_to_csv_success(dataset, tmp_path):
    """Test the _jsonl_to_csv method with valid JSONL data"""
    # Create test JSONL file
    test_jsonl = tmp_path / "test.jsonl"
    jsonl_content = '{"col1": "value1", "col2": "value2"}\n{"col1": "value3", "col2": "value4"}'
    test_jsonl.write_text(jsonl_content)
    csv_output = tmp_path / "output.csv"
    
    # Mock print function to check it's called
    with patch('builtins.print') as mock_print:
        dataset._jsonl_to_csv(str(test_jsonl), str(csv_output))
        
        # Verify print is called with expected message about conversion
        mock_print.assert_called_once()
        assert "Converted" in mock_print.call_args[0][0]
        
        # Verify CSV file was created and has content
        assert csv_output.exists()
        content = csv_output.read_text()
        assert "col1,col2" in content
        assert "value1,value2" in content.replace('\r', '')

def test_create_from_jsonl_io_error(dataset):
    """Test create_from_jsonl with an IO error"""
    with patch.object(dataset, '_jsonl_to_csv') as mock_jsonl_to_csv, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        # Mock an IO error when converting JSONL to CSV
        mock_jsonl_to_csv.side_effect = IOError("File not found")
        
        # Call the method and verify it raises the expected error
        with pytest.raises(IOError):
            dataset.create_from_jsonl("nonexistent.jsonl", "test_dataset", {})
        
        # Verify logger.error is called with expected message
        mock_logger.error.assert_called_once()
        assert "Error converting JSONL to CSV" in mock_logger.error.call_args[0][0]

def test_create_from_jsonl_file_removal_error(dataset, tmp_path):
    """Test create_from_jsonl error in file removal"""
    # Create a mock temporary path that will be used
    test_path = str(tmp_path / "test_dataset.csv")
    
    # Setup the scenario: create_from_csv succeeds but file removal fails
    with patch.object(dataset, '_jsonl_to_csv') as mock_jsonl_to_csv, \
         patch.object(dataset, 'create_from_csv') as mock_create_from_csv, \
         patch('os.path.exists') as mock_exists, \
         patch('os.remove') as mock_remove, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger, \
         patch('tempfile.gettempdir', return_value=str(tmp_path)):
        
        # Configure mocks
        mock_exists.return_value = True
        mock_remove.side_effect = Exception("Permission denied")
        
        # Call the method
        dataset.create_from_jsonl("test.jsonl", "test_dataset", {})
        
        # Verify logger.error is called for file removal error
        mock_logger.error.assert_called_once()
        assert "Error removing temporary CSV file" in mock_logger.error.call_args[0][0]

def test_add_rows_from_jsonl_exception(dataset):
    """Test add_rows_from_jsonl with an exception"""
    with patch.object(dataset, '_jsonl_to_csv') as mock_jsonl_to_csv, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        # Mock a Unicode error when converting JSONL to CSV
        mock_jsonl_to_csv.side_effect = UnicodeError("Encoding error", 0, 1, 0, 1)
        
        # Call the method and verify it raises the expected error
        with pytest.raises(UnicodeError):
            dataset.add_rows_from_jsonl("test.jsonl", "test_dataset")
        
        # Verify logger.error is called with expected message
        mock_logger.error.assert_called_once()
        assert "Error converting JSONL to CSV" in mock_logger.error.call_args[0][0]

def test_create_from_df_io_error(dataset):
    """Test create_from_df with an IOError"""
    test_df = pd.DataFrame({'col1': ['val1', 'val2'], 'col2': ['val3', 'val4']})
    
    with patch.object(test_df, 'to_csv') as mock_to_csv, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        # Mock an IO error when writing DataFrame to CSV
        mock_to_csv.side_effect = IOError("Disk full")
        
        # Call the method and verify it raises the expected error
        with pytest.raises(IOError):
            dataset.create_from_df(test_df, "test_dataset", TEST_SCHEMA_MAPPING)
        
        # Verify logger.error is called with expected message
        mock_logger.error.assert_called_once()
        assert "Error converting DataFrame to CSV" in mock_logger.error.call_args[0][0]

def test_create_from_df_file_removal_error(dataset, tmp_path):
    """Test create_from_df with file removal error"""
    test_df = pd.DataFrame({'col1': ['val1', 'val2'], 'col2': ['val3', 'val4']})
    
    with patch.object(dataset, 'create_from_csv') as mock_create_from_csv, \
         patch('os.path.exists') as mock_exists, \
         patch('os.remove') as mock_remove, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger, \
         patch('tempfile.gettempdir', return_value=str(tmp_path)):
        
        # Configure mocks
        mock_exists.return_value = True
        mock_remove.side_effect = Exception("Permission denied")
        
        # Call the method
        dataset.create_from_df(test_df, "test_dataset", TEST_SCHEMA_MAPPING)
        
        # Verify logger.error is called for file removal error
        mock_logger.error.assert_called_once()
        assert "Error removing temporary CSV file" in mock_logger.error.call_args[0][0]

def test_add_rows_from_df_exception(dataset):
    """Test add_rows_from_df with an exception"""
    test_df = pd.DataFrame({'col1': ['val1', 'val2'], 'col2': ['val3', 'val4']})
    
    with patch.object(test_df, 'to_csv') as mock_to_csv, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        # Mock a UnicodeError when writing DataFrame to CSV
        mock_to_csv.side_effect = UnicodeError("Encoding error", 0, 1, 0, 1)
        
        # Call the method and verify it raises the expected error
        with pytest.raises(UnicodeError):
            dataset.add_rows_from_df(test_df, "test_dataset")
        
        # Verify logger.error is called with expected message
        mock_logger.error.assert_called_once()
        assert "Error converting DataFrame to CSV" in mock_logger.error.call_args[0][0]

# Tests for get_status functionality

def test_get_status_completed(dataset):
    """Test getting job status when completed"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('builtins.print') as mock_print, \
         patch.object(Dataset, 'BASE_URL', 'https://api.catalyst.raga.ai/api'):
        
        # Mock the status API call
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {
                "content": [
                    {"id": "test-job-id", "status": "Completed"} # Maps to JOB_STATUS_COMPLETED
                ]
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_COMPLETED, "Should return success status"
        assert mock_print.call_count >= 1

def test_get_status_in_progress(dataset):
    """Test getting job status when in progress"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('builtins.print') as mock_print, \
         patch.object(Dataset, 'BASE_URL', 'https://api.catalyst.raga.ai/api'):
        
        # Mock the status API call
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {
                "content": [
                    {"id": "test-job-id", "status": "In Progress"} # Maps to JOB_STATUS_IN_PROGRESS
                ]
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_IN_PROGRESS, "Should return in_progress status"
        assert mock_print.call_count >= 1

def test_get_status_failed(dataset):
    """Test getting job status when failed"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('builtins.print') as mock_print, \
         patch.object(Dataset, 'BASE_URL', 'https://api.catalyst.raga.ai/api'):
        
        # Mock the status API call
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {
                "content": [
                    {"id": "test-job-id", "status": "Failed"} # Maps to JOB_STATUS_FAILED
                ]
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_FAILED, "Should return failed status"
        
        # Print is called for Failed status
        mock_print.assert_called_once()

def test_get_status_request_exception(dataset):
    """Test getting job status with request exception"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        
        # Mock the status API call to raise an exception
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_FAILED, "Should return failed status on exception"
        mock_logger.error.assert_called_once()

def test_get_status_http_error(dataset):
    """Test getting job status with HTTP error"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        
        # Mock the status API call to raise an HTTP error
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_FAILED, "Should return failed status on HTTP error"
        mock_logger.error.assert_called_once()
        assert "HTTP error" in mock_logger.error.call_args[0][0]

def test_get_status_connection_error(dataset):
    """Test getting job status with connection error"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        
        # Mock the status API call to raise a connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_FAILED, "Should return failed status on connection error"
        mock_logger.error.assert_called_once()
        assert "Connection error" in mock_logger.error.call_args[0][0]

def test_get_status_timeout_error(dataset):
    """Test getting job status with timeout error"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        
        # Mock the status API call to raise a timeout error
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_FAILED, "Should return failed status on timeout"
        mock_logger.error.assert_called_once()
        assert "Timeout error" in mock_logger.error.call_args[0][0]

def test_get_status_general_exception(dataset):
    """Test getting job status with unexpected exception"""
    # Set job ID
    dataset.jobId = "test-job-id"
    
    with patch('requests.get') as mock_get, \
         patch('ragaai_catalyst.dataset.logger') as mock_logger:
        
        # Mock the status API call to raise an unexpected exception
        mock_get.side_effect = Exception("Some unexpected error")
        
        # Call the method under test
        status = dataset.get_status()
        
        # Verify expected behavior
        assert status == JOB_STATUS_FAILED, "Should return failed status on any exception"
        mock_logger.error.assert_called_once()
        assert "unexpected error" in mock_logger.error.call_args[0][0].lower()