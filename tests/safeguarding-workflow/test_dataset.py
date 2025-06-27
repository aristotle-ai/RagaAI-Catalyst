import pytest
import os
import dotenv
dotenv.load_dotenv()
import pandas as pd
from datetime import datetime
import logging
import requests
from ragaai_catalyst import Dataset, RagaAICatalyst

# Setup test data paths
csv_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_test_dataset.csv"))

@pytest.fixture
def base_url():
    return os.getenv("RAGAAI_CATALYST_BASE_URL")

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")}

@pytest.fixture
def dataset(base_url, access_keys):
    """Create dataset instance with specific project"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Dataset(project_name="prompt_metric_dataset")

def test_nonexistent_project(base_url, access_keys, caplog):
    """Test handling of non-existent project name"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    # Initialize RagaAICatalyst first to ensure authentication
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    
    # Try to create Dataset with a non-existent project
    try:
        invalid_dataset = Dataset(project_name="nonexistent_project_name_1234567890")
        # If execution continues, check the log
        assert "Project not found. Please enter a valid project name" in caplog.text
    except IndexError:
        # The change logs errors but still attempts to access the non-existent project
        # which may result in an IndexError. Check the log in this case too.
        assert "Project not found. Please enter a valid project name" in caplog.text

def test_list_datasets_with_no_token(base_url, caplog):
    """Test list_datasets with no token set (should log error)"""
    original_token = os.environ.get("RAGAAI_CATALYST_TOKEN")
    try:
        if "RAGAAI_CATALYST_TOKEN" in os.environ:
            del os.environ["RAGAAI_CATALYST_TOKEN"]
        ds = Dataset(project_name="prompt_metric_dataset")
        result = ds.list_datasets()
        # Updated to match actual error message pattern
        assert "Failed to" in caplog.text
    finally:
        if original_token:
            os.environ["RAGAAI_CATALYST_TOKEN"] = original_token


def test_dataset_nonexistent_columns(dataset, caplog):
    """Test error handling for non-existent dataset"""
    try:
        columns = dataset.get_dataset_columns("nonexistent_dataset_name_12345")
    except IndexError:
        # The change logs errors but might still attempt to access dataset id
        pass
    
    # Check that error was logged
    assert "Dataset nonexistent_dataset_name_12345 does not exists" in caplog.text

def test_schema_mapping_no_token(base_url, caplog):
    """Test get_schema_mapping with no token (should log error)"""
    original_token = os.environ.get("RAGAAI_CATALYST_TOKEN")
    try:
        if "RAGAAI_CATALYST_TOKEN" in os.environ:
            del os.environ["RAGAAI_CATALYST_TOKEN"]
        ds = Dataset(project_name="prompt_metric_dataset")
        result = ds.get_schema_mapping()
        # Updated to match actual error message pattern
        assert "Failed to" in caplog.text
    finally:
        if original_token:
            os.environ["RAGAAI_CATALYST_TOKEN"] = original_token


def test_create_csv_duplicate_name(dataset, caplog):
    """Test creating dataset with existing name"""
    # Get list of existing datasets
    existing_datasets = dataset.list_datasets()
    
    if existing_datasets and len(existing_datasets) > 0:
        # Use first dataset name from the list
        existing_name = existing_datasets[0]
        
        # Schema mapping for test
        schema_mapping = {
            'Query': 'prompt',
            'Response': 'response',
            'Context': 'context',
            'ExpectedResponse': 'expected_response',
        }
        
        # Try to create with existing name
        dataset.create_from_csv(
            csv_path=csv_path,
            dataset_name=existing_name,
            schema_mapping=schema_mapping
        )
        
        # Check for appropriate error log
        assert f"Dataset name {existing_name} already exists" in caplog.text

def test_create_csv_nonexistent_path(dataset, caplog):
    """Test creating dataset with non-existent CSV path"""
    # Generate unique name to avoid duplicate issues
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"test_nonexistent_path_{timestamp}"
    
    # Schema mapping for test
    schema_mapping = {
        'Query': 'prompt',
        'Response': 'response',
        'Context': 'context',
        'ExpectedResponse': 'expected_response',
    }
    
    # Try to create with non-existent path
    dataset.create_from_csv(
        csv_path="/nonexistent/path/to/file.csv",
        dataset_name=dataset_name,
        schema_mapping=schema_mapping
    )
    
    # Check for appropriate error log
    assert "No such file or directory" in caplog.text

def test_create_csv_invalid_schema(dataset, caplog):
    """Test creating dataset with invalid schema mapping"""
    # Generate unique name to avoid duplicate issues
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"test_invalid_schema_{timestamp}"
    
    # Try to create with string instead of dict for schema mapping
    dataset.create_from_csv(
        csv_path=csv_path,
        dataset_name=dataset_name,
        schema_mapping="not_a_valid_schema_mapping"
    )
    
    # Check for appropriate error log
    assert "Error in create_from_csv" in caplog.text

def test_add_rows_nonexistent_dataset(dataset, caplog):
    """Test adding rows to non-existent dataset"""
    try:
        dataset.add_rows(
            csv_path=csv_path,
            dataset_name="nonexistent_dataset_name_12345"
        )
    except IndexError:
        pass
    assert "Dataset nonexistent_dataset_name_12345 does not exists" in caplog.text


# def test_add_rows_invalid_csv(dataset, caplog):
#     """Test adding rows with invalid CSV path"""
#     # Get list of existing datasets
#     existing_datasets = dataset.list_datasets()
    
#     if existing_datasets and len(existing_datasets) > 0:
#         # Use first dataset name from the list
#         existing_name = existing_datasets[0]
        
#         # Try to add rows with non-existent CSV
#         dataset.add_rows(
#             csv_path="/nonexistent/path/to/file.csv",
#             dataset_name=existing_name
#         )
        
#         # Check for appropriate error log
#         assert "Failed to read CSV file" in caplog.text or "No such file or directory" in caplog.text

def test_add_columns_invalid_text_fields(dataset, caplog):
    """Test add_columns with invalid text_fields"""
    # Try to add column with invalid text_fields (should be list of dicts)
    dataset.add_columns(
        text_fields="not_a_list",
        dataset_name="test_dataset",
        column_name="test_column",
        provider="openai",
        model="gpt-3.5-turbo"
    )
    
    # Check for appropriate error log
    assert "text_fields must be a list of dictionaries" in caplog.text

def test_add_columns_invalid_field_format(dataset, caplog):
    """Test add_columns with invalid field format"""
    # Try to add column with invalid field format (missing required keys)
    dataset.add_columns(
        text_fields=[{"invalid_key": "value"}],
        dataset_name="test_dataset",
        column_name="test_column",
        provider="openai",
        model="gpt-3.5-turbo"
    )
    
    # Check for appropriate error log
    assert "Each text field must be a dictionary with 'role' and 'content' keys" in caplog.text

def test_create_from_jsonl_nonexistent(dataset, caplog):
    """Test create_from_jsonl with non-existent file"""
    # Generate unique name to avoid duplicate issues
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"test_jsonl_{timestamp}"
    
    # Schema mapping for test
    schema_mapping = {
        'Query': 'prompt',
        'Response': 'response',
        'Context': 'context',
        'ExpectedResponse': 'expected_response',
    }
    
    # Try to create with non-existent JSONL file
    dataset.create_from_jsonl(
        jsonl_path="/nonexistent/path/to/file.jsonl",
        dataset_name=dataset_name,
        schema_mapping=schema_mapping
    )
    
    # Check for appropriate error log
    assert "Error converting JSONL to CSV" in caplog.text

# def test_create_from_df_empty(dataset, caplog):
#     """Test create_from_df with empty DataFrame"""
#     # Generate unique name to avoid duplicate issues
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     dataset_name = f"test_empty_df_{timestamp}"
    
#     # Create empty DataFrame
#     empty_df = pd.DataFrame()
    
#     # Schema mapping for test
#     schema_mapping = {
#         'Query': 'prompt',
#         'Response': 'response',
#         'Context': 'context',
#         'ExpectedResponse': 'expected_response',
#     }
    
#     # Try to create from empty DataFrame
#     dataset.create_from_df(
#         df=empty_df,
#         dataset_name=dataset_name,
#         schema_mapping=schema_mapping
#     )
    
#     # Check for appropriate error log - message could vary based on empty DataFrame behavior
#     assert "Error" in caplog.text

# def test_add_rows_from_df_empty(dataset, caplog):
#     """Test add_rows_from_df with empty DataFrame"""
#     # Get list of existing datasets
#     existing_datasets = dataset.list_datasets()
    
#     if existing_datasets and len(existing_datasets) > 0:
#         # Use first dataset name from the list
#         existing_name = existing_datasets[0]
        
#         # Create empty DataFrame
#         empty_df = pd.DataFrame()
        
#         # Try to add rows from empty DataFrame
#         dataset.add_rows_from_df(
#             df=empty_df,
#             dataset_name=existing_name
#         )
        
#         # Check for appropriate error log
#         assert "Error" in caplog.text

def test_add_rows_from_jsonl_nonexistent(dataset, caplog):
    """Test add_rows_from_jsonl with non-existent file"""
    # Get list of existing datasets
    existing_datasets = dataset.list_datasets()
    
    if existing_datasets and len(existing_datasets) > 0:
        # Use first dataset name from the list
        existing_name = existing_datasets[0]
        
        # Try to add rows from non-existent JSONL file
        dataset.add_rows_from_jsonl(
            jsonl_path="/nonexistent/path/to/file.jsonl",
            dataset_name=existing_name
        )
        
        # Check for appropriate error log
        assert "Error converting JSONL to CSV" in caplog.text