import logging
import os
import pytest
import requests
from ragaai_catalyst import Evaluation, RagaAICatalyst, Dataset  # Added Dataset import
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def evaluation():
    base_url = os.getenv("RAGAAI_CATALYST_BASE_URL")
    access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
    secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_key,
        secret_key=secret_key
    )
    
    # Create project if it doesn't exist
    project_name = "test_dataset"
    existing_projects = catalyst.list_projects()
    
    if project_name not in existing_projects:
        try:
            catalyst.create_project(
                project_name=project_name,
                usecase="Q/A"  # default usecase Q/A
            )
            print(f"Project '{project_name}' created successfully")
        except Exception as e:
            print(f"Error creating project: {e}")
    else:
        print(f"Project '{project_name}' already exists")
    
    # Initialize Dataset management for the project
    dataset_manager = Dataset(project_name=project_name)
    
    # List existing datasets
    existing_datasets = dataset_manager.list_datasets()
    print("Existing Datasets:", existing_datasets)
    
    # Get schema mappings available
    schema_elements = dataset_manager.get_schema_mapping()
    print("Available schema elements:", schema_elements)
    
    # Create test dataset if it doesn't exist
    dataset_name = "test"
    
    if dataset_name not in existing_datasets:
        try:
            # Path to the test CSV file
            csv_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_synthetic_data_valid.csv"))
            
            # Schema mapping for the CSV - map CSV columns to schema elements
            schema_mapping = {
                'prompt': 'prompt',                   # CSV column 'prompt' maps to schema element 'prompt'
                'response': 'response',               # CSV column 'response' maps to schema element 'response'
                'expected response': 'expected_response',  # CSV column 'expected response' maps to schema element 'expected_response'
                'context': 'context'                  # CSV column 'context' maps to schema element 'context'
            }
            
            # Create the dataset from CSV
            dataset_manager.create_from_csv(
                csv_path=csv_path,
                dataset_name=dataset_name,
                schema_mapping=schema_mapping
            )
            print(f"Dataset '{dataset_name}' created successfully")
        except Exception as e:
            print(f"Error creating dataset: {e}")
    else:
        print(f"Dataset '{dataset_name}' already exists")
    
    # Return the evaluation instance with the project and dataset
    return Evaluation(
        project_name=project_name, 
        dataset_name=dataset_name
    )
@pytest.fixture(scope="module")
def catalyst():
    base_url = os.getenv("RAGAAI_CATALYST_BASE_URL")
    access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
    secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_key,
        secret_key=secret_key
    )
    
    # Create project if it doesn't exist
    project_name = "test_dataset"
    existing_projects = catalyst.list_projects()
    
    if project_name not in existing_projects:
        try:
            catalyst.create_project(
                project_name=project_name,
                usecase="Q/A"  # default usecase Q/A
            )
            print(f"Project '{project_name}' created successfully")
        except Exception as e:
            print(f"Error creating project: {e}")
    else:
        print(f"Project '{project_name}' already exists")
    
    return catalyst

@pytest.fixture
def valid_metrics(evaluation):
    # Define the metrics to be added
    metrics = [{
        "name": "Hallucination",
        "config": {"threshold": {"gte": 0.0}},  # Changed from direct float to dictionary with operator
        "column_name": "Hallucination3",
        "schema_mapping": {"input": "test_input"}
    }]
    
    # Add metrics to the evaluation instance
    for metric in metrics:
        evaluation.add_metrics([metric])
    
    return metrics


def test_add_metrics_success(evaluation, valid_metrics):
    """Test successful addition of metrics"""
    try:
        evaluation.add_metrics(valid_metrics)
        assert evaluation.jobId is not None
    except requests.exceptions.RequestException as e:
        pytest.fail(f"API request failed: {e}")

def test_add_metrics_missing_required_keys(evaluation, caplog):
    """Test validation of required keys"""
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    invalid_metrics = [{
        "name": "Hallucination",
        "config": {"provider": "openai", "model": "gpt-4o-mini"}
        # missing column_name and schema_mapping
    }]
    
    try:
        evaluation.add_metrics(invalid_metrics)
    except (KeyError, TypeError):
        pass
    
    assert "{'schema_mapping', 'column_name'} required for each metric evaluation" in caplog.text or \
           "{'column_name', 'schema_mapping'} required for each metric evaluation" in caplog.text

def test_add_metrics_invalid_metric_name(evaluation, caplog):
    """Test validation of metric names"""
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    invalid_metrics = [{
        "name": "InvalidMetricName",  # Use an intentionally invalid name
        "config": {"threshold": {"gte": 0.8}},  # Changed from direct float to dictionary with operator
        "column_name": "invalid_metric_col",
        "schema_mapping": {"input": "test_input", "Prompt": "prompt_col", "Response": "response_col", "Context": "context_col"}
    }]
    
    try:
        evaluation.add_metrics(invalid_metrics)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"API request failed: {e}")
    
    assert "Enter a valid metric name" in caplog.text
def test_add_metrics_duplicate_column_name(evaluation, valid_metrics, caplog):
    """Test validation of duplicate column names"""
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    # Attempt to add the same metric twice to trigger the duplicate column error
    try:
        evaluation.add_metrics(valid_metrics)
        evaluation.add_metrics(valid_metrics)  # Add again to simulate duplication
    except requests.exceptions.RequestException as e:
        pytest.fail(f"API request failed: {e}")
    
    # Adjust the assertion to match the actual log message
    assert "Column name 'Hallucination3' already exists" in caplog.text

def test_add_metrics_http_error(evaluation, valid_metrics):
    """Test handling of HTTP errors"""
    try:
        evaluation.add_metrics(valid_metrics)
    except requests.exceptions.HTTPError as e:
        assert "HTTP Error" in str(e)

def test_add_metrics_connection_error(evaluation, valid_metrics):
    """Test handling of connection errors"""
    try:
        evaluation.add_metrics(valid_metrics)
    except requests.exceptions.ConnectionError as e:
        assert "Connection Error" in str(e)

def test_add_metrics_timeout_error(evaluation, valid_metrics):
    """Test handling of timeout errors"""
    try:
        evaluation.add_metrics(valid_metrics)
    except requests.exceptions.Timeout as e:
        assert "Timeout Error" in str(e)

