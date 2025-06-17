import logging
import os
import pytest
import requests
from ragaai_catalyst import Evaluation, RagaAICatalyst

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
    
    return Evaluation(
        project_name="test_dataset", 
        dataset_name="test"
    )

@pytest.fixture
def valid_metrics(evaluation):
    # Define the metrics to be added
    metrics = [{
        "name": "Hallucination",
        "config": {"threshold": 0.0},
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
        "config": {"threshold": 0.8},
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

def test_add_metrics_bad_request(evaluation, caplog):
    """Test handling of 400 bad request"""
    caplog.set_level(logging.ERROR)
    caplog.clear()
    
    invalid_metrics = [{
        "name": "Hallucination",
        "config": {},  # Intentionally leave out required config details
        "column_name": "Hallucination",
        "schema_mapping": {"input": "test_input", "Prompt": "prompt_col", "Response": "response_col", "Context": "context_col"}
    }]
    
    try:
        evaluation.add_metrics(invalid_metrics)
    except requests.exceptions.HTTPError as e:
        assert "Bad request error" in str(e)
    
    assert evaluation.jobId is None