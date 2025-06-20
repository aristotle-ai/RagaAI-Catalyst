from unittest.mock import patch
import time
import pytest
import os
import dotenv
dotenv.load_dotenv()
import pandas as pd
from datetime import datetime 
from typing import Dict, List
from ragaai_catalyst import Evaluation, RagaAICatalyst

# Simplified model configurations
MODEL_CONFIGS = [
    {"provider": "openai", "model": "gpt-4"},  # Only one OpenAI model
    # {"provider": "gemini", "model": "gemini-1.5-flash"}  # Only one Gemini model
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
        project_name="bug_test2", 
        dataset_name="legal_research_rag"
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
        project_name="bug_test2", 
        dataset_name="legal_research_rag"
    )

# Basic initialization tests
def test_evaluation_initialization(evaluation):
    """Test if evaluation is initialized correctly"""
    assert evaluation.project_name == "bug_test2"
    assert evaluation.dataset_name == "legal_research_rag"
import logging

def test_project_does_not_exist(caplog):
    """Test initialization with non-existent project"""
    # Set caplog to capture logging at ERROR level
    caplog.set_level(logging.ERROR)
    
    # This should raise an IndexError because after logging the error,
    # it still tries to access the project_id from an empty list
    with pytest.raises(IndexError):
        Evaluation(project_name="non_existent_project_12345", dataset_name="dataset")
    
    # Verify the error message was logged
    assert "Project not found. Please enter a valid project name" in caplog.text

@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_metric_validation_checks(evaluation, provider_config, caplog):
    """Test all validation checks in one parameterized test"""
    # Set caplog to capture logging at ERROR level
    caplog.set_level(logging.ERROR)
    
    schema_mapping = {
        'Query': 'Prompt',
        'Response': 'Response',
        'Context': 'Context',
    }
    
    # Test missing schema_mapping
    caplog.clear()
    with pytest.raises(KeyError):  # Will raise KeyError when trying to access schema_mapping
        evaluation.add_metrics([{
            "name": "Hallucination",
            "config": provider_config,
            "column_name": "test_column"
        }])
    assert "{'schema_mapping'} required for each metric evaluation" in caplog.text
    
    # Test missing column_name
    caplog.clear()
    try:
        evaluation.add_metrics([{
            "name": "Hallucination",
            "config": provider_config,
            "schema_mapping": schema_mapping
        }])
    except (KeyError, AttributeError):
        # Expected error when accessing missing key
        pass
    assert "{'column_name'} required for each metric evaluation" in caplog.text
    
    # Test missing metric name
    caplog.clear()
    try:
        evaluation.add_metrics([{
            "config": provider_config,
            "column_name": "test_column",
            "schema_mapping": schema_mapping
        }])
    except (KeyError, AttributeError):
        # Expected error when accessing missing key
        pass
    assert "{'name'} required for each metric evaluation" in caplog.text