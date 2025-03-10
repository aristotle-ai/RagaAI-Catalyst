import pytest
from pytest_mock import mocker
from ragaai_catalyst.redteaming import RedTeaming
import os
import dotenv
import pandas as pd

dotenv.load_dotenv("/Users/siddharthakosti/Downloads/catalyst_new_github_repo/RagaAI-Catalyst/.env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@pytest.fixture
def red_teaming():
    return RedTeaming(api_key=OPENAI_API_KEY)

def test_missing_api_key():
    """Test behavior when API key is missing"""
    with pytest.raises(ValueError, match="Api Key is required"):
        RedTeaming(api_key="")

def test_supported_detectors_loading(red_teaming):
    """Test loading of supported detectors"""
    assert isinstance(red_teaming.get_supported_detectors(), list), "Supported detectors should be a list"

def test_validate_detectors(red_teaming):
    """Test validation of detectors"""
    with pytest.raises(ValueError, match="Unsupported detectors"):
        red_teaming.validate_detectors(["unsupported_detector"])

def test_run_with_examples(red_teaming, mocker):
    """Test running with examples"""
    mock_response_model = mocker.Mock(return_value="mock response")
    mocker.patch('ragaai_catalyst.redteaming.data_generator.scenario_generator.ScenarioGenerator.generate_scenarios', return_value=["scenario1", "scenario2"])
    mocker.patch('ragaai_catalyst.redteaming.data_generator.test_case_generator.TestCaseGenerator.generate_test_cases', return_value={"inputs": [{"user_input": "test input"}]})
    mocker.patch('ragaai_catalyst.redteaming.evaluator.Evaluator.evaluate_conversation', return_value={"eval_passed": True, "reason": "mock reason"})

    df, save_path = red_teaming.run(
        description="Test app",
        detectors=["stereotypes"],
        response_model=mock_response_model,
        examples=["example1", "example2"],
        scenarios_per_detector=2
    )
    assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"

def test_run_without_examples(red_teaming, mocker):
    """Test running without examples"""
    mock_response_model = mocker.Mock(return_value="mock response")
    mocker.patch('ragaai_catalyst.redteaming.data_generator.scenario_generator.ScenarioGenerator.generate_scenarios', return_value=["scenario1", "scenario2"])
    mocker.patch('ragaai_catalyst.redteaming.data_generator.test_case_generator.TestCaseGenerator.generate_test_cases', return_value={"inputs": [{"user_input": "test input"}]})
    mocker.patch('ragaai_catalyst.redteaming.evaluator.Evaluator.evaluate_conversation', return_value={"eval_passed": True, "reason": "mock reason"})

    df, save_path = red_teaming.run(
        description="Test app",
        detectors=["stereotypes"],
        response_model=mock_response_model,
        scenarios_per_detector=2,
        examples_per_scenario=2
    )
    assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"

def test_invalid_detector_format(red_teaming):
    """Test invalid detector format"""
    with pytest.raises(ValueError, match='Detector must be a string or a dictionary with only key "custom" and a string as a value'):
        red_teaming.run(
            description="Test app",
            detectors=[123],  # Invalid format
            response_model=lambda x: "response"
        )

def test_custom_detector(red_teaming, mocker):
    """Test running with a custom detector"""
    mock_response_model = mocker.Mock(return_value="mock response")
    mocker.patch('ragaai_catalyst.redteaming.data_generator.scenario_generator.ScenarioGenerator.generate_scenarios', return_value=["scenario1"])
    mocker.patch('ragaai_catalyst.redteaming.evaluator.Evaluator.evaluate_conversation', return_value={"eval_passed": True, "reason": "mock reason"})

    df, save_path = red_teaming.run(
        description="Test app",
        detectors=[{"custom": "Prevent AI from discussing killing anything"}],
        response_model=mock_response_model,
        examples=["example1"],
        scenarios_per_detector=1
    )
    assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"
    

def test_upload_result_without_run(red_teaming):
    """Test uploading results without running"""
    with pytest.raises(Exception, match="Please execute the RedTeaming run() method before uploading the result"):
        red_teaming.upload_result("project_name", "dataset_name")