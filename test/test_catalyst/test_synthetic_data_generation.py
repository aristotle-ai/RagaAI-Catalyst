import sys
# sys.path.append('/Users/ritikagoel/workspace/synthetic-catalyst-internal-api2/ragaai-catalyst')

import pytest
from pytest_mock import mocker
from ragaai_catalyst import SyntheticDataGeneration
import os
import dotenv
import pandas as pd
import json
dotenv.load_dotenv("/Users/siddharthakosti/Downloads/catalyst_new_github_repo/RagaAI-Catalyst/.env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@pytest.fixture
def synthetic_gen():
    return SyntheticDataGeneration()

@pytest.fixture
def sample_text(synthetic_gen):
    text_file = "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/ai_document_061023_2.pdf"  # Update this path as needed
    return synthetic_gen.process_document(input_data=text_file)

def test_invalid_csv_processing(synthetic_gen):
    """Test processing an invalid CSV file"""
    # Verify that an exception is raised when trying to process an invalid CSV file
    with pytest.raises(Exception):
        synthetic_gen.process_document(input_data="/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/OG1.csv")

def test_special_chars_csv_processing(synthetic_gen):
    """Test processing CSV with special characters"""
    # Verify that an exception is raised when trying to process a CSV file with special characters
    with pytest.raises(Exception):
        synthetic_gen.process_document(input_data="/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/OG1.csv")

def test_missing_llm_proxy(synthetic_gen, sample_text):
    """Test behavior when internal_llm_proxy is not provided"""
    # Test that the function works with OpenAI API key from environment
    result = synthetic_gen.generate_qna(
        text=sample_text,
        question_type='mcq',
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        n=20,
        user_id="1"
    )
    assert len(result) == 20, "Should generate exactly 20 questions"

def test_llm_proxy(synthetic_gen, sample_text):
    """Test behavior with valid internal_llm_proxy URL"""
    # Verify that the method returns 15 questions when internal_llm_proxy is provided
    result = synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "gemini", "model": "gemini-1.5-flash"},
            n=15,
            internal_llm_proxy="http://4.240.49.141:4000/chat/completions",
            user_id="1"
        )
    assert len(result) == 15 

def test_invalid_llm_proxy(synthetic_gen, sample_text):
    """Test behavior with invalid internal_llm_proxy URL"""
    # Verify that an Exception is raised when internal_llm_proxy is invalid
    with pytest.raises(Exception, match="No connection adapters were found for"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "openai", "model": "gpt-4o-mini"},
            n=2,
            internal_llm_proxy="tp://invalid.url",
            user_id="1"
        )

def test_missing_model_config(synthetic_gen, sample_text):
    """Test behavior when model_config is not provided"""
    # Verify that a ValueError is raised when model_config is not provided
    with pytest.raises(ValueError, match="Model configuration must be provided with a valid provider and model"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            n=2,
            internal_llm_proxy="http://20.244.126.4:4000/chat/completions",
            user_id="1"
        )

def test_missing_api_key_for_external_provider(synthetic_gen, sample_text, monkeypatch):
    """Test behavior when API key is missing for external provider"""
    # Ensure no environment variables are set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    # Verify that a ValueError is raised when API key is missing for external provider
    with pytest.raises(ValueError, match="API key must be provided for Gemini"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "gemini", "model": "gemini-pro"},
            n=5,
            api_key=None
        )

def test_invalid_api_key(synthetic_gen, sample_text, monkeypatch):
    """Test behavior with invalid API key"""
    # Ensure no environment variables are set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    # Test with an invalid API key
    with pytest.raises(Exception, match="Failed to generate valid response after 3 attempts"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "gemini", "model": "gemini-pro"},
            n=5,
            api_key="invalid_api_key_123"
        )

def test_default_question_count(synthetic_gen, sample_text, mocker):
    """Test default number of questions when n is not provided"""
    # Mock the batch response to return 5 questions
    mock_response = [
        {"Question": f"Question {i}", "Answer": f"Answer {i}"} 
        for i in range(5)
    ]
    mock_df = pd.DataFrame(mock_response)
    mocker.patch(
        'ragaai_catalyst.synthetic_data_generation.SyntheticDataGeneration._generate_batch_response',
        return_value=mock_df
    )
    
    # Generate questions without specifying n
    result = synthetic_gen.generate_qna(
        text=sample_text,
        question_type='simple',
        model_config={"provider": "openai", "model": "gpt-3.5-turbo"}
    )
    
    # Verify that exactly 5 questions are generated (default value)
    assert len(result) == 5, f"Expected 5 questions but got {len(result)}"
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert all(col in result.columns for col in ["Question", "Answer"]), "DataFrame should have Question and Answer columns"

def test_default_question_type(synthetic_gen, sample_text, mocker):
    """Test default question type when question_type is not provided"""
    # Mock the API response
    mock_response = pd.DataFrame({
        'Question': ['Simple question 1', 'Simple question 2', 'Simple question 3', 'Simple question 4', 'Simple question 5'],
        'Answer': ['Answer 1', 'Answer 2', 'Answer 3', 'Answer 4', 'Answer 5']
    })
    
    mocker.patch.object(
        synthetic_gen, 
        '_generate_internal_response',
        return_value=mock_response
    )
    
    # Verify that simple questions are generated when question_type is not provided
    result = synthetic_gen.generate_qna(
        text=sample_text,
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        n=5,
        internal_llm_proxy="http://20.244.126.4:4000/chat/completions",
        user_id="1"
    )
    
    # Verify result contains simple Q/A format without multiple choice options
    assert isinstance(result, pd.DataFrame)
    assert all(isinstance(q, str) for q in result['Question'])
    assert all(isinstance(a, str) for a in result['Answer'])
    assert len(result) == 5

def test_question_count_matches_n(synthetic_gen, sample_text):
    """Test if number of generated questions matches n"""
    # Verify that the number of generated questions matches n
    n = 2
    result = synthetic_gen.generate_qna(
        text=sample_text,
        question_type='mcq',
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        n=n,
        internal_llm_proxy="http://4.240.49.141:4000/chat/completions",
        user_id="1"
    )
    assert len(result) == n

def test_proxy_call_check(synthetic_gen, sample_text, mocker):
    """Test compatibility when proxy script called"""
    # Mock the internal API completion
    mock_response = pd.DataFrame({
        'Question': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'Answer': ['A1', 'A2', 'A3', 'A4', 'A5']
    })
    mocker.patch('ragaai_catalyst.synthetic_data_generation.internal_api_completion', return_value=mock_response)
    
    # Verify that the method returns 5 questions when proxy script is called
    result = synthetic_gen.generate_qna(
            text=sample_text,
            question_type='simple',
            model_config={"provider": "gemini", "model": "gemini-1.5-flash", "api_base": "http://172.172.11.158:8000/v1alpha1/v1alpha1/predictions"},
            n=5,
            internal_llm_proxy="http://172.172.11.158:8000/v1alpha1/v1alpha1/predictions",
            user_id="1"
        )
    assert len(result) == 5

def test_generate_examples(synthetic_gen, mocker):
    """Test the generate_examples functionality"""
    # Mock environment variables
    mocker.patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key'
    })
    
    # Mock the input function to simulate user interaction
    mocker.patch('builtins.input', side_effect=['1,2', '3,4'])
    
    # Mock the _generate_examples method to return predictable results
    mock_examples_3 = "Example 1\nExample 2\nExample 3"
    mock_examples_2 = "Example 1\nExample 2"
    
    def mock_generate_examples(*args, **kwargs):
        if kwargs.get('no_examples', 3) == 3:
            return mock_examples_3
        return mock_examples_2
    
    mocker.patch.object(synthetic_gen, '_generate_examples', side_effect=mock_generate_examples)
    mocker.patch.object(synthetic_gen, '_generate_examples_iter', side_effect=mock_generate_examples)
    
    # Test with required parameters
    result = synthetic_gen.generate_examples(
        user_instruction="Generate test examples",
        no_examples=3,
        model_config={"provider": "openai", "model": "gpt-4"}
    )
    
    # Verify the results
    assert result is not None
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 3, "Should generate exactly 3 examples"
    
    # Test with optional parameters
    result_with_context = synthetic_gen.generate_examples(
        user_instruction="Generate test examples",
        user_examples=["Test example 1", "Test example 2"],
        user_context="Test context",
        no_examples=2,
        model_config={"provider": "openai", "model": "gpt-4"}
    )
    
    assert len(result_with_context) == 2, "Should generate exactly 2 examples"

def test_input_validation(synthetic_gen):
    """Test input validation for generate_qna"""
    # Test empty text
    with pytest.raises(ValueError):
        synthetic_gen.generate_qna(
            text="",
            model_config={"provider": "openai", "model": "gpt-4o-mini"},
            internal_llm_proxy="http://4.240.49.141:4000/chat/completions",
            user_id="1"
        )
    
    # Test invalid question type
    with pytest.raises(ValueError):
        synthetic_gen.generate_qna(
            text="Sample text",
            question_type="invalid_type",
            model_config={"provider": "openai", "model": "gpt-4o-mini"},
            internal_llm_proxy="http://4.240.49.141:4000/chat/completions",
            user_id="1"
        )

def test_batch_processing(synthetic_gen, sample_text):
    """Test batch processing functionality"""
    # Test with large n to trigger multiple batches
    n = 12  # Should trigger at least 3 batches with default BATCH_SIZE of 5
    result = synthetic_gen.generate_qna(
        text=sample_text,
        n=n,
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        internal_llm_proxy="http://4.240.49.141:4000/chat/completions",
        user_id="1"
    )
    assert len(result) == n
    assert isinstance(result, pd.DataFrame)
    assert all(isinstance(qa, dict) for qa in result.to_dict('records'))

def test_error_handling_malformed_response(synthetic_gen, sample_text, mocker):
    """Test handling of malformed responses from the API"""
    # Mock the API response to return malformed data
    mocker.patch('ragaai_catalyst.synthetic_data_generation.SyntheticDataGeneration._generate_batch_response',
                 side_effect=json.JSONDecodeError("Malformed response from API", "test", 0))
    
    # Should not raise an exception for non-failure case errors, but should retry
    synthetic_gen.generate_qna(
        text=sample_text,
        n=2,
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        internal_llm_proxy="http://4.240.49.141:4000/chat/completions",
        user_id="1"
    )
