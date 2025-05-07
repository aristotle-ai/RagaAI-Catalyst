import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ragaai_catalyst.internal_api_completion import api_completion, get_username, convert_input


@pytest.fixture
def sample_messages():
    return [
        {
            "role": "system",
            "content": "you are a helpful assistant"
        },
        {
            "role": "user",
            "content": "write a test query"
        }
    ]


@pytest.fixture
def sample_model_config():
    return {
        "model": "test_model",
        "provider": "openai",
        "max_tokens": 100
    }


@pytest.fixture
def sample_kwargs():
    return {
        "internal_llm_proxy": "http://test-proxy.com/chat/completions",
        "user_id": "test_user",
        "log_level": "debug"
    }


@pytest.fixture
def mock_response_json_success():
    return {
        "choices": [
            {
                "message": {
                    "content": '{"column1": [1, 2, 3], "column2": ["a", "b", "c"]}'
                }
            }
        ]
    }


@pytest.fixture
def mock_response_error():
    return {
        "error": {
            "message": "Test error message"
        }
    }


@pytest.fixture
def mock_invalid_json_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "This is not a valid JSON"
                }
            }
        ]
    }


class TestApiCompletion:
    
    @patch('ragaai_catalyst.internal_api_completion.requests.request')
    def test_api_completion_success(self, mock_request, sample_messages, sample_model_config, sample_kwargs, mock_response_json_success):
        # Setup the mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_json_success
        mock_request.return_value = mock_response
        
        # Call the function
        result = api_completion(sample_messages, sample_model_config, sample_kwargs)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["column1", "column2"]
        assert len(result) == 3
        mock_request.assert_called_once()
    
    @patch('ragaai_catalyst.internal_api_completion.requests.request')
    def test_api_completion_error_response(self, mock_request, sample_messages, sample_model_config, sample_kwargs):
        # Setup the mock to return a non-200 status code
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error from API"
        mock_request.return_value = mock_response
        
        # Call the function and check for exception
        with pytest.raises(ValueError, match="Error from API"):
            api_completion(sample_messages, sample_model_config, sample_kwargs)
    
    @patch('ragaai_catalyst.internal_api_completion.requests.request')
    def test_api_completion_error_in_response(self, mock_request, sample_messages, sample_model_config, sample_kwargs, mock_response_error):
        # Setup the mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_error
        mock_request.return_value = mock_response
        
        # Call the function and check for exception
        with pytest.raises(ValueError, match="Test error message"):
            api_completion(sample_messages, sample_model_config, sample_kwargs)
    
    @patch('ragaai_catalyst.internal_api_completion.requests.request')
    def test_api_completion_invalid_json(self, mock_request, sample_messages, sample_model_config, sample_kwargs, mock_invalid_json_response):
        # Setup the mock with invalid JSON in the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_invalid_json_response
        mock_request.return_value = mock_response
        
        # After 3 attempts, it should raise an exception
        with pytest.raises(Exception, match="Failed to generate a valid response after multiple attempts"):
            api_completion(sample_messages, sample_model_config, sample_kwargs)
        
        # Verify it made 3 attempts
        assert mock_request.call_count == 3
    
    @patch('ragaai_catalyst.internal_api_completion.requests.request')
    def test_api_completion_exception_during_request(self, mock_request, sample_messages, sample_model_config, sample_kwargs):
        # Setup the mock to raise an exception
        mock_request.side_effect = Exception("Network error")
        
        # Call the function and check for exception
        with pytest.raises(ValueError, match="Network error"):
            api_completion(sample_messages, sample_model_config, sample_kwargs)
    
    @patch('ragaai_catalyst.internal_api_completion.requests.request')
    def test_api_completion_with_job_id(self, mock_request, sample_messages, sample_kwargs, mock_response_json_success):
        # Setup model_config with job_id
        model_config = {
            "model": "test_model",
            "provider": "openai",
            "max_tokens": 100,
            "job_id": "test_job_123"
        }
        
        # Setup the mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_json_success
        mock_request.return_value = mock_response
        
        # Call the function
        result = api_completion(sample_messages, model_config, sample_kwargs)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        mock_request.assert_called_once()


class TestGetUsername:
    
    @patch('ragaai_catalyst.internal_api_completion.subprocess.run')
    def test_get_username(self, mock_run):
        # Setup the mock
        mock_process = MagicMock()
        mock_process.stdout = "test_user\n"
        mock_run.return_value = mock_process
        
        # Call the function
        result = get_username()
        
        # Assertions
        assert result == "test_user\n"
        mock_run.assert_called_with(['whoami'], capture_output=True, text=True)


class TestConvertInput:
    
    def test_convert_input_basic(self, sample_messages, sample_model_config):
        # Call the function
        user_id = "test_user_id"
        result = convert_input(sample_messages, sample_model_config, user_id)
        
        # Assertions
        assert result["model"] == sample_model_config["model"]
        assert result["provider"] == sample_model_config["provider"]
        assert result["max_tokens"] == sample_model_config["max_tokens"]
        assert result["messages"] == sample_messages
        assert result["user_id"] == user_id
    
    def test_convert_input_empty_messages(self, sample_model_config):
        # Call the function with empty messages
        user_id = "test_user_id"
        result = convert_input([], sample_model_config, user_id)
        
        # Assertions
        assert result["messages"] == []
    
    def test_convert_input_additional_config(self, sample_messages):
        # Setup model_config with additional parameters
        model_config = {
            "model": "test_model",
            "temperature": 0.7,
            "top_p": 0.9,
            "custom_param": "custom_value"
        }
        
        # Call the function
        user_id = "test_user_id"
        result = convert_input(sample_messages, model_config, user_id)
        
        # Assertions
        assert result["model"] == model_config["model"]
        assert result["temperature"] == model_config["temperature"]
        assert result["top_p"] == model_config["top_p"]
        assert result["custom_param"] == model_config["custom_param"]
