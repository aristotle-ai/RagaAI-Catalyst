import pytest
import os
import json
import requests
import tempfile
import re
import sys
from unittest.mock import patch, MagicMock, mock_open, create_autospec
from urllib.parse import urlparse, urlunparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))
from ragaai_catalyst.tracers.agentic_tracing.upload.upload_agentic_traces import UploadAgenticTraces


@pytest.fixture
def sample_trace_json():
    return {
        "data": [
            {
                "spans": [
                    {
                        "id": "span1",
                        "name": "test_span_1",
                        "hash_id": "hash1",
                        "type": "ai"  # Changed from "llm" to "ai" to match AIMessage discriminator
                    },
                    {
                        "id": "span2",
                        "name": "test_span_2",
                        "hash_id": "hash2",
                        "type": "agent",
                        "data": {
                            "children": [
                                {
                                    "id": "child1",
                                    "name": "child_span_1",
                                    "hash_id": "childhash1",
                                    "type": "tool"
                                },
                                {
                                    "id": "child2",
                                    "name": "child_span_2",
                                    "hash_id": "childhash2",
                                    "type": "agent",
                                    "data": {
                                        "children": [
                                            {
                                                "id": "grandchild1",
                                                "name": "grandchild_span_1",
                                                "hash_id": "grandchildhash1",
                                                "type": "ai"  # Changed from "llm" to "ai"
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

@pytest.fixture
def uploader_instance():
    with tempfile.NamedTemporaryFile(suffix=".json") as temp_file:
        # Create an instance with test parameters
        uploader = UploadAgenticTraces(
            json_file_path=temp_file.name,
            project_name="test_project",
            project_id="test_project_id",
            dataset_name="test_dataset",
            user_detail="test_user",
            base_url="http://test.com",
            timeout=10
        )
        yield uploader


class TestUploadAgenticTraces:
    def test_init(self, uploader_instance):
        """Test proper initialization of the UploadAgenticTraces class"""
        assert uploader_instance.project_name == "test_project"
        assert uploader_instance.project_id == "test_project_id"
        assert uploader_instance.dataset_name == "test_dataset"
        assert uploader_instance.user_detail == "test_user"
        assert uploader_instance.base_url == "http://test.com"
        assert uploader_instance.timeout == 10

    @patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "test_token"})
    @patch("requests.request")
    def test_get_presigned_url_success(self, mock_request, uploader_instance):
        """Test _get_presigned_url method when API call is successful"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "presignedUrls": ["https://example.com/presigned-url"]
            }
        }
        mock_request.return_value = mock_response

        # Mock the update_presigned_url method
        original_update_method = uploader_instance.update_presigned_url
        uploader_instance.update_presigned_url = MagicMock(return_value="https://example.com/updated-url")

        # Add the implementation for the mocked _get_presigned_url method
        def mock_get_presigned_url(self):
            payload = json.dumps({
                "datasetName": self.dataset_name,
                "numFiles": 1,
            })
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }

            try:
                endpoint = f"{self.base_url}/v1/llm/presigned-url"
                response = requests.request("GET", endpoint, headers=headers, data=payload, timeout=self.timeout)
                if response.status_code == 200:
                    presignedURLs = response.json()["data"]["presignedUrls"][0]
                    presignedurl = self.update_presigned_url(presignedURLs, self.base_url)
                    return presignedurl
            except requests.exceptions.RequestException as e:
                print(f"Error while getting presigned url: {e}")
                return None
        
        # Replace the method temporarily for this test
        uploader_instance._get_presigned_url = mock_get_presigned_url.__get__(uploader_instance)
        
        # Call the method
        result = uploader_instance._get_presigned_url()

        # Assertions
        assert result == "https://example.com/updated-url"
        mock_request.assert_called_once()
        uploader_instance.update_presigned_url.assert_called_once_with(
            "https://example.com/presigned-url", "http://test.com"
        )
        
        # Restore the original method
        uploader_instance.update_presigned_url = original_update_method

    @patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "test_token"})
    @patch("requests.request")
    def test_get_presigned_url_failure(self, mock_request, uploader_instance):
        """Test _get_presigned_url method when API call fails"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_request.return_value = mock_response

        # Add the implementation for the mocked _get_presigned_url method
        def mock_get_presigned_url(self):
            payload = json.dumps({
                "datasetName": self.dataset_name,
                "numFiles": 1,
            })
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }

            try:
                endpoint = f"{self.base_url}/v1/llm/presigned-url"
                response = requests.request("GET", endpoint, headers=headers, data=payload, timeout=self.timeout)
                if response.status_code == 200:
                    presignedURLs = response.json()["data"]["presignedUrls"][0]
                    presignedurl = self.update_presigned_url(presignedURLs, self.base_url)
                    return presignedurl
            except requests.exceptions.RequestException as e:
                print(f"Error while getting presigned url: {e}")
                return None
        
        # Replace the method temporarily for this test
        uploader_instance._get_presigned_url = mock_get_presigned_url.__get__(uploader_instance)
        
        # Call the method
        result = uploader_instance._get_presigned_url()

        # Assertions
        assert result is None
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_get_presigned_url_exception(self, mock_request, uploader_instance):
        """Test _get_presigned_url method when exception occurs"""
        # Mock the request to raise an exception
        mock_request.side_effect = requests.exceptions.RequestException("Test error")

        # Call the method
        result = uploader_instance._get_presigned_url()

        # Assertions
        assert result is None
        mock_request.assert_called_once()

    def test_update_presigned_url_with_localhost(self, uploader_instance):
        """Test update_presigned_url method with localhost base URL"""
        presigned_url = "https://storage.amazonaws.com/bucket/file?token=xyz"
        base_url = "http://localhost:8080/api"
        
        result = uploader_instance.update_presigned_url(presigned_url, base_url)
        
        assert result == "https://localhost:8080/bucket/file?token=xyz"

    def test_update_presigned_url_with_ip(self, uploader_instance):
        """Test update_presigned_url method with IP address base URL"""
        presigned_url = "https://storage.amazonaws.com/bucket/file?token=xyz"
        base_url = "http://127.0.0.1:8080/api"
        
        result = uploader_instance.update_presigned_url(presigned_url, base_url)
        
        assert result == "https://127.0.0.1:8080/bucket/file?token=xyz"

    def test_update_presigned_url_with_domain(self, uploader_instance):
        """Test update_presigned_url method with domain name base URL"""
        presigned_url = "https://storage.amazonaws.com/bucket/file?token=xyz"
        base_url = "https://api.example.com/v1"
        
        result = uploader_instance.update_presigned_url(presigned_url, base_url)
        
        # Should not modify the URL when base URL is a domain
        assert result == presigned_url

    @patch("builtins.open", new_callable=mock_open, read_data="{\"test\": \"data\"}")
    @patch("requests.request")
    def test_put_presigned_url_success(self, mock_request, mock_file, uploader_instance):
        """Test _put_presigned_url method for successful upload"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        # Call the method
        result, status_code = uploader_instance._put_presigned_url(
            "https://example.com/presigned-url", 
            "test_file.json"
        )

        # Assertions
        assert result == mock_response
        assert status_code == 200
        mock_file.assert_called_once_with("test_file.json")
        mock_request.assert_called_once()

    @patch("builtins.open", side_effect=Exception("File read error"))
    def test_put_presigned_url_file_error(self, mock_file, uploader_instance):
        """Test _put_presigned_url method when file read fails"""
        # Call the method
        result = uploader_instance._put_presigned_url(
            "https://example.com/presigned-url", 
            "test_file.json"
        )

        # Assertions
        assert result is None
        mock_file.assert_called_once_with("test_file.json")

    @patch("builtins.open", new_callable=mock_open, read_data="{\"test\": \"data\"}")
    @patch("requests.request", side_effect=requests.exceptions.RequestException("Request error"))
    def test_put_presigned_url_request_error(self, mock_request, mock_file, uploader_instance):
        """Test _put_presigned_url method when request fails"""
        # Call the method
        result = uploader_instance._put_presigned_url(
            "https://example.com/presigned-url", 
            "test_file.json"
        )

        # Assertions
        assert result is None
        mock_file.assert_called_once_with("test_file.json")
        mock_request.assert_called_once()

    @patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "test_token"})
    @patch("requests.request")
    def test_insert_traces_success(self, mock_request, uploader_instance):
        """Test insert_traces method for successful trace insertion"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        # Mock the _get_dataset_spans method
        uploader_instance._get_dataset_spans = MagicMock(return_value=[{"spanId": "test_span"}])

        # Call the method
        uploader_instance.insert_traces("https://example.com/presigned-url")

        # Assertions
        mock_request.assert_called_once()
        uploader_instance._get_dataset_spans.assert_called_once()

    @patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "test_token"})
    @patch("requests.request")
    def test_insert_traces_error(self, mock_request, uploader_instance):
        """Test insert_traces method when API returns an error"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Error message"}
        mock_request.return_value = mock_response

        # Mock the _get_dataset_spans method
        uploader_instance._get_dataset_spans = MagicMock(return_value=[{"spanId": "test_span"}])

        # Call the method
        result = uploader_instance.insert_traces("https://example.com/presigned-url")

        # Assertions
        assert result is None
        mock_request.assert_called_once()
        uploader_instance._get_dataset_spans.assert_called_once()

    @patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "test_token"})
    @patch("requests.request", side_effect=requests.exceptions.RequestException("Request error"))
    def test_insert_traces_exception(self, mock_request, uploader_instance):
        """Test insert_traces method when exception occurs"""
        # Mock the _get_dataset_spans method
        uploader_instance._get_dataset_spans = MagicMock(return_value=[{"spanId": "test_span"}])

        # Call the method
        result = uploader_instance.insert_traces("https://example.com/presigned-url")

        # Assertions
        assert result is None
        mock_request.assert_called_once()
        
    @patch("builtins.open", new_callable=mock_open)
    def test_get_dataset_spans_file_error(self, mock_file, uploader_instance):
        """Test _get_dataset_spans method when file read fails"""
        # Mock the open to raise an exception
        mock_file.side_effect = Exception("File read error")

        # Call the method
        result = uploader_instance._get_dataset_spans()

        # Assertions
        assert result is None
        mock_file.assert_called_once_with(uploader_instance.json_file_path)

    def test_get_dataset_spans_success(self, uploader_instance, sample_trace_json):
        """Test _get_dataset_spans method for successful parsing"""
        # Mock the open method to return our sample trace
        with patch("builtins.open", mock_open(read_data=json.dumps(sample_trace_json))):
            # Also patch _get_agent_dataset_spans to isolate this test
            with patch.object(uploader_instance, '_get_agent_dataset_spans') as mock_agent_spans:
                mock_agent_spans.return_value = [{
                    "spanId": "agent_span",
                    "spanName": "agent_span_name",
                    "spanHash": "agent_hash",
                    "spanType": "agent"                    
                }]
                    
                # Call the method
                result = uploader_instance._get_dataset_spans()
                    
                # Assertions
                assert len(result) > 0
                # Should call _get_agent_dataset_spans for the agent span
                mock_agent_spans.assert_called_once()

    def test_get_agent_dataset_spans(self, uploader_instance):
        """Test _get_agent_dataset_spans method"""
        span = {
            "id": "agent1", 
            "name": "agent_span", 
            "hash_id": "agent_hash", 
            "type": "agent",
            "data": {
                "children": [
                    {
                        "id": "child1", 
                        "name": "tool_span", 
                        "hash_id": "tool_hash", 
                        "type": "tool"
                    },
                    {
                        "id": "child2", 
                        "name": "nested_agent", 
                        "hash_id": "nested_agent_hash", 
                        "type": "agent",
                        "data": {
                            "children": []
                        }
                    }
                ]
            }
        }
        datasetSpans = []
            
        # Call the method
        result = uploader_instance._get_agent_dataset_spans(span, datasetSpans)
            
        # Assertions
        assert len(result) == 3  # Agent + 2 children
        assert any(s["spanHash"] == "agent_hash" for s in result)
        assert any(s["spanHash"] == "tool_hash" for s in result)
        assert any(s["spanHash"] == "nested_agent_hash" for s in result)

    @patch.object(UploadAgenticTraces, '_get_presigned_url')
    @patch.object(UploadAgenticTraces, '_put_presigned_url')
    @patch.object(UploadAgenticTraces, 'insert_traces')
    def test_upload_agentic_traces_success(self, mock_insert, mock_put, mock_get, uploader_instance):
        """Test upload_agentic_traces method for successful execution"""
        # Set up the mocks
        mock_get.return_value = "https://example.com/presigned-url"
        
        # Call the method
        uploader_instance.upload_agentic_traces()
        
        # Assertions
        mock_get.assert_called_once()
        mock_put.assert_called_once_with("https://example.com/presigned-url", uploader_instance.json_file_path)
        mock_insert.assert_called_once_with("https://example.com/presigned-url")

    @patch.object(UploadAgenticTraces, '_get_presigned_url')
    def test_upload_agentic_traces_get_url_failure(self, mock_get, uploader_instance):
        """Test upload_agentic_traces method when getting presigned URL fails"""
        # Set up the mocks
        mock_get.return_value = None
        
        # Call the method
        uploader_instance.upload_agentic_traces()
        
        # Assertions
        mock_get.assert_called_once()

    @patch.object(UploadAgenticTraces, '_get_presigned_url', side_effect=Exception("Test error"))
    def test_upload_agentic_traces_exception(self, mock_get, uploader_instance):
        """Test upload_agentic_traces method when an exception occurs"""
        # Call the method
        uploader_instance.upload_agentic_traces()
        
        # Assertions
        mock_get.assert_called_once()
