import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from ragaai_catalyst.ragaai_catalyst import RagaAICatalyst

# Fixture for basic RagaAICatalyst instance setup
@pytest.fixture
def catalyst_instance():
    with patch('requests.post') as mock_post:
        # Mock successful token response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "success": True,
            "data": {"token": "test_token"}
        }
        
        instance = RagaAICatalyst(
            access_key="test_access_key",
            secret_key="test_secret_key"
        )
        return instance

# Test initialization
class TestRagaAICatalystInit:
    def test_init_with_valid_credentials(self):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "success": True,
                "data": {"token": "test_token"}
            }
            
            catalyst = RagaAICatalyst(
                access_key="test_access_key",
                secret_key="test_secret_key"
            )
            
            assert catalyst.access_key == "test_access_key"
            assert catalyst.secret_key == "test_secret_key"
            assert os.getenv("RAGAAI_CATALYST_TOKEN") == "test_token"

    def test_init_with_invalid_credentials(self):
        with pytest.raises(ValueError):
            RagaAICatalyst(access_key="", secret_key="")

    def test_init_with_custom_base_url(self):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "success": True,
                "data": {"token": "test_token"}
            }
            
            catalyst = RagaAICatalyst(
                access_key="test_access_key",
                secret_key="test_secret_key",
                base_url="https://custom.url"
            )
            
            assert RagaAICatalyst.BASE_URL == "https://custom.url/api"

    def test_init_with_invalid_base_url(self):
        with patch('requests.post') as mock_post:
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.RequestException
            
            with pytest.raises(ConnectionError):
                RagaAICatalyst(
                    access_key="test_access_key",
                    secret_key="test_secret_key",
                    base_url="https://invalid.url"
                )

# Test API key management
class TestRagaAICatalystAPIKeys:
    def test_add_and_get_api_key(self, catalyst_instance):
        catalyst_instance.add_api_key("openai", "test-key-123")
        assert catalyst_instance.get_api_key("openai") == "test-key-123"

    def test_get_nonexistent_api_key(self, catalyst_instance):
        assert catalyst_instance.get_api_key("nonexistent") is None

    def test_upload_api_keys(self, catalyst_instance):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            catalyst_instance.api_keys = {"openai": "test-key-123"}
            catalyst_instance._upload_keys()
            
            mock_post.assert_called_once()

# Test project operations
class TestRagaAICatalystProjects:
    def test_create_project_success(self, catalyst_instance):
        with patch('requests.post') as mock_post, \
             patch.object(catalyst_instance, 'list_projects') as mock_list_projects, \
             patch.object(catalyst_instance, 'project_use_cases') as mock_use_cases:
            
            mock_list_projects.return_value = []
            mock_use_cases.return_value = ["Q/A"]
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "data": {"name": "test_project"}
            }
            
            result = catalyst_instance.create_project("test_project", usecase="Q/A")
            assert "Successfully" in result

    def test_create_project_duplicate(self, catalyst_instance):
        with patch.object(catalyst_instance, 'list_projects') as mock_list_projects:
            mock_list_projects.return_value = ["test_project"]
            
            with pytest.raises(ValueError) as exc_info:
                catalyst_instance.create_project("test_project")
            assert "already exists" in str(exc_info.value)

    def test_create_project_invalid_usecase(self, catalyst_instance):
        with patch.object(catalyst_instance, 'list_projects') as mock_list_projects, \
             patch.object(catalyst_instance, 'project_use_cases') as mock_use_cases:
            
            mock_list_projects.return_value = []
            mock_use_cases.return_value = ["Q/A"]
            
            with pytest.raises(ValueError) as exc_info:
                catalyst_instance.create_project("test_project", usecase="invalid")
            assert "valid usecase" in str(exc_info.value)

    def test_list_projects(self, catalyst_instance):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "data": {
                    "content": [
                        {"name": "project1"},
                        {"name": "project2"}
                    ]
                }
            }
            
            projects = catalyst_instance.list_projects()
            assert projects == ["project1", "project2"]

# Test metrics operations
class TestRagaAICatalystMetrics:
    def test_list_metrics(self, catalyst_instance):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "data": {
                    "metrics": [
                        {"name": "metric1"},
                        {"name": "metric2"}
                    ]
                }
            }
            
            metrics = catalyst_instance.list_metrics()
            assert metrics == ["metric1", "metric2"]

# Test URL normalization
class TestRagaAICatalystUtils:
    def test_normalize_base_url(self):
        test_cases = [
            ("https://example.com", "https://example.com/api"),
            ("https://example.com/", "https://example.com/api"),
            ("https://example.com//", "https://example.com/api"),
            ("https://example.com/api", "https://example.com/api"),
            ("https://example.com/api/", "https://example.com/api"),
        ]
        
        for input_url, expected_url in test_cases:
            assert RagaAICatalyst._normalize_base_url(input_url) == expected_url