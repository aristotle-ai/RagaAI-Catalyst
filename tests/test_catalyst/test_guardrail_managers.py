import pytest
from unittest.mock import patch, MagicMock
import os

from ragaai_catalyst.guardrails_manager import GuardrailsManager

@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "fake-token"}):
        yield

@pytest.fixture
def mock_project_list():
    return {
        "data": {
            "content": [
                {"id": "proj-1", "name": "TestProject"},
                {"id": "proj-2", "name": "OtherProject"}
            ]
        }
    }

@pytest.fixture
def mock_guardrails_manager(mock_env, mock_project_list):
    with patch("ragaai_catalyst.guardrails_manager.requests.request") as mock_request:
        # Patch project list for __init__
        mock_request.return_value.json.return_value = mock_project_list
        mock_request.return_value.status_code = 200
        yield GuardrailsManager("TestProject")

@patch("ragaai_catalyst.guardrails_manager.requests.request")
def test_init_project_not_found(mock_request, mock_env, mock_project_list):
    # Remove TestProject to trigger ValueError
    mock_project_list["data"]["content"] = [{"id": "proj-2", "name": "OtherProject"}]
    mock_request.return_value.json.return_value = mock_project_list
    with pytest.raises(ValueError):
        GuardrailsManager("TestProject")

@patch("ragaai_catalyst.guardrails_manager.requests.request")
def test_list_deployment_ids(mock_request, mock_guardrails_manager):
    mock_request.return_value.json.return_value = {
        "data": {
            "content": [
                {"id": "dep-1", "name": "Deployment1"},
                {"id": "dep-2", "name": "Deployment2"}
            ]
        }
    }
    result = mock_guardrails_manager.list_deployment_ids()
    assert result == [{"id": "dep-1", "name": "Deployment1"}, {"id": "dep-2", "name": "Deployment2"}]

@patch("ragaai_catalyst.guardrails_manager.requests.request")
def test_get_deployment_success(mock_request, mock_guardrails_manager):
    mock_request.return_value.json.return_value = {"success": True, "data": {"name": "Deployment1", "guardrailsResponse": []}}
    result = mock_guardrails_manager.get_deployment("dep-1")
    assert result["success"]

@patch("ragaai_catalyst.guardrails_manager.requests.request")
def test_get_deployment_failure(mock_request, mock_guardrails_manager):
    mock_request.return_value.json.return_value = {"success": False, "message": "fail"}
    result = mock_guardrails_manager.get_deployment("dep-1")
    assert result is None

@patch("ragaai_catalyst.guardrails_manager.requests.request")
def test_list_guardrails(mock_request, mock_guardrails_manager):
    mock_request.return_value.json.return_value = {
        "data": {
            "metrics": [
                {"name": "Guardrail1"},
                {"name": "Guardrail2"}
            ]
        }
    }
    result = mock_guardrails_manager.list_guardrails()
    assert result == ["Guardrail1", "Guardrail2"]

@patch("ragaai_catalyst.guardrails_manager.requests.request")
def test_list_fail_condition(mock_request, mock_guardrails_manager):
    mock_request.return_value.json.return_value = {"data": ["fail1", "fail2"]}
    result = mock_guardrails_manager.list_fail_condition()
    assert result == ["fail1", "fail2"]

@patch("ragaai_catalyst.guardrails_manager.requests.post")
@patch("ragaai_catalyst.guardrails_manager.response_checker")
def test_list_datasets(mock_checker, mock_post, mock_guardrails_manager):
    mock_post.return_value.json.return_value = {
        "data": {
            "content": [
                {"name": "Dataset1"},
                {"name": "Dataset2"}
            ]
        }
    }
    mock_post.return_value.status_code = 200
    result = mock_guardrails_manager.list_datasets()
    assert result == ["Dataset1", "Dataset2"]

# @patch("ragaai_catalyst.guardrails_manager.requests.request")
# @patch("ragaai_catalyst.guardrails_manager.GuardrailsManager.list_datasets")
# def test_create_deployment(mock_list_datasets, mock_request, mock_guardrails_manager):
#     # This function will return the correct response for each call in order
#     def json_side_effect(*args, **kwargs):
#         if not hasattr(json_side_effect, "call_count"):
#             json_side_effect.call_count = 0
#         json_side_effect.call_count += 1
#         if json_side_effect.call_count == 1:
#             return {"data": {"content": []}}  # list_deployment_ids
#         elif json_side_effect.call_count == 2:
#             return {"data": {"content": [{"name": "Dataset1"}]}}  # list_datasets
#         elif json_side_effect.call_count == 3:
#             return {"message": "Created"}  # POST create - removed success key to match actual API
#         elif json_side_effect.call_count == 4:
#             return {"data": {"content": [{"id": "dep-1", "name": "NewDeployment"}]}}  # list_deployment_ids again
#         else:
#             return {}

#     # Set status code to 201 (Created) for the POST request to create a deployment
#     def status_code_side_effect(*args, **kwargs):
#         if args and args[0] == "POST" and "/guardrail/deployment" in args[1]:
#             return 201
#         return 200
        
#     mock_request.return_value.status_code = property(lambda self: status_code_side_effect(*mock_request.call_args[0]))
#     mock_request.return_value.json.side_effect = json_side_effect
#     mock_list_datasets.return_value = ["Dataset1"]
#     dep_id = mock_guardrails_manager.create_deployment("NewDeployment", "Dataset1")
#     assert dep_id == "dep-1"

@patch("ragaai_catalyst.guardrails_manager.requests.request")
@patch("ragaai_catalyst.guardrails_manager.GuardrailsManager.get_deployment")
@patch("ragaai_catalyst.guardrails_manager.GuardrailsManager.list_guardrails")
def test_add_guardrails(mock_list_guardrails, mock_get_deployment, mock_request, mock_guardrails_manager):
    mock_get_deployment.return_value = {
        "data": {
            "name": "Deployment1",
            "guardrailsResponse": []
        }
    }
    mock_list_guardrails.return_value = ["GuardrailType"]
    mock_request.return_value.json.return_value = {"success": True, "message": "Added"}
    guardrails = [{"name": "GuardrailType", "displayName": "GuardrailType"}]
    mock_guardrails_manager.add_guardrails("dep-1", guardrails)

def test_get_guardrail_config_payload(mock_guardrails_manager):
    config = {"isActive": True, "guardrailFailConditions": ["FAIL"], "deploymentFailCondition": "ALL_FAIL", "alternateResponse": "Alt"}
    payload = mock_guardrails_manager._get_guardrail_config_payload(config)
    assert payload["isActive"] is True
    assert payload["failAction"]["args"] == '{"alternateResponse": "Alt"}'

def test_get_guardrail_list_payload_and_one_guardrail_data(mock_guardrails_manager):
    guardrail = {"name": "G", "displayName": "G", "config": {"mappings": [{"schemaName": "Prompt", "variableName": "Prompt"}]}}
    payload = mock_guardrails_manager._get_guardrail_list_payload([guardrail])
    assert isinstance(payload, list)
    assert payload[0]["name"] == "G"
    one = mock_guardrails_manager._get_one_guardrail_data(guardrail)
    assert one["name"] == "G"