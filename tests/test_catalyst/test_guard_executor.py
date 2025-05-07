import pytest
from unittest.mock import MagicMock, patch
import os

from ragaai_catalyst.guard_executor import GuardExecutor

@pytest.fixture
def mock_guard_manager():
    gm = MagicMock()
    gm.base_url = "http://fake-url"
    gm.project_id = "proj-123"
    gm.timeout = 5
    gm.get_deployment.side_effect = lambda x: {
        "data": {
            "datasetId": "ds-1",
            "guardrailsResponse": [
                {
                    "metricSpec": {
                        "config": {
                            "mappings": [
                                {"schemaName": "Prompt"},
                                {"schemaName": "Context"}
                            ]
                        }
                    }
                }
            ]
        }
    }
    return gm

@pytest.fixture
def executor(mock_guard_manager):
    return GuardExecutor(
        guard_manager=mock_guard_manager,
        input_deployment_id="input-id",
        output_deployment_id="output-id",
        field_map={"prompt": "prompt", "context": "context"}
    )

@patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "fake-token"})
@patch("requests.request")
def test_execute_deployment_success(mock_request, executor):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": {"status": "PASS", "results": [{"executionId": "trace-1"}]}}
    mock_request.return_value = mock_response

    payload = {"prompt": "test"}
    result = executor.execute_deployment("input-id", payload)
    assert result["success"]

@patch.dict(os.environ, {"RAGAAI_CATALYST_TOKEN": "fake-token"})
@patch("requests.request")
def test_execute_deployment_failure(mock_request, executor):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"success": False, "message": "fail"}
    mock_request.return_value = mock_response

    payload = {"prompt": "test"}
    result = executor.execute_deployment("input-id", payload)
    assert result is None

@patch("ragaai_catalyst.guard_executor.litellm.completion")
def test_llm_executor_litellm(mock_completion, executor):
    mock_completion.return_value = {"choices": [MagicMock(message=MagicMock(content="response"))]}
    result = executor.llm_executor("prompt", {}, "litellm")
    assert result == "response"

@patch("ragaai_catalyst.guard_executor.genai.Client")
def test_llm_executor_genai(mock_genai_client, executor):
    mock_instance = mock_genai_client.return_value
    mock_instance.models.generate.return_value = MagicMock(text="genai response")
    result = executor.llm_executor("prompt", {}, "genai")
    assert result == "genai response"

def test_set_input_params(executor):
    executor.set_input_params(prompt="p", context="c", instruction="i")
    assert executor.id_2_doc["latest"]["prompt"] == "p"
    assert executor.id_2_doc["latest"]["context"] == "c"
    assert executor.id_2_doc["latest"]["instruction"] == "i"

@patch.object(GuardExecutor, "execute_input_guardrails")
@patch.object(GuardExecutor, "llm_executor")
@patch.object(GuardExecutor, "execute_output_guardrails")
def test_call_success(mock_output, mock_llm, mock_input, executor):
    mock_input.return_value = (None, {"data": {"status": "PASS"}})
    mock_llm.return_value = "llm response"
    mock_output.return_value = (None, {"data": {"status": "PASS"}})
    alt, llm_resp, out = executor("prompt", {"context": "ctx"}, {}, "litellm")
    assert llm_resp == "llm response"

def test_set_variables(executor):
    doc = executor.set_variables("prompt", {"context": "ctx"})
    assert doc["prompt"] == "prompt"
    assert doc["context"] == "ctx"

@patch.object(GuardExecutor, "execute_deployment")
def test_execute_input_guardrails(mock_exec, executor):
    mock_exec.return_value = {"data": {"status": "PASS", "results": [{"executionId": "trace-1"}]}}
    alt, resp = executor.execute_input_guardrails("prompt", {"context": "ctx"})
    assert resp["data"]["status"] == "PASS"

@patch.object(GuardExecutor, "execute_deployment")
def test_execute_output_guardrails(mock_exec, executor):
    mock_exec.return_value = {"data": {"status": "PASS"}}
    executor.current_trace_id = "trace-1"
    executor.id_2_doc["trace-1"] = {"prompt": "p", "context": "c"}
    alt, resp = executor.execute_output_guardrails("llm response")
    assert resp["data"]["status"] == "PASS"