import pytest
import os
import json
import time
from ragaai_catalyst.guard_executor import GuardExecutor
from ragaai_catalyst.guardrails_manager import GuardrailsManager
from ragaai_catalyst import RagaAICatalyst

# Skip tests if credentials are not available
pytest.importorskip("dotenv").load_dotenv()

# Project configuration for tests
PROJECT_NAME = os.getenv("TEST_PROJECT_NAME", "guard")
DATASET_NAME = os.getenv("TEST_DATASET_NAME", "abcd")
INPUT_DEPLOYMENT_NAME = "input"
OUTPUT_DEPLOYMENT_NAME = "output"

@pytest.fixture(scope="module")
def raga_catalyst():
    """Initialize RagaAICatalyst client for testing"""
    access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
    secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    
    if not access_key or not secret_key:
        pytest.skip("API credentials not found in environment variables")
    
    return RagaAICatalyst(
        access_key=access_key,
        secret_key=secret_key
    )

@pytest.fixture(scope="module")
def guard_manager(raga_catalyst):
    """Create a real guard manager for testing"""
    try:
        return GuardrailsManager(PROJECT_NAME)
    except ValueError as e:
        pytest.skip(f"Error initializing GuardrailsManager: {e}")

@pytest.fixture(scope="module")
def deployments(guard_manager):
    """Create test deployments for input and output guardrails"""
    input_deployment_id = 381
    output_deployment_id = 382
    
    # Check if deployments already exist
    deployments = guard_manager.list_deployment_ids()
    deployment_names = {d["name"]: d["id"] for d in deployments}
    
    if INPUT_DEPLOYMENT_NAME in deployment_names:
        input_deployment_id = deployment_names[INPUT_DEPLOYMENT_NAME]
    else:
        try:
            input_deployment_id = guard_manager.create_deployment(INPUT_DEPLOYMENT_NAME, DATASET_NAME)
        except Exception as e:
            pytest.skip(f"Failed to create input deployment: {e}")
    
    if OUTPUT_DEPLOYMENT_NAME in deployment_names:
        output_deployment_id = deployment_names[OUTPUT_DEPLOYMENT_NAME]
    else:
        try:
            output_deployment_id = guard_manager.create_deployment(OUTPUT_DEPLOYMENT_NAME, DATASET_NAME)
        except Exception as e:
            pytest.skip(f"Failed to create output deployment: {e}")
    
    # Add some basic guardrails to both deployments
    input_guardrails = [
        {
            "displayName": "Test Prompt Quality",
            "name": "PromptQuality",
            "config": {
                "mappings": [
                    {"schemaName": "Prompt", "variableName": "Prompt"},
                    {"schemaName": "Context", "variableName": "Context"}
                ]
            }
        }
    ]
    
    output_guardrails = [
        {
            "displayName": "Test Response Quality",
            "name": "ResponseQuality",
            "config": {
                "mappings": [
                    {"schemaName": "Prompt", "variableName": "Prompt"},
                    {"schemaName": "Context", "variableName": "Context"},
                    {"schemaName": "Response", "variableName": "Response"}
                ]
            }
        }
    ]
    
    # Configure guardrails with basic settings
    guardrails_config = {
        "isActive": True,
        "guardrailFailConditions": ["FAIL"],
        "deploymentFailCondition": "ONE_FAIL",
        "alternateResponse": "This is an alternate response due to guardrail failure"
    }
    
    try:
        # Add guardrails to deployments if they don't already have them
        input_deployment = guard_manager.get_deployment(input_deployment_id)
        if not input_deployment["data"]["guardrailsResponse"]:
            guard_manager.add_guardrails(input_deployment_id, input_guardrails, guardrails_config)
        
        output_deployment = guard_manager.get_deployment(output_deployment_id)
        if not output_deployment["data"]["guardrailsResponse"]:
            guard_manager.add_guardrails(output_deployment_id, output_guardrails, guardrails_config)
    except Exception as e:
        pytest.skip(f"Failed to configure guardrails: {e}")
    
    return {"input_id": input_deployment_id, "output_id": output_deployment_id}

@pytest.fixture
def executor(guard_manager, deployments):
    """Create a GuardExecutor instance with real deployments"""
    return GuardExecutor(
        guard_manager=guard_manager,
        input_deployment_id=deployments["input_id"],
        output_deployment_id=deployments["output_id"],
        field_map={"prompt": "prompt", "context": "context"}
    )

def test_execute_deployment_success(executor):
    """Test successful execution of a deployment with real API call"""
    payload = {"prompt": "Test prompt for guardrail evaluation", "context": "This is a context for testing"}
    result = executor.execute_deployment(executor.input_deployment_id, payload)
    
    assert result is not None
    assert result["success"]
    assert "data" in result
    assert "status" in result["data"]
    assert result["data"]["status"] in ["Pass", "FAIL"]

def test_set_input_params(executor):
    """Test setting input parameters"""
    executor.set_input_params(prompt="Test prompt", context="Test context", instruction="Test instruction")
    
    assert executor.id_2_doc["latest"]["prompt"] == "Test prompt"
    assert executor.id_2_doc["latest"]["context"] == "Test context"
    assert executor.id_2_doc["latest"]["instruction"] == "Test instruction"

def test_set_variables(executor):
    """Test setting variables from prompt and params"""
    doc = executor.set_variables("Test prompt", {"context": "Test context"})
    
    assert doc["prompt"] == "Test prompt"
    assert doc["context"] == "Test context"

def test_execute_input_guardrails(executor):
    """Test execution of input guardrails with real API call"""
    alt, resp = executor.execute_input_guardrails(
        "Test prompt for input guardrails", 
        {"context": "This is a context for input guardrails testing"}
    )
    
    assert resp is not None
    assert "data" in resp
    assert "status" in resp["data"]
    assert resp["data"]["status"] in ["Pass", "Fail"]
    
    # If the guardrail failed, check that we have an alternate response
    if resp["data"]["status"] == "Fail":
        assert alt is not None

def test_execute_output_guardrails(executor):
    """Test execution of output guardrails with real API call"""
    # First run input guardrails to set up the trace ID
    _, input_resp = executor.execute_input_guardrails(
        "Test prompt for output guardrails", 
        {"context": "This is a context for output guardrails testing"}
    )
    
    assert input_resp is not None
    
    # Now test output guardrails
    alt, resp = executor.execute_output_guardrails("This is a test LLM response")
    
    assert resp is not None
    assert "data" in resp
    assert "status" in resp["data"]
    assert resp["data"]["status"] in ["Pass", "Fail"]
    
    # If the guardrail failed, check that we have an alternate response
    if resp["data"]["status"] == "Fail":
        assert alt is not None

def test_call_success(executor):
    """Test the __call__ method with real API calls"""
    prompt = "Test prompt for full execution"
    prompt_params = {"context": "This is a context for full execution testing"}
    model_params = {
        "model": "gpt-4o-mini",
        "max_tokens": 100
    }
    
    # Skip this test if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment variables")
        
    # Set up LiteLLM environment
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    alt, llm_resp, out = executor(prompt, prompt_params, model_params, "litellm")
    
    assert out is not None
    assert "data" in out
    assert "status" in out["data"]
    assert out["data"]["status"] in ["Pass", "Fail"]
    
    # If input guardrails passed, we should have an LLM response
    if out["data"]["status"] == "Pass" or llm_resp is not None:
        assert llm_resp is not None
        assert isinstance(llm_resp, str)

def test_llm_executor_litellm(executor):
    """Test the LLM executor with LiteLLM"""
    # Skip this test if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment variables")
        
    # Set up LiteLLM environment
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    model_params = {
        "model": "gpt-4o-mini",
        "max_tokens": 100
    }
    
    result = executor.llm_executor("What is the capital of France?", model_params, "litellm")
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
