import pytest
import os
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

# Test responses from tracer_responses.json
TEST_RESPONSES = {
    "initialization": {
        "success": True,
        "tracer_attrs": {
            "project_name": "agentic_tracer_sk_v3",
            "dataset_name": "pytest_dataset",
            "tracer_type": "agentic/crewai",
            "timeout": 120,
            "project_id": 767
        }
    },
    "set_model_cost": {
        "success": True,
        "model_cost": {
            "gpt-4": {
                "input_cost_per_token": 6e-06,
                "output_cost_per_token": 2.4e-06
            }
        }
    },
    "set_dataset_name": {
        "success": True,
        "original_dataset": "pytest_dataset",
        "new_dataset": "pytest_dataset_new"
    },
    "add_context": {
        "success": False,
        "error": "add_context is only supported for 'langchain' and 'llamaindex' tracer types"
    }
}

@pytest.fixture(scope="module")
def setup_environment():
    load_dotenv()
    catalyst = RagaAICatalyst(
        access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'),
        secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'),
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    return catalyst

@pytest.fixture
def tracer(setup_environment):
    tracer = Tracer(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        tracer_type="agentic/crewai",
    )
    init_tracing(catalyst=setup_environment, tracer=tracer)
    return tracer

def test_tracer_initialization(tracer):
    expected_attrs = TEST_RESPONSES["initialization"]["tracer_attrs"]
    assert tracer.project_name == expected_attrs["project_name"]
    assert tracer.dataset_name == expected_attrs["dataset_name"]
    assert tracer.tracer_type == expected_attrs["tracer_type"]
    assert tracer.timeout == expected_attrs["timeout"]

def test_set_model_cost(tracer):
    cost_config = {
        "model_name": "gpt-4",
        "input_cost_per_million_token": 6,
        "output_cost_per_million_token": 2.40
    }
    tracer.set_model_cost(cost_config)
    assert tracer.model_custom_cost == TEST_RESPONSES["set_model_cost"]["model_cost"]

def test_set_dataset_name(tracer):
    new_dataset = "pytest_dataset_new"
    tracer.set_dataset_name(new_dataset)
    assert tracer.dataset_name == TEST_RESPONSES["set_dataset_name"]["new_dataset"]

def test_add_context_unsupported(tracer):
    with pytest.raises(ValueError) as exc_info:
        tracer.add_context("test context")
    assert str(exc_info.value) == TEST_RESPONSES["add_context"]["error"]