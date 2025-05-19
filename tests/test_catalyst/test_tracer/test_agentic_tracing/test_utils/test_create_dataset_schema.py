import pytest
import os
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst.tracers.agentic_tracing.utils.create_dataset_schema import create_dataset_schema_with_trace

@pytest.fixture(scope="module")
def setup_environment():
    # Load environment variables
    load_dotenv()
    
    # Initialize catalyst and tracer
    catalyst = RagaAICatalyst(
        access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'),
        secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'),
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    
    tracer = Tracer(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        tracer_type="agentic/crewai",
    )
    
    init_tracing(catalyst=catalyst, tracer=tracer)
    return catalyst

def test_create_dataset_schema_with_trace(setup_environment):
    # Arrange
    project_name = "agentic_tracer_sk_v3"
    dataset_name = "pytest_dataset2"
    
    # Act
    response = create_dataset_schema_with_trace(
        project_name=project_name,
        dataset_name=dataset_name
    )
    # import pdb; pdb.set_trace()
    
    # Assert
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"