import pytest
import os
import tempfile
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst.tracers.exporters.dynamic_trace_exporter import DynamicTraceExporter

# Test responses from dynamic_trace_exporter_responses.json
TEST_RESPONSES = {
    "initialization": {
        "success": True,
        "exporter_attrs": {
            "files_to_zip": 3,
            "project_name": "agentic_tracer_sk_v3",
            "dataset_name": "pytest_dataset",
            "base_url": "https://catalyst.raga.ai/api",
            "max_upload_workers": 30
        }
    },
    "property_update": {
        "success": True,
        "updated_values": {
            "dataset_name": {
                "original": "pytest_dataset",
                "updated": "pytest_dataset_new"
            },
            "project_name": {
                "original": "agentic_tracer_sk_v3",
                "updated": "new_project_name"
            }
        }
    },
    "user_context_update": {
        "success": True,
        "value": "test context"
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
def mock_files():
    temp_dir = tempfile.mkdtemp()
    files = []
    for i in range(3):
        file_path = os.path.join(temp_dir, f"test_file_{i}.py")
        with open(file_path, 'w') as f:
            f.write(f'print("Hello World from file {i}")\n')
        files.append(file_path)
    return files

@pytest.fixture
def exporter(setup_environment, mock_files):
    tracer = Tracer(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        tracer_type="agentic/crewai",
    )
    init_tracing(catalyst=setup_environment, tracer=tracer)
    
    return DynamicTraceExporter(
        tracer_type="agentic/crewai",
        files_to_zip=mock_files,
        project_name='agentic_tracer_sk_v3',
        project_id=tracer.project_id,
        dataset_name='pytest_dataset',
        user_details=tracer._pass_user_data(),
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL'),
        custom_model_cost=None,
        timeout=120
    )

def test_initialization(exporter):
    expected_attrs = TEST_RESPONSES["initialization"]["exporter_attrs"]
    assert len(exporter.files_to_zip) == expected_attrs["files_to_zip"]
    assert exporter.project_name == expected_attrs["project_name"]
    assert exporter.dataset_name == expected_attrs["dataset_name"]
    assert exporter.max_upload_workers == expected_attrs["max_upload_workers"]

def test_property_updates(exporter):
    # Test dataset_name update
    original_dataset = exporter.dataset_name
    exporter.dataset_name = "pytest_dataset_new"
    assert original_dataset == TEST_RESPONSES["property_update"]["updated_values"]["dataset_name"]["original"]
    assert exporter.dataset_name == TEST_RESPONSES["property_update"]["updated_values"]["dataset_name"]["updated"]

    # Test project_name update
    original_project = exporter.project_name
    exporter.project_name = "new_project_name"
    assert original_project == TEST_RESPONSES["property_update"]["updated_values"]["project_name"]["original"]
    assert exporter.project_name == TEST_RESPONSES["property_update"]["updated_values"]["project_name"]["updated"]

def test_user_context_update(exporter):
    exporter.user_context = "test context"
    assert exporter.user_context == TEST_RESPONSES["user_context_update"]["value"]